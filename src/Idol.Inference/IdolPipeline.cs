using System.Numerics;
using System.Text.Json;
using Idol.Core;
using Idol.Inference.Encoders;
using Idol.Inference.Imaging;
using Idol.Inference.Models;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference;

/// <summary>The complete local C# pipeline. Each phase releases its weights before the next phase loads.</summary>
public sealed class IdolPipeline : IDisposable
{
    private readonly string modelDirectory;
    private readonly Device device;
    private readonly bool useHalf;
    private bool disposed;
    // LibTorch's RNG is process-wide. Serialize generations to preserve the selected seed.
    private static readonly SemaphoreSlim GenerationGate = new(1, 1);

    private IdolPipeline(string directory, Device device, bool useHalf)
    {
        modelDirectory = directory;
        this.device = device;
        this.useHalf = useHalf;
    }

    public static async Task<IdolPipeline> LoadAsync(ModelSetupOptions options,
        IProgress<InferenceProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        progress?.Report(new("Checking model files", 0, "Validating access and SHA-256 checksums"));
        using var downloader = new ModelDownloader(options.HuggingFaceToken);
        await downloader.EnsureAsync(options.ModelDirectory, options.DownloadMissing, progress, cancellationToken).ConfigureAwait(false);
        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            progress?.Report(new("Starting native runtime", 1, "Loading LibTorch; the first launch can take a moment"));
            if (!options.UseCpu && !cuda.is_available())
                throw new InvalidOperationException("CUDA is unavailable. Install a current NVIDIA driver or select CPU in Setup. CPU inference is substantially slower.");
            var target = options.UseCpu ? CPU : CUDA;
            // Initialize eagerly, before beginning image processing, so missing native dependencies are actionable.
            using var check = zeros([1L], device: target);
            return new IdolPipeline(Path.GetFullPath(options.ModelDirectory), target, !options.UseCpu);
        }, cancellationToken).ConfigureAwait(false);
    }

    public async Task<IdolResult> GenerateAsync(string imagePath, GenerationOptions options,
        IProgress<InferenceProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (!File.Exists(imagePath)) throw new FileNotFoundException("Select an existing image.", imagePath);
        await GenerationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await Task.Run(() => GenerateCore(imagePath, options, progress, cancellationToken), cancellationToken).ConfigureAwait(false);
        }
        finally { GenerationGate.Release(); }
    }

    private IdolResult GenerateCore(string imagePath, GenerationOptions options,
        IProgress<InferenceProgress>? progress, CancellationToken ct)
    {
        using var noGrad = no_grad();
        using var scope = NewDisposeScope();
        using var config = JsonDocument.Parse(File.ReadAllText(PathFor("idol/config.json")));
        var root = config.RootElement;
        progress?.Report(new("Preparing image", 0, "Segmenting the object and framing it on black"));
        RasterImage prepared;
        if (ImagePreprocessor.NeedsBackgroundRemovalAsync(imagePath, ct).GetAwaiter().GetResult())
        {
            using var weights = WeightStore.Load(PathFor("idol/birefnet.safetensors"), device,
                _ => useHalf ? ScalarType.Float16 : ScalarType.Float32, ct);
            prepared = ImagePreprocessor.PrepareAsync(imagePath, new BiRefNetMask(weights), options.RealPhoto, ct).GetAwaiter().GetResult();
        }
        else prepared = ImagePreprocessor.PrepareAsync(imagePath, null, options.RealPhoto, ct).GetAwaiter().GetResult();
        using (prepared)
        {
            var outputDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Idol", "outputs", DateTime.UtcNow.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N")[..8]);
            Directory.CreateDirectory(outputDirectory);
            var preparedPath = Path.Combine(outputDirectory, "input.png");
            ct.ThrowIfCancellationRequested();
            prepared.SavePng(preparedPath);
            var pixels = ImagePreprocessor.ToTensor(prepared, device);
            manual_seed(options.Seed);

            Tensor feature1, feature2;
            progress?.Report(new("Encoding image", .1, "DINOv3 ViT-H+"));
            using (var weights = WeightStore.Load(PathFor("dinov3/model.safetensors"), device,
                _ => useHalf ? ScalarType.BFloat16 : ScalarType.Float32, ct))
                feature1 = new DinoV3Encoder(weights, PathFor("dinov3/config.json")).Encode(pixels, ct);

            ct.ThrowIfCancellationRequested();
            progress?.Report(new("Encoding image", .2, "FLUX.2 VAE"));
            using (var weights = WeightStore.Load(PathFor("flux2/vae/diffusion_pytorch_model.safetensors"), device,
                _ => useHalf ? ScalarType.BFloat16 : ScalarType.Float32, ct))
                feature2 = new Flux2VaeEncoder(weights, PathFor("flux2/vae/config.json")).Encode(pixels, ct);
            pixels.Dispose();
            if (feature1.shape[1] != feature2.shape[1])
                throw new InvalidDataException($"Encoder token counts differ: DINOv3={feature1.shape[1]}, FLUX.2={feature2.shape[1]}. Check the pinned model files.");

            Tensor latent, cameraLatent;
            progress?.Report(new("Generating 3D", .25, "Loading the Idol diffusion transformer"));
            using (var weights = WeightStore.Load(PathFor("idol/denoiser.safetensors"), device,
                name => IdolDenoiser.CheckpointType(name, useHalf), ct))
            {
                var denoiser = new IdolDenoiser(weights, root.GetProperty("denoiser").GetProperty("args"));
                (latent, cameraLatent) = FlowSampler.Sample(denoiser, feature1, feature2, options.Seed, options.Steps,
                    options.Guidance, options.TimeShift,
                    (step, total) => progress?.Report(new("Generating 3D", .25 + .5 * step / total, $"Sampling step {step} / {total}")),
                    ct, resetSeed: false);
            }
            feature1.Dispose(); feature2.Dispose();
            var camera = CameraDecoder.Decode(cameraLatent);
            cameraLatent.Dispose();

            progress?.Report(new("Placing Gaussians", .77, "Sampling the octree"));
            Tensor anchors;
            using (var weights = WeightStore.Load(PathFor("idol/decoder.safetensors"), device,
                name => IdolDenoiser.CheckpointType(name, useHalf), ct))
            {
                var decoder = new OctreeDecoder(weights, root.GetProperty("decoder").GetProperty("args"));
                anchors = decoder.Sample(latent, options.NumGaussians / 32, root.GetProperty("max_voxel_level").GetInt32(), ct);
            }
            progress?.Report(new("Decoding Gaussians", .88, $"Building {options.NumGaussians:N0} colored Gaussians"));
            GaussianCloud cloud;
            using (var weights = WeightStore.Load(PathFor("idol/decoder_gs.safetensors"), device,
                name => IdolDenoiser.CheckpointType(name, useHalf), ct))
            {
                var decoder = new GaussianDecoder(weights, root.GetProperty("decoder_gs").GetProperty("args"));
                using var gaussians = decoder.Decode(anchors, latent, ct);
                cloud = ToCloud(gaussians, ct);
            }
            var info = new CameraInfo(camera.Yaw, camera.Pitch, camera.Radius, camera.Fov);
            File.WriteAllText(Path.Combine(outputDirectory, "camera.json"), JsonSerializer.Serialize(info, new JsonSerializerOptions { WriteIndented = true }));
            progress?.Report(new("Complete", 1, $"{cloud.Count:N0} Gaussians ready"));
            return new IdolResult(cloud, preparedPath, info);
        }
    }

    internal static GaussianCloud ToCloud(GaussianTensors tensors, CancellationToken ct)
    {
        using var scope = NewDisposeScope();
        var xyz = tensors.Xyz.to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();
        var dc = tensors.FeaturesDc.to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();
        var scales = tensors.Scaling.to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();
        var rotations = tensors.Rotation.to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();
        var opacity = tensors.Opacity.to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();
        var points = new Gaussian[opacity.Length];
        // Native Z-up -> desktop/Unity Y-up. The SDK's +90 X export convention
        // points positive native Z downward in these viewers; use -90 X here.
        var x90 = Quaternion.CreateFromAxisAngle(Vector3.UnitX, -MathF.PI / 2);
        for (int i = 0; i < points.Length; i++)
        {
            if ((i & 4095) == 0) ct.ThrowIfCancellationRequested();
            float alpha = Math.Clamp(opacity[i], 1e-6f, 1 - 1e-6f);
            var rotation = new Quaternion(rotations[i * 4 + 1], rotations[i * 4 + 2], rotations[i * 4 + 3], rotations[i * 4]);
            points[i] = new Gaussian(new(xyz[i * 3], xyz[i * 3 + 2], -xyz[i * 3 + 1]),
                new(dc[i * 3], dc[i * 3 + 1], dc[i * 3 + 2]), MathF.Log(alpha / (1 - alpha)),
                new(MathF.Log(Math.Max(scales[i * 3], 1e-6f)), MathF.Log(Math.Max(scales[i * 3 + 1], 1e-6f)), MathF.Log(Math.Max(scales[i * 3 + 2], 1e-6f))),
                Quaternion.Normalize(x90 * rotation));
            if (!points[i].IsFinite) throw new InvalidDataException($"The model generated a non-finite Gaussian at index {i}. Try CPU mode to diagnose device precision support.");
        }
        return new GaussianCloud(points);
    }

    private string PathFor(string relative) => Path.Combine(modelDirectory, relative.Replace('/', Path.DirectorySeparatorChar));
    public void Dispose() => disposed = true;
}
