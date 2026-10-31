using System.Diagnostics;
using Idol.Inference.Encoders;
using Idol.Inference.Imaging;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Cli;

/// <summary>Exercises the real published BiRefNet checkpoint independently of the gated image encoders.</summary>
public static class BackgroundCheck
{
    public static Task<int> RunAsync(string imagePath, string modelDirectory, string outputDirectory,
        bool cpu, CancellationToken cancellationToken = default) => Task.Run(() =>
            Run(imagePath, modelDirectory, outputDirectory, cpu, cancellationToken), cancellationToken);

    private static int Run(string imagePath, string modelDirectory, string outputDirectory,
        bool cpu, CancellationToken cancellationToken)
    {
        string path = Path.Combine(modelDirectory, "idol", "birefnet.safetensors");
        if (!File.Exists(path)) throw new FileNotFoundException("Run model setup to download BiRefNet first.", path);
        var device = cpu ? CPU : CUDA;
        if (!cpu && !cuda.is_available()) throw new InvalidOperationException("CUDA is unavailable. Pass --cpu or install the CUDA build.");
        Directory.CreateDirectory(outputDirectory);
        var watch = Stopwatch.StartNew();
        Console.WriteLine($"Loading published BiRefNet weights on {device}...");
        using var weights = WeightStore.Load(path, device, _ => cpu ? ScalarType.Float32 : ScalarType.Float16, cancellationToken);
        var network = new BiRefNetMask(weights);
        using var noGrad = no_grad();
        using var scope = NewDisposeScope();
        using var source = RasterImage.Load(imagePath);
        var input = ImagePreprocessor.ToTensor(source, device);
        Console.WriteLine("Predicting the foreground alpha map...");
        var mask = network.Predict(input, cancellationToken);
        if (!isfinite(mask).all().item<bool>()) throw new InvalidDataException("Background model returned non-finite values.");
        double minimum = mask.min().item<float>(), maximum = mask.max().item<float>();
        var data = (mask.clamp(0, 1) * 255).to_type(ScalarType.Byte).cpu().contiguous().data<byte>().ToArray();
        using var alpha = new RasterImage(source.Width, source.Height);
        for (int i = 0; i < data.Length; i++)
        {
            alpha.Pixels[i * 4] = data[i]; alpha.Pixels[i * 4 + 1] = data[i];
            alpha.Pixels[i * 4 + 2] = data[i]; alpha.Pixels[i * 4 + 3] = 255;
        }
        string maskPath = Path.Combine(outputDirectory, "foreground-mask.png");
        alpha.SavePng(maskPath);
        for (int i = 0; i < data.Length; i++) source.Pixels[i * 4 + 3] = data[i];
        using var prepared = ImagePreprocessor.FrameAlpha(source);
        string preparedPath = Path.Combine(outputDirectory, "prepared-input.png");
        prepared.SavePng(preparedPath);
        Console.WriteLine($"BiRefNet passed: alpha range [{minimum:G6}, {maximum:G6}], shape [{string.Join(",", mask.shape)}], elapsed {watch.Elapsed.TotalSeconds:F1}s.");
        Console.WriteLine($"Mask: {maskPath}");
        Console.WriteLine($"Prepared input: {preparedPath}");
        return 0;
    }
}
