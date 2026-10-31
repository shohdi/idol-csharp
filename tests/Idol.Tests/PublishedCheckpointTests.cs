using System.Text.Json;
using Idol.Inference.Models;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Tests;

/// <summary>Opt-in integration checks against downloaded original, unconverted checkpoint files.</summary>
public sealed class PublishedCheckpointTests
{
    private static string ModelDirectory => Environment.GetEnvironmentVariable("IDOL_MODEL_DIRECTORY") ??
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Idol", "models", "idol");

    [PublishedModelFact]
    public void PublishedOctreeCheckpointProducesFiniteLogitsAndConservesAnchors()
    {
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        var device = cuda.is_available() ? CUDA : CPU;
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(ModelDirectory, "config.json")));
        using var w = WeightStore.Load(Path.Combine(ModelDirectory, "decoder.safetensors"), device,
            name => IdolDenoiser.CheckpointType(name, device.type == DeviceType.CUDA));
        var decoder = new OctreeDecoder(w, config.RootElement.GetProperty("decoder").GetProperty("args"));
        var latent = randn(1, 8, 16, device: device);
        var output = decoder.Forward(rand(1, 3, 3, device: device), full([1L], 4, device: device), latent);
        Assert.Equal(new long[] { 1, 3, 8 }, output.shape);
        Assert.True(output.isfinite().all().item<bool>());
        var anchors = decoder.Sample(latent, 64, 3);
        Assert.Equal(new long[] { 1, 64, 3 }, anchors.shape);
        Assert.True(anchors.ge(0).logical_and(anchors.lt(1)).all().item<bool>());
    }

    [PublishedModelFact]
    public void PublishedGaussianCheckpointProducesActivatedSplats()
    {
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        var device = cuda.is_available() ? CUDA : CPU;
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(ModelDirectory, "config.json")));
        using var w = WeightStore.Load(Path.Combine(ModelDirectory, "decoder_gs.safetensors"), device,
            name => IdolDenoiser.CheckpointType(name, device.type == DeviceType.CUDA));
        var decoder = new GaussianDecoder(w, config.RootElement.GetProperty("decoder_gs").GetProperty("args"));
        using var result = decoder.Decode(rand(1, 4, 3, device: device), randn(1, 8, 16, device: device));
        Assert.Equal(new long[] { 128, 3 }, result.Xyz.shape);
        Assert.True(result.Xyz.isfinite().all().item<bool>());
        Assert.True(result.Scaling.gt(0).all().item<bool>());
        Assert.True(result.Opacity.ge(0).logical_and(result.Opacity.le(1)).all().item<bool>());
        Assert.True(allclose(result.Rotation.norm(-1), ones(128, device: device), 1e-4, 1e-5));
    }

    [PublishedModelFact(RequiresCuda = true)]
    public void PublishedDenoiserRunsAll8192LatentTokensOnCuda()
    {
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(ModelDirectory, "config.json")));
        using var w = WeightStore.Load(Path.Combine(ModelDirectory, "denoiser.safetensors"), CUDA,
            name => IdolDenoiser.CheckpointType(name, true));
        var denoiser = new IdolDenoiser(w, config.RootElement.GetProperty("denoiser").GetProperty("args"));
        var result = denoiser.Forward(randn(1, 8192, 16, device: CUDA), randn(1, 1, 5, device: CUDA),
            full([1L], 500, device: CUDA), zeros(1, 9, 1280, device: CUDA), zeros(1, 9, 128, device: CUDA));
        Assert.Equal(new long[] { 1, 8192, 16 }, result.Latent.shape);
        Assert.True(result.Latent.isfinite().all().item<bool>());
        Assert.True(result.Camera.isfinite().all().item<bool>());
    }
}

public sealed class PublishedModelFactAttribute : FactAttribute
{
    private bool requiresCuda;
    public bool RequiresCuda
    {
        get => requiresCuda;
        set
        {
            requiresCuda = value;
            if (Skip is null && value && !cuda.is_available()) Skip = "This full-size checkpoint check requires CUDA.";
        }
    }
    public PublishedModelFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("IDOL_MODEL_TESTS") != "1")
            Skip = "Set IDOL_MODEL_TESTS=1 after downloading the published checkpoints to run this integration check.";
    }
}
