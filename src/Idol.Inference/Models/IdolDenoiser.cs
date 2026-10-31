// C# port of speridlabs/idol-sdk, Apache-2.0. The Wan/Hugging Face attribution is in NOTICE.
using System.Text.Json;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Models;

public sealed class IdolDenoiser
{
    private readonly WeightStore w;
    private readonly int blocks, refiners, heads, channels;
    private readonly ScalarType computeType;
    public int TokenLength { get; }
    public int InChannels { get; }
    public int CameraChannels { get; }
    public Device Device => w.Device;
    public IdolDenoiser(WeightStore weights, JsonElement args)
    {
        w = weights;
        TokenLength = args.GetProperty("q_token_length").GetInt32();
        InChannels = args.GetProperty("in_channels").GetInt32();
        CameraChannels = args.GetProperty("cam_channels").GetInt32();
        blocks = args.GetProperty("num_blocks").GetInt32();
        refiners = args.GetProperty("num_refiner_blocks").GetInt32();
        heads = args.GetProperty("num_heads").GetInt32();
        channels = args.GetProperty("model_channels").GetInt32();
        computeType = w.Get("blocks.0.attn.qkv.weight").dtype;
    }

    public (Tensor Latent, Tensor Camera) Forward(Tensor latent, Tensor camera, Tensor time,
        Tensor feature1, Tensor feature2, CancellationToken cancellationToken = default)
    {
        using var scope = NewDisposeScope();
        var hx = w.Linear(latent, "input_layer").to(computeType) +
            ModelLayers.Position(w.Get("pos_pe"), channels, version2: false).to(computeType);
        var hc = (w.Linear(feature1, "cond_embedder") + w.Linear(feature2, "cond_embedder2")).to(computeType);
        var embedding = ModelLayers.Sinusoidal(w, time, "t_embedder");
        var modulation = w.Linear(embedding.silu(), "adaLN_modulation.1").to(computeType);
        for (int i = 0; i < refiners; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var iteration = NewDisposeScope();
            var next = ModelLayers.UnifiedBlock(w, hx, modulation, ModelLayers.Rotary(w, hx, $"noise_repo_layers.{i}", heads), $"noise_refiner.{i}", heads);
            hx.Dispose(); hx = next.MoveToOuterDisposeScope();
        }
        for (int i = 0; i < refiners; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var iteration = NewDisposeScope();
            var next = ModelLayers.UnifiedBlock(w, hc, null, ModelLayers.Rotary(w, hc, $"context_repo_layers.{i}", heads), $"context_refiner.{i}", heads);
            hc.Dispose(); hc = next.MoveToOuterDisposeScope();
        }
        var hcam = ModelLayers.FeedForward(w, camera, "cam_refiner.mlp").to(computeType);
        var h = cat([hx, hc, hcam], 1);
        hx.Dispose(); hc.Dispose(); hcam.Dispose();
        for (int i = 0; i < blocks; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var iteration = NewDisposeScope();
            var next = ModelLayers.UnifiedBlock(w, h, modulation, ModelLayers.Rotary(w, h, $"repo_layers.{i}", heads), $"blocks.{i}", heads);
            h.Dispose(); h = next.MoveToOuterDisposeScope();
        }
        hx = w.LayerNorm(h.narrow(1, 0, latent.shape[1]).to(latent.dtype), affine: false);
        hcam = w.LayerNorm(h.narrow(1, h.shape[1] - camera.shape[1], camera.shape[1]).to(camera.dtype), affine: false);
        var finalMod = (w.Get("shift_table") + embedding.unsqueeze(1)).chunk(2, 1);
        var output = w.Linear(hx * (1 + finalMod[1]) + finalMod[0], "out_layer");
        var outputCamera = w.Linear(hcam * (1 + finalMod[1]) + finalMod[0], "cam_out_layer");
        return (output.MoveToOuterDisposeScope(), outputCamera.MoveToOuterDisposeScope());
    }

    /// <summary>Matches the SDK's half-precision Linear modules in transformer blocks.</summary>
    public static ScalarType CheckpointType(string name, bool useHalf)
    {
        if (!useHalf) return ScalarType.Float32;
        bool inBlock = name.StartsWith("blocks.", StringComparison.Ordinal) ||
            name.StartsWith("noise_refiner.", StringComparison.Ordinal) || name.StartsWith("context_refiner.", StringComparison.Ordinal);
        bool linear = name.Contains(".attn.qkv.", StringComparison.Ordinal) || name.Contains(".attn.out.", StringComparison.Ordinal) ||
            name.Contains(".mlp.mlp.", StringComparison.Ordinal) || name.Contains(".to_q.", StringComparison.Ordinal) ||
            name.Contains(".to_kv.", StringComparison.Ordinal) || name.Contains(".to_qkv.", StringComparison.Ordinal) ||
            name.Contains(".to_out.", StringComparison.Ordinal);
        return inBlock && linear ? ScalarType.Float16 : ScalarType.Float32;
    }
}
