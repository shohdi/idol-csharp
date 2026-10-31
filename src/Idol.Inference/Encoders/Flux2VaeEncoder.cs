// Adapted from Hugging Face Diffusers AutoencoderKLFlux2 and VAE Encoder (Apache-2.0).
using System.Text.Json;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Encoders;

/// <summary>The FLUX.2 image VAE encoder and packed, batch-normalized conditioning tokens used by Idol.</summary>
public sealed class Flux2VaeEncoder
{
    private readonly WeightStore weights;
    private readonly int blockCount, layersPerBlock, groups;
    private readonly double bnEps;
    private readonly string activation;
    private readonly bool midAttention;

    public Flux2VaeEncoder(WeightStore weights, string configPath)
    {
        this.weights = weights;
        using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
        var c = doc.RootElement;
        blockCount = c.GetProperty("block_out_channels").GetArrayLength();
        layersPerBlock = c.TryGetProperty("layers_per_block", out var l) ? l.GetInt32() : 2;
        groups = c.TryGetProperty("norm_num_groups", out var g) ? g.GetInt32() : 32;
        activation = c.TryGetProperty("act_fn", out var a) ? a.GetString()! : "silu";
        bnEps = c.TryGetProperty("batch_norm_eps", out var e) ? e.GetDouble() : 1e-4;
        midAttention = !c.TryGetProperty("mid_block_add_attention", out var attn) || attn.GetBoolean();
        if (c.TryGetProperty("down_block_types", out var types) && types.EnumerateArray().Any(x => x.GetString() != "DownEncoderBlock2D"))
            throw new NotSupportedException("FLUX.2 VAE checkpoint uses an unsupported encoder block type.");
    }

    public Tensor Encode(Tensor pixels, CancellationToken cancellationToken = default)
    {
        using var scope = NewDisposeScope();
        var dtype = weights.Get("encoder.conv_in.weight").dtype;
        var x = EncoderOps.Conv(weights, pixels.to_type(dtype) * 2 - 1, "encoder.conv_in", padding: 1);
        for (int block = 0; block < blockCount; block++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int i = 0; i < layersPerBlock; i++)
            {
                var next = Resnet(x, $"encoder.down_blocks.{block}.resnets.{i}");
                x.Dispose(); x = next;
            }
            if (block + 1 < blockCount)
            {
                var padded = nn.functional.pad(x, new long[] { 0, 1, 0, 1 });
                var next = EncoderOps.Conv(weights, padded, $"encoder.down_blocks.{block}.downsamplers.0.conv", stride: 2);
                x.Dispose(); padded.Dispose(); x = next;
            }
        }
        var middle = Resnet(x, "encoder.mid_block.resnets.0"); x.Dispose(); x = middle;
        if (midAttention) { var next = Attention(x, "encoder.mid_block.attentions.0"); x.Dispose(); x = next; }
        middle = Resnet(x, "encoder.mid_block.resnets.1"); x.Dispose(); x = middle;
        var encoded = EncoderOps.Conv(weights, EncoderOps.Activate(EncoderOps.GroupNorm(weights, x, "encoder.conv_norm_out", groups), activation), "encoder.conv_out", padding: 1);
        if (weights.Contains("quant_conv.weight")) encoded = EncoderOps.Conv(weights, encoded, "quant_conv");
        var moments = encoded.chunk(2, 1);
        // Diffusers DiagonalGaussianDistribution samples the posterior, rather than using its mean.
        var latent = moments[0] + (moments[1].clamp(-30, 20) * .5).exp() * randn_like(moments[0]);
        long b = latent.shape[0], c = latent.shape[1], h = latent.shape[2], w = latent.shape[3];
        if ((h % 2) != 0 || (w % 2) != 0) throw new InvalidDataException("VAE latent dimensions must be even.");
        var packed = latent.view(b, c, h / 2, 2, w / 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h / 2, w / 2);
        var mean = weights.Get("bn.running_mean").view(1, -1, 1, 1);
        var std = (weights.Get("bn.running_var").view(1, -1, 1, 1) + bnEps).sqrt();
        var tokens = ((packed - mean) / std).to_type(ScalarType.Float32).flatten(2).transpose(1, 2).contiguous();
        return cat(new[] { zeros(new long[] { b, 5, tokens.shape[2] }, device: pixels.device), tokens }, 1).MoveToOuterDisposeScope();
    }

    private Tensor Resnet(Tensor x, string p)
    {
        using var scope = NewDisposeScope();
        var h = EncoderOps.Conv(weights, EncoderOps.Activate(EncoderOps.GroupNorm(weights, x, p + ".norm1", groups), activation), p + ".conv1", padding: 1);
        h = EncoderOps.Conv(weights, EncoderOps.Activate(EncoderOps.GroupNorm(weights, h, p + ".norm2", groups), activation), p + ".conv2", padding: 1);
        var residual = weights.Contains(p + ".conv_shortcut.weight") ? EncoderOps.Conv(weights, x, p + ".conv_shortcut") : x;
        return (residual + h).MoveToOuterDisposeScope();
    }

    private Tensor Attention(Tensor x, string p)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
        var normalized = EncoderOps.GroupNorm(weights, x, p + ".group_norm", groups).flatten(2).transpose(1, 2);
        var q = weights.Linear(normalized, p + ".to_q").unsqueeze(1);
        var k = weights.Linear(normalized, p + ".to_k").unsqueeze(1);
        var v = weights.Linear(normalized, p + ".to_v").unsqueeze(1);
        var attn = nn.functional.scaled_dot_product_attention(q, k, v).squeeze(1);
        return (x + weights.Linear(attn, p + ".to_out.0").transpose(1, 2).reshape(b, c, h, w)).MoveToOuterDisposeScope();
    }
}
