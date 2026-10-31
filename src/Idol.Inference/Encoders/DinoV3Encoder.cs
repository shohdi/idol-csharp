// Adapted from Hugging Face Transformers DINOv3 ViT (Apache-2.0).
// Copyright 2025 Meta AI and The HuggingFace Inc. team.
using System.Text.Json;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Encoders;

/// <summary>Native DINOv3 ViT inference, retaining CLS and register tokens as required by Idol.</summary>
public sealed class DinoV3Encoder
{
    private readonly WeightStore weights;
    private readonly int patch, heads, layers, hidden;
    private readonly double eps, theta;
    private readonly bool gated;
    private readonly string activation, layerPrefix;

    public DinoV3Encoder(WeightStore weights, string configPath)
    {
        this.weights = weights;
        using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
        var c = doc.RootElement;
        patch = c.TryGetProperty("patch_size", out var p) ? p.GetInt32() : 16;
        heads = c.GetProperty("num_attention_heads").GetInt32();
        layers = c.GetProperty("num_hidden_layers").GetInt32();
        hidden = c.GetProperty("hidden_size").GetInt32();
        eps = c.TryGetProperty("layer_norm_eps", out var e) ? e.GetDouble() : 1e-5;
        theta = c.TryGetProperty("rope_theta", out var t) ? t.GetDouble() : 100;
        gated = c.TryGetProperty("use_gated_mlp", out var g) && g.GetBoolean();
        activation = c.TryGetProperty("hidden_act", out var a) ? a.GetString()! : "gelu";
        // Transformers renamed encoder.layer to model.layer. Accommodate both official layouts.
        layerPrefix = weights.Contains("model.layer.0.norm1.weight") ? "model.layer" : "encoder.layer";
    }

    /// <param name="pixels">RGB [B,3,H,W] in [0,1], normally 1024x1024.</param>
    public Tensor Encode(Tensor pixels, CancellationToken cancellationToken = default)
    {
        using var scope = NewDisposeScope();
        var dtype = weights.Get("embeddings.patch_embeddings.weight").dtype;
        var image = EncoderOps.ImageNetNormalize(pixels).to_type(dtype);
        var x = EncoderOps.Conv(weights, image, "embeddings.patch_embeddings", patch).flatten(2).transpose(1, 2);
        x = cat(new[] { weights.Get("embeddings.cls_token").expand(pixels.shape[0], -1, -1),
            weights.Get("embeddings.register_tokens").expand(pixels.shape[0], -1, -1), x }, 1);
        var (cos, sin) = PositionEmbedding(image);
        for (int i = 0; i < layers; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var next = Block(x, $"{layerPrefix}.{i}", cos, sin);
            x.Dispose();
            x = next;
        }
        var result = weights.LayerNorm(x, "norm", eps).to_type(ScalarType.Float32);
        // Idol applies a second, non-affine LayerNorm after the model's learned final norm.
        return weights.LayerNorm(result, null, 1e-5, false).MoveToOuterDisposeScope();
    }

    private (Tensor Cos, Tensor Sin) PositionEmbedding(Tensor image)
    {
        using var scope = NewDisposeScope();
        int h = (int)image.shape[2] / patch, w = (int)image.shape[3] / patch, d = hidden / heads;
        var positions = new float[h * w * d];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int j = 0; j < d / 4; j++)
                {
                    double inv = 1.0 / Math.Pow(theta, j * 4.0 / d);
                    float ay = (float)(2 * Math.PI * (2.0 * (y + .5) / h - 1) * inv);
                    float ax = (float)(2 * Math.PI * (2.0 * (x + .5) / w - 1) * inv);
                    int k = (y * w + x) * d;
                    positions[k + j] = positions[k + d / 2 + j] = ay;
                    positions[k + d / 4 + j] = positions[k + 3 * d / 4 + j] = ax;
                }
        var angles = tensor(positions, device: image.device).reshape(1, 1, h * w, d);
        return (angles.cos().to_type(image.dtype).MoveToOuterDisposeScope(),
            angles.sin().to_type(image.dtype).MoveToOuterDisposeScope());
    }

    private Tensor Block(Tensor x, string p, Tensor cos, Tensor sin)
    {
        using var scope = NewDisposeScope();
        var h = weights.LayerNorm(x, p + ".norm1", eps);
        long b = x.shape[0], n = x.shape[1];
        var q = weights.Linear(h, p + ".attention.q_proj").reshape(b, n, heads, hidden / heads).transpose(1, 2);
        var k = weights.Linear(h, p + ".attention.k_proj").reshape(b, n, heads, hidden / heads).transpose(1, 2);
        var v = weights.Linear(h, p + ".attention.v_proj").reshape(b, n, heads, hidden / heads).transpose(1, 2);
        q = Rotate(q, cos, sin);
        k = Rotate(k, cos, sin);
        var attn = nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, hidden);
        var residual = x + weights.Linear(attn, p + ".attention.o_proj") * weights.Get(p + ".layer_scale1.lambda1");
        h = weights.LayerNorm(residual, p + ".norm2", eps);
        var up = weights.Linear(h, p + ".mlp.up_proj");
        var ff = gated ? EncoderOps.Activate(weights.Linear(h, p + ".mlp.gate_proj"), activation) * up : EncoderOps.Activate(up, activation);
        return (residual + weights.Linear(ff, p + ".mlp.down_proj") * weights.Get(p + ".layer_scale2.lambda1")).MoveToOuterDisposeScope();
    }

    private static Tensor Rotate(Tensor x, Tensor cos, Tensor sin)
    {
        using var scope = NewDisposeScope();
        long patchCount = cos.shape[2], prefix = x.shape[2] - patchCount, d = x.shape[3];
        var prefixTokens = x.narrow(2, 0, prefix);
        var patches = x.narrow(2, prefix, patchCount);
        var rotated = cat(new[] { -patches.narrow(3, d / 2, d / 2), patches.narrow(3, 0, d / 2) }, 3);
        return cat(new[] { prefixTokens, patches * cos + rotated * sin }, 2).MoveToOuterDisposeScope();
    }
}
