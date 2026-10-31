// C# port of speridlabs/idol-sdk (Apache-2.0), idol/models/layers.py.
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace Idol.Inference.Models;

public static class ModelLayers
{
    public static Tensor Mlp(WeightStore w, Tensor x, string prefix) =>
        FeedForward(w, x, prefix + ".mlp");

    public static Tensor FeedForward(WeightStore w, Tensor x, string prefix)
    {
        using var scope = NewDisposeScope();
        return w.Linear(w.Linear(x, prefix + ".0").gelu(TorchSharp.Modules.GELU.Approximate.tanh), prefix + ".2")
            .MoveToOuterDisposeScope();
    }

    public static Tensor Sinusoidal(WeightStore w, Tensor t, string prefix,
        double maxPeriod = 10000, bool twoPi = false)
    {
        using var scope = NewDisposeScope();
        int size = checked((int)w.Get(prefix + ".mlp.0.weight").shape[1]);
        var freq = (-Math.Log(maxPeriod) * arange(size / 2, dtype: ScalarType.Float32, device: t.device) / (size / 2)).exp();
        var angles = t.to(ScalarType.Float32).unsqueeze(1) * freq.unsqueeze(0);
        if (twoPi) angles = angles * (2 * Math.PI);
        var embedded = cat([angles.cos(), angles.sin()], -1);
        return w.Linear(w.Linear(embedded, prefix + ".mlp.0").silu(), prefix + ".mlp.2")
            .MoveToOuterDisposeScope();
    }

    public static Tensor Position(Tensor points, int channels, bool version2)
    {
        using var scope = NewDisposeScope();
        int frequencyDim = channels / 3 / 2;
        Tensor frequencies;
        if (version2)
            frequencies = (linspace(0, 10, frequencyDim, dtype: points.dtype, device: points.device) * Math.Log(2)).exp();
        else
        {
            int first = Math.Min(16, frequencyDim), remaining = Math.Max(0, frequencyDim - 16);
            frequencies = (cat([arange(first, dtype: ScalarType.Float32, device: points.device),
                arange(remaining, dtype: ScalarType.Float32, device: points.device) / Math.Max(1, remaining) * 16]) * Math.Log(2)).exp();
        }
        var angles = points.unsqueeze(-1) * frequencies * (version2 ? Math.PI : 2 * Math.PI);
        var shape = points.shape[..^1].Concat(new long[] { -1 }).ToArray();
        var embedded = cat([angles.sin(), angles.cos()], -1).reshape(shape);
        if (embedded.shape[^1] < channels)
        {
            var paddingShape = points.shape[..^1].Concat(new long[] { channels - embedded.shape[^1] }).ToArray();
            embedded = cat([embedded, zeros(paddingShape, dtype: embedded.dtype, device: embedded.device)], -1);
        }
        return embedded.MoveToOuterDisposeScope();
    }

    public static Tensor Rotary(WeightStore w, Tensor x, string prefix, int heads)
    {
        using var scope = NewDisposeScope();
        var normalized = w.LayerNorm(x.to(ScalarType.Float32), prefix + ".norm");
        var h = w.Linear(w.Linear(normalized, prefix + ".gate_map").silu() *
            w.Linear(normalized, prefix + ".content_map"), prefix + ".final_map");
        var delta = h.reshape(h.shape[0], h.shape[1], heads, 3);
        // clamp_mul in the SDK has an identity forward; its clamp only affects training gradients.
        var a0 = delta.select(-1, 0).unsqueeze(-1) * w.Get(prefix + ".freqs_0") * Math.PI;
        var a1 = delta.select(-1, 1).unsqueeze(-1) * w.Get(prefix + ".freqs_1") * Math.PI;
        var a2 = delta.select(-1, 2).unsqueeze(-1) * w.Get(prefix + ".freqs_2") * Math.PI;
        var angles = cat([a0, a1, a2], -1);
        return stack([angles.cos(), angles.sin()], -1).MoveToOuterDisposeScope();
    }

    public static Tensor ApplyRotary(Tensor x, Tensor rotary)
    {
        using var scope = NewDisposeScope();
        var pairs = x.to(ScalarType.Float32).reshape(x.shape[..^1].Concat(new long[] { -1, 2 }).ToArray());
        var real = pairs.select(-1, 0); var imaginary = pairs.select(-1, 1);
        var cos = rotary.select(-1, 0); var sin = rotary.select(-1, 1);
        return stack([real * cos - imaginary * sin, real * sin + imaginary * cos], -1)
            .flatten(-2).to(x.dtype).MoveToOuterDisposeScope();
    }

    public static Tensor Rms(WeightStore w, Tensor x, string prefix)
    {
        using var scope = NewDisposeScope();
        var normalized = x.to(ScalarType.Float32);
        normalized = normalized / normalized.norm(-1, keepdim: true).clamp_min(1e-12);
        return (normalized * w.Get(prefix + ".gamma") * Math.Sqrt(x.shape[^1]))
            .to(x.dtype).MoveToOuterDisposeScope();
    }

    /// <summary>Exact attention, chunked by query to bound memory even on LibTorch's math backend.</summary>
    public static Tensor Attention(Tensor q, Tensor k, Tensor v, int queryChunkSize = 512)
    {
        using var scope = NewDisposeScope();
        var qt = q.permute(0, 2, 1, 3); var kt = k.permute(0, 2, 1, 3); var vt = v.permute(0, 2, 1, 3);
        var chunks = new List<Tensor>();
        for (long start = 0; start < q.shape[1]; start += queryChunkSize)
            chunks.Add(F.scaled_dot_product_attention(qt.narrow(2, start, Math.Min(queryChunkSize, q.shape[1] - start)), kt, vt));
        return cat(chunks.ToArray(), 2).permute(0, 2, 1, 3).MoveToOuterDisposeScope();
    }

    public static Tensor RopeAttention(WeightStore w, Tensor x, Tensor rotary, string prefix, int heads)
    {
        using var scope = NewDisposeScope();
        var qkv = w.Linear(x, prefix + ".qkv").reshape(x.shape[0], x.shape[1], 3, heads, -1);
        var q = Rms(w, ApplyRotary(qkv.select(2, 0), rotary), prefix + ".q_norm");
        var k = Rms(w, ApplyRotary(qkv.select(2, 1), rotary), prefix + ".k_norm");
        var result = Attention(q, k, qkv.select(2, 2)).reshape(x.shape);
        return w.Linear(result, prefix + ".out").MoveToOuterDisposeScope();
    }

    public static Tensor MultiHeadAttention(WeightStore w, Tensor x, Tensor? context, string prefix, int heads)
    {
        using var scope = NewDisposeScope();
        Tensor q, k, v;
        if (context is not null)
        {
            q = w.Linear(x, prefix + ".to_q").reshape(x.shape[0], x.shape[1], heads, -1);
            var kv = w.Linear(context, prefix + ".to_kv").reshape(context.shape[0], context.shape[1], 2, heads, -1);
            k = kv.select(2, 0); v = kv.select(2, 1);
        }
        else
        {
            var qkv = w.Linear(x, prefix + ".to_qkv").reshape(x.shape[0], x.shape[1], 3, heads, -1);
            q = qkv.select(2, 0); k = qkv.select(2, 1); v = qkv.select(2, 2);
        }
        var output = Attention(Rms(w, q, prefix + ".q_rms_norm"), Rms(w, k, prefix + ".k_rms_norm"), v);
        return w.Linear(output.reshape(x.shape[0], x.shape[1], -1), prefix + ".to_out").MoveToOuterDisposeScope();
    }

    public static Tensor UnifiedBlock(WeightStore w, Tensor x, Tensor? mod, Tensor rotary, string prefix, int heads)
    {
        using var scope = NewDisposeScope();
        if (mod is null)
        {
            var h = x + RopeAttention(w, w.LayerNorm(x, prefix + ".norm1", 1e-6), rotary, prefix + ".attn", heads);
            return (h + Mlp(w, w.LayerNorm(h, prefix + ".norm2", 1e-6), prefix + ".mlp"))
                .MoveToOuterDisposeScope();
        }
        var modulation = (mod + w.Get(prefix + ".shift_table").to(mod.dtype)).chunk(6, 1);
        var h1 = w.LayerNorm(x, eps: 1e-6, affine: false) * (1 + modulation[1].unsqueeze(1)) + modulation[0].unsqueeze(1);
        var residual = x + RopeAttention(w, h1, rotary, prefix + ".attn", heads) * modulation[2].unsqueeze(1);
        var h2 = w.LayerNorm(residual, eps: 1e-6, affine: false) * (1 + modulation[4].unsqueeze(1)) + modulation[3].unsqueeze(1);
        return (residual + Mlp(w, h2, prefix + ".mlp") * modulation[5].unsqueeze(1)).MoveToOuterDisposeScope();
    }

    public static Tensor CrossOnlyBlock(WeightStore w, Tensor x, Tensor mod, Tensor context, string prefix, int heads)
    {
        using var scope = NewDisposeScope();
        var m = mod.chunk(6, 1);
        var h = w.LayerNorm(x, eps: 1e-6, affine: false) * (1 + m[1].unsqueeze(1)) + m[0].unsqueeze(1);
        var residual = x + MultiHeadAttention(w, h, context, prefix + ".cross_attn", heads) * m[2].unsqueeze(1);
        h = w.LayerNorm(residual, eps: 1e-6, affine: false) * (1 + m[4].unsqueeze(1)) + m[3].unsqueeze(1);
        return (residual + Mlp(w, h, prefix + ".mlp") * m[5].unsqueeze(1)).MoveToOuterDisposeScope();
    }

    public static Tensor CrossBlock(WeightStore w, Tensor x, Tensor context, string prefix, int heads)
    {
        using var scope = NewDisposeScope();
        var h = x + MultiHeadAttention(w, w.LayerNorm(x, eps: 1e-6, affine: false), null, prefix + ".self_attn", heads);
        h = h + MultiHeadAttention(w, w.LayerNorm(h, prefix + ".norm2", 1e-6), context, prefix + ".cross_attn", heads);
        return (h + Mlp(w, w.LayerNorm(h, eps: 1e-6, affine: false), prefix + ".mlp")).MoveToOuterDisposeScope();
    }
}
