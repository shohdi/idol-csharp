// Native inference helpers shared by the image encoders. No Python process is used.
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Encoders;

internal static class EncoderOps
{
    public static Tensor Conv(WeightStore w, Tensor x, string p, long stride = 1, long padding = 0)
    {
        var weight = w.Get(p + ".weight");
        return nn.functional.conv2d(x, weight, w.Contains(p + ".bias") ? w.Get(p + ".bias") : null,
            strides: new[] { stride, stride }, padding: new[] { padding, padding });
    }

    public static Tensor GroupNorm(WeightStore w, Tensor x, string p, long groups, double eps = 1e-6) =>
        nn.functional.group_norm(x, groups, w.Get(p + ".weight"), w.Get(p + ".bias"), eps);

    public static Tensor BatchNorm(WeightStore w, Tensor x, string p, double eps = 1e-5) =>
        nn.functional.batch_norm(x, w.Get(p + ".running_mean"), w.Get(p + ".running_var"),
            w.Get(p + ".weight"), w.Get(p + ".bias"), training: false, eps: eps);

    public static Tensor Resize(Tensor x, long h, long width, bool alignCorners = true) =>
        nn.functional.interpolate(x, size: new[] { h, width }, mode: InterpolationMode.Bilinear,
            align_corners: alignCorners);

    public static Tensor ImageNetNormalize(Tensor x)
    {
        using var scope = NewDisposeScope();
        var mean = tensor(new[] { .485f, .456f, .406f }, device: x.device).view(1, 3, 1, 1);
        var std = tensor(new[] { .229f, .224f, .225f }, device: x.device).view(1, 3, 1, 1);
        return ((x.to_type(ScalarType.Float32) - mean) / std).MoveToOuterDisposeScope();
    }

    public static Tensor Activate(Tensor x, string name) => name switch
    {
        "silu" or "swish" => nn.functional.silu(x),
        "gelu" => nn.functional.gelu(x),
        "gelu_new" or "gelu_pytorch_tanh" => nn.functional.gelu(x, TorchSharp.Modules.GELU.Approximate.tanh),
        "relu" => nn.functional.relu(x),
        _ => throw new NotSupportedException($"Unsupported encoder activation '{name}'.")
    };
}
