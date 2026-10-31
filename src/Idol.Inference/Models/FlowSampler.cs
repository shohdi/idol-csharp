// C# port of speridlabs/idol-sdk, Apache-2.0, idol/sampler.py and camera.py.
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Models;

public static class FlowSampler
{
    public static double[] TimeGrid(int steps, double timeShift)
    {
        if (steps < 1 || steps > 1000 || !double.IsFinite(timeShift) || timeShift <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps));
        return Enumerable.Range(0, steps + 1).Select(i =>
        {
            double t = 1 - i / (double)steps;
            return timeShift * t / (1 + (timeShift - 1) * t);
        }).ToArray();
    }

    public static (Tensor Latent, Tensor Camera) Sample(IdolDenoiser model, Tensor feature1, Tensor feature2,
        int seed = 42, int steps = 50, double guidance = 3, double timeShift = 3,
        Action<int, int>? progress = null, CancellationToken cancellationToken = default, bool resetSeed = true)
    {
        using var scope = NewDisposeScope();
        if (resetSeed) manual_seed(seed);
        var latent = randn([1L, model.TokenLength, model.InChannels], device: model.Device);
        var camera = randn([1L, 1, model.CameraChannels], device: model.Device);
        var negative1 = zeros_like(feature1); var negative2 = zeros_like(feature2);
        var times = TimeGrid(steps, timeShift);
        for (int i = 0; i < steps; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var iteration = NewDisposeScope();
            var time = full([1L], 1000 * times[i], dtype: ScalarType.Float32, device: model.Device);
            var velocity = model.Forward(latent, camera, time, feature1, feature2, cancellationToken);
            var vl = velocity.Latent; var vc = velocity.Camera;
            if (guidance > 0)
            {
                var negative = model.Forward(latent, camera, time, negative1, negative2, cancellationToken);
                vl = (1 + guidance) * vl - guidance * negative.Latent;
                vc = (1 + guidance) * vc - guidance * negative.Camera;
            }
            var nextLatent = latent - vl * (times[i] - times[i + 1]);
            var nextCamera = camera - vc * (times[i] - times[i + 1]);
            latent.Dispose(); camera.Dispose();
            latent = nextLatent.MoveToOuterDisposeScope(); camera = nextCamera.MoveToOuterDisposeScope();
            progress?.Invoke(i + 1, steps);
        }
        return (latent.MoveToOuterDisposeScope(), camera.MoveToOuterDisposeScope());
    }
}

public sealed record PredictedCamera(double Yaw, double Pitch, double Radius, double Fov);

public static class CameraDecoder
{
    public static PredictedCamera Decode(float[] latent) => Decode((ReadOnlySpan<float>)latent);

    public static PredictedCamera Decode(Tensor latent)
    {
        using var scope = NewDisposeScope();
        return Decode(latent.to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray());
    }

    public static PredictedCamera Decode(ReadOnlySpan<float> latent)
    {
        if (latent.Length != 5) throw new ArgumentException("Camera latent must have five elements.");
        double dx = latent[0], dy = latent[1], dz = latent[2];
        double length = Math.Sqrt(dx * dx + dy * dy + dz * dz);
        if (length > 1e-6) { dx /= length; dy /= length; dz /= length; }
        else { dx = dy = 0; dz = 1; }
        double inverseDistance = latent[3], halfWidth = latent[4];
        double radius = inverseDistance < 1e-3 ? 1000 : 1 / inverseDistance;
        double fov = 2 * Math.Atan(halfWidth / radius) * 180 / Math.PI;
        return new PredictedCamera(Math.Atan2(dx, dy), Math.Asin(Math.Clamp(dz, -1, 1)), radius, fov);
    }
}
