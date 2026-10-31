using System.Numerics;
using Idol.Core;
using Idol.Core.Rendering;

namespace Idol.Tests;

public sealed class GaussianRasterizerTests
{
    private static readonly SplatCamera Camera = new(new(0, 0, 2), Vector3.UnitX, Vector3.UnitY, -Vector3.UnitZ);

    [Fact]
    public void DenseSurfaceKeepsEveryGaussianAndClosesSamplingHoles()
    {
        const int side = 256;
        var points = new Gaussian[side * side];
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++)
                points[y * side + x] = Splat(new((x / (float)(side - 1) - .5f) * .85f,
                    (y / (float)(side - 1) - .5f) * .85f, 0), Vector3.One, new(.001f), .9f);
        var complete = GaussianRasterizer.Render(new(points), Camera, 128, 128);
        // Match the previous maximum-detail preview's approximately 1-in-52 point selection.
        var sampled = GaussianRasterizer.Render(new(points.Where((_, i) => i % 52 == 0).ToArray()), Camera, 128, 128);
        Assert.Equal(points.Length, complete.SubmittedCount);
        Assert.Equal(points.Length, complete.VisibleCount);
        double denseAlpha = MeanAlpha(complete.Pixels, 128, 40, 40, 88, 88);
        double sparseAlpha = MeanAlpha(sampled.Pixels, 128, 40, 40, 88, 88);
        Assert.True(denseAlpha > .97, $"Dense surface opacity was {denseAlpha:F3}.");
        Assert.True(denseAlpha > sparseAlpha + .3, $"Dense={denseAlpha:F3}; sparse={sparseAlpha:F3}.");
    }

    [Fact]
    public void DepthSortingCompositesTheNearestColorOverTheFartherColor()
    {
        var near = Splat(new(0, 0, .1f), Vector3.UnitX, new(.1f), .5f);
        var far = Splat(new(0, 0, -.1f), Vector3.UnitZ, new(.1f), .5f);
        var result = GaussianRasterizer.Render(new([near, far]), Camera, 65, 65);
        int center = (32 * 65 + 32) * 4;
        Assert.InRange(result.Pixels[center], (byte)63, (byte)65); // Far blue contributes 0.25.
        Assert.Equal(0, result.Pixels[center + 1]);
        Assert.InRange(result.Pixels[center + 2], (byte)127, (byte)129); // Near red contributes 0.5.
        Assert.InRange(result.Pixels[center + 3], (byte)190, (byte)192);
        var reversed = GaussianRasterizer.Render(new([far, near]), Camera, 65, 65);
        Assert.Equal(result.Pixels, reversed.Pixels);
    }

    [Fact]
    public void QuaternionRotatesTheProjectedAnisotropicEllipse()
    {
        var horizontal = Splat(Vector3.Zero, Vector3.One, new(.2f, .005f, .005f));
        var vertical = horizontal with { Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, MathF.PI / 2) };
        var h = Bounds(GaussianRasterizer.Render(new([horizontal]), Camera, 101, 101).Pixels, 101);
        var v = Bounds(GaussianRasterizer.Render(new([vertical]), Camera, 101, 101).Pixels, 101);
        Assert.True(h.Width > 3 * h.Height);
        Assert.True(v.Height > 3 * v.Width);
    }

    [Fact]
    public void BroadGaussianRetainsItsSupportBeyondTheOld180PixelLimit()
    {
        var broad = Splat(Vector3.Zero, Vector3.UnitZ, new(.5f), .8f);
        var foreground = Splat(new(0, 0, .1f), Vector3.UnitX, new(.1f), .5f);
        var result = GaussianRasterizer.Render(new([foreground, broad]), Camera, 512, 512);
        int edge = (256 * 512 + 20) * 4;
        Assert.True(result.Pixels[edge + 3] > 10, "A broad Gaussian's visible tail was clipped.");
        int center = (256 * 512 + 256) * 4;
        Assert.True(result.Pixels[center + 2] > result.Pixels[center], "Tile/global merging lost depth ordering.");
    }

    [Fact]
    public void WeakGaussiansAboveThePixelAlphaThresholdRemainVisible()
    {
        var result = GaussianRasterizer.Render(new([Splat(Vector3.Zero, Vector3.One, new(.01f), .006f)]), Camera, 33, 33);
        Assert.Equal(1, result.VisibleCount);
        Assert.Equal(2, result.Pixels[(16 * 33 + 16) * 4 + 3]);
    }

    [Fact]
    public void VisibilityCullingDoesNotChangeTheSubmittedCount()
    {
        var points = new[]
        {
            Splat(Vector3.Zero, Vector3.One, new(.01f)),
            Splat(new(0, 0, 3), Vector3.One, new(.01f)), // Behind the camera.
            Splat(new(100, 0, 0), Vector3.One, new(.01f)),
            Splat(new(float.NaN, 0, 0), Vector3.One, new(.01f)),
            Splat(Vector3.Zero, Vector3.One, new(.01f), .0001f)
        };
        var result = GaussianRasterizer.Render(new(points), Camera, 33, 33);
        Assert.Equal(5, result.SubmittedCount);
        Assert.Equal(1, result.VisibleCount);
    }

    [Fact]
    public void ParallelTilesProduceDeterministicPixelsIncludingEqualDepthTies()
    {
        var random = new Random(37);
        var points = Enumerable.Range(0, 1000).Select(i => Splat(new(
            (float)random.NextDouble() - .5f, (float)random.NextDouble() - .5f, i % 3 * .05f),
            new((float)random.NextDouble(), (float)random.NextDouble(), (float)random.NextDouble()), new(.015f), .3f)).ToList();
        points.Add(Splat(Vector3.Zero, new(.2f), new(1f), .2f));
        var cloud = new GaussianCloud(points.ToArray());
        Assert.Equal(GaussianRasterizer.Render(cloud, Camera, 513, 257).Pixels,
            GaussianRasterizer.Render(cloud, Camera, 513, 257).Pixels);
    }

    [Fact]
    public void CancellationAndInvalidViewportsFailBeforeRendering()
    {
        var cloud = new GaussianCloud([Splat(Vector3.Zero, Vector3.One, new(.01f))]);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => GaussianRasterizer.Render(cloud, Camera, 32, 32, cancellation.Token));
        Assert.Throws<ArgumentOutOfRangeException>(() => GaussianRasterizer.Render(cloud, Camera, 65536, 65536));
        Assert.Throws<ArgumentException>(() => GaussianRasterizer.Render(cloud, Camera with { Forward = Vector3.Zero }, 32, 32));
    }

    private static Gaussian Splat(Vector3 position, Vector3 color, Vector3 scale, float opacity = .9f) =>
        new(position, (color - new Vector3(.5f)) / Gaussian.ShC0, MathF.Log(opacity / (1 - opacity)),
            new(MathF.Log(scale.X), MathF.Log(scale.Y), MathF.Log(scale.Z)), Quaternion.Identity);

    private static double MeanAlpha(byte[] pixels, int width, int left, int top, int right, int bottom)
    {
        long total = 0;
        for (int y = top; y < bottom; y++) for (int x = left; x < right; x++) total += pixels[(y * width + x) * 4 + 3];
        return total / (255.0 * (right - left) * (bottom - top));
    }

    private static (int Width, int Height) Bounds(byte[] pixels, int width)
    {
        int minX = width, minY = pixels.Length / (width * 4), maxX = -1, maxY = -1;
        for (int y = 0; y < pixels.Length / (width * 4); y++) for (int x = 0; x < width; x++)
            if (pixels[(y * width + x) * 4 + 3] > 4)
            { minX = Math.Min(minX, x); maxX = Math.Max(maxX, x); minY = Math.Min(minY, y); maxY = Math.Max(maxY, y); }
        return (maxX - minX + 1, maxY - minY + 1);
    }
}
