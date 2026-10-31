using System.Buffers;
using System.Numerics;

namespace Idol.Core.Rendering;

public readonly record struct SplatCamera(Vector3 Eye, Vector3 Right, Vector3 Up, Vector3 Forward,
    float VerticalFovDegrees = 45);

/// <summary>Premultiplied BGRA pixels and counts for a render that submits the complete input cloud.</summary>
public sealed record RenderedSplats(byte[] Pixels, int SubmittedCount, int VisibleCount);

/// <summary>
/// Projects every Gaussian into its anisotropic screen-space ellipse. Independent tiles composite the
/// same deterministic back-to-front ordering, so parallelism never changes blending or drops splats.
/// </summary>
public static class GaussianRasterizer
{
    private const int TileSize = 32;
    private const float MinimumAlpha = 1f / 255;
    private const int MaximumPixels = 16_777_216;
    private const int MaximumGaussians = 10_000_000;
    private const int MaximumTileReferences = 8_388_608;
    private const int MaximumTilesPerSplat = 64;
    private const int FalloffSamples = 4096;
    private static readonly float[] Falloff = Enumerable.Range(0, FalloffSamples + 1)
        .Select(i => MathF.Exp(-.5f * i * 9 / FalloffSamples)).ToArray();

    public static RenderedSplats Render(GaussianCloud cloud, SplatCamera camera, int width, int height,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(cloud);
        if (width < 1 || height < 1 || (long)width * height > MaximumPixels)
            throw new ArgumentOutOfRangeException(nameof(width), "The viewport must contain 1–16,777,216 pixels.");
        if (cloud.Count > MaximumGaussians)
            throw new ArgumentOutOfRangeException(nameof(cloud), "The renderer supports at most 10 million Gaussians.");
        if (!Finite(camera.Eye) || !Finite(camera.Right) || !Finite(camera.Up) || !Finite(camera.Forward) ||
            camera.Right.LengthSquared() < 1e-12f || camera.Up.LengthSquared() < 1e-12f || camera.Forward.LengthSquared() < 1e-12f ||
            !float.IsFinite(camera.VerticalFovDegrees) || camera.VerticalFovDegrees <= 0 || camera.VerticalFovDegrees >= 180)
            throw new ArgumentException("The camera must have finite, nonzero axes and a field of view between 0 and 180 degrees.", nameof(camera));
        cancellationToken.ThrowIfCancellationRequested();

        var pixels = new byte[checked(width * height * 4)];
        if (cloud.Count == 0) return new(pixels, 0, 0);
        float focal = height / (2 * MathF.Tan(camera.VerticalFovDegrees * MathF.PI / 360));
        var projected = new Projected[cloud.Count];
        var parallel = new ParallelOptions { CancellationToken = cancellationToken };
        // Each projection has a fixed input slot. Compaction afterwards preserves original indices,
        // including the tie-breaker for Gaussians with the same depth.
        Parallel.For(0, cloud.Count, parallel, i =>
        {
            if (TryProject(cloud.Points[i], camera, width, height, focal, i, out var splat))
                projected[i] = splat;
        });
        int count = 0;
        for (int i = 0; i < projected.Length; i++)
        {
            if ((i & 4095) == 0) cancellationToken.ThrowIfCancellationRequested();
            if (projected[i].Depth > 0) projected[count++] = projected[i];
        }
        if (count == 0) return new(pixels, cloud.Count, 0);
        Array.Sort(projected, 0, count, DepthOrder.Instance);
        cancellationToken.ThrowIfCancellationRequested();

        int columns = (width + TileSize - 1) / TileSize, rows = (height + TileSize - 1) / TileSize;
        var bins = new List<int>?[checked(columns * rows)];
        var large = new List<int>();
        int references = 0;
        for (int i = 0; i < count; i++)
        {
            if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
            ref readonly var splat = ref projected[i];
            int left = splat.Left / TileSize, right = splat.Right / TileSize;
            int top = splat.Top / TileSize, bottom = splat.Bottom / TileSize;
            int coverage = checked((right - left + 1) * (bottom - top + 1));
            // Broad Gaussians are merged into each tile's ordered stream without duplicating their
            // indices thousands of times. The cap bounds memory; it never caps visible Gaussians.
            if (coverage > MaximumTilesPerSplat || references > MaximumTileReferences - coverage)
            {
                large.Add(i);
                continue;
            }
            references += coverage;
            for (int y = top; y <= bottom; y++)
                for (int x = left; x <= right; x++)
                    (bins[y * columns + x] ??= []).Add(i);
        }

        Parallel.For(0, bins.Length, parallel, tile =>
        {
            var indices = bins[tile];
            if (indices is null && large.Count == 0) return;
            int left = tile % columns * TileSize, top = tile / columns * TileSize;
            int tileWidth = Math.Min(TileSize, width - left), tileHeight = Math.Min(TileSize, height - top);
            var color = ArrayPool<float>.Shared.Rent(TileSize * TileSize * 4);
            Array.Clear(color, 0, TileSize * TileSize * 4);
            try
            {
                int local = 0, broad = 0, examined = 0;
                while (local < (indices?.Count ?? 0) || broad < large.Count)
                {
                    if ((examined++ & 127) == 0) cancellationToken.ThrowIfCancellationRequested();
                    int index;
                    if (local < (indices?.Count ?? 0) && (broad == large.Count || indices![local] < large[broad]))
                        index = indices![local++];
                    else index = large[broad++];
                    ref readonly var splat = ref projected[index];
                    if (splat.Right < left || splat.Left >= left + tileWidth || splat.Bottom < top || splat.Top >= top + tileHeight) continue;
                    Draw(in splat, color, left, top, tileWidth, tileHeight, cancellationToken);
                }
                for (int y = 0; y < tileHeight; y++)
                    for (int x = 0; x < tileWidth; x++)
                    {
                        int from = (y * TileSize + x) * 4, to = ((top + y) * width + left + x) * 4;
                        for (int c = 0; c < 4; c++) pixels[to + c] = (byte)Math.Clamp((int)(color[from + c] * 255 + .5f), 0, 255);
                    }
            }
            finally { ArrayPool<float>.Shared.Return(color); }
        });
        return new(pixels, cloud.Count, count);
    }

    private static bool TryProject(in Gaussian gaussian, in SplatCamera camera, int width, int height,
        float focal, int ordinal, out Projected projected)
    {
        projected = default;
        if (!gaussian.IsFinite) return false;
        float opacity = Math.Min(.99f, gaussian.Opacity);
        if (opacity < MinimumAlpha) return false;
        var relative = gaussian.Position - camera.Eye;
        float x = Vector3.Dot(relative, camera.Right), y = Vector3.Dot(relative, camera.Up), z = Vector3.Dot(relative, camera.Forward);
        if (z <= .01f) return false;
        float sx = width * .5f + focal * x / z, sy = height * .5f - focal * y / z;
        if (!float.IsFinite(sx) || !float.IsFinite(sy)) return false;
        var jx = camera.Right * (focal / z) - camera.Forward * (focal * x / (z * z));
        var jy = -camera.Up * (focal / z) + camera.Forward * (focal * y / (z * z));
        var scale = gaussian.Scale;
        var rotation = gaussian.Rotation;
        float norm = rotation.LengthSquared();
        if (norm < 1e-12f) rotation = Quaternion.Identity;
        else if (Math.Abs(norm - 1) > 1e-5f) rotation = Quaternion.Normalize(rotation);
        // Expand the quaternion once for the three scaled covariance axes.
        float xx = rotation.X * rotation.X, yy = rotation.Y * rotation.Y, zz = rotation.Z * rotation.Z;
        float xy = rotation.X * rotation.Y, xz = rotation.X * rotation.Z, yz = rotation.Y * rotation.Z;
        float wx = rotation.W * rotation.X, wy = rotation.W * rotation.Y, wz = rotation.W * rotation.Z;
        var ax = new Vector3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)) * scale.X;
        var ay = new Vector3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)) * scale.Y;
        var az = new Vector3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)) * scale.Z;
        var dx = new Vector3(Vector3.Dot(ax, jx), Vector3.Dot(ay, jx), Vector3.Dot(az, jx));
        var dy = new Vector3(Vector3.Dot(ax, jy), Vector3.Dot(ay, jy), Vector3.Dot(az, jy));
        float cxx = Vector3.Dot(dx, dx) + .3f, cyy = Vector3.Dot(dy, dy) + .3f, cxy = Vector3.Dot(dx, dy);
        float determinant = cxx * cyy - cxy * cxy;
        if (!float.IsFinite(determinant) || determinant < 1e-12f) return false;
        float radiusX = 3 * MathF.Sqrt(cxx), radiusY = 3 * MathF.Sqrt(cyy);
        if (sx + radiusX < .5f || sy + radiusY < .5f || sx - radiusX > width - .5f || sy - radiusY > height - .5f) return false;
        // Clamp as floats before conversion; a valid large ellipse may extend far beyond int range.
        int left = (int)MathF.Ceiling(Math.Clamp(sx - radiusX - .5f, 0, width - 1));
        int right = (int)MathF.Floor(Math.Clamp(sx + radiusX - .5f, 0, width - 1));
        int top = (int)MathF.Ceiling(Math.Clamp(sy - radiusY - .5f, 0, height - 1));
        int bottom = (int)MathF.Floor(Math.Clamp(sy + radiusY - .5f, 0, height - 1));
        if (left > right || top > bottom) return false;
        projected = new(sx, sy, z, cyy / determinant, -cxy / determinant, cxx / determinant,
            gaussian.Color, opacity, left, right, top, bottom, ordinal);
        return true;
    }

    private static void Draw(in Projected splat, float[] target, int tileLeft, int tileTop,
        int tileWidth, int tileHeight, CancellationToken cancellationToken)
    {
        int left = Math.Max(tileLeft, splat.Left), right = Math.Min(tileLeft + tileWidth - 1, splat.Right);
        int top = Math.Max(tileTop, splat.Top), bottom = Math.Min(tileTop + tileHeight - 1, splat.Bottom);
        for (int y = top; y <= bottom; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float dy = y + .5f - splat.Y;
            for (int x = left; x <= right; x++)
            {
                float dx = x + .5f - splat.X;
                float q = splat.A * dx * dx + 2 * splat.B * dx * dy + splat.C * dy * dy;
                if (q > 9 || q < 0) continue;
                float alpha = splat.Opacity * Falloff[Math.Clamp((int)(q * (FalloffSamples / 9f)), 0, FalloffSamples)];
                if (alpha < MinimumAlpha) continue;
                int pixel = ((y - tileTop) * TileSize + x - tileLeft) * 4;
                float remaining = 1 - alpha;
                target[pixel] = splat.Color.Z * alpha + target[pixel] * remaining;
                target[pixel + 1] = splat.Color.Y * alpha + target[pixel + 1] * remaining;
                target[pixel + 2] = splat.Color.X * alpha + target[pixel + 2] * remaining;
                target[pixel + 3] = alpha + target[pixel + 3] * remaining;
            }
        }
    }

    private static bool Finite(Vector3 value) => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
    private readonly record struct Projected(float X, float Y, float Depth, float A, float B, float C,
        Vector3 Color, float Opacity, int Left, int Right, int Top, int Bottom, int Ordinal);
    private sealed class DepthOrder : IComparer<Projected>
    {
        public static DepthOrder Instance { get; } = new();
        public int Compare(Projected a, Projected b)
        {
            int depth = b.Depth.CompareTo(a.Depth);
            return depth != 0 ? depth : a.Ordinal.CompareTo(b.Ordinal);
        }
    }
}
