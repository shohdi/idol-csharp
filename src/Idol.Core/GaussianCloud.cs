using System.Numerics;

namespace Idol.Core;

/// <summary>A degree-zero Gaussian in standard 3DGS PLY coordinates (y-up). Rotation is XYZW.</summary>
public readonly record struct Gaussian(Vector3 Position, Vector3 FeaturesDc, float OpacityLogit, Vector3 LogScale, Quaternion Rotation)
{
    public const float ShC0 = 0.28209479177387814f;
    public Vector3 Color => Vector3.Clamp(FeaturesDc * ShC0 + new Vector3(0.5f), Vector3.Zero, Vector3.One);
    public float Opacity => OpacityLogit >= 0 ? 1 / (1 + MathF.Exp(-OpacityLogit)) : MathF.Exp(OpacityLogit) / (1 + MathF.Exp(OpacityLogit));
    public Vector3 Scale => new(MathF.Exp(LogScale.X), MathF.Exp(LogScale.Y), MathF.Exp(LogScale.Z));
    public bool IsFinite => Finite(Position) && Finite(FeaturesDc) && Finite(LogScale) && float.IsFinite(OpacityLogit)
        && float.IsFinite(Rotation.X) && float.IsFinite(Rotation.Y) && float.IsFinite(Rotation.Z) && float.IsFinite(Rotation.W);
    private static bool Finite(Vector3 v) => float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z);
}

public sealed class GaussianCloud
{
    public Gaussian[] Points { get; }
    public int Count => Points.Length;
    public GaussianCloud(Gaussian[] points) => Points = points ?? throw new ArgumentNullException(nameof(points));

    public (Vector3 Min, Vector3 Max) GetBounds()
    {
        if (Count == 0) return (new(-0.5f), new(0.5f));
        var min = new Vector3(float.PositiveInfinity);
        var max = new Vector3(float.NegativeInfinity);
        foreach (var g in Points)
        {
            if (!g.IsFinite || g.Opacity < 0.01f) continue;
            min = Vector3.Min(min, g.Position);
            max = Vector3.Max(max, g.Position);
        }
        return float.IsFinite(min.X) ? (min, max) : (new(-0.5f), new(0.5f));
    }
}
