namespace Idol.Inference;

public sealed record ModelSetupOptions
{
    public string ModelDirectory { get; init; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Idol", "models");
    public string? HuggingFaceToken { get; init; }
    public bool UseCpu { get; init; }
    public bool DownloadMissing { get; init; } = true;
}

public sealed record GenerationOptions
{
    public int Seed { get; init; } = 42;
    public int Steps { get; init; } = 50;
    public int NumGaussians { get; init; } = 65_536;
    public double Guidance { get; init; } = 3;
    public double TimeShift { get; init; } = 3;
    public bool RealPhoto { get; init; } = true;

    public void Validate()
    {
        if (Steps is < 1 or > 200) throw new ArgumentOutOfRangeException(nameof(Steps), "Use 1–200 sampling steps.");
        if (NumGaussians is < 32_768 or > 1_048_576 || NumGaussians % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(NumGaussians), "Use 32,768–1,048,576 Gaussians in multiples of 32.");
        if (!double.IsFinite(Guidance) || Guidance < 0 || Guidance > 20)
            throw new ArgumentOutOfRangeException(nameof(Guidance));
        if (!double.IsFinite(TimeShift) || TimeShift <= 0 || TimeShift > 100)
            throw new ArgumentOutOfRangeException(nameof(TimeShift));
    }
}

public sealed record InferenceProgress(string Stage, double Fraction = 0, string? Detail = null);
public sealed record CameraInfo(double Yaw, double Pitch, double Radius, double Fov);
public sealed record IdolResult(Idol.Core.GaussianCloud Cloud, string? PreparedImagePath, CameraInfo Camera);
