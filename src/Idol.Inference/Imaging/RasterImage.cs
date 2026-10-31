using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Idol.Inference.Imaging;

/// <summary>RGBA pixels with Windows image codecs and a C# Lanczos resampler.</summary>
public sealed class RasterImage : IDisposable
{
    public int Width { get; }
    public int Height { get; }
    public byte[] Pixels { get; }
    public RasterImage(int width, int height)
    {
        if (width < 1 || height < 1 || (long)width * height > 64L * 1024 * 1024)
            throw new ArgumentException("Image dimensions must be positive and below 64 megapixels.");
        Width = width; Height = height; Pixels = new byte[checked(width * height * 4)];
    }
    public Color this[int x, int y]
    {
        get { int i = (y * Width + x) * 4; return Color.FromArgb(Pixels[i + 3], Pixels[i], Pixels[i + 1], Pixels[i + 2]); }
        set { int i = (y * Width + x) * 4; Pixels[i] = value.R; Pixels[i + 1] = value.G; Pixels[i + 2] = value.B; Pixels[i + 3] = value.A; }
    }
    public static RasterImage Load(string path)
    {
        using var source = new Bitmap(path);
        if (source.PropertyIdList.Contains(0x112))
        {
            int orientation = source.GetPropertyItem(0x112)?.Value?[0] ?? 1;
            source.RotateFlip(orientation switch { 2 => RotateFlipType.RotateNoneFlipX, 3 => RotateFlipType.Rotate180FlipNone,
                4 => RotateFlipType.Rotate180FlipX, 5 => RotateFlipType.Rotate90FlipX, 6 => RotateFlipType.Rotate90FlipNone,
                7 => RotateFlipType.Rotate270FlipX, 8 => RotateFlipType.Rotate270FlipNone, _ => RotateFlipType.RotateNoneFlipNone });
        }
        var image = new RasterImage(source.Width, source.Height);
        using var bitmap = new Bitmap(source.Width, source.Height, PixelFormat.Format32bppArgb);
        using (var graphics = Graphics.FromImage(bitmap)) { graphics.CompositingMode = CompositingMode.SourceCopy; graphics.DrawImageUnscaled(source, 0, 0); }
        var data = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            var row = new byte[bitmap.Width * 4];
            for (int y = 0; y < bitmap.Height; y++)
            {
                Marshal.Copy(data.Scan0 + y * data.Stride, row, 0, row.Length);
                for (int x = 0; x < bitmap.Width; x++)
                {
                    int s = x * 4, d = (y * bitmap.Width + x) * 4;
                    image.Pixels[d] = row[s + 2]; image.Pixels[d + 1] = row[s + 1];
                    image.Pixels[d + 2] = row[s]; image.Pixels[d + 3] = row[s + 3];
                }
            }
        }
        finally { bitmap.UnlockBits(data); }
        return image;
    }
    public void SavePng(string path)
    {
        using var bitmap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);
        var data = bitmap.LockBits(new Rectangle(0, 0, Width, Height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
        try
        {
            var row = new byte[Width * 4];
            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    int s = (y * Width + x) * 4, d = x * 4;
                    row[d] = Pixels[s + 2]; row[d + 1] = Pixels[s + 1]; row[d + 2] = Pixels[s]; row[d + 3] = Pixels[s + 3];
                }
                Marshal.Copy(row, 0, data.Scan0 + y * data.Stride, row.Length);
            }
        }
        finally { bitmap.UnlockBits(data); }
        bitmap.Save(path, ImageFormat.Png);
    }
    public RasterImage Resize(int width, int height, CancellationToken ct = default)
    {
        var output = new RasterImage(width, height);
        // Premultiply before filtering so hidden RGB does not leak across transparent edges.
        var horizontal = new float[checked(width * Height * 4)];
        var wx = Coefficients(Width, width); var wy = Coefficients(Height, height);
        for (int y = 0; y < Height; y++)
        {
            ct.ThrowIfCancellationRequested();
            for (int x = 0; x < width; x++)
            {
                int d = (y * width + x) * 4;
                foreach (var (index, weight) in wx[x])
                {
                    int s = (y * Width + index) * 4; float a = Pixels[s + 3] / 255f;
                    for (int c = 0; c < 3; c++) horizontal[d + c] += Pixels[s + c] * a * weight;
                    horizontal[d + 3] += Pixels[s + 3] * weight;
                }
            }
        }
        for (int y = 0; y < height; y++)
        {
            ct.ThrowIfCancellationRequested();
            for (int x = 0; x < width; x++)
            {
                float r = 0, g = 0, b = 0, a = 0;
                foreach (var (index, weight) in wy[y])
                {
                    int s = (index * width + x) * 4;
                    r += horizontal[s] * weight; g += horizontal[s + 1] * weight;
                    b += horizontal[s + 2] * weight; a += horizontal[s + 3] * weight;
                }
                int d = (y * width + x) * 4;
                output.Pixels[d] = Byte(a > 0 ? r * 255 / a : 0); output.Pixels[d + 1] = Byte(a > 0 ? g * 255 / a : 0);
                output.Pixels[d + 2] = Byte(a > 0 ? b * 255 / a : 0); output.Pixels[d + 3] = Byte(a);
            }
        }
        return output;
    }
    private static byte Byte(float f) => (byte)Math.Clamp((int)MathF.Round(f), 0, 255);
    private static (int Index, float Weight)[][] Coefficients(int source, int target)
    {
        var output = new (int, float)[target][];
        double scale = (double)source / target, filter = Math.Max(1, scale);
        for (int i = 0; i < target; i++)
        {
            double center = (i + .5) * scale - .5, sum = 0;
            var terms = new List<(int, float)>();
            for (int j = (int)Math.Ceiling(center - 3 * filter); j <= (int)Math.Floor(center + 3 * filter); j++)
            {
                double d = (j - center) / filter;
                double weight = Math.Abs(d) < 1e-9 ? 1 : Math.Abs(d) >= 3 ? 0 :
                    3 * Math.Sin(Math.PI * d) * Math.Sin(Math.PI * d / 3) / (Math.PI * Math.PI * d * d);
                terms.Add((Math.Clamp(j, 0, source - 1), (float)weight)); sum += weight;
            }
            output[i] = terms.Select(t => (t.Item1, (float)(t.Item2 / sum))).ToArray();
        }
        return output;
    }
    public void Dispose() { }
}
