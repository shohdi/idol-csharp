// Image framing follows speridlabs/idol-sdk preprocess.py (Apache-2.0).
using Idol.Inference.Encoders;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Imaging;

public static class ImagePreprocessor
{
    public static Task<bool> NeedsBackgroundRemovalAsync(string path, CancellationToken cancellationToken = default) =>
        Task.Run(() => { using var image = RasterImage.Load(path); return !HasRealAlpha(image); }, cancellationToken);

    public static Task<RasterImage> PrepareAsync(string path, BiRefNetMask? remover, bool realPhoto = true,
        CancellationToken cancellationToken = default) => Task.Run(() =>
    {
        using var source = RasterImage.Load(path);
        double scale = 1024.0 / Math.Min(source.Width, source.Height);
        using var image = source.Resize(Math.Max(1, (int)Math.Round(source.Width * scale)),
            Math.Max(1, (int)Math.Round(source.Height * scale)), cancellationToken);
        if (!HasRealAlpha(image))
        {
            if (remover is null) throw new InvalidOperationException("This photograph needs the BiRefNet background-removal weights.");
            using var noGrad = no_grad();
            using var scope = NewDisposeScope();
            var input = ToTensor(image, remover.Device);
            var alpha = remover.Predict(input, cancellationToken).clamp(0, 1).mul(255).to_type(ScalarType.Byte).cpu().contiguous();
            var values = alpha.data<byte>().ToArray();
            for (int i = 0; i < values.Length; i++) image.Pixels[i * 4 + 3] = values[i];
        }
        return FrameAlpha(image, realPhoto ? 1.7 : 1.2, cancellationToken);
    }, cancellationToken);

    public static RasterImage FrameAlpha(RasterImage image, double cropMargin = 1.7, CancellationToken cancellationToken = default)
    {
        if (!double.IsFinite(cropMargin) || cropMargin <= 0) throw new ArgumentOutOfRangeException(nameof(cropMargin));
        int width = image.Width, height = image.Height;
        var alpha = new byte[checked(width * height)];
        for (int i = 0; i < alpha.Length; i++) alpha[i] = image.Pixels[i * 4 + 3];
        int left = width, right = -1, top = height, bottom = -1;
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int x = 0; x < width; x++)
            {
                byte minimum = 255;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                    minimum = Math.Min(minimum, alpha[Math.Clamp(y + dy, 0, height - 1) * width + Math.Clamp(x + dx, 0, width - 1)]);
                image.Pixels[(y * width + x) * 4 + 3] = minimum;
                if (minimum == 0) continue;
                left = Math.Min(left, x); right = Math.Max(right, x); top = Math.Min(top, y); bottom = Math.Max(bottom, y);
            }
        }
        if (right < left) throw new ArgumentException("No object was found in the image: its alpha channel is empty after erosion.");
        double centerX = (left + right) / 2.0, centerY = (top + bottom) / 2.0;
        double half = Math.Max(right - left, bottom - top) / 2.0 * cropMargin;
        int cropLeft = (int)(centerX - half), cropTop = (int)(centerY - half);
        int cropRight = (int)(centerX + half), cropBottom = (int)(centerY + half);
        using var cropped = new RasterImage(Math.Max(1, cropRight - cropLeft), Math.Max(1, cropBottom - cropTop));
        for (int y = 0; y < cropped.Height; y++) for (int x = 0; x < cropped.Width; x++)
        {
            int sx = x + cropLeft, sy = y + cropTop;
            if (sx >= 0 && sx < width && sy >= 0 && sy < height)
                Array.Copy(image.Pixels, (sy * width + sx) * 4, cropped.Pixels, (y * cropped.Width + x) * 4, 4);
        }
        var result = cropped.Resize(1024, 1024, cancellationToken);
        for (int i = 0; i < result.Pixels.Length; i += 4)
        {
            for (int c = 0; c < 3; c++) result.Pixels[i + c] = (byte)((result.Pixels[i + c] * result.Pixels[i + 3] + 127) / 255);
            result.Pixels[i + 3] = 255;
        }
        return result;
    }

    public static Tensor ToTensor(RasterImage image, Device device)
    {
        using var scope = NewDisposeScope();
        int plane = image.Width * image.Height;
        var values = new float[checked(plane * 3)];
        for (int i = 0; i < plane; i++) for (int c = 0; c < 3; c++) values[c * plane + i] = image.Pixels[i * 4 + c] / 255f;
        return tensor(values, device: device).view(1, 3, image.Height, image.Width).MoveToOuterDisposeScope();
    }

    private static bool HasRealAlpha(RasterImage image)
    {
        for (int i = 3; i < image.Pixels.Length; i += 4) if (image.Pixels[i] < 255) return true;
        return false;
    }
}
