using System.Numerics;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Idol.Core;

namespace Idol.Desktop.Controls;

/// <summary>Dependency-free orbit viewer with covariance-projected, depth-sorted Gaussian splats.</summary>
public sealed class GaussianViewport : FrameworkElement
{
    private GaussianCloud? _cloud;
    private Vector3 _target;
    private (Vector3 Min, Vector3 Max) _bounds = (new(-.5f), new(.5f));
    private float _yaw = .5f, _pitch = .22f, _distance = 2;
    private Point _last;
    private bool _dragging, _panning;
    private BitmapSource? _bitmap;
    private CancellationTokenSource? _renderCancellation;
    private static readonly float[] Falloff = Enumerable.Range(0, 2049).Select(i => MathF.Exp(-.5f * i * 9 / 2048)).ToArray();
    public int PreviewBudget { get; set; } = 20000;
    public int DisplayedCount { get; private set; }
    public event EventHandler? ViewChanged;

    public GaussianViewport()
    {
        Focusable = true;
        ClipToBounds = true;
        Cursor = Cursors.Hand;
        SizeChanged += (_, _) => RequestRender();
        Unloaded += (_, _) => _renderCancellation?.Cancel();
    }

    public void SetCloud(GaussianCloud? cloud)
    {
        _cloud = cloud; _bitmap = null; _bounds = cloud?.GetBounds() ?? (new Vector3(-.5f), new Vector3(.5f));
        ResetCamera();
    }

    public void ResetCamera()
    {
        var bounds = _bounds;
        _target = (bounds.Item1 + bounds.Item2) * .5f;
        _distance = Math.Max(.2f, (bounds.Item2 - bounds.Item1).Length() * 1.45f);
        _yaw = .5f; _pitch = .22f;
        RequestRender();
    }

    private (Vector3 Eye, Vector3 Right, Vector3 Up, Vector3 Forward) Camera()
    {
        var eye = _target + _distance * new Vector3(MathF.Cos(_pitch) * MathF.Sin(_yaw), MathF.Sin(_pitch), MathF.Cos(_pitch) * MathF.Cos(_yaw));
        var forward = Vector3.Normalize(_target - eye);
        var right = Vector3.Normalize(Vector3.Cross(forward, Vector3.UnitY));
        var up = Vector3.Cross(right, forward);
        return (eye, right, up, forward);
    }

    protected override void OnRender(DrawingContext dc)
    {
        base.OnRender(dc);
        dc.DrawRectangle(new LinearGradientBrush(Color.FromRgb(17, 24, 37), Color.FromRgb(9, 15, 25), 90), null, new Rect(RenderSize));
        DrawGrid(dc);
        if (_bitmap != null) dc.DrawImage(_bitmap, new Rect(RenderSize));
        if (_cloud == null)
        {
            Text(dc, "Your object appears here", 22, Color.FromRgb(177, 190, 209), ActualWidth / 2, ActualHeight * .43);
            Text(dc, "Choose a photo to start, or open a Gaussian PLY", 13, Color.FromRgb(107, 125, 149), ActualWidth / 2, ActualHeight * .43 + 38);
        }
        Text(dc, "Y", 11, Color.FromRgb(104, 207, 170), 30, ActualHeight - 65);
        Text(dc, "X", 11, Color.FromRgb(227, 131, 129), 53, ActualHeight - 38);
        Text(dc, "Z", 11, Color.FromRgb(120, 163, 238), 17, ActualHeight - 36);
        dc.DrawLine(new Pen(new SolidColorBrush(Color.FromRgb(104, 207, 170)), 1.5), new Point(30, ActualHeight - 42), new Point(30, ActualHeight - 63));
        dc.DrawLine(new Pen(new SolidColorBrush(Color.FromRgb(227, 131, 129)), 1.5), new Point(30, ActualHeight - 42), new Point(49, ActualHeight - 37));
        dc.DrawLine(new Pen(new SolidColorBrush(Color.FromRgb(120, 163, 238)), 1.5), new Point(30, ActualHeight - 42), new Point(19, ActualHeight - 34));
    }

    private void DrawGrid(DrawingContext dc)
    {
        var camera = Camera();
        var f = (float)(ActualHeight * 1.2071);
        bool Project(Vector3 p, out Point screen)
        {
            var d = p - camera.Eye; var z = Vector3.Dot(d, camera.Forward);
            screen = z <= .01f ? default : new Point(ActualWidth * .5 + f * Vector3.Dot(d, camera.Right) / z, ActualHeight * .5 - f * Vector3.Dot(d, camera.Up) / z);
            return z > .01f;
        }
        var pen = new Pen(new SolidColorBrush(Color.FromArgb(45, 77, 100, 131)), 1);
        var bottom = _bounds.Min.Y;
        for (int i = -10; i <= 10; ++i)
        {
            float u = i * .2f;
            if (Project(new(u, bottom, -2), out var a) && Project(new(u, bottom, 2), out var b)) dc.DrawLine(pen, a, b);
            if (Project(new(-2, bottom, u), out a) && Project(new(2, bottom, u), out b)) dc.DrawLine(pen, a, b);
        }
    }

    private void Text(DrawingContext dc, string text, double size, Color color, double x, double y)
    {
        var formatted = new FormattedText(text, System.Globalization.CultureInfo.CurrentUICulture, FlowDirection.LeftToRight,
            new Typeface("Segoe UI"), size, new SolidColorBrush(color), VisualTreeHelper.GetDpi(this).PixelsPerDip);
        dc.DrawText(formatted, new Point(x - formatted.Width * .5, y));
    }

    protected override void OnMouseDown(MouseButtonEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.ChangedButton is not (MouseButton.Left or MouseButton.Right or MouseButton.Middle)) return;
        Focus(); _dragging = true;
        _panning = e.ChangedButton != MouseButton.Left || Keyboard.Modifiers.HasFlag(ModifierKeys.Shift);
        _last = e.GetPosition(this); CaptureMouse(); e.Handled = true;
    }
    protected override void OnMouseUp(MouseButtonEventArgs e)
    {
        base.OnMouseUp(e); _dragging = false; ReleaseMouseCapture(); RequestRender();
    }
    protected override void OnLostMouseCapture(MouseEventArgs e) { base.OnLostMouseCapture(e); _dragging = false; }
    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e); if (!_dragging) return;
        var p = e.GetPosition(this); var delta = p - _last; _last = p;
        if (_panning)
        {
            var camera = Camera();
            var scale = _distance / (float)Math.Max(1, ActualHeight) * .8f;
            _target += camera.Right * (float)-delta.X * scale + camera.Up * (float)delta.Y * scale;
        }
        else { _yaw -= (float)delta.X * .008f; _pitch = Math.Clamp(_pitch + (float)delta.Y * .008f, -1.48f, 1.48f); }
        RequestRender(); e.Handled = true;
    }
    protected override void OnMouseWheel(MouseWheelEventArgs e)
    {
        base.OnMouseWheel(e); _distance = Math.Clamp(_distance * MathF.Pow(.88f, e.Delta / 120f), .025f, 100);
        RequestRender(); e.Handled = true;
    }
    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e); if (e.Key == Key.Home || e.Key == Key.F) { ResetCamera(); e.Handled = true; }
    }

    private async void RequestRender()
    {
        InvalidateVisual();
        _renderCancellation?.Cancel();
        if (_cloud == null || ActualWidth < 1 || ActualHeight < 1) return;
        var cancellation = new CancellationTokenSource(); _renderCancellation = cancellation;
        var camera = Camera(); var cloud = _cloud;
        var width = Math.Max(1, Math.Min(1100, (int)ActualWidth));
        var height = Math.Max(1, (int)(ActualHeight * width / ActualWidth));
        var budget = _dragging ? Math.Min(5000, PreviewBudget) : PreviewBudget;
        try
        {
            var result = await Task.Run(() => Render(cloud, camera, width, height, budget, cancellation.Token), cancellation.Token);
            if (cancellation.IsCancellationRequested || _cloud != cloud) return;
            var bitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Pbgra32, null, result.Pixels, width * 4);
            bitmap.Freeze(); _bitmap = bitmap; DisplayedCount = result.Count;
            InvalidateVisual(); ViewChanged?.Invoke(this, EventArgs.Empty);
        }
        catch (OperationCanceledException) { }
        finally { cancellation.Dispose(); if (_renderCancellation == cancellation) _renderCancellation = null; }
    }

    private readonly record struct Projected(float X, float Y, float Depth, float A, float B, float C, float RadiusX, float RadiusY, Vector3 Color, float Opacity);
    private static (byte[] Pixels, int Count) Render(GaussianCloud cloud, (Vector3 Eye, Vector3 Right, Vector3 Up, Vector3 Forward) camera, int width, int height, int budget, CancellationToken cancellation)
    {
        var list = new List<Projected>(Math.Min(budget, cloud.Count));
        float f = height * 1.2071f;
        var stride = Math.Max(1, (int)Math.Ceiling(cloud.Count / (double)budget));
        for (int i = 0; i < cloud.Count; i += stride)
        {
            if ((i & 63) == 0) cancellation.ThrowIfCancellationRequested();
            var g = cloud.Points[i]; var opacity = Math.Min(.99f, g.Opacity);
            if (!g.IsFinite || opacity < .01f) continue;
            var relative = g.Position - camera.Eye;
            float x = Vector3.Dot(relative, camera.Right), y = Vector3.Dot(relative, camera.Up), z = Vector3.Dot(relative, camera.Forward);
            if (z < .01f) continue;
            var sx = width * .5f + f * x / z; var sy = height * .5f - f * y / z;
            var jx = camera.Right * (f / z) - camera.Forward * (f * x / (z * z));
            var jy = -camera.Up * (f / z) + camera.Forward * (f * y / (z * z));
            var s = g.Scale;
            var ax = Vector3.Transform(new Vector3(s.X, 0, 0), g.Rotation);
            var ay = Vector3.Transform(new Vector3(0, s.Y, 0), g.Rotation);
            var az = Vector3.Transform(new Vector3(0, 0, s.Z), g.Rotation);
            var dx = new Vector3(Vector3.Dot(ax, jx), Vector3.Dot(ay, jx), Vector3.Dot(az, jx));
            var dy = new Vector3(Vector3.Dot(ax, jy), Vector3.Dot(ay, jy), Vector3.Dot(az, jy));
            var xx = Vector3.Dot(dx, dx) + .3f; var yy = Vector3.Dot(dy, dy) + .3f; var xy = Vector3.Dot(dx, dy);
            var det = xx * yy - xy * xy;
            if (!float.IsFinite(det) || det < 1e-12f) continue;
            var rx = Math.Min(180, 3 * MathF.Sqrt(xx)); var ry = Math.Min(180, 3 * MathF.Sqrt(yy));
            if (sx + rx < 0 || sy + ry < 0 || sx - rx >= width || sy - ry >= height) continue;
            list.Add(new(sx, sy, z, yy / det, -xy / det, xx / det, rx, ry, g.Color, opacity));
        }
        list.Sort((a, b) => b.Depth.CompareTo(a.Depth));
        var pixels = new byte[checked(width * height * 4)];
        var rgb = new float[pixels.Length];
        int rendered = 0;
        foreach (var g in list)
        {
            if ((rendered++ & 63) == 0) cancellation.ThrowIfCancellationRequested();
            int x0 = Math.Max(0, (int)(g.X - g.RadiusX)), x1 = Math.Min(width - 1, (int)(g.X + g.RadiusX));
            int y0 = Math.Max(0, (int)(g.Y - g.RadiusY)), y1 = Math.Min(height - 1, (int)(g.Y + g.RadiusY));
            for (int y = y0; y <= y1; ++y)
            {
                var dy = y + .5f - g.Y;
                for (int x = x0; x <= x1; ++x)
                {
                    var dx = x + .5f - g.X;
                    var q = g.A * dx * dx + 2 * g.B * dx * dy + g.C * dy * dy;
                    if (q > 9 || q < 0) continue;
                    var alpha = g.Opacity * Falloff[Math.Clamp((int)(q * (2048f / 9)), 0, 2048)];
                    if (alpha < 1f / 255) continue;
                    var p = (y * width + x) * 4; var inv = 1 - alpha;
                    rgb[p] = g.Color.Z * alpha + rgb[p] * inv;
                    rgb[p + 1] = g.Color.Y * alpha + rgb[p + 1] * inv;
                    rgb[p + 2] = g.Color.X * alpha + rgb[p + 2] * inv;
                    rgb[p + 3] = alpha + rgb[p + 3] * inv;
                }
            }
        }
        for (int i = 0; i < pixels.Length; ++i) pixels[i] = (byte)Math.Clamp((int)(rgb[i] * 255 + .5f), 0, 255);
        return (pixels, list.Count);
    }
}
