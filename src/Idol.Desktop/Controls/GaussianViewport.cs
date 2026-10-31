using System.Numerics;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Idol.Core;
using Idol.Core.Rendering;

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
    private bool _renderAgain;
    private TaskCompletionSource _renderIdle = CompletedRenderSignal();
    public bool IsRendering { get; private set; }
    public bool IsInteractiveFrame { get; private set; }
    public int SubmittedCount { get; private set; }
    public string? RenderError { get; private set; }
    public int DisplayedCount { get; private set; }
    public event EventHandler? ViewChanged;

    public GaussianViewport()
    {
        Focusable = true;
        ClipToBounds = true;
        Cursor = Cursors.Hand;
        SizeChanged += (_, _) => RequestRender();
        Loaded += (_, _) => RequestRender();
        Unloaded += (_, _) => { _renderAgain = false; _renderCancellation?.Cancel(); };
    }

    public void SetCloud(GaussianCloud? cloud)
    {
        _renderCancellation?.Cancel();
        SubmittedCount = DisplayedCount = 0;
        RenderError = null;
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
    protected override void OnLostMouseCapture(MouseEventArgs e)
    {
        base.OnLostMouseCapture(e);
        if (_dragging) { _dragging = false; RequestRender(); }
    }
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

    private void RequestRender()
    {
        InvalidateVisual();
        if (!IsLoaded || _cloud == null || ActualWidth < 1 || ActualHeight < 1) return;
        _renderAgain = true;
        if (_renderIdle.Task.IsCompleted) _renderIdle = new(TaskCreationOptions.RunContinuationsAsynchronously);
        if (!IsRendering) RenderPendingFrames();
    }

    public async Task WaitForRenderAsync()
    {
        await _renderIdle.Task;
        if (RenderError != null) throw new InvalidOperationException(RenderError);
    }

    private static TaskCompletionSource CompletedRenderSignal()
    {
        var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        completion.SetResult(); return completion;
    }

    // Coalesce camera changes instead of continuously canceling a million-point
    // projection. Every frame submits the whole cloud. Dragging reduces image
    // resolution, then the last camera pose is refined at normal resolution.
    private async void RenderPendingFrames()
    {
        IsRendering = true;
        ViewChanged?.Invoke(this, EventArgs.Empty);
        try
        {
            while (_renderAgain && _cloud != null && ActualWidth >= 1 && ActualHeight >= 1)
            {
                _renderAgain = false;
                var cloud = _cloud;
                var camera = Camera();
                var interactive = _dragging;
                var scale = Math.Min(1, Math.Min(1100 / ActualWidth, 1100 / ActualHeight)) * (interactive ? .4 : 1);
                int width = Math.Max(1, (int)(ActualWidth * scale));
                int height = Math.Max(1, (int)(ActualHeight * scale));
                using var cancellation = new CancellationTokenSource();
                _renderCancellation = cancellation;
                try
                {
                    var result = await Task.Run(() => GaussianRasterizer.Render(cloud,
                        new SplatCamera(camera.Eye, camera.Right, camera.Up, camera.Forward),
                        width, height, cancellation.Token), cancellation.Token);
                    if (cancellation.IsCancellationRequested || _cloud != cloud) continue;
                    var bitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Pbgra32, null, result.Pixels, width * 4);
                    bitmap.Freeze(); _bitmap = bitmap;
                    SubmittedCount = result.SubmittedCount; DisplayedCount = result.VisibleCount;
                    IsInteractiveFrame = interactive; RenderError = null;
                    InvalidateVisual(); ViewChanged?.Invoke(this, EventArgs.Empty);
                }
                catch (OperationCanceledException) { }
                catch (Exception error) { RenderError = error.Message; _renderAgain = false; }
                finally { if (_renderCancellation == cancellation) _renderCancellation = null; }
            }
        }
        finally
        {
            IsRendering = false;
            _renderIdle.TrySetResult();
            ViewChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
