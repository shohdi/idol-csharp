using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using Idol.Core;
using Idol.Core.Export;
using Idol.Inference;
using Microsoft.Win32;

namespace Idol.Desktop;

public partial class MainWindow : Window
{
    private IdolPipeline? _pipeline;
    private ModelSetupOptions? _loadedSetup;
    private GaussianCloud? _cloud;
    private string? _imagePath;
    private CancellationTokenSource? _operation;
    private bool _closeWhenIdle;
    private bool _busy;

    public MainWindow()
    {
        InitializeComponent();
        ModelDirectoryBox.Text = new ModelSetupOptions().ModelDirectory;
#if IDOL_CPU
        CpuBox.IsChecked = true;
        CpuBox.IsEnabled = false;
        CpuBox.ToolTip = "This build includes the CPU runtime. Build with IdolUseCuda=true to enable CUDA.";
#endif
        Viewport.ViewChanged += (_, _) => UpdateObjectInfo();
        Closing += OnClosing;
    }

    private async void Browse_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Choose a photo of one object",
            Filter = "Photos|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff|All files|*.*"
        };
        if (dialog.ShowDialog(this) != true) return;
        await GenerateImagePathAsync(dialog.FileName);
    }

    public async Task GenerateImagePathAsync(string path)
    {
        if (_busy) return;
        try
        {
            var image = new BitmapImage();
            using var stream = File.OpenRead(path);
            image.BeginInit(); image.CacheOption = BitmapCacheOption.OnLoad; image.DecodePixelWidth = 520;
            image.StreamSource = stream; image.EndInit(); image.Freeze();
            PhotoPreview.Source = image;
        }
        catch { PhotoPreview.Source = null; }
        PhotoPlaceholder.Visibility = Visibility.Collapsed;
        _imagePath = path; PhotoName.Text = Path.GetFileName(_imagePath);
        await GenerateAsync();
    }

    private async void Regenerate_Click(object sender, RoutedEventArgs e) => await GenerateAsync();

    private async Task GenerateAsync()
    {
        if (_imagePath == null || _operation != null) return;
        try
        {
            if (!int.TryParse(StepsBox.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var steps)) throw new ArgumentException("Enter a whole number for steps (1–200).");
            if (!int.TryParse(SeedBox.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var seed)) throw new ArgumentException("Enter a whole number for the seed.");
            var options = new GenerationOptions
            {
                Steps = steps, Seed = seed,
                NumGaussians = int.Parse((string)((ComboBoxItem)CountBox.SelectedItem).Tag, CultureInfo.InvariantCulture),
                RealPhoto = RealPhotoBox.IsChecked == true
            };
            options.Validate();
            var setup = new ModelSetupOptions
            {
                ModelDirectory = Path.GetFullPath(ModelDirectoryBox.Text.Trim()),
                HuggingFaceToken = string.IsNullOrWhiteSpace(TokenBox.Password) ? Environment.GetEnvironmentVariable("HF_TOKEN") : TokenBox.Password.Trim(),
                UseCpu = CpuBox.IsChecked == true
            };
            using var cancellation = new CancellationTokenSource(); _operation = cancellation;
            SetBusy(true); StageText.Text = "Preparing Idol"; DetailText.Text = "Checking model files and encoder access…";
            var progress = new Progress<InferenceProgress>(p =>
            {
                if (_operation != cancellation) return;
                StageText.Text = p.Stage;
                DetailText.Text = p.Detail ?? "Please keep the application open.";
                Progress.IsIndeterminate = p.Fraction <= 0;
                Progress.Value = Math.Clamp(p.Fraction * 100, 0, 100);
            });
            if (_pipeline == null || setup != _loadedSetup)
            {
                _pipeline?.Dispose(); _pipeline = null; _loadedSetup = null;
                _pipeline = await Task.Run(() => IdolPipeline.LoadAsync(setup, progress, cancellation.Token), cancellation.Token);
                _loadedSetup = setup;
            }
            var result = await Task.Run(() => _pipeline.GenerateAsync(_imagePath, options, progress, cancellation.Token), cancellation.Token);
            _cloud = result.Cloud; Viewport.SetCloud(_cloud);
            StageText.Text = "Object ready";
            DetailText.Text = $"{_cloud.Count:N0} Gaussians · Full object retained for PLY and Unity export";
            Progress.IsIndeterminate = false; Progress.Value = 100;
            UpdateObjectInfo();
        }
        catch (OperationCanceledException)
        {
            StageText.Text = "Conversion canceled"; DetailText.Text = "Choose another photo or generate again when ready.";
            Progress.IsIndeterminate = false; Progress.Value = 0;
        }
        catch (Exception ex)
        {
            StageText.Text = "Conversion could not finish";
            DetailText.Text = SafeError(ex);
            Progress.IsIndeterminate = false; Progress.Value = 0;
            MessageBox.Show(this, SafeError(ex), "Idol setup / conversion", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        finally
        {
            _operation = null; SetBusy(false);
            if (_closeWhenIdle) Close();
        }
    }

    private string SafeError(Exception error)
    {
        var message = error.GetBaseException().Message;
        foreach (var token in new[] { TokenBox.Password, Environment.GetEnvironmentVariable("HF_TOKEN") })
            if (!string.IsNullOrEmpty(token)) message = message.Replace(token, "[redacted]", StringComparison.Ordinal);
        return message;
    }

    private void SetBusy(bool busy)
    {
        _busy = busy;
        SettingsPanel.IsEnabled = !busy; OpenPlyButton.IsEnabled = !busy;
        ExportPanel.IsEnabled = !busy && _cloud != null;
        RegenerateButton.IsEnabled = !busy && _imagePath != null;
        CancelButton.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
        CancelButton.IsEnabled = true;
        Progress.IsIndeterminate = busy;
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        _operation?.Cancel(); StageText.Text = "Canceling at the next safe step…";
        CancelButton.IsEnabled = false;
    }

    private async void OpenPly_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Title = "Open a Gaussian splat", Filter = "Gaussian PLY|*.ply;*.idolply|All files|*.*" };
        if (dialog.ShowDialog(this) == true) await OpenPlyPathAsync(dialog.FileName);
    }

    public async Task OpenPlyPathAsync(string path)
    {
        if (_operation != null) return;
        SetBusy(true); StageText.Text = "Opening Gaussian PLY";
        try
        {
            _cloud = await Task.Run(() => PlyFile.Read(path));
            Viewport.SetCloud(_cloud); UpdateObjectInfo();
            StageText.Text = "Object opened"; DetailText.Text = $"{Path.GetFileName(path)} · {_cloud.Count:N0} Gaussians";
            Progress.Value = 100;
        }
        catch (Exception error)
        {
            StageText.Text = "Could not open PLY"; DetailText.Text = error.Message;
            MessageBox.Show(this, error.Message, "Open PLY", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        finally { SetBusy(false); Progress.IsIndeterminate = false; if (_closeWhenIdle) Close(); }
    }

    private void UpdateObjectInfo()
    {
        ObjectInfo.Text = _cloud == null ? "3D VIEWPORT" :
            Viewport.RenderError != null ? "VIEWPORT ERROR: " + Viewport.RenderError :
            $"{_cloud.Count:N0} GAUSSIANS  /  " + (Viewport.IsRendering ? "RENDERING FULL DENSITY" :
                Viewport.IsInteractiveFrame ? "FULL DENSITY · REFINING" : "FULL DENSITY");
    }

    private void ResetView_Click(object sender, RoutedEventArgs e) => Viewport.ResetCamera();
    private string ObjectName => _imagePath == null ? "IdolObject" : Path.GetFileNameWithoutExtension(_imagePath);

    private async void SavePly_Click(object sender, RoutedEventArgs e)
    {
        if (_cloud == null) return;
        var dialog = new SaveFileDialog { Title = "Save full Gaussian object", Filter = "3D Gaussian PLY|*.ply", FileName = ObjectName + ".ply" };
        if (dialog.ShowDialog(this) == true) await ExportAsync(() => PlyFile.Write(dialog.FileName, _cloud), dialog.FileName, "Full Gaussian PLY saved");
    }

    private async void ExportUnity_Click(object sender, RoutedEventArgs e)
    {
        if (_cloud == null) return;
        var dialog = new SaveFileDialog { Title = "Export Unity Assets folder", Filter = "Unity import ZIP|*.zip", FileName = ObjectName + "-Unity.zip" };
        if (dialog.ShowDialog(this) == true)
            await ExportAsync(() => UnityExport.WriteZip(dialog.FileName, _cloud, ObjectName), dialog.FileName, "Unity export saved. Extract and copy its Assets folder into your Unity project.");
    }

    private async void SaveObj_Click(object sender, RoutedEventArgs e)
    {
        if (_cloud == null) return;
        var dialog = new SaveFileDialog { Title = "Save coarse ellipsoid proxy (appearance differs from splats)", Filter = "OBJ ellipsoid proxy|*.obj", FileName = ObjectName + "-proxy.obj" };
        if (dialog.ShowDialog(this) == true)
            await ExportAsync(() => ObjExporter.Write(dialog.FileName, _cloud), dialog.FileName, "OBJ proxy and MTL saved. This is coarse ellipsoid geometry, not a reconstructed surface.");
    }

    private async Task ExportAsync(Action write, string path, string success)
    {
        SetBusy(true); CancelButton.Visibility = Visibility.Collapsed;
        StageText.Text = "Saving object…"; DetailText.Text = path;
        try { await Task.Run(write); StageText.Text = success; Progress.Value = 100; }
        catch (Exception error)
        {
            StageText.Text = "Export could not finish"; DetailText.Text = error.Message;
            MessageBox.Show(this, error.Message, "Export", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        finally { SetBusy(false); Progress.IsIndeterminate = false; if (_closeWhenIdle) Close(); }
    }

    private void ModelLink_RequestNavigate(object sender, RequestNavigateEventArgs e)
    {
        Process.Start(new ProcessStartInfo(e.Uri.AbsoluteUri) { UseShellExecute = true }); e.Handled = true;
    }

    public async Task CaptureAsync(string path)
    {
        UpdateLayout();
        await Viewport.WaitForRenderAsync();
        await System.Windows.Threading.Dispatcher.Yield(System.Windows.Threading.DispatcherPriority.Render);
        await Viewport.WaitForRenderAsync();
        var bitmap = new RenderTargetBitmap((int)ActualWidth, (int)ActualHeight, 96, 96, PixelFormats.Pbgra32);
        bitmap.Render(this);
        var encoder = new PngBitmapEncoder(); encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using var stream = File.Create(path); encoder.Save(stream);
    }

    private void OnClosing(object? sender, CancelEventArgs e)
    {
        if (_busy)
        {
            _closeWhenIdle = true; _operation?.Cancel(); e.Cancel = true;
            StageText.Text = "Finishing the current operation before closing…";
        }
        else { _pipeline?.Dispose(); _pipeline = null; }
    }
}
