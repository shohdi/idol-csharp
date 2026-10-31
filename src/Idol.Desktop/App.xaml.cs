using System.Windows;

namespace Idol.Desktop;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        var window = new MainWindow(); MainWindow = window;
        window.Loaded += async (_, _) =>
        {
            string? Argument(string name)
            {
                var index = Array.IndexOf(e.Args, name);
                return index >= 0 && index + 1 < e.Args.Length ? e.Args[index + 1] : null;
            }
            try
            {
                var ply = Argument("--open-ply");
                if (ply != null) await window.OpenPlyPathAsync(ply);
                var capture = Argument("--capture");
                if (capture != null) await window.CaptureAsync(capture);
                if (e.Args.Contains("--exit-after-capture")) window.Close();
            }
            catch (Exception error)
            {
                System.IO.File.WriteAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "idol-desktop-startup-error.txt"), error.ToString());
                Shutdown(1);
            }
        };
        window.Show();
    }
}
