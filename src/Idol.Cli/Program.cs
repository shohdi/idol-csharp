// See https://aka.ms/new-console-template for more information
using Idol.Cli;
using Idol.Core;
using Idol.Core.Export;
using Idol.Inference;

using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
var arguments = args.Where(a => a != "--cpu").ToArray();
bool cpu = args.Contains("--cpu");
var defaultModels = new ModelSetupOptions().ModelDirectory;
// User-supplied credentials remain in the ignored local file and are never included in an export.
if (File.Exists("token.txt") && string.IsNullOrEmpty(Environment.GetEnvironmentVariable("HF_TOKEN_PATH")))
    Environment.SetEnvironmentVariable("HF_TOKEN_PATH", Path.GetFullPath("token.txt"));
try
{
    if (arguments.Length >= 2 && arguments[0] == "background-check")
        return await BackgroundCheck.RunAsync(arguments[1], arguments.ElementAtOrDefault(2) ?? defaultModels,
            arguments.ElementAtOrDefault(3) ?? Path.GetFullPath("artifacts/background-check"), cpu, cancellation.Token);
    if (arguments.Length >= 3 && arguments[0] == "generate")
    {
        var progress = new Progress<InferenceProgress>(p => Console.WriteLine($"{p.Stage} {p.Fraction:P0}: {p.Detail}"));
        using var pipeline = await IdolPipeline.LoadAsync(new ModelSetupOptions
        {
            UseCpu = cpu, ModelDirectory = arguments.ElementAtOrDefault(3) ?? defaultModels
        }, progress, cancellation.Token);
        var result = await pipeline.GenerateAsync(arguments[1], new GenerationOptions(), progress, cancellation.Token);
        var output = Path.GetFullPath(arguments[2]);
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        PlyFile.Write(output, result.Cloud);
        UnityExport.WriteZip(Path.ChangeExtension(output, "unity.zip"), result.Cloud, Path.GetFileNameWithoutExtension(output));
        Console.WriteLine($"Saved {result.Cloud.Count:N0} Gaussians to {output}");
        return 0;
    }
    Console.WriteLine("Idol C# — local image-to-3D");
    Console.WriteLine("  generate <photo> <output.ply> [model-folder] [--cpu]");
    Console.WriteLine("  background-check <photo> [model-folder] [output-folder] [--cpu]");
    Console.WriteLine("Token: HF_TOKEN, HF_TOKEN_PATH, or an ignored token.txt in the current folder.");
    return arguments.Length == 0 ? 0 : 2;
}
catch (OperationCanceledException) { Console.Error.WriteLine("Canceled."); return 130; }
catch (Exception error)
{
    // Do not print request headers, environment variables or credential files.
    Console.Error.WriteLine(error.GetBaseException().Message);
    return 1;
}
