using System.Buffers;
using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace Idol.Inference;

/// <summary>Downloads only pinned Idol and encoder artifacts. Never stores an access token.</summary>
public sealed class ModelDownloader : IDisposable
{
    private readonly HttpClient http;
    private readonly string? token;

    public ModelDownloader(string? token = null, HttpMessageHandler? handler = null)
    {
        this.token = string.IsNullOrWhiteSpace(token) ? FindCachedToken() : token;
        http = handler is null ? new HttpClient() : new HttpClient(handler);
        http.Timeout = TimeSpan.FromMinutes(90);
        http.DefaultRequestHeaders.UserAgent.ParseAdd("IdolCSharp/0.1");
    }

    private static string? FindCachedToken()
    {
        var fromEnvironment = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrWhiteSpace(fromEnvironment)) return fromEnvironment.Trim();
        var home = Environment.GetEnvironmentVariable("HF_HOME") ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface");
        var path = Environment.GetEnvironmentVariable("HF_TOKEN_PATH") ?? Path.Combine(home, "token");
        try { return File.Exists(path) ? File.ReadAllText(path).Trim() : null; }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
    }

    public async Task EnsureAsync(string directory, bool allowDownload,
        IProgress<InferenceProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        directory = Path.GetFullPath(directory);
        Directory.CreateDirectory(directory);
        var missing = ModelManifest.Files.Where(f => !File.Exists(LocalPath(directory, f)) ||
            new FileInfo(LocalPath(directory, f)).Length != f.Size).ToArray();
        if (missing.Length != 0 && !allowDownload)
            throw new FileNotFoundException($"Missing model files in {directory}. Enable downloads in Setup. First missing: {missing[0].LocalPath}");

        // Authorize every gated component before any multi-GB public download starts.
        foreach (var file in missing.Where(f => f.Gated && f.RemotePath.EndsWith("config.json", StringComparison.Ordinal)))
        {
            using var request = Request(HttpMethod.Head, file.DownloadUri);
            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            CheckResponse(response, file);
        }
        // A cached config alone is not proof that the account can still download its weights.
        foreach (var file in missing.Where(f => f.Gated && f.Sha256 is not null))
        {
            using var request = Request(HttpMethod.Head, file.DownloadUri);
            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            CheckResponse(response, file);
        }

        foreach (var (file, index) in ModelManifest.Files.Select((f, i) => (f, i)))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var path = LocalPath(directory, file);
            var portion = 1.0 / ModelManifest.Files.Count;
            if (File.Exists(path) && new FileInfo(path).Length == file.Size &&
                await VerifyHashAsync(path, file.Sha256, cancellationToken).ConfigureAwait(false))
            {
                progress?.Report(new("Models", (index + 1) * portion, $"Verified {file.LocalPath}"));
                continue;
            }
            if (!allowDownload) throw new InvalidDataException($"Model checksum does not match: {file.LocalPath}. Enable downloads to repair it.");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            await DownloadAsync(file, path, new ProgressAdapter(p => progress?.Report(
                new("Downloading models", (index + p) * portion, file.LocalPath))), cancellationToken).ConfigureAwait(false);
        }
    }

    public async Task DownloadAsync(ModelFile file, string path, IProgress<double>? progress,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var partial = path + ".part";
        var offset = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (offset > file.Size) { File.Delete(partial); offset = 0; }
        if (offset < file.Size)
        {
            using var request = Request(HttpMethod.Get, file.DownloadUri);
            if (offset > 0) request.Headers.Range = new RangeHeaderValue(offset, null);
            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            CheckResponse(response, file);
            if (response.StatusCode != HttpStatusCode.PartialContent) offset = 0;
            else if (response.Content.Headers.ContentRange?.From != offset)
                throw new InvalidDataException("The model host returned an incorrect download range.");
            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            await using var destination = new FileStream(partial, offset == 0 ? FileMode.Create : FileMode.Append,
                FileAccess.Write, FileShare.None, 1 << 20, true);
            var buffer = ArrayPool<byte>.Shared.Rent(1 << 20);
            try
            {
                var total = offset;
                var lastReport = Stopwatch.GetTimestamp();
                while (true)
                {
                    var count = await source.ReadAsync(buffer.AsMemory(), cancellationToken).ConfigureAwait(false);
                    if (count == 0) break;
                    total += count;
                    if (total > file.Size) throw new InvalidDataException($"Unexpected download size: {file.LocalPath}.");
                    await destination.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
                    if (Stopwatch.GetElapsedTime(lastReport) >= TimeSpan.FromMilliseconds(250))
                    {
                        progress?.Report((double)total / file.Size);
                        lastReport = Stopwatch.GetTimestamp();
                    }
                }
            }
            finally { ArrayPool<byte>.Shared.Return(buffer); }
        }
        if (new FileInfo(partial).Length != file.Size)
            throw new IOException($"Download interrupted for {file.LocalPath}. Retry to resume.");
        if (!await VerifyHashAsync(partial, file.Sha256, cancellationToken).ConfigureAwait(false))
        {
            File.Delete(partial);
            throw new InvalidDataException($"Checksum failed for {file.LocalPath}. Retry the download.");
        }
        File.Move(partial, path, true);
        progress?.Report(1);
    }

    private HttpRequestMessage Request(HttpMethod method, Uri uri)
    {
        var request = new HttpRequestMessage(method, uri);
        if (!string.IsNullOrEmpty(token)) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        return request;
    }

    private static void CheckResponse(HttpResponseMessage response, ModelFile file)
    {
        if (response.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
        {
            var reason = response.Headers.TryGetValues("X-Error-Message", out var values) ? values.FirstOrDefault() ?? "" : "";
            string detail = reason.Contains("awaiting a review", StringComparison.OrdinalIgnoreCase)
                ? "Your model access request is awaiting review by the model authors.\n"
                : reason.Contains("enable access to public gated", StringComparison.OrdinalIgnoreCase)
                    ? "In your fine-grained token's Repositories section, check 'Read contents of public gated repos you can access' and save the change. Inference-provider permission alone is insufficient.\n"
                    : "Check that your token is valid and can read the approved gated repositories.\n";
            throw new ModelAccessException($"Hugging Face access is required for {file.Repository}.\n" + detail +
                $"Accept the model's terms at https://huggingface.co/{file.Repository}, then enter a read token in Setup or set HF_TOKEN.\n" +
                "Both DINOv3 and FLUX.2-dev need approval. Tokens are never saved by this app.");
        }
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"Cannot download {file.LocalPath}: HTTP {(int)response.StatusCode} ({response.ReasonPhrase}).", null, response.StatusCode);
    }

    public static async Task<bool> VerifyHashAsync(string path, string? expectedHash, CancellationToken cancellationToken = default)
    {
        if (expectedHash is null) return true;
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, true);
        var hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexString(hash).Equals(expectedHash, StringComparison.OrdinalIgnoreCase);
    }

    private static string LocalPath(string directory, ModelFile file) => Path.Combine(directory, file.LocalPath.Replace('/', Path.DirectorySeparatorChar));
    public void Dispose() => http.Dispose();
    private sealed class ProgressAdapter(Action<double> report) : IProgress<double> { public void Report(double value) => report(value); }
}

public sealed class ModelAccessException(string message) : Exception(message);
