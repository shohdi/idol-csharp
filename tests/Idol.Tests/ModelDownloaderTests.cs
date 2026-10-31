using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using Idol.Inference;

namespace Idol.Tests;

public sealed class ModelDownloaderTests
{
    [Fact]
    public async Task GateDenialStopsBeforeDownloadingAnyWeights()
    {
        var calls = new List<HttpMethod>();
        using var handler = new StubHandler(request =>
        {
            calls.Add(request.Method);
            return new HttpResponseMessage(HttpStatusCode.Forbidden);
        });
        using var downloader = new ModelDownloader("test-token", handler);
        using var temp = new TemporaryDirectory();
        var error = await Assert.ThrowsAsync<ModelAccessException>(() => downloader.EnsureAsync(temp.Path, true));
        Assert.Contains("Accept", error.Message);
        Assert.DoesNotContain("test-token", error.Message);
        Assert.Single(calls);
        Assert.Equal(HttpMethod.Head, calls[0]);
        Assert.Empty(Directory.GetFiles(temp.Path, "*", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task InterruptedDownloadResumesThenVerifiesChecksum()
    {
        byte[] content = [1, 3, 5, 7, 9, 11];
        using var temp = new TemporaryDirectory();
        string path = System.IO.Path.Combine(temp.Path, "model.safetensors");
        await File.WriteAllBytesAsync(path + ".part", content[..3]);
        var descriptor = new ModelFile("test/repo", "revision", "model.safetensors", "model.safetensors",
            content.Length, Convert.ToHexString(SHA256.HashData(content)));
        using var handler = new StubHandler(request =>
        {
            Assert.Equal(3, request.Headers.Range!.Ranges.Single().From);
            var response = new HttpResponseMessage(HttpStatusCode.PartialContent) { Content = new ByteArrayContent(content[3..]) };
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(3, 5, 6);
            return response;
        });
        using var downloader = new ModelDownloader("test-token", handler);
        await downloader.DownloadAsync(descriptor, path, null);
        Assert.Equal(content, await File.ReadAllBytesAsync(path));
        Assert.False(File.Exists(path + ".part"));
    }

    [Fact]
    public async Task CorruptWeightIsNeverPromotedToFinalFile()
    {
        using var temp = new TemporaryDirectory();
        var path = System.IO.Path.Combine(temp.Path, "bad.safetensors");
        var descriptor = new ModelFile("test/repo", "revision", "bad.safetensors", "bad.safetensors", 2, new string('0', 64));
        using var handler = new StubHandler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent([4, 5]) });
        using var downloader = new ModelDownloader("test-token", handler);
        await Assert.ThrowsAsync<InvalidDataException>(() => downloader.DownloadAsync(descriptor, path, null));
        Assert.False(File.Exists(path));
        Assert.False(File.Exists(path + ".part"));
    }

    [Fact]
    public void ManifestContainsOnlyNeededFluxVaeAndPinnedRevisions()
    {
        Assert.All(ModelManifest.Files, f => Assert.Matches("^[a-f0-9]{40}$", f.Revision));
        Assert.All(ModelManifest.Files.Where(f => f.Repository == "black-forest-labs/FLUX.2-dev"),
            f => Assert.StartsWith("vae/", f.RemotePath));
        Assert.Equal(9, ModelManifest.Files.Count);
        Assert.InRange(ModelManifest.Files.Sum(f => f.Size), 6_000_000_000, 7_000_000_000);
    }

    [Theory]
    [InlineData(0, 65536)]
    [InlineData(50, 65537)]
    [InlineData(50, 1024)]
    public void InvalidGenerationSettingsFailBeforeInference(int steps, int count) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenerationOptions { Steps = steps, NumGaussians = count }.Validate());

    private sealed class StubHandler(Func<HttpRequestMessage, HttpResponseMessage> run) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => Task.FromResult(run(request));
    }
    private sealed class TemporaryDirectory : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "idol-test-" + Guid.NewGuid().ToString("N"));
        public TemporaryDirectory() => Directory.CreateDirectory(Path);
        public void Dispose() => Directory.Delete(Path, true);
    }
}
