namespace Idol.Inference;

public sealed record ModelFile(string Repository, string Revision, string RemotePath,
    string LocalPath, long Size, string? Sha256 = null, bool Gated = false)
{
    public Uri DownloadUri => new($"https://huggingface.co/{Repository}/resolve/{Revision}/{RemotePath}");
}

public static class ModelManifest
{
    public const string IdolRevision = "2731c2291fc65a797c5cb627a5e5526fc1490902";
    public const string DinoRevision = "c807c9eeea853df70aec4069e6f56b28ddc82acc";
    public const string FluxRevision = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c";
    public static IReadOnlyList<ModelFile> Files { get; } =
    [
        new("speridlabs/idol", IdolRevision, "config.json", "idol/config.json", 1335),
        new("speridlabs/idol", IdolRevision, "denoiser.safetensors", "idol/denoiser.safetensors", 1482258308,
            "ded2cacf671bf3978ebc62729087abdbd318ef4752e6805b4c9fc8ffa51fe716"),
        new("speridlabs/idol", IdolRevision, "decoder.safetensors", "idol/decoder.safetensors", 203171696,
            "9da4d7676bfcadb1c3fdfbff796e6257dadf82d26ec05be7717c79b2b13a1c4e"),
        new("speridlabs/idol", IdolRevision, "decoder_gs.safetensors", "idol/decoder_gs.safetensors", 949083036,
            "3087eb8ee5763206a53e4078acf2a1bf382dab5147d407ec13a2cf39c04cc544"),
        new("speridlabs/idol", IdolRevision, "birefnet.safetensors", "idol/birefnet.safetensors", 444466856,
            "44bbc97b041f879975d0d6d538b8e66a25eb04d1f49aab40697544cdd97eaa8e"),
        new("facebook/dinov3-vith16plus-pretrain-lvd1689m", DinoRevision, "config.json", "dinov3/config.json", 744, Gated: true),
        new("facebook/dinov3-vith16plus-pretrain-lvd1689m", DinoRevision, "model.safetensors", "dinov3/model.safetensors", 3362432800,
            "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d", true),
        new("black-forest-labs/FLUX.2-dev", FluxRevision, "vae/config.json", "flux2/vae/config.json", 770, Gated: true),
        new("black-forest-labs/FLUX.2-dev", FluxRevision, "vae/diffusion_pytorch_model.safetensors", "flux2/vae/diffusion_pytorch_model.safetensors", 336213556,
            "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5", true)
    ];
}
