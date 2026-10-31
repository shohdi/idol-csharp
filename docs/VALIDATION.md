# Validation record

Recorded on 2026-10-31. Windows x64; .NET SDK 9.0.200; TorchSharp 0.107.0 / LibTorch 2.10.0. The machine has an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM and driver 592.82.

## Checks performed

- Release solution build with the CPU runtime: **passed, zero warnings/errors**.
- Native component tests: safetensors F32/F16/BF16 and integer buffers, invalid checkpoint ranges, attention chunking, rotary/absolute embeddings, tiny full denoiser, Euler time grid/camera math, octree resampling, deformable convolution, DINO prefix tokens/norm and VAE posterior/packing.
- Preprocessing tests: empty foreground rejection and transparent-image framing/black compositing.
- Download tests: gated access preflight, resumed HTTP range downloads, SHA-256 verification and corrupt-file rejection.
- Export tests: PLY roundtrip/quaternion ordering, malformed/truncated PLY, finite data, Unity ZIP contents/full counts, OBJ indices/material references.
- Real public `decoder.safetensors`: all octree decoder blocks executed; finite logits and 64 anchors within the unit cube.
- Real public `decoder_gs.safetensors`: all 16 Gaussian decoder blocks executed with four anchors/eight context tokens; output contained 128 finite Gaussians with positive scales, bounded opacities and unit quaternions. This is a component check with synthetic latent input, not an image reconstruction.
- Real public `birefnet.safetensors`: native C# background removal ran on the PyTorch Hub dog photograph. The foreground mask was finite in [0,1], with original-image shape [1,1,1213,1546]. CPU execution took about 14 seconds. The resulting black-background crop was visually inspected. The photograph and outputs are local artifacts, not repository assets.
- WPF app: launched, loaded a synthetic 24,000-Gaussian PLY, rendered its preview, captured its window and exited successfully. [Screenshot](images/desktop-viewer.png). The screenshot is a viewer fixture, not a claimed model result.
- Unity 6000.2.6f1: batch import created a prefab with the expected two-Gaussian data asset; quaternion convention was checked; shader reported zero compilation errors; scene was saved. A Direct3D11 camera render produced 1,142 non-black splat pixels.

## Limits

The gated DINOv3 and FLUX.2 model files require both account approval and gated-repository **read** permission on the token. At the last access check, the supplied account's approval was accepted but its fine-grained token still denied public gated repository reads. The app reports this specific distinction. Credentials are not part of this record or the repository.

Until those weights can be downloaded, full encoder-checkpoint compatibility and end-to-end photo quality remain **unverified**. Native tiny-encoder tests establish code execution and selected equations, not full-model parity. No reference Python implementation was run. No generated sphere or zero-conditioned latent is presented as a successful photo reconstruction.

Bit-identical results with the SDK are not promised: the octree samples the same stratified distribution but draws random offsets in a different order, image resampling uses a C# implementation, and backend precision/kernels differ. Performance, full-size GPU peak memory, SRP/XR rendering, multiple overlapping splat objects and Unity player builds need further validation.

## Reproduce

```powershell
dotnet build Idol.sln -c Release -p:IdolUseCuda=false
dotnet test tests/Idol.Tests -c Release -p:IdolUseCuda=false

# Public checkpoint checks; IDOL_MODEL_DIRECTORY optionally points to the idol/ subfolder.
$env:IDOL_MODEL_TESTS = '1'
dotnet test tests/Idol.Tests -c Release --filter FullyQualifiedName~PublishedCheckpointTests

dotnet run --project src/Idol.Cli -c Release -- background-check photo.jpg
dotnet run --project src/Idol.Desktop -c Release -- --open-ply object.ply --capture screenshot.png --exit-after-capture
```

The normal test suite skips the large downloaded-checkpoint tests. It does not download model weights or use a Hugging Face token.
