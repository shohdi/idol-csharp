# Validation record

Recorded on 2026-10-31. Windows x64; .NET SDK 9.0.200; TorchSharp 0.107.0 / LibTorch 2.10.0. The machine has an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM and driver 592.82.

## Checks performed

- Release solution build with the CPU runtime: **passed, zero warnings/errors**.
- Release solution build with the CUDA runtime: **passed, zero warnings/errors**. All **42 tests passed** with public-checkpoint integration enabled, including the full 8,192-token denoiser on the RTX 4060. Tests also cover all three DINO checkpoint naming layouts and the positive-Z to positive-Y conversion.
- Native component tests: safetensors F32/F16/BF16 and integer buffers, invalid checkpoint ranges, attention chunking, rotary/absolute embeddings, tiny full denoiser, Euler time grid/camera math, octree resampling, deformable convolution, DINO prefix tokens/norm and VAE posterior/packing.
- Preprocessing tests: empty foreground rejection and transparent-image framing/black compositing.
- Download tests: gated access preflight, resumed HTTP range downloads, SHA-256 verification and corrupt-file rejection.
- Export tests: PLY roundtrip/quaternion ordering, malformed/truncated PLY, finite data, Unity ZIP contents/full counts, OBJ indices/material references.
- Real public `decoder.safetensors`: all octree decoder blocks executed; finite logits and 64 anchors within the unit cube.
- Real public `decoder_gs.safetensors`: all 16 Gaussian decoder blocks executed with four anchors/eight context tokens; output contained 128 finite Gaussians with positive scales, bounded opacities and unit quaternions. This is a component check with synthetic latent input, not an image reconstruction.
- Real public `denoiser.safetensors`: all 24 joint transformer blocks and both stream refiners executed on CUDA with 8,192 latent tokens plus nine synthetic conditioning tokens. Both latent and camera outputs were finite.
- Full photo conversion: all nine pinned model files downloaded and weight checksums verified. The real DINOv3 ViT-H+ and FLUX.2 VAE checkpoints encoded a photograph; Idol ran 50 guided Euler sampling steps and decoded **65,536 Gaussians**. Binary PLY and a Unity ZIP were saved successfully. The generated object was visually inspected in the desktop viewer. Output coordinates use -90 degrees about X so positive native Z becomes positive Y in the desktop viewer and Unity.
- Real public `birefnet.safetensors`: native C# background removal ran on the PyTorch Hub dog photograph. The foreground mask was finite in [0,1], with original-image shape [1,1,1213,1546]. Execution took about 14 seconds on CPU and 2.8 seconds on CUDA. The resulting black-background crop was visually inspected. The photograph and outputs are local artifacts, not repository assets.
- WPF app: launched, loaded a synthetic 24,000-Gaussian PLY, rendered its preview, captured its window and exited successfully. [Screenshot](images/desktop-viewer.png). The screenshot is a viewer fixture, not a claimed model result.
- WPF conversion flow: selected a real photograph through the same method called by Browse, ran the complete 50-step GPU pipeline, displayed the input and 65,536-Gaussian result, captured the window and exited successfully. The generated object stood upright. About 6 GB total device memory use was observed during sampling; this is not a measured peak-memory benchmark.
- Unity 6000.2.6f1: batch import created a prefab with the expected two-Gaussian data asset; quaternion convention was checked; shader reported zero compilation errors; scene was saved. A Direct3D11 camera render produced 1,142 non-black splat pixels. The actual photo-converted object also imported as a prefab with all 65,536 Gaussians and rendered successfully.

## Full-density rendering correction

The original desktop viewer submitted only about 20,000 splats even for a million-Gaussian result. This produced conspicuous holes that were display artifacts. The replacement renderer projects all input Gaussians, sorts deterministically and composites image tiles in parallel. It retains low-alpha splats down to the shader's per-pixel 1/255 threshold and removes the old 180-pixel support cap. During interaction only image resolution changes; point density is retained.

An exact reproduction at 1,048,576 Gaussians, real mode, seed 42 and 50 steps was generated and saved locally. Rendering that same cloud with the same camera submitted 19,785 points for the old stride-53 preview and 1,048,576 points for full density. The sparse view reproduced the reported perforation; full density showed a smooth object. Full rendering at 900×620 took approximately 0.53 seconds on this machine. Eight rendering regression tests cover dense coverage, alpha/depth ordering, rotated covariance, large splat support, low-alpha contributions, culling counts, determinism and cancellation.

All six cached model checkpoint hashes were rechecked against the original pinned manifest and matched. This correction changes rendering only; inference equations, model weights and generation settings were left intact. User images, model outputs and comparison artifacts remain local and are excluded from the public repository.

## Limits

The gated DINOv3 and FLUX.2 model files require both account approval and gated-repository **read** permission on the token. Both access checks passed for the full conversion above. Credentials are not part of this record or the repository.

The full pipeline ran on a photograph and a generated character image; these examples do not establish general reconstruction quality or numerical parity with the original SDK. No reference Python implementation was run. The README sphere remains an explicitly labeled synthetic viewer fixture. Outputs retain partial objects and source-image cropping; hidden surfaces are generated guesses, and results are not watertight meshes.

Bit-identical results with the SDK are not promised: the octree samples the same stratified distribution but draws random offsets in a different order, image resampling uses a C# implementation, and backend precision/kernels differ. Performance, complete image-conditioned GPU peak memory, SRP/XR rendering, multiple overlapping splat objects and Unity player builds need further validation.

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
