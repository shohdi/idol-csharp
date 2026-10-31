using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using Idol.Inference.Models;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Tests;

public sealed class NativeModelTests
{
    [Fact]
    public void NativePositiveZBecomesPositiveYWithConsistentQuaternion()
    {
        using var scope = NewDisposeScope();
        using var splats = new GaussianTensors(tensor(new[] { 0f, 0f, .25f }).reshape(1, 3),
            zeros(1, 1, 3), ones(1, 3), tensor(new[] { 1f, 0f, 0f, 0f }).reshape(1, 4), full([1L, 1L], .5));
        var cloud = Idol.Inference.IdolPipeline.ToCloud(splats, CancellationToken.None);
        Assert.Equal(new System.Numerics.Vector3(0, .25f, 0), cloud.Points[0].Position);
        var up = System.Numerics.Vector3.Transform(System.Numerics.Vector3.UnitZ, cloud.Points[0].Rotation);
        Assert.True(System.Numerics.Vector3.Distance(up, System.Numerics.Vector3.UnitY) < 1e-5);
    }
    [Fact]
    public void SafetensorsReadsPublishedNumericTypesAndPreservesIntegerBuffers()
    {
        string path = Path.Combine(Path.GetTempPath(), $"idol-{Guid.NewGuid():N}.safetensors");
        var entries = new Dictionary<string, object>
        {
            ["full"] = new { dtype = "F32", shape = new[] { 2 }, data_offsets = new[] { 0, 8 } },
            ["half"] = new { dtype = "F16", shape = new[] { 2 }, data_offsets = new[] { 8, 12 } },
            ["brain"] = new { dtype = "BF16", shape = new[] { 2 }, data_offsets = new[] { 12, 16 } },
            ["index"] = new { dtype = "I64", shape = new[] { 1 }, data_offsets = new[] { 16, 24 } }
        };
        using (var writer = new BinaryWriter(File.Create(path)))
        {
            byte[] header = JsonSerializer.SerializeToUtf8Bytes(entries);
            writer.Write((ulong)header.Length); writer.Write(header);
            writer.Write(1.25f); writer.Write(-2.5f);
            writer.Write(BitConverter.HalfToUInt16Bits((Half)3.5f)); writer.Write(BitConverter.HalfToUInt16Bits((Half)(-.5f)));
            writer.Write((ushort)(BitConverter.SingleToInt32Bits(4.25f) >> 16));
            writer.Write((ushort)(BitConverter.SingleToInt32Bits(-1.75f) >> 16));
            writer.Write(42L);
        }
        try
        {
            using var weights = WeightStore.Load(path, CPU, _ => ScalarType.Float32);
            Assert.Equal(new[] { 1.25f, -2.5f }, weights.Get("full").data<float>().ToArray());
            Assert.Equal(new[] { 3.5f, -.5f }, weights.Get("half").data<float>().ToArray());
            Assert.Equal(new[] { 4.25f, -1.75f }, weights.Get("brain").data<float>().ToArray());
            Assert.Equal(ScalarType.Int64, weights.Get("index").dtype);
            Assert.Equal(42L, weights.Get("index").item<long>());
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SafetensorsRejectsOffsetsOutsideFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"idol-{Guid.NewGuid():N}.safetensors");
        byte[] header = Encoding.UTF8.GetBytes("{\"bad\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}");
        using (var writer = new BinaryWriter(File.Create(path))) { writer.Write((ulong)header.Length); writer.Write(header); }
        try { Assert.Throws<InvalidDataException>(() => WeightStore.Load(path, CPU)); }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SystematicResamplingConservesCountsAndHandlesEmptyProbabilityRows()
    {
        var probabilities = new float[24];
        probabilities[0] = 1; probabilities[8] = .5f; probabilities[9] = .5f;
        var result = OctreeDecoder.SystematicSample(probabilities, [32, 11, 8], [.25f, .75f, .1f]);
        Assert.Equal(32, result[0]);
        Assert.All(result[..8].Skip(1), n => Assert.Equal(0, n));
        Assert.Equal(11, result[8..16].Sum());
        Assert.Equal(new[] { 1, 1, 1, 1, 1, 1, 1, 1 }, result[16..24]);
    }

    [Fact]
    public void SystematicResamplingKeepsBoundaryTiesInTheLeftBin()
    {
        var result = OctreeDecoder.SystematicSample([.5f, .5f, 0, 0, 0, 0, 0, 0], [2], [0]);
        Assert.Equal(new[] { 2, 0, 0, 0, 0, 0, 0, 0 }, result);
    }

    [Fact]
    public void TimeGridAndCameraFollowUpstreamEquations()
    {
        Assert.Equal(new[] { 1.0, .75, 0.0 }, FlowSampler.TimeGrid(2, 3));
        var c = CameraDecoder.Decode(new float[] { 0, 1, 0, .5f, 1 });
        Assert.Equal(0, c.Yaw, 6); Assert.Equal(0, c.Pitch, 6);
        Assert.Equal(2, c.Radius, 6); Assert.Equal(53.13010235415598, c.Fov, 6);
    }

    [Fact]
    public void AttentionQueryChunkingMatchesUnchunkedComputation()
    {
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        manual_seed(8);
        var q = randn(1, 7, 2, 4); var k = randn(1, 9, 2, 4); var v = randn(1, 9, 2, 4);
        var chunked = ModelLayers.Attention(q, k, v, 2);
        var full = ModelLayers.Attention(q, k, v, 100);
        Assert.True(allclose(chunked, full, 1e-5, 1e-6));
    }

    [Fact]
    public void RotaryZeroAnglesPreserveInputAndPositionPaddingIsZero()
    {
        using var scope = NewDisposeScope();
        var x = randn(1, 3, 2, 6);
        var angles = zeros(1, 3, 2, 3);
        var rotary = stack([angles.cos(), angles.sin()], -1);
        Assert.True(allclose(x, ModelLayers.ApplyRotary(x, rotary)));
        var position = ModelLayers.Position(zeros(1, 2, 3), 16, version2: true);
        Assert.Equal(new long[] { 1, 2, 16 }, position.shape);
        Assert.Equal(0, position.narrow(-1, 12, 4).abs().sum().item<float>());
    }

    [Fact]
    public void TinyDenoiserExecutesEveryBranchWithCheckpointNamedWeights()
    {
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        using var w = new WeightStore(CPU);
        const int c = 12, heads = 2;
        void Linear(string p, int input, int output, float bias = 0)
        {
            w.Add(p + ".weight", zeros(output, input)); w.Add(p + ".bias", full([output], bias));
        }
        void Norm(string p) { w.Add(p + ".weight", ones(c)); w.Add(p + ".bias", zeros(c)); }
        void Repo(string p)
        {
            Norm(p + ".norm");
            w.Add(p + ".gate_map.weight", zeros(2, c)); w.Add(p + ".content_map.weight", zeros(2, c));
            w.Add(p + ".final_map.weight", zeros(heads * 3, 2));
            for (int i = 0; i < 3; i++) w.Add(p + $".freqs_{i}", ones(1));
        }
        void Block(string p, bool modulated)
        {
            Linear(p + ".attn.qkv", c, 3 * c); Linear(p + ".attn.out", c, c);
            w.Add(p + ".attn.q_norm.gamma", ones(heads, c / heads));
            w.Add(p + ".attn.k_norm.gamma", ones(heads, c / heads));
            Linear(p + ".mlp.mlp.0", c, 4 * c); Linear(p + ".mlp.mlp.2", 4 * c, c);
            if (modulated) w.Add(p + ".shift_table", zeros(1, 6 * c));
            else { Norm(p + ".norm1"); Norm(p + ".norm2"); }
        }
        Linear("input_layer", 4, c); Linear("cond_embedder", 6, c); Linear("cond_embedder2", 8, c);
        Linear("t_embedder.mlp.0", 256, c); Linear("t_embedder.mlp.2", c, c);
        Linear("adaLN_modulation.1", c, c * 6);
        w.Add("pos_pe", zeros(1, 4, 3));
        for (int i = 0; i < 2; i++)
        {
            Repo($"noise_repo_layers.{i}"); Repo($"context_repo_layers.{i}");
            Block($"noise_refiner.{i}", true); Block($"context_refiner.{i}", false);
        }
        Linear("cam_refiner.mlp.0", 5, c); Linear("cam_refiner.mlp.2", c, c);
        Repo("repo_layers.0"); Block("blocks.0", true);
        w.Add("shift_table", zeros(1, 2, c));
        Linear("out_layer", c, 4, .25f); Linear("cam_out_layer", c, 5, -.5f);
        using var config = JsonDocument.Parse("{\"q_token_length\":4,\"in_channels\":4,\"cam_channels\":5,\"model_channels\":12,\"cond_channels\":6,\"cond2_channels\":8,\"num_blocks\":1,\"num_refiner_blocks\":2,\"num_heads\":2}");
        var model = new IdolDenoiser(w, config.RootElement);
        var output = model.Forward(randn(1, 4, 4), randn(1, 1, 5), full([1L], 500), randn(1, 3, 6), randn(1, 3, 8));
        Assert.True(allclose(output.Latent, full([1L, 4, 4], .25f)));
        Assert.True(allclose(output.Camera, full([1L, 1, 5], -.5f)));
    }
}
