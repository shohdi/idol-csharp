using System.Text.Json;
using Idol.Inference.Encoders;
using Idol.Inference.Imaging;
using Idol.Inference.Tensors;
using System.Drawing;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Tests;

public sealed class EncoderTests
{
    [Fact]
    public void DeformableConvolutionWithZeroOffsetsEqualsOrdinaryConvolution()
    {
        using var scope = NewDisposeScope();
        using var weights = new WeightStore(CPU);
        weights.Add("test.offset_conv.weight", zeros(18, 1, 3, 3));
        weights.Add("test.offset_conv.bias", zeros(18));
        weights.Add("test.modulator_conv.weight", zeros(9, 1, 3, 3));
        weights.Add("test.modulator_conv.bias", zeros(9));
        weights.Add("test.regular_conv.weight", arange(1, 10, dtype: ScalarType.Float32).reshape(1, 1, 3, 3));
        var input = arange(1, 26, dtype: ScalarType.Float32).reshape(1, 1, 5, 5);
        var expected = nn.functional.conv2d(input, weights.Get("test.regular_conv.weight"), padding: [1, 1]);
        var actual = new BiRefNetMask(weights).DeformConv(input, "test");
        Assert.True(allclose(expected, actual, atol: 1e-4));
    }

    [Fact]
    public void DeformableConvolutionUsesPairedOffsetsAndModulation()
    {
        using var scope = NewDisposeScope();
        using var weights = new WeightStore(CPU);
        weights.Add("test.offset_conv.weight", zeros(2, 1, 1, 1));
        // One pixel to the right, no vertical offset.
        weights.Add("test.offset_conv.bias", tensor(new[] { 0f, 1f }));
        weights.Add("test.modulator_conv.weight", zeros(1, 1, 1, 1));
        weights.Add("test.modulator_conv.bias", zeros(1));
        weights.Add("test.regular_conv.weight", ones(1, 1, 1, 1));
        var input = tensor(new[] { 1f, 2f, 3f, 4f, 5f, 6f }).reshape(1, 1, 2, 3);
        var actual = new BiRefNetMask(weights).DeformConv(input, "test");
        var expected = tensor(new[] { 2f, 3f, 0f, 5f, 6f, 0f }).reshape(1, 1, 2, 3);
        Assert.True(allclose(expected, actual, atol: 1e-5));
    }

    [Fact]
    public void AlphaPreprocessingFramesObjectAndCompositesBlack()
    {
        using var image = new RasterImage(32, 32);
        for (int y = 8; y < 24; y++)
            for (int x = 8; x < 24; x++) image[x, y] = Color.FromArgb(255, 255, 0, 0);
        using var prepared = ImagePreprocessor.FrameAlpha(image, 1.7);
        Assert.Equal(1024, prepared.Width);
        Assert.Equal(1024, prepared.Height);
        Assert.Equal(Color.FromArgb(255, 0, 0, 0), prepared[0, 0]);
        Assert.Equal(Color.FromArgb(255, 255, 0, 0), prepared[512, 512]);
    }

    [Fact]
    public void AlphaPreprocessingRejectsEmptyForeground()
    {
        using var empty = new RasterImage(16, 16);
        Assert.Throws<ArgumentException>(() => ImagePreprocessor.FrameAlpha(empty));
    }

    [Fact]
    public void DinoRetainsPrefixTokensAndAppliesFinalNormalization()
    {
        string config = Path.GetTempFileName();
        try
        {
            File.WriteAllText(config, JsonSerializer.Serialize(new
            {
                patch_size = 2, num_attention_heads = 2, num_hidden_layers = 1, hidden_size = 8,
                layer_norm_eps = 1e-5, rope_theta = 100, use_gated_mlp = false, hidden_act = "gelu"
            }));
            using var scope = NewDisposeScope();
            using var weights = new WeightStore(CPU);
            weights.Add("embeddings.patch_embeddings.weight", zeros(8, 3, 2, 2));
            weights.Add("embeddings.patch_embeddings.bias", arange(8, dtype: ScalarType.Float32));
            weights.Add("embeddings.cls_token", arange(8, dtype: ScalarType.Float32).reshape(1, 1, 8));
            weights.Add("embeddings.register_tokens", arange(8, dtype: ScalarType.Float32).reshape(1, 1, 8).expand(1, 4, 8).clone());
            const string p = "model.layer.0";
            AddNorm(weights, p + ".norm1", 8); AddNorm(weights, p + ".norm2", 8); AddNorm(weights, "norm", 8);
            foreach (var projection in new[] { "q_proj", "k_proj", "v_proj", "o_proj" }) AddLinear(weights, p + ".attention." + projection, 8, 8);
            weights.Add(p + ".layer_scale1.lambda1", ones(8)); weights.Add(p + ".layer_scale2.lambda1", ones(8));
            AddLinear(weights, p + ".mlp.up_proj", 8, 16); AddLinear(weights, p + ".mlp.down_proj", 16, 8);
            var actual = new DinoV3Encoder(weights, config).Encode(zeros(1, 3, 4, 4));
            Assert.Equal(new long[] { 1, 9, 8 }, actual.shape);
            var expected = nn.functional.layer_norm(arange(8, dtype: ScalarType.Float32), [8]);
            Assert.True(allclose(actual.select(0, 0).select(0, 0), expected, atol: 1e-4));
            Assert.True(isfinite(actual).all().item<bool>());
        }
        finally { File.Delete(config); }
    }

    [Fact]
    public void FluxEncoderPacksFourSpatialValuesAndAddsFivePrefixTokens()
    {
        string config = Path.GetTempFileName();
        try
        {
            File.WriteAllText(config, JsonSerializer.Serialize(new
            {
                block_out_channels = new[] { 4 }, layers_per_block = 1, norm_num_groups = 1,
                act_fn = "silu", batch_norm_eps = 1e-4, mid_block_add_attention = false,
                down_block_types = new[] { "DownEncoderBlock2D" }
            }));
            using var scope = NewDisposeScope();
            using var weights = new WeightStore(CPU);
            AddConv(weights, "encoder.conv_in", 3, 4, 3);
            AddResnet(weights, "encoder.down_blocks.0.resnets.0", 4);
            AddResnet(weights, "encoder.mid_block.resnets.0", 4);
            AddResnet(weights, "encoder.mid_block.resnets.1", 4);
            AddNorm(weights, "encoder.conv_norm_out", 4);
            AddConv(weights, "encoder.conv_out", 4, 4, 3);
            weights.Get("encoder.conv_out.bias").copy_(tensor(new[] { 1f, 2f, -30f, -30f }));
            weights.Add("bn.running_mean", zeros(8)); weights.Add("bn.running_var", ones(8));
            var actual = new Flux2VaeEncoder(weights, config).Encode(zeros(1, 3, 4, 4));
            Assert.Equal(new long[] { 1, 9, 8 }, actual.shape);
            Assert.Equal(0f, actual.narrow(1, 0, 5).abs().max().item<float>());
            Assert.True(allclose(actual.narrow(1, 5, 4).select(2, 0), full(new long[] { 1, 4 }, 1 / Math.Sqrt(1.0001)), atol: 1e-5));
            Assert.True(allclose(actual.narrow(1, 5, 4).select(2, 4), full(new long[] { 1, 4 }, 2 / Math.Sqrt(1.0001)), atol: 1e-5));
        }
        finally { File.Delete(config); }
    }

    private static void AddNorm(WeightStore w, string p, int channels)
    { w.Add(p + ".weight", ones(channels)); w.Add(p + ".bias", zeros(channels)); }
    private static void AddLinear(WeightStore w, string p, int input, int output)
    { w.Add(p + ".weight", zeros(output, input)); w.Add(p + ".bias", zeros(output)); }
    private static void AddConv(WeightStore w, string p, int input, int output, int kernel)
    { w.Add(p + ".weight", zeros(output, input, kernel, kernel)); w.Add(p + ".bias", zeros(output)); }
    private static void AddResnet(WeightStore w, string p, int channels)
    { AddNorm(w, p + ".norm1", channels); AddNorm(w, p + ".norm2", channels); AddConv(w, p + ".conv1", channels, channels, 3); AddConv(w, p + ".conv2", channels, channels, 3); }
}
