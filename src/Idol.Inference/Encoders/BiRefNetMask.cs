// Native C# port of speridlabs/idol-sdk BiRefNet (Apache-2.0).
// Deformable convolution is evaluated with LibTorch bilinear grid sampling; no torchvision/Python is required.
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Encoders;

/// <summary>Idol's Swin-L BiRefNet background removal network.</summary>
public sealed class BiRefNetMask
{
    private readonly WeightStore weights;
    private static readonly int[] Depths = [2, 2, 18, 2];
    private static readonly int[] Heads = [6, 12, 24, 48];
    private const int Window = 12;

    public BiRefNetMask(WeightStore weights) => this.weights = weights;
    public Device Device => weights.Device;

    /// <summary>Predicts an alpha mask in the original input dimensions. Input is RGB [B,3,H,W] in [0,1].</summary>
    public Tensor Predict(Tensor pixels, CancellationToken cancellationToken = default)
    {
        using var scope = NewDisposeScope();
        long sourceH = pixels.shape[2], sourceW = pixels.shape[3];
        var dtype = weights.Get("bb.patch_embed.proj.weight").dtype;
        var x = EncoderOps.ImageNetNormalize(EncoderOps.Resize(pixels, 1024, 1024)).to_type(dtype);
        var features = Backbone(x, cancellationToken);
        var small = Backbone(EncoderOps.Resize(x, 512, 512), cancellationToken);
        var combined = new Tensor[4];
        for (int i = 0; i < 4; i++)
            combined[i] = cat(new[] { features[i], EncoderOps.Resize(small[i], features[i].shape[2], features[i].shape[3]) }, 1);
        var f1 = combined[0]; var f2 = combined[1]; var f3 = combined[2]; var f4 = combined[3];
        var context = cat(new[] { EncoderOps.Resize(f1, f4.shape[2], f4.shape[3]),
            EncoderOps.Resize(f2, f4.shape[2], f4.shape[3]), EncoderOps.Resize(f3, f4.shape[2], f4.shape[3]), f4 }, 1);
        var squeezed = DecodeBlock(context, "squeeze_module.0", cancellationToken);
        var p4 = DecodeBlock(cat(new[] { squeezed, InputBlock(x, squeezed, "decoder.ipt_blk5") }, 1), "decoder.decoder_block4", cancellationToken);
        p4 = Gate(p4, 4);
        var p3 = EncoderOps.Resize(p4, f3.shape[2], f3.shape[3]) + EncoderOps.Conv(weights, f3, "decoder.lateral_block4.conv");
        p3 = DecodeBlock(cat(new[] { p3, InputBlock(x, p3, "decoder.ipt_blk4") }, 1), "decoder.decoder_block3", cancellationToken);
        p3 = Gate(p3, 3);
        var p2 = EncoderOps.Resize(p3, f2.shape[2], f2.shape[3]) + EncoderOps.Conv(weights, f2, "decoder.lateral_block3.conv");
        p2 = DecodeBlock(cat(new[] { p2, InputBlock(x, p2, "decoder.ipt_blk3") }, 1), "decoder.decoder_block2", cancellationToken);
        p2 = Gate(p2, 2);
        var p1 = EncoderOps.Resize(p2, f1.shape[2], f1.shape[3]) + EncoderOps.Conv(weights, f1, "decoder.lateral_block2.conv");
        p1 = DecodeBlock(cat(new[] { p1, InputBlock(x, p1, "decoder.ipt_blk2") }, 1), "decoder.decoder_block1", cancellationToken);
        p1 = EncoderOps.Resize(p1, 1024, 1024);
        var logits = EncoderOps.Conv(weights, cat(new[] { p1, InputBlock(x, p1, "decoder.ipt_blk1") }, 1), "decoder.conv_out1.0");
        return EncoderOps.Resize(logits.sigmoid().to_type(ScalarType.Float32), sourceH, sourceW).MoveToOuterDisposeScope();
    }

    private Tensor[] Backbone(Tensor input, CancellationToken cancellationToken)
    {
        using var scope = NewDisposeScope();
        var projected = EncoderOps.Conv(weights, input, "bb.patch_embed.proj", stride: 4);
        long h = projected.shape[2], w = projected.shape[3];
        var x = weights.LayerNorm(projected.flatten(2).transpose(1, 2), "bb.patch_embed.norm");
        var outs = new Tensor[4];
        for (int stage = 0; stage < 4; stage++)
        {
            var mask = ShiftMask(h, w, x);
            for (int i = 0; i < Depths[stage]; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var next = SwinBlock(x, h, w, $"bb.layers.{stage}.blocks.{i}", Heads[stage], i % 2 == 1, mask);
                x.Dispose(); x = next;
            }
            var normalized = weights.LayerNorm(x, $"bb.norm{stage}");
            outs[stage] = normalized.view(x.shape[0], h, w, x.shape[2]).permute(0, 3, 1, 2).contiguous();
            if (stage < 3)
            {
                var next = PatchMerge(x, h, w, $"bb.layers.{stage}.downsample");
                x.Dispose(); x = next; h = (h + 1) / 2; w = (w + 1) / 2;
            }
            mask.Dispose();
        }
        foreach (var output in outs) output.MoveToOuterDisposeScope();
        return outs;
    }

    private Tensor ShiftMask(long h, long w, Tensor sample)
    {
        using var scope = NewDisposeScope();
        int hp = (int)((h + Window - 1) / Window * Window), wp = (int)((w + Window - 1) / Window * Window);
        var ids = new float[hp * wp];
        for (int y = 0; y < hp; y++)
            for (int x = 0; x < wp; x++)
            {
                int r = y < hp - Window ? 0 : y < hp - Window / 2 ? 1 : 2;
                int c = x < wp - Window ? 0 : x < wp - Window / 2 ? 1 : 2;
                ids[y * wp + x] = r * 3 + c;
            }
        var windows = Partition(tensor(ids, device: sample.device).view(1, hp, wp, 1)).view(-1, Window * Window);
        return ((windows.unsqueeze(1) - windows.unsqueeze(2)).ne(0).to_type(sample.dtype) * -100).MoveToOuterDisposeScope();
    }

    private static Tensor Partition(Tensor x) => x.view(x.shape[0], x.shape[1] / Window, Window, x.shape[2] / Window, Window, x.shape[3])
        .permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Window, Window, x.shape[3]);

    private Tensor SwinBlock(Tensor input, long h, long w, string p, int heads, bool shifted, Tensor mask)
    {
        using var scope = NewDisposeScope();
        long b = input.shape[0], c = input.shape[2];
        var x = weights.LayerNorm(input, p + ".norm1").view(b, h, w, c);
        long padR = (Window - w % Window) % Window, padB = (Window - h % Window) % Window;
        x = nn.functional.pad(x, new long[] { 0, 0, 0, padR, 0, padB });
        long hp = h + padB, wp = w + padR;
        if (shifted) x = x.roll(new long[] { -Window / 2, -Window / 2 }, new long[] { 1, 2 });
        var windows = Partition(x).view(-1, Window * Window, c);
        var a = WindowAttention(windows, p + ".attn", heads, shifted ? mask : null).view(-1, Window, Window, c);
        x = a.view(b, hp / Window, wp / Window, Window, Window, c).permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, c);
        if (shifted) x = x.roll(new long[] { Window / 2, Window / 2 }, new long[] { 1, 2 });
        x = x.narrow(1, 0, h).narrow(2, 0, w).contiguous().view(b, h * w, c) + input;
        var mlp = weights.Linear(nn.functional.gelu(weights.Linear(weights.LayerNorm(x, p + ".norm2"), p + ".mlp.fc1")), p + ".mlp.fc2");
        return (x + mlp).MoveToOuterDisposeScope();
    }

    private Tensor WindowAttention(Tensor x, string p, int heads, Tensor? mask)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], n = x.shape[1], c = x.shape[2];
        var qkv = weights.Linear(x, p + ".qkv").reshape(b, n, 3, heads, c / heads).permute(2, 0, 3, 1, 4).unbind(0);
        var scores = qkv[0].matmul(qkv[1].transpose(-2, -1)) * Math.Pow(c / heads, -.5);
        var index = weights.Get(p + ".relative_position_index").flatten().to_type(ScalarType.Int64);
        var bias = weights.Get(p + ".relative_position_bias_table").index_select(0, index).view(n, n, heads).permute(2, 0, 1).unsqueeze(0);
        scores = scores + bias;
        if (mask is not null)
            scores = (scores.view(b / mask.shape[0], mask.shape[0], heads, n, n) + mask.unsqueeze(1).unsqueeze(0)).view(b, heads, n, n);
        var output = scores.softmax(-1).matmul(qkv[2]).transpose(1, 2).reshape(b, n, c);
        return weights.Linear(output, p + ".proj").MoveToOuterDisposeScope();
    }

    private Tensor PatchMerge(Tensor x, long h, long w, string p)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], c = x.shape[2];
        var map = x.view(b, h, w, c);
        var evenH = arange(0, h, 2, dtype: ScalarType.Int64, device: x.device);
        var oddH = arange(1, h, 2, dtype: ScalarType.Int64, device: x.device);
        var evenW = arange(0, w, 2, dtype: ScalarType.Int64, device: x.device);
        var oddW = arange(1, w, 2, dtype: ScalarType.Int64, device: x.device);
        var merged = cat(new[] { map.index_select(1, evenH).index_select(2, evenW),
            map.index_select(1, oddH).index_select(2, evenW), map.index_select(1, evenH).index_select(2, oddW),
            map.index_select(1, oddH).index_select(2, oddW) }, -1).view(b, -1, 4 * c);
        return weights.Linear(weights.LayerNorm(merged, p + ".norm"), p + ".reduction").MoveToOuterDisposeScope();
    }

    private Tensor DecodeBlock(Tensor x, string p, CancellationToken cancellationToken)
    {
        using var scope = NewDisposeScope();
        var y = nn.functional.relu(EncoderOps.BatchNorm(weights, EncoderOps.Conv(weights, x, p + ".conv_in", padding: 1), p + ".bn_in"));
        y = Aspp(y, p + ".dec_att", cancellationToken);
        return EncoderOps.BatchNorm(weights, EncoderOps.Conv(weights, y, p + ".conv_out", padding: 1), p + ".bn_out").MoveToOuterDisposeScope();
    }

    private Tensor Aspp(Tensor x, string p, CancellationToken cancellationToken)
    {
        using var scope = NewDisposeScope();
        var features = new List<Tensor> { AsppBranch(x, p + ".aspp1", cancellationToken) };
        for (int i = 0; i < 3; i++) features.Add(AsppBranch(x, $"{p}.aspp_deforms.{i}", cancellationToken));
        var pooled = x.mean(new long[] { 2, 3 }, keepdim: true);
        pooled = nn.functional.relu(EncoderOps.BatchNorm(weights, EncoderOps.Conv(weights, pooled, p + ".global_avg_pool.1"), p + ".global_avg_pool.2"));
        features.Add(EncoderOps.Resize(pooled, x.shape[2], x.shape[3]));
        return nn.functional.relu(EncoderOps.BatchNorm(weights, EncoderOps.Conv(weights, cat(features.ToArray(), 1), p + ".conv1"), p + ".bn1")).MoveToOuterDisposeScope();
    }

    private Tensor AsppBranch(Tensor x, string p, CancellationToken cancellationToken)
    {
        using var scope = NewDisposeScope();
        return nn.functional.relu(EncoderOps.BatchNorm(weights, DeformConv(x, p + ".atrous_conv", cancellationToken), p + ".bn")).MoveToOuterDisposeScope();
    }

    // torchvision deform_conv2d: paired (dy,dx) offsets and 2*sigmoid modulation.
    internal Tensor DeformConv(Tensor x, string p, CancellationToken cancellationToken = default)
    {
        using var scope = NewDisposeScope();
        var weight = weights.Get(p + ".regular_conv.weight");
        int kh = (int)weight.shape[2], kw = (int)weight.shape[3], padding = kh / 2;
        var offset = EncoderOps.Conv(weights, x, p + ".offset_conv", padding: padding);
        var mask = EncoderOps.Conv(weights, x, p + ".modulator_conv", padding: padding).sigmoid() * 2;
        long h = offset.shape[2], w = offset.shape[3];
        var baseY = arange(h, device: x.device, dtype: x.dtype).view(1, h, 1).expand(x.shape[0], h, w);
        var baseX = arange(w, device: x.device, dtype: x.dtype).view(1, 1, w).expand(x.shape[0], h, w);
        Tensor? result = null;
        for (int y = 0; y < kh; y++)
            for (int column = 0; column < kw; column++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var kernelScope = NewDisposeScope();
                int k = y * kw + column;
                var py = baseY + y - padding + offset.select(1, 2 * k);
                var px = baseX + column - padding + offset.select(1, 2 * k + 1);
                var grid = stack(new[] { (px + .5) * (2.0 / x.shape[3]) - 1,
                    (py + .5) * (2.0 / x.shape[2]) - 1 }, -1);
                var sampled = nn.functional.grid_sample(x, grid, align_corners: false) * mask.narrow(1, k, 1);
                var kernel = weight.narrow(2, y, 1).narrow(3, column, 1).contiguous();
                var contribution = nn.functional.conv2d(sampled, kernel);
                if (result is null) result = contribution.MoveToOuterDisposeScope();
                else { var next = (result + contribution).MoveToOuterDisposeScope(); result.Dispose(); result = next; }
            }
        if (weights.Contains(p + ".regular_conv.bias")) result = result! + weights.Get(p + ".regular_conv.bias").view(1, -1, 1, 1);
        return result!.MoveToOuterDisposeScope();
    }

    private Tensor Gate(Tensor x, int level)
    {
        using var scope = NewDisposeScope();
        string p = $"decoder.gdt_convs_{level}";
        var h = nn.functional.relu(EncoderOps.BatchNorm(weights, EncoderOps.Conv(weights, x, p + ".0", padding: 1), p + ".1"));
        return (x * EncoderOps.Conv(weights, h, $"decoder.gdt_convs_attn_{level}.0").sigmoid()).MoveToOuterDisposeScope();
    }

    private Tensor InputBlock(Tensor image, Tensor reference, string p)
    {
        using var scope = NewDisposeScope();
        long b = image.shape[0], c = image.shape[1], h = reference.shape[2], w = reference.shape[3];
        long hg = image.shape[2] / h, wg = image.shape[3] / w;
        var patches = image.view(b, c, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).reshape(b, c * hg * wg, h, w);
        return EncoderOps.Conv(weights, EncoderOps.Conv(weights, patches, p + ".conv1", padding: 1), p + ".conv_out", padding: 1).MoveToOuterDisposeScope();
    }
}
