// C# port of speridlabs/idol-sdk, Apache-2.0, idol/models/decoder.py and gaussian.py.
using System.Text.Json;
using Idol.Inference.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Idol.Inference.Models;

public sealed class OctreeDecoder
{
    private readonly WeightStore w;
    private readonly int blocks, heads, channels;
    private readonly ScalarType computeType;
    public OctreeDecoder(WeightStore weights, JsonElement args)
    {
        w = weights;
        blocks = args.GetProperty("num_blocks").GetInt32();
        heads = args.GetProperty("num_heads").GetInt32();
        channels = args.GetProperty("model_channels").GetInt32();
        computeType = w.Get("blocks.0.cross_attn.to_q.weight").dtype;
    }

    public Tensor Forward(Tensor centers, Tensor level, Tensor condition, CancellationToken cancellationToken = default)
    {
        using var scope = NewDisposeScope();
        var h = w.Linear(centers, "in_proj") + ModelLayers.Position(centers, channels, version2: true);
        h = w.Linear(h, "input_layer");
        var mod = w.Linear(ModelLayers.Sinusoidal(w, level.to(h.dtype), "l_embedder", 1024, true).silu(), "adaLN_modulation.1").to(computeType);
        var output = h.to(computeType);
        var context = condition.to(computeType);
        for (int i = 0; i < blocks; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var next = ModelLayers.CrossOnlyBlock(w, output, mod, context, $"blocks.{i}", heads);
            output.Dispose(); output = next;
        }
        return w.Linear(w.LayerNorm(output.to(condition.dtype), affine: false), "out_proj").MoveToOuterDisposeScope();
    }

    public Tensor Sample(Tensor latent, int numPoints, int level, CancellationToken cancellationToken = default,
        Action<int, int>? progress = null)
    {
        if (latent.shape[0] != 1) throw new ArgumentException("Desktop reconstruction expects a single image batch.");
        if (numPoints < 1 || level < 1 || level > 20) throw new ArgumentOutOfRangeException(nameof(numPoints));
        using var scope = NewDisposeScope();
        var cells = new List<(int X, int Y, int Z, int Count)> { (0, 0, 0, numPoints) };
        for (int l = 1; l <= level; l++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var iteration = NewDisposeScope();
            float resolution = 1 << (l - 1);
            var coordinates = new float[cells.Count * 3];
            for (int i = 0; i < cells.Count; i++)
            {
                coordinates[3 * i] = (cells[i].X + .5f) / resolution;
                coordinates[3 * i + 1] = (cells[i].Y + .5f) / resolution;
                coordinates[3 * i + 2] = (cells[i].Z + .5f) / resolution;
            }
            var centers = tensor(coordinates, device: latent.device).reshape(1, cells.Count, 3);
            var levels = full([1L], 1 << l, dtype: ScalarType.Float32, device: latent.device);
            var probabilities = Forward(centers, levels, latent, cancellationToken).softmax(-1).to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();
            var offsets = rand(cells.Count, dtype: ScalarType.Float32, device: latent.device).cpu().data<float>().ToArray();
            var counts = SystematicSample(probabilities, cells.Select(c => c.Count).ToArray(), offsets);
            var children = new List<(int, int, int, int)>();
            for (int i = 0; i < cells.Count; i++)
                for (int child = 0; child < 8; child++)
                {
                    int n = counts[8 * i + child];
                    if (n == 0) continue;
                    children.Add((cells[i].X * 2 + (child & 1), cells[i].Y * 2 + ((child >> 1) & 1),
                        cells[i].Z * 2 + ((child >> 2) & 1), n));
                }
            cells = children;
            progress?.Invoke(l, level);
        }
        var positions = new float[checked(numPoints * 3)];
        int index = 0;
        foreach (var c in cells)
            for (int j = 0; j < c.Count; j++)
            {
                positions[index++] = c.X; positions[index++] = c.Y; positions[index++] = c.Z;
            }
        if (index != positions.Length) throw new InvalidOperationException("Octree lost anchor counts.");
        var integerCoordinates = tensor(positions, device: latent.device).reshape(1, numPoints, 3);
        return ((integerCoordinates + rand_like(integerCoordinates)) / (1 << level)).MoveToOuterDisposeScope();
    }

    /// <summary>Systematic resampling; one uniform [0,1) offset per parent, eight children per row.</summary>
    public static int[] SystematicSample(float[] probabilities, int[] parentCounts, float[] offsets)
    {
        if (probabilities.Length != parentCounts.Length * 8 || offsets.Length != parentCounts.Length)
            throw new ArgumentException("Eight probabilities and one random offset are required per parent.");
        var result = new int[probabilities.Length];
        for (int i = 0; i < parentCounts.Length; i++)
        {
            int n = parentCounts[i], previous = 0;
            double sum = 0, cumulative = 0;
            for (int k = 0; k < 8; k++) sum += Math.Max(0, probabilities[i * 8 + k]);
            for (int k = 0; k < 8; k++)
            {
                cumulative += sum > 0 ? Math.Max(0, probabilities[i * 8 + k]) / sum : .125;
                // torch.searchsorted uses the left insertion point: a sample equal to
                // a CDF boundary belongs to that bin, including an exact zero offset.
                int at = k == 7 ? n : Math.Clamp((int)Math.Floor(cumulative * n - offsets[i]) + 1, 0, n);
                result[8 * i + k] = at - previous;
                previous = at;
            }
        }
        return result;
    }
}

/// <summary>Activated, z-up Gaussian data. Rotation is (w,x,y,z), features are degree-zero SH.</summary>
public sealed record GaussianTensors(Tensor Xyz, Tensor FeaturesDc, Tensor Scaling, Tensor Rotation, Tensor Opacity) : IDisposable
{
    public void Dispose() { Xyz.Dispose(); FeaturesDc.Dispose(); Scaling.Dispose(); Rotation.Dispose(); Opacity.Dispose(); }
}

public sealed class GaussianDecoder
{
    private readonly WeightStore w;
    private readonly int blocks, heads, channels, perAnchor;
    private readonly JsonElement representation;
    private readonly ScalarType computeType;
    public GaussianDecoder(WeightStore weights, JsonElement args)
    {
        w = weights;
        blocks = args.GetProperty("num_blocks").GetInt32();
        heads = args.GetProperty("num_heads").GetInt32();
        channels = args.GetProperty("model_channels").GetInt32();
        representation = args.GetProperty("representation_config").Clone();
        perAnchor = representation.GetProperty("num_gaussians").GetInt32();
        computeType = w.Get("blocks.0.self_attn.to_qkv.weight").dtype;
    }

    public Tensor Forward(Tensor points, Tensor latent, CancellationToken cancellationToken = default,
        Action<int, int>? progress = null)
    {
        using var scope = NewDisposeScope();
        var h = w.Linear(points, "in_proj") + ModelLayers.Position(points, channels, version2: true);
        h = w.Linear(h, "input_layer");
        var output = h.to(computeType);
        var context = latent.to(computeType);
        for (int i = 0; i < blocks; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var next = ModelLayers.CrossBlock(w, output, context, $"blocks.{i}", heads);
            output.Dispose(); output = next;
            progress?.Invoke(i + 1, blocks);
        }
        return w.Linear(w.LayerNorm(output.to(latent.dtype), affine: false), "out_proj").MoveToOuterDisposeScope();
    }

    public GaussianTensors Decode(Tensor points, Tensor latent, CancellationToken cancellationToken = default,
        Action<int, int>? progress = null)
    {
        using var scope = NewDisposeScope();
        var features = Forward(points, latent, cancellationToken, progress);
        if (features.shape[0] != 1) throw new ArgumentException("Expected a single object.");
        var lr = representation.GetProperty("lr");
        Tensor Slice(int offset, int count) => features.narrow(2, offset * perAnchor, count * perAnchor)
            .reshape(1, -1, perAnchor, count);
        var offsetScale = (Slice(14, 1) + w.Get("base_offset_scale")).softplus();
        var offset = Slice(0, 3) * lr.GetProperty("_xyz").GetDouble() + w.Get("points_offset_perturbation");
        offset = offset.tanh() * .5 * representation.GetProperty("perturbe_size").GetDouble() * offsetScale;
        var xyz = (offset + points.unsqueeze(2)).reshape(-1, 3) - .5;
        var dc = (Slice(3, 3) * lr.GetProperty("_features_dc").GetDouble()).reshape(-1, 1, 3);
        double scaleBias = representation.GetProperty("scaling_bias").GetDouble();
        double scaleOffset = Math.Log(Math.Exp(scaleBias) - 1);
        double kernel = representation.GetProperty("3d_filter_kernel_size").GetDouble();
        var scale = (Slice(6, 3) * lr.GetProperty("_scaling").GetDouble() + scaleOffset).softplus();
        scale = (scale.square() + kernel * kernel).sqrt().reshape(-1, 3);
        var rotation = (Slice(9, 4) * lr.GetProperty("_rotation").GetDouble()).reshape(-1, 4) +
            tensor(new float[] { 1, 0, 0, 0 }, device: latent.device);
        rotation = rotation / rotation.norm(-1, keepdim: true).clamp_min(1e-12);
        double opacityBias = representation.GetProperty("opacity_bias").GetDouble();
        var opacity = (Slice(13, 1) * lr.GetProperty("_opacity").GetDouble() + Math.Log(opacityBias / (1 - opacityBias)))
            .sigmoid().reshape(-1, 1);
        return new GaussianTensors(xyz.MoveToOuterDisposeScope(), dc.MoveToOuterDisposeScope(), scale.MoveToOuterDisposeScope(),
            rotation.MoveToOuterDisposeScope(), opacity.MoveToOuterDisposeScope());
    }
}
