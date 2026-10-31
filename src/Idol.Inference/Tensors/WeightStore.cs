using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace Idol.Inference.Tensors;

/// <summary>Reads the published safetensors checkpoints directly; no Python/pickle converter is involved.</summary>
public sealed class WeightStore : IDisposable
{
    private readonly Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
    public Device Device { get; }
    public IEnumerable<string> Names => weights.Keys;
    public WeightStore(Device device) => Device = device;
    public bool Contains(string name) => weights.ContainsKey(name);
    public Tensor Get(string name) => weights.TryGetValue(name, out var value)
        ? value : throw new InvalidDataException($"Missing checkpoint tensor '{name}'.");

    /// <summary>Adds an owned tensor, detached from any ambient disposal scope.</summary>
    public void Add(string name, Tensor value)
    {
        if (!weights.TryAdd(name, value.DetachFromDisposeScope()))
            throw new InvalidDataException($"Duplicate checkpoint tensor '{name}'.");
    }

    public static WeightStore Load(string path, Device device,
        Func<string, ScalarType>? dtypeSelector = null, CancellationToken cancellationToken = default)
    {
        var store = new WeightStore(device);
        try
        {
            using var file = File.OpenRead(path);
            Span<byte> prefix = stackalloc byte[8];
            file.ReadExactly(prefix);
            var headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
            if (headerLength == 0 || headerLength > 100_000_000 || headerLength > (ulong)(file.Length - 8))
                throw new InvalidDataException("Invalid safetensors header length.");
            var header = new byte[checked((int)headerLength)];
            file.ReadExactly(header);
            using var json = JsonDocument.Parse(header);
            long dataStart = checked(8 + (long)headerLength);
            foreach (var item in json.RootElement.EnumerateObject())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (item.Name == "__metadata__") continue;
                using var scope = NewDisposeScope();
                var shape = item.Value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt64()).ToArray();
                if (shape.Any(d => d < 0)) throw new InvalidDataException("Negative tensor dimension.");
                var offsets = item.Value.GetProperty("data_offsets");
                long start = offsets[0].GetInt64(), end = offsets[1].GetInt64();
                var type = item.Value.GetProperty("dtype").GetString();
                int elementSize = type switch { "F64" or "I64" => 8, "F32" or "I32" => 4,
                    "F16" or "BF16" or "I16" => 2, "U8" or "I8" or "BOOL" => 1,
                    _ => throw new NotSupportedException($"Unsupported safetensors dtype {type}.") };
                long count = shape.Aggregate(1L, (a, b) => checked(a * b));
                if (start < 0 || end < start || end - start != checked(count * elementSize) || end > file.Length - dataStart)
                    throw new InvalidDataException($"Invalid tensor offsets for '{item.Name}'.");
                var bytes = new byte[checked((int)(end - start))];
                file.Position = dataStart + start;
                file.ReadExactly(bytes);
                Tensor value = type switch
                {
                    "F32" => tensor(MemoryMarshal.Cast<byte, float>(bytes).ToArray()),
                    "F64" => tensor(MemoryMarshal.Cast<byte, double>(bytes).ToArray()),
                    "F16" => tensor(MemoryMarshal.Cast<byte, Half>(bytes).ToArray()),
                    "BF16" => ReadBFloat16(bytes),
                    "I64" => tensor(MemoryMarshal.Cast<byte, long>(bytes).ToArray()),
                    "I32" => tensor(MemoryMarshal.Cast<byte, int>(bytes).ToArray()),
                    "I16" => tensor(MemoryMarshal.Cast<byte, short>(bytes).ToArray()),
                    "U8" => tensor(bytes),
                    "I8" => tensor(MemoryMarshal.Cast<byte, sbyte>(bytes).ToArray()),
                    "BOOL" => tensor(bytes.Select(b => b != 0).ToArray()),
                    _ => throw new InvalidDataException()
                };
                bool floating = type is "F16" or "BF16" or "F32" or "F64";
                var dtype = floating && dtypeSelector != null ? dtypeSelector(item.Name) : value.dtype;
                store.Add(item.Name, value.reshape(shape).to(dtype, device, copy: true));
            }
            return store;
        }
        catch { store.Dispose(); throw; }
    }

    private static Tensor ReadBFloat16(byte[] bytes)
    {
        var bits = MemoryMarshal.Cast<byte, ushort>(bytes);
        var floats = new float[bits.Length];
        for (int i = 0; i < bits.Length; i++) floats[i] = BitConverter.Int32BitsToSingle(bits[i] << 16);
        return tensor(floats).to(ScalarType.BFloat16);
    }

    public Tensor Linear(Tensor input, string prefix)
    {
        using var scope = NewDisposeScope();
        var weight = Get(prefix + ".weight");
        var bias = Contains(prefix + ".bias") ? Get(prefix + ".bias") : null;
        return F.linear(input.to(weight.dtype), weight, bias).MoveToOuterDisposeScope();
    }

    /// <summary>LayerNorm32 from the SDK: compute in float32, then restore the input dtype.</summary>
    public Tensor LayerNorm(Tensor input, string? prefix = null, double eps = 1e-5, bool affine = true)
    {
        using var scope = NewDisposeScope();
        Tensor? weight = null, bias = null;
        if (affine && prefix != null)
        {
            if (Contains(prefix + ".weight")) weight = Get(prefix + ".weight").to(ScalarType.Float32);
            if (Contains(prefix + ".bias")) bias = Get(prefix + ".bias").to(ScalarType.Float32);
        }
        return F.layer_norm(input.to(ScalarType.Float32), [input.shape[^1]], weight, bias, eps)
            .to(input.dtype).MoveToOuterDisposeScope();
    }

    public void Dispose()
    {
        foreach (var value in weights.Values) value.Dispose();
        weights.Clear();
    }
}
