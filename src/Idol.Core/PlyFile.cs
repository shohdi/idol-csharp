using System.Globalization;
using System.Numerics;
using System.Text;

namespace Idol.Core;

/// <summary>Reads ASCII and binary little-endian 3DGS PLY. Unknown scalar properties are preserved by skipping.</summary>
public static class PlyFile
{
    private static readonly string[] Fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];

    public static void Write(string path, GaussianCloud cloud)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Write(stream, cloud);
    }

    public static void Write(Stream stream, GaussianCloud cloud)
    {
        if (cloud.Points.Any(g => !g.IsFinite)) throw new InvalidDataException("The object contains non-finite Gaussian values.");
        var header = new StringBuilder("ply\nformat binary_little_endian 1.0\ncomment Idol CSharp; y-up; quaternion wxyz\n");
        header.Append("element vertex ").Append(cloud.Count.ToString(CultureInfo.InvariantCulture)).Append('\n');
        foreach (var field in Fields) header.Append("property float ").Append(field).Append('\n');
        header.Append("end_header\n");
        stream.Write(Encoding.ASCII.GetBytes(header.ToString()));
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
        foreach (var g in cloud.Points)
        {
            writer.Write(g.Position.X); writer.Write(g.Position.Y); writer.Write(g.Position.Z);
            writer.Write(0f); writer.Write(0f); writer.Write(0f);
            writer.Write(g.FeaturesDc.X); writer.Write(g.FeaturesDc.Y); writer.Write(g.FeaturesDc.Z);
            writer.Write(g.OpacityLogit);
            writer.Write(g.LogScale.X); writer.Write(g.LogScale.Y); writer.Write(g.LogScale.Z);
            writer.Write(g.Rotation.W); writer.Write(g.Rotation.X); writer.Write(g.Rotation.Y); writer.Write(g.Rotation.Z);
        }
    }

    public static GaussianCloud Read(string path)
    {
        using var stream = File.OpenRead(path);
        return Read(stream);
    }

    public static GaussianCloud Read(Stream stream)
    {
        if (ReadHeaderLine(stream) != "ply") throw new InvalidDataException("This is not a PLY file.");
        string? format = null, element = null;
        var properties = new List<(string Type, string Name)>();
        int count = 0, headerBytes = 4;
        while (true)
        {
            var line = ReadHeaderLine(stream);
            headerBytes += line.Length + 1;
            if (headerBytes > 1_048_576) throw new InvalidDataException("PLY header is too large.");
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) continue;
            if (parts[0] == "end_header") break;
            if (parts[0] == "format" && parts.Length >= 2) format = parts[1];
            if (parts[0] == "element" && parts.Length == 3)
            {
                if (element == null && parts[1] != "vertex") throw new InvalidDataException("PLY vertex data must be the first element.");
                element = parts[1];
                if (element == "vertex") count = int.Parse(parts[2], CultureInfo.InvariantCulture);
            }
            if (parts[0] == "property" && element == "vertex")
            {
                if (parts.Length != 3 || parts[1] == "list") throw new InvalidDataException("List vertex properties are unsupported.");
                _ = ScalarBytes(parts[1]);
                properties.Add((parts[1], parts[2]));
            }
        }
        if (format is not ("binary_little_endian" or "ascii")) throw new InvalidDataException("Use ASCII or binary little-endian PLY.");
        if (count <= 0 || count > 10_000_000) throw new InvalidDataException("PLY must contain between 1 and 10 million Gaussians.");
        foreach (var field in Fields.Where(f => f is not ("nx" or "ny" or "nz")))
            if (!properties.Any(p => p.Name == field)) throw new InvalidDataException($"Missing Gaussian PLY property '{field}'. This viewer needs a 3DGS PLY, not a mesh PLY.");
        var indices = properties.Select((p, i) => (p.Name, Index: i)).ToDictionary(p => p.Name, p => p.Index);
        var points = new Gaussian[count];
        var values = new float[properties.Count];
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        for (var i = 0; i < count; ++i)
        {
            if (format == "ascii")
            {
                var row = ReadHeaderLine(stream).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (row.Length != properties.Count) throw new InvalidDataException($"Unexpected PLY vertex length at row {i}.");
                for (var j = 0; j < values.Length; ++j) values[j] = float.Parse(row[j], CultureInfo.InvariantCulture);
            }
            else for (var j = 0; j < values.Length; ++j) values[j] = ReadScalar(reader, properties[j].Type);
            float F(string name) => values[indices[name]];
            var rotation = new Quaternion(F("rot_1"), F("rot_2"), F("rot_3"), F("rot_0"));
            rotation = rotation.LengthSquared() > 1e-12f ? Quaternion.Normalize(rotation) : Quaternion.Identity;
            var g = new Gaussian(new(F("x"), F("y"), F("z")), new(F("f_dc_0"), F("f_dc_1"), F("f_dc_2")),
                F("opacity"), new(F("scale_0"), F("scale_1"), F("scale_2")), rotation);
            if (!g.IsFinite) throw new InvalidDataException($"Invalid Gaussian values at vertex {i}.");
            points[i] = g;
        }
        return new GaussianCloud(points);
    }

    private static string ReadHeaderLine(Stream stream)
    {
        var bytes = new List<byte>();
        for (var c = stream.ReadByte(); c != '\n'; c = stream.ReadByte())
        {
            if (c < 0)
            {
                if (bytes.Count > 0) break;
                throw new EndOfStreamException("Unexpected end of PLY file.");
            }
            if (bytes.Count > 65536) throw new InvalidDataException("PLY line is too long.");
            if (c != '\r') bytes.Add((byte)c);
        }
        return Encoding.ASCII.GetString(bytes.ToArray());
    }

    private static int ScalarBytes(string type) => type switch
    {
        "char" or "int8" or "uchar" or "uint8" => 1,
        "short" or "int16" or "ushort" or "uint16" => 2,
        "int" or "int32" or "uint" or "uint32" or "float" or "float32" => 4,
        "double" or "float64" => 8,
        _ => throw new InvalidDataException($"Unsupported PLY scalar type '{type}'.")
    };

    private static float ReadScalar(BinaryReader reader, string type) => type switch
    {
        "char" or "int8" => reader.ReadSByte(), "uchar" or "uint8" => reader.ReadByte(),
        "short" or "int16" => reader.ReadInt16(), "ushort" or "uint16" => reader.ReadUInt16(),
        "int" or "int32" => reader.ReadInt32(), "uint" or "uint32" => reader.ReadUInt32(),
        "float" or "float32" => reader.ReadSingle(), "double" or "float64" => (float)reader.ReadDouble(),
        _ => throw new InvalidDataException($"Unsupported PLY scalar type '{type}'.")
    };
}
