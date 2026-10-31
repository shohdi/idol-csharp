using System.Globalization;
using System.Numerics;
using System.Text;

namespace Idol.Core.Export;

/// <summary>Exports a coarse ellipsoid proxy, not a reconstructed surface. Use UnityExport to preserve splats.</summary>
public static class ObjExporter
{
    public static void Write(string path, GaussianCloud cloud, int maximumEllipsoids = 16000)
    {
        if (maximumEllipsoids <= 0) throw new ArgumentOutOfRangeException(nameof(maximumEllipsoids));
        var mtlPath = Path.ChangeExtension(path, ".mtl");
        using var obj = new StreamWriter(path, false, Encoding.ASCII);
        using var mtl = new StreamWriter(mtlPath, false, Encoding.ASCII);
        obj.WriteLine("# Idol CSharp coarse ellipsoid proxy; not a watertight surface or the original splat appearance");
        obj.WriteLine("# y-up, metres; import with scale 1");
        obj.WriteLine("mtllib " + Path.GetFileName(mtlPath));
        var materials = new HashSet<int>();
        var stride = Math.Max(1, (int)Math.Ceiling(cloud.Count / (double)maximumEllipsoids));
        int vertex = 1;
        for (var i = 0; i < cloud.Count; i += stride)
        {
            var g = cloud.Points[i];
            if (!g.IsFinite || g.Opacity < 0.1f) continue;
            var rgb = g.Color;
            var r = Math.Clamp((int)MathF.Round(rgb.X * 7), 0, 7);
            var b = Math.Clamp((int)MathF.Round(rgb.Z * 7), 0, 7);
            var green = Math.Clamp((int)MathF.Round(rgb.Y * 7), 0, 7);
            var material = (r << 6) | (green << 3) | b;
            if (materials.Add(material))
            {
                mtl.WriteLine($"newmtl c{material}");
                mtl.WriteLine(FormattableString.Invariant($"Kd {r / 7f:F5} {green / 7f:F5} {b / 7f:F5}"));
                mtl.WriteLine("Ka 0 0 0\nKs 0 0 0\nd 1\nillum 1\n");
            }
            obj.WriteLine($"usemtl c{material}");
            var scale = Vector3.Min(g.Scale * 2, new Vector3(0.1f));
            Vector3[] axes = [new(scale.X, 0, 0), new(-scale.X, 0, 0), new(0, scale.Y, 0), new(0, -scale.Y, 0), new(0, 0, scale.Z), new(0, 0, -scale.Z)];
            foreach (var axis in axes)
            {
                var p = Vector3.Transform(axis, g.Rotation) + g.Position;
                obj.WriteLine(FormattableString.Invariant($"v {p.X:G9} {p.Y:G9} {p.Z:G9}"));
            }
            int[] faces = [0, 2, 4, 4, 2, 1, 1, 2, 5, 5, 2, 0, 4, 3, 0, 1, 3, 4, 5, 3, 1, 0, 3, 5];
            for (var j = 0; j < faces.Length; j += 3) obj.WriteLine($"f {vertex + faces[j]} {vertex + faces[j + 1]} {vertex + faces[j + 2]}");
            vertex += 6;
        }
    }
}
