using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Idol.Unity;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Idol.Unity.Editor
{
    // .idolply contains standard binary little-endian 3DGS PLY. Custom extension avoids other PLY importer conflicts.
    [ScriptedImporter(1, "idolply")]
    public sealed class IdolPlyImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            using (var stream = File.OpenRead(ctx.assetPath))
            using (var reader = new BinaryReader(stream))
            {
                if (Line(stream) != "ply") throw new InvalidDataException("Not a PLY file.");
                int count = 0;
                string element = null, format = null;
                var fields = new List<string>();
                for (var line = Line(stream); line != "end_header"; line = Line(stream))
                {
                    var p = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length < 2) continue;
                    if (p[0] == "format") format = p[1];
                    if (p[0] == "element") { element = p[1]; if (element == "vertex") count = int.Parse(p[2], CultureInfo.InvariantCulture); }
                    if (p[0] == "property" && element == "vertex")
                    {
                        if (p.Length != 3 || (p[1] != "float" && p[1] != "float32")) throw new InvalidDataException("Export from Idol CSharp as float32 3DGS PLY.");
                        fields.Add(p[2]);
                    }
                }
                if (format != "binary_little_endian" || count < 1 || count > 10000000) throw new InvalidDataException("Expected binary little-endian Gaussian PLY.");
                string[] required = { "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };
                foreach (var f in required) if (!fields.Contains(f)) throw new InvalidDataException("Missing property " + f);
                var asset = ScriptableObject.CreateInstance<IdolGaussianAsset>();
                asset.name = Path.GetFileNameWithoutExtension(ctx.assetPath) + " Data";
                asset.positions = new Vector3[count]; asset.scales = new Vector3[count]; asset.rotations = new Quaternion[count]; asset.colors = new Color[count];
                var values = new float[fields.Count];
                float F(string key) => values[fields.IndexOf(key)];
                var bounds = new Bounds();
                for (int i = 0; i < count; ++i)
                {
                    for (int j = 0; j < values.Length; ++j) values[j] = reader.ReadSingle();
                    // Reflect Z from right-handed PLY to Unity's left-handed space. R' = S R S.
                    var pos = new Vector3(F("x"), F("y"), -F("z"));
                    var q = new Quaternion(-F("rot_1"), -F("rot_2"), F("rot_3"), F("rot_0")).normalized;
                    var scale = new Vector3(Mathf.Exp(F("scale_0")), Mathf.Exp(F("scale_1")), Mathf.Exp(F("scale_2")));
                    var color = new Color(Mathf.Clamp01(.5f + .2820947918f * F("f_dc_0")), Mathf.Clamp01(.5f + .2820947918f * F("f_dc_1")), Mathf.Clamp01(.5f + .2820947918f * F("f_dc_2")), 1 / (1 + Mathf.Exp(-F("opacity"))));
                    asset.positions[i] = pos; asset.scales[i] = scale; asset.rotations[i] = q; asset.colors[i] = color;
                    if (i == 0) bounds = new Bounds(pos, Vector3.zero); else bounds.Encapsulate(pos);
                }
                bounds.Expand(.2f); asset.bounds = bounds;
                var go = new GameObject(Path.GetFileNameWithoutExtension(ctx.assetPath));
                var renderer = go.AddComponent<IdolGaussianRenderer>(); renderer.asset = asset;
                renderer.shader = Shader.Find("Idol/GaussianSplat");
                ctx.AddObjectToAsset("data", asset); ctx.AddObjectToAsset("prefab", go); ctx.SetMainObject(go);
            }
        }
        private static string Line(Stream stream)
        {
            var bytes = new List<byte>();
            for (int c = stream.ReadByte(); c != 10; c = stream.ReadByte())
            {
                if (c < 0) throw new EndOfStreamException();
                if (bytes.Count > 65536) throw new InvalidDataException("Header line too long.");
                if (c != 13) bytes.Add((byte)c);
            }
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
    }
}
