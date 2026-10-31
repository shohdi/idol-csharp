using System.IO.Compression;
using System.Numerics;
using System.Text;
using Idol.Core;
using Idol.Core.Export;

namespace Idol.Tests;

public sealed class CoreExportTests
{
    private static GaussianCloud Fixture() => new([
        new(new(.12f, -.2f, .31f), new(.3f, -.4f, 1.7f), 2.1f, new(-5.2f, -3.7f, -4.8f), Quaternion.Normalize(new(.2f, .3f, .4f, .5f))),
        new(new(-.22f, .4f, .13f), new(-.1f, .6f, .2f), -1.5f, new(-4.1f), Quaternion.Identity)
    ]);

    [Fact]
    public void PlyRoundTripKeepsEveryGaussianAttribute()
    {
        var cloud = Fixture();
        using var stream = new MemoryStream();
        PlyFile.Write(stream, cloud); stream.Position = 0;
        var restored = PlyFile.Read(stream);
        Assert.Equal(cloud.Count, restored.Count);
        for (int i = 0; i < cloud.Count; ++i)
        {
            Assert.Equal(cloud.Points[i].Position, restored.Points[i].Position);
            Assert.Equal(cloud.Points[i].FeaturesDc, restored.Points[i].FeaturesDc);
            Assert.Equal(cloud.Points[i].LogScale, restored.Points[i].LogScale);
            Assert.Equal(cloud.Points[i].OpacityLogit, restored.Points[i].OpacityLogit);
            Assert.True(MathF.Abs(Quaternion.Dot(cloud.Points[i].Rotation, restored.Points[i].Rotation)) > .999999f);
        }
    }

    [Fact]
    public void PlyRot0IsQuaternionW()
    {
        var cloud = new GaussianCloud([new(Vector3.Zero, Vector3.Zero, 0, new(-4), Quaternion.Identity)]);
        using var stream = new MemoryStream(); PlyFile.Write(stream, cloud);
        var bytes = stream.ToArray();
        Assert.Equal(1f, BitConverter.ToSingle(bytes, bytes.Length - 4 * 4));
        Assert.Equal(0f, BitConverter.ToSingle(bytes, bytes.Length - 3 * 4));
    }

    [Fact]
    public void TruncatedPlyIsRejected()
    {
        using var stream = new MemoryStream(); PlyFile.Write(stream, Fixture());
        using var truncated = new MemoryStream(stream.ToArray()[..^4]);
        Assert.Throws<EndOfStreamException>(() => PlyFile.Read(truncated));
    }

    [Fact]
    public void MeshPlyReportsMissingSplatAttributes()
    {
        using var stream = new MemoryStream(Encoding.ASCII.GetBytes("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n"));
        Assert.Contains("f_dc_0", Assert.Throws<InvalidDataException>(() => PlyFile.Read(stream)).Message);
    }

    [Fact]
    public void NonFiniteDataCannotBeExported()
    {
        var cloud = new GaussianCloud([new(new(float.NaN, 0, 0), Vector3.Zero, 0, new(-4), Quaternion.Identity)]);
        using var stream = new MemoryStream();
        Assert.Throws<InvalidDataException>(() => PlyFile.Write(stream, cloud));
    }

    [Fact]
    public void UnityArchiveHasImportableAssetsAndCompletePly()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".zip");
        try
        {
            var cloud = Fixture(); UnityExport.WriteZip(path, cloud, "test object");
            // Overwriting through a Save dialog is supported.
            UnityExport.WriteZip(path, cloud, "test object");
            using var archive = ZipFile.OpenRead(path);
            Assert.NotNull(archive.GetEntry("Assets/Idol/Runtime/IdolGaussian.shader"));
            Assert.NotNull(archive.GetEntry("Assets/Idol/Runtime/IdolGaussianRenderer.cs"));
            Assert.NotNull(archive.GetEntry("Assets/Idol/Editor/IdolPlyImporter.cs"));
            using var ply = archive.GetEntry("Assets/Idol/Objects/test_object.idolply")!.Open();
            Assert.Equal(cloud.Count, PlyFile.Read(ply).Count);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ObjProxyFacesReferenceValidVerticesAndMtl()
    {
        var folder = Path.Combine(Path.GetTempPath(), "idol-export-" + Guid.NewGuid()); Directory.CreateDirectory(folder);
        var path = Path.Combine(folder, "proxy.obj");
        try
        {
            ObjExporter.Write(path, Fixture());
            var lines = File.ReadAllLines(path); var vertices = lines.Count(l => l.StartsWith("v "));
            Assert.True(vertices > 0);
            foreach (var face in lines.Where(l => l.StartsWith("f ")))
                foreach (var index in face.Split(' ').Skip(1)) Assert.InRange(int.Parse(index), 1, vertices);
            Assert.Contains("mtllib proxy.mtl", lines);
            Assert.True(File.Exists(Path.ChangeExtension(path, ".mtl")));
        }
        finally { Directory.Delete(folder, true); }
    }
}
