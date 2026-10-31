using System.IO.Compression;
using System.Reflection;
using System.Text;

namespace Idol.Core.Export;

public static class UnityExport
{
    /// <summary>Creates a ZIP whose Assets folder can be copied into a Unity project. All Gaussians are retained.</summary>
    public static void WriteZip(string path, GaussianCloud cloud, string objectName = "IdolObject")
    {
        var safeName = new string(objectName.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_').ToArray());
        if (string.IsNullOrWhiteSpace(safeName)) safeName = "IdolObject";
        using var file = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var archive = new ZipArchive(file, ZipArchiveMode.Create);
        var assembly = typeof(UnityExport).Assembly;
        var resources = assembly.GetManifestResourceNames().Where(n => n.StartsWith("Unity/", StringComparison.Ordinal)).ToArray();
        if (resources.Length == 0) throw new InvalidOperationException("Unity templates are missing from the application build.");
        foreach (var name in resources)
        {
            using var source = assembly.GetManifestResourceStream(name)!;
            var entry = archive.CreateEntry("Assets/Idol/" + name["Unity/".Length..].Replace('\\', '/'), CompressionLevel.Optimal);
            using var destination = entry.Open();
            source.CopyTo(destination);
        }
        var ply = archive.CreateEntry($"Assets/Idol/Objects/{safeName}.idolply", CompressionLevel.Fastest);
        using (var stream = ply.Open()) PlyFile.Write(stream, cloud);
        var readme = archive.CreateEntry("IMPORT.txt");
        using var writer = new StreamWriter(readme.Open(), Encoding.UTF8);
        writer.WriteLine("Copy this Assets folder into a Unity 2022.3+ project. Unity imports .idolply as a prefab automatically. Drag the imported object into your scene. Keep Assets/Idol/Runtime and Editor with it. See Assets/Idol/README.md for renderer limitations. The .idolply content is standard binary 3DGS PLY and can be renamed to .ply for other viewers.");
    }
}
