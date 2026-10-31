using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace Idol.Unity
{
    [ExecuteAlways]
    public sealed class IdolGaussianRenderer : MonoBehaviour
    {
        public IdolGaussianAsset asset;
        public Shader shader;
        [Range(.1f, 3f)] public float opacity = 1;
        private Material material;
        private ComputeBuffer splats, order;
        private IdolGaussianAsset loadedAsset;
        private int[] indices;
        private float[] depths;
        [StructLayout(LayoutKind.Sequential)]
        private struct Splat { public Vector4 positionOpacity, scale, rotation, color; }

        private void OnEnable() { RenderPipelineManager.endCameraRendering += RenderSrp; }
        private void OnDisable() { RenderPipelineManager.endCameraRendering -= RenderSrp; Release(); }
        private void OnDestroy() { Release(); }
        private void Release()
        {
            splats?.Release(); order?.Release(); splats = null; order = null; loadedAsset = null;
            if (material != null) { if (Application.isPlaying) Destroy(material); else DestroyImmediate(material); material = null; }
        }
        private bool Prepare(Camera camera)
        {
            if (asset == null || asset.Count == 0 || camera == null || !SystemInfo.supportsComputeShaders) return false;
            if (loadedAsset != asset || splats == null)
            {
                Release();
                var selected = shader != null ? shader : Shader.Find("Idol/GaussianSplat");
                if (selected == null || !selected.isSupported) return false;
                material = new Material(selected) { hideFlags = HideFlags.HideAndDontSave };
                var data = new Splat[asset.Count]; indices = new int[asset.Count]; depths = new float[asset.Count];
                for (int i = 0; i < data.Length; ++i)
                {
                    var p = asset.positions[i]; var s = asset.scales[i]; var q = asset.rotations[i]; var c = asset.colors[i];
                    data[i] = new Splat { positionOpacity = new Vector4(p.x, p.y, p.z, c.a), scale = new Vector4(s.x, s.y, s.z, 0), rotation = new Vector4(q.x, q.y, q.z, q.w), color = new Vector4(c.r, c.g, c.b, 1) };
                }
                splats = new ComputeBuffer(asset.Count, 64); splats.SetData(data);
                order = new ComputeBuffer(asset.Count, sizeof(int)); loadedAsset = asset;
            }
            var view = camera.worldToCameraMatrix * transform.localToWorldMatrix;
            for (int i = 0; i < indices.Length; ++i) { indices[i] = i; depths[i] = view.MultiplyPoint3x4(asset.positions[i]).z; }
            // Unity camera space points in front have negative Z. Ascending is back to front.
            Array.Sort(depths, indices); order.SetData(indices);
            material.SetBuffer("_Splats", splats); material.SetBuffer("_Order", order);
            material.SetMatrix("_ObjectToWorld", transform.localToWorldMatrix);
            material.SetMatrix("_View", camera.worldToCameraMatrix);
            material.SetMatrix("_Projection", GL.GetGPUProjectionMatrix(camera.projectionMatrix, camera.targetTexture != null));
            material.SetVector("_Viewport", new Vector4(camera.pixelWidth, camera.pixelHeight, 0, 0));
            material.SetFloat("_Opacity", opacity);
            return true;
        }
        private void OnRenderObject()
        {
            if (GraphicsSettings.currentRenderPipeline != null || !Prepare(Camera.current)) return;
            material.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Triangles, 6, asset.Count);
        }
        private void RenderSrp(ScriptableRenderContext context, Camera camera)
        {
            if (!isActiveAndEnabled || !Prepare(camera)) return;
            var command = new CommandBuffer { name = "Idol Gaussian splats" };
            command.DrawProcedural(Matrix4x4.identity, material, 0, MeshTopology.Triangles, 6, asset.Count);
            context.ExecuteCommandBuffer(command); command.Release();
        }
    }
}
