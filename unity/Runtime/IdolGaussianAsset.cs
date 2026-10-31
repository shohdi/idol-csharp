using UnityEngine;

namespace Idol.Unity
{
    public sealed class IdolGaussianAsset : ScriptableObject
    {
        public Vector3[] positions;
        public Vector3[] scales;
        public Quaternion[] rotations;
        public Color[] colors;
        public Bounds bounds;
        public int Count => positions == null ? 0 : positions.Length;
    }
}
