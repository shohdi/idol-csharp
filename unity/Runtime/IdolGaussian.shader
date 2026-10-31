Shader "Idol/GaussianSplat"
{
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Pass
        {
            ZWrite Off
            ZTest LEqual
            Cull Off
            Blend One OneMinusSrcAlpha
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #include "UnityCG.cginc"
            struct Splat { float4 positionOpacity; float4 scale; float4 rotation; float4 color; };
            StructuredBuffer<Splat> _Splats;
            StructuredBuffer<uint> _Order;
            float4x4 _ObjectToWorld, _View, _Projection;
            float4 _Viewport;
            float _Opacity;
            struct Varyings { float4 position : SV_POSITION; float2 local : TEXCOORD0; float4 color : COLOR0; };
            float3 Rotate(float3 p, float4 q) { return p + 2 * cross(q.xyz, cross(q.xyz, p) + q.w * p); }
            Varyings Vert(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
            {
                Splat s = _Splats[_Order[instance]];
                float4 center = mul(_View, mul(_ObjectToWorld, float4(s.positionOpacity.xyz, 1)));
                float4 clip = mul(_Projection, center);
                Varyings o;
                o.position = float4(2,2,2,1); o.local = 0; o.color = 0;
                if (clip.w <= 0 || center.z >= -.001) return o;
                float3 a = mul((float3x3)_View, mul((float3x3)_ObjectToWorld, Rotate(float3(s.scale.x,0,0), s.rotation)));
                float3 b = mul((float3x3)_View, mul((float3x3)_ObjectToWorld, Rotate(float3(0,s.scale.y,0), s.rotation)));
                float3 c = mul((float3x3)_View, mul((float3x3)_ObjectToWorld, Rotate(float3(0,0,s.scale.z), s.rotation)));
                // Derivative of perspective projection in pixel units. Each projected axis contributes v*v^T.
                float2 focal = float2(_Projection[0][0], _Projection[1][1]) * _Viewport.xy * .5;
                float iz = -1 / center.z;
                float2 pa = focal * (a.xy * iz + center.xy * a.z * iz * iz);
                float2 pb = focal * (b.xy * iz + center.xy * b.z * iz * iz);
                float2 pc = focal * (c.xy * iz + center.xy * c.z * iz * iz);
                float xx = dot(float3(pa.x,pb.x,pc.x), float3(pa.x,pb.x,pc.x)) + .3;
                float yy = dot(float3(pa.y,pb.y,pc.y), float3(pa.y,pb.y,pc.y)) + .3;
                float xy = dot(float3(pa.x,pb.x,pc.x), float3(pa.y,pb.y,pc.y));
                float mid = (xx + yy) * .5;
                float d = sqrt(max(0, (xx-yy)*(xx-yy)*.25 + xy*xy));
                float l1 = max(.1, mid+d), l2 = max(.1, mid-d);
                float2 e1 = abs(xy) > 1e-5 ? normalize(float2(xy,l1-xx)) : (xx >= yy ? float2(1,0) : float2(0,1));
                float2 e2 = float2(-e1.y,e1.x);
                float2 corners[6] = { float2(-1,-1),float2(1,-1),float2(1,1),float2(-1,-1),float2(1,1),float2(-1,1) };
                float2 uv = corners[vertex];
                float2 pixels = 3 * (e1 * sqrt(l1) * uv.x + e2 * sqrt(l2) * uv.y);
                clip.xy += pixels / _Viewport.xy * 2 * clip.w;
                o.position = clip; o.local = uv * 3;
                #ifndef UNITY_COLORSPACE_GAMMA
                s.color.rgb = GammaToLinearSpace(s.color.rgb);
                #endif
                o.color = float4(s.color.rgb, s.positionOpacity.w * _Opacity);
                return o;
            }
            float4 Frag(Varyings i) : SV_Target
            {
                float alpha = min(.99, exp(-.5 * dot(i.local,i.local)) * i.color.a);
                clip(alpha - 1.0/255.0);
                return float4(i.color.rgb * alpha, alpha);
            }
            ENDHLSL
        }
    }
}
