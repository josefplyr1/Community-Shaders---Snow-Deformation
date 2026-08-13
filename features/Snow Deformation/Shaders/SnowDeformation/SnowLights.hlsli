#ifndef __SNOW_LIGHTS_DEPENDENCY_HLSL__
#define __SNOW_LIGHTS_DEPENDENCY_HLSL__

// Point lights for the snow shells, from Light Limit Fix's clustered
// visible-light list. The shells draw in the deferred pass, outside the
// game's per-geometry light plumbing, so the strict-light cbuffer (b3) is
// not available; the cluster list carries every visible placed light and
// is the sole source here. Shadow-casting lights multiply in their channel
// of the game's screen-space shadow mask.

#include "Common/BRDF.hlsli"
#include "Common/Color.hlsli"
#include "Common/SharedData.hlsli"

namespace LightLimitFix
{
#include "LightLimitFix/Common.hlsli"

	// Same registers LLF uses in the forward pass; bound by the CPU side
	// for the shell draws.
	StructuredBuffer<Light> lights : register(t35);
	StructuredBuffer<uint> lightList : register(t36);
	StructuredBuffer<LightGrid> lightGrid : register(t37);
}

#include "InverseSquareLighting/InverseSquareLighting.hlsli"

namespace SnowLights
{
	// The game's deferred shadow mask (one channel per shadow-casting
	// light), valid for the depth-prepass surface under the shell. Same
	// register Lighting.hlsl uses; free in the shell pass.
	Texture2D<float4> ShadowMask : register(t14);

	// LightLimitFix::GetClusterIndex reads FrameBuffer (b12) for its
	// first-person fix; b12 is not bound in this pass and the shell is
	// never first person, so the cluster math is local.
	bool GetClusterIndex(float2 uv, float z, out uint clusterIndex)
	{
		clusterIndex = 0;
		const uint3 clusterSize = SharedData::lightLimitFixSettings.ClusterSize.xyz;
		z = max(z, SharedData::CameraData.y);
		uint clusterZ = log(z / SharedData::CameraData.y) * clusterSize.z / log(SharedData::CameraData.x / SharedData::CameraData.y);
		uint3 cluster = uint3(uint2(uv * clusterSize.xy), clusterZ);
		if (any(cluster >= clusterSize))
			return false;
		clusterIndex = cluster.x + (clusterSize.x * cluster.y) + (clusterSize.x * clusterSize.y * cluster.z);
		return true;
	}

	// Adds the clustered point lights to the shell's direct lobes. All
	// positions camera-relative. clusterUV/maskUV are projection-space
	// screen UVs: clusterUV for the shell pixel itself, maskUV for the
	// ground point beneath it (the mask was rendered against the prepass
	// depth; sampling at the under-point keeps fire shadows from sliding
	// with camera motion). Room/portal culling is skipped: the shell only
	// exists in exteriors. GetAttenuation self-selects inverse-square vs
	// vanilla falloff per light flags, so ISL parity is automatic.
	void AccumulatePointLights(
		float3 worldPos, float3 normalWS, float3 V, float viewZ,
		float2 clusterUV, float2 maskUV, float2 dynResScale,
		float3 albedo, float3 F0, float roughness,
		inout float3 diffuse, inout float3 specular)
	{
		uint clusterIndex = 0;
		[branch] if (!GetClusterIndex(clusterUV, viewZ, clusterIndex))
			return;

		LightLimitFix::LightGrid grid = LightLimitFix::lightGrid[clusterIndex];

		float4 shadowMask = 1.0;
		bool maskLoaded = false;

		[loop] for (uint i = 0; i < grid.lightCount; i++)
		{
			LightLimitFix::Light light = LightLimitFix::lights[LightLimitFix::lightList[grid.offset + i]];

			float3 lightDirection = light.positionWS.xyz - worldPos;
			float lightDist = length(lightDirection);
			float attenuation = InverseSquareLighting::GetAttenuation(lightDist, light);
			if (attenuation < 1e-5)
				continue;

			const bool isPointLightLinear = light.lightFlags & LightLimitFix::LightFlags::Linear;
			float3 lightColor = Color::PointLight(light.color.xyz, isPointLightLinear) * attenuation * light.fade;

			float lightShadow = 1.0;
			[branch] if (light.lightFlags & LightLimitFix::LightFlags::Shadow)
			{
				[branch] if (!maskLoaded)
				{
					// Upscaling renders into the top-left sub-rect of the
					// full-size target; full-screen UV must be scaled by the
					// dynamic-resolution ratio (Lighting.hlsl's
					// GetDynamicResolutionAdjustedScreenPosition, b12-free).
					// The under-point can also project past the viewport
					// edge; clamp to the sub-rect instead of reading outside
					// the rendered region.
					int2 renderDims = int2(SharedData::BufferDim.xy * dynResScale);
					int2 maskPixel = clamp(int2(maskUV * dynResScale * SharedData::BufferDim.xy), int2(0, 0), renderDims - 1);
					shadowMask = ShadowMask.Load(int3(maskPixel, 0));
					maskLoaded = true;
				}
				lightShadow = shadowMask[light.shadowLightIndex];
			}

			float3 L = normalize(lightDirection);
			float satNdotL = saturate(dot(normalWS, L));
			if (satNdotL <= 0.0 || lightShadow <= 0.0)
				continue;

			float3 H = normalize(V + L);
			float satNdotV = saturate(abs(dot(normalWS, V)) + 1e-5);
			float satNdotH = saturate(dot(normalWS, H));
			float satVdotH = saturate(dot(V, H));

			float3 F = BRDF::F_Schlick(F0, satVdotH);
			float specD = BRDF::D_GGX(roughness, satNdotH);
			float specV = BRDF::Vis_SmithJointApprox(roughness, satNdotV, satNdotL);

			float3 lit = lightColor * lightShadow * satNdotL;
			diffuse += lit * (1.0 - F) * albedo;
			specular += specD * specV * F * lit;
		}
	}
}

#endif  //__SNOW_LIGHTS_DEPENDENCY_HLSL__
