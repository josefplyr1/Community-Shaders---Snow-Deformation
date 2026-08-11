// Crisp cascaded shadows for the snow shell.
//
// The Volumetric Shadows feature exposes the sun cascades only as a 512px
// blurred VSM moments copy — fine for volumetrics, but on the shell it turns
// tree branches and actor silhouettes into smudges while the bare ground next
// to it (vanilla forward path) shows crisp cascade shadows. This samples the
// game's RAW shadow atlas instead: cascade = array slice, per-slice [0,1] UVs
// matching the ShadowProj matrices from DirectionalShadowLights (t98), with
// hardware comparison PCF at full atlas resolution.
//
// Two source textures, min'd like VolumetricShadows' downsample does: the
// main atlas (grabbed at EarlyPrepass from PS t4, right after the game
// rendered the cascades) and the ESRAM partner target. Cascade selection,
// blend and distance fade mirror VolumetricShadows::GetVSMShadow2D so the
// crisp and fallback paths agree about where shadows exist.
//
// Include AFTER Common/ShadowSampling.hlsli — needs DirectionalShadowLights,
// SharedData and FrameBuffer from it.

Texture2DArray<float> SnowShadowAtlas : register(t22);
Texture2DArray<float> SnowShadowAtlasESRAM : register(t23);
SamplerComparisonState SnowShadowCmpSampler : register(s2);

namespace SnowShadow
{
	float SampleCascadeCmp(float3 posLS, uint cascade)
	{
		float lit = SnowShadowAtlas.SampleCmpLevelZero(SnowShadowCmpSampler, float3(posLS.xy, cascade), posLS.z);
		float litEsram = SnowShadowAtlasESRAM.SampleCmpLevelZero(SnowShadowCmpSampler, float3(posLS.xy, cascade), posLS.z);
		return min(lit, litEsram);
	}

	// Center + 4 rotated taps, each hardware-bilinear 2x2 comparison — a
	// small soft edge like the vanilla receiver, nothing like the VSM blur.
	// a_spread widens the tap ring: >1 turns the far cascade's blocky
	// texels into soft penumbra blobs WITHOUT dropping the shadows (LOD
	// trees live in these cascades — fading them out erased their shadows
	// from distant snow).
	float SampleCascadePCF(float3 posLS, uint cascade, float2 texel, float a_spread)
	{
		float shadow = SampleCascadeCmp(posLS, cascade);
		const float2 kTaps[4] = { { 1.4, 0.4 }, { -0.4, 1.4 }, { -1.4, -0.4 }, { 0.4, -1.4 } };
		[unroll] for (uint tapI = 0; tapI < 4; tapI++)
			shadow += SampleCascadeCmp(float3(posLS.xy + kTaps[tapI] * texel * a_spread, posLS.z), cascade);
		return shadow * 0.2;
	}

	// positionRel is camera-relative. The receiver is offset along the normal
	// (instead of a large depth bias) so flat sunlit snow shows no acne while
	// contact shadows stay attached.
	// The shared DirectionalShadowLights buffer carries only cascades 0/1;
	// slice 2 is the game's LOD SHADOW cascade (distant LOD trees/objects
	// render into it — bare ground samples it in the vanilla shadow mask).
	// Its projection arrives via a_lodProj/a_lodEnd from ShellCB so distant
	// shell snow receives the same LOD shadows the ground does.
	float GetCascadeShadow(float3 positionRel, float3 normalWS, float a_spread,
		column_major float4x4 a_lodProj, float a_lodEnd, float a_lodActive)
	{
		DirectionalShadowLightData sd = DirectionalShadowLights[0];

		float shadowMapDepth = SharedData::GetScreenDepth(FrameBuffer::GetShadowDepth(positionRel));

		float3 positionWS = positionRel + FrameBuffer::CameraPosAdjust.xyz + normalWS * 3.0;

		float atlasW, atlasH, atlasSlices;
		SnowShadowAtlas.GetDimensions(atlasW, atlasH, atlasSlices);
		float2 texel = 1.0 / float2(atlasW, atlasH);

		[branch] if (shadowMapDepth >= sd.EndSplitDistances.y)
		{
			[branch] if (a_lodActive > 0.5 && atlasSlices >= 2.5 && shadowMapDepth < a_lodEnd)
			{
				float3 posLOD = mul(a_lodProj, float4(positionWS, 1)).xyz;
				posLOD.xy = saturate(posLOD.xy);
				posLOD.z -= 0.0024;  // bias pattern: 0.0008 * (cascade + 1)
				float lodShadow = SampleCascadePCF(posLOD, 2, texel, a_spread);
				float lodFade = saturate(shadowMapDepth / a_lodEnd);
				return lerp(1.0, lodShadow, 1.0 - pow(lodFade * lodFade, 8));
			}
			return 1.0;
		}

		// Cascade selection and blend — identical to GetVSMShadow2D.
		float cascadeSelect = saturate((shadowMapDepth - sd.StartSplitDistances.y) / (sd.EndSplitDistances.x - sd.StartSplitDistances.y));
		uint primaryCascade = uint(cascadeSelect);

		float3 posLS = mul(sd.ShadowProj[primaryCascade], float4(positionWS, 1)).xyz;
		posLS.xy = saturate(posLS.xy);
		posLS.z -= 0.0008 * (primaryCascade + 1.0);
		float shadow = SampleCascadePCF(posLS, primaryCascade, texel, a_spread);

		[branch] if (cascadeSelect > 0.0 && cascadeSelect < 1.0)
		{
			uint secondaryCascade = 1 - primaryCascade;
			posLS = mul(sd.ShadowProj[secondaryCascade], float4(positionWS, 1)).xyz;
			posLS.xy = saturate(posLS.xy);
			posLS.z -= 0.0008 * (secondaryCascade + 1.0);
			float shadowBlend = SampleCascadePCF(posLS, secondaryCascade, texel, a_spread);
			shadow = lerp(shadow, shadowBlend, smoothstep(0, 1, cascadeSelect));
		}

		// Distance fade matching the VSM path: beyond the cascades the world
		// shadows (terrain/cloud) carry on alone, with no visible boundary.
		float fade = saturate(shadowMapDepth / sd.EndSplitDistances.y);
		return lerp(1.0, shadow, 1.0 - pow(fade * fade, 8));
	}
}
