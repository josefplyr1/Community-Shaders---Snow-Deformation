// Snow shell renderer.
//
// A vertex-buffer-less, camera-following grid of real snow geometry: the
// grid is generated from SV_VertexID (6 vertices per quad), conforms to the
// baked terrain data window, is displaced by the per-texture-class snow
// depth, and carved by the deformation map. Per-pixel normals come from the
// same height/deformation fields, so trench walls shade smoothly even where
// the geometry is coarse.
//
// Camera matrices arrive via the private ShellCB (b0), copied from the same
// per-frame data the game uploads to b12. SharedData (b5) is bound by the
// CPU side for lighting.

#include "Common/BRDF.hlsli"
#include "Common/Color.hlsli"
#include "Common/GBuffer.hlsli"
#include "Common/Random.hlsli"
#include "Common/SharedData.hlsli"

cbuffer ShellCB : register(b0)
{
	// row_major matches the game's FrameBuffer.hlsli declarations — the CPU
	// side copies the b12 bytes verbatim, which are row-major packed.
	row_major float4x4 CameraViewProj;
	row_major float4x4 CameraViewProjUnjittered;
	row_major float4x4 CameraPreviousViewProjUnjittered;
	row_major float4x4 CameraView;

	float4 ShellCameraPosAdjust;
	float4 ShellCameraPreviousPosAdjust;

	float2 GridOrigin;
	float GridSpacing;
	float TerrainTexelSize;

	// Grid-local sampling offsets, precomputed on CPU: absolute world XY at
	// ~1e5 magnitude destroys float32 finite differences (shimmer).
	float2 GridToTerrainOffset;
	float2 GridToDeformOffset;

	float WarpedHalfSpan;
	uint GridDim;
	uint TerrainDim;
	uint ShellDebugData;

	float DeformInvWorldSize;
	float3 padShell;
}

Texture2D<float4> TerrainWindow : register(t0);
Texture2D<float> DeformationMap : register(t1);
// Full-scene depth copy (Terrain Blending's blended depth when available) —
// never the bound DSV, so sampling during the shell draw is legal.
Texture2D<float> SceneDepth : register(t3);

// Distance warp: inner kWarpInnerVerts vertices per side keep linear
// GridSpacing; beyond them each ring's spacing grows by kWarpGrowth so the
// grid stretches ~26k units from the camera. Constants MUST match
// SnowDeformation.h (kShellWarpInnerVerts / kShellWarpGrowth).
static const float kWarpInnerVerts = 256.0;
static const float kWarpGrowth = 1.0902;

// Maps a vertex coordinate relative to the grid center (in vertex units)
// to a world-unit offset from the center.
float WarpAxis(float u)
{
	float a = abs(u);
	float lin = min(a, kWarpInnerVerts);
	float ext = max(a - kWarpInnerVerts, 0.0);
	float outer = kWarpGrowth * (pow(kWarpGrowth, ext) - 1.0) / (kWarpGrowth - 1.0);
	return sign(u) * (lin + outer) * GridSpacing;
}

struct VS_OUTPUT
{
	float4 Position : SV_POSITION;
	float4 CurrentClip : TEXCOORD0;
	float4 PreviousClip : TEXCOORD1;
	float3 WorldPos : TEXCOORD2;
	float2 GridLocal : TEXCOORD3;
	float Snowness : TEXCOORD4;
	float DebugHeight : TEXCOORD5;
	// xyz: smooth per-vertex terrain normal, w: coverage alpha (taper*fade).
	float4 TerrainNormalAlpha : TEXCOORD6;
};

// Manual bilinear over the terrain window (Load-based, deterministic).
// Returns (height, rampDepth, coverage): rampDepth is the per-texture-class
// depth blend in world units, precomputed at window-rebuild time on the CPU.
// gridLocal = world XY relative to GridOrigin (small, precision-safe).
float3 SampleTerrain(float2 gridLocal)
{
	float2 t = (GridToTerrainOffset + gridLocal) / TerrainTexelSize;
	t = clamp(t, 0.0, (float)(TerrainDim - 1) - 0.001);
	int2 t0 = (int2)t;
	float2 f = t - t0;
	int2 t1 = min(t0 + 1, int2(TerrainDim - 1, TerrainDim - 1));

	float3 s00 = TerrainWindow.Load(int3(t0.x, t0.y, 0)).xyz;
	float3 s10 = TerrainWindow.Load(int3(t1.x, t0.y, 0)).xyz;
	float3 s01 = TerrainWindow.Load(int3(t0.x, t1.y, 0)).xyz;
	float3 s11 = TerrainWindow.Load(int3(t1.x, t1.y, 0)).xyz;

	return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

// Bilinear helper at fractional texel coordinates (Load-based).
float SampleDeformationBilinear(float2 t, float2 dims)
{
	t = clamp(t, 0.0, dims - 1.001);
	int2 t0 = (int2)t;
	float2 f = t - t0;
	int2 t1 = min(t0 + 1, int2(dims) - 1);

	float s00 = DeformationMap.Load(int3(t0.x, t0.y, 0));
	float s10 = DeformationMap.Load(int3(t1.x, t0.y, 0));
	float s01 = DeformationMap.Load(int3(t0.x, t1.y, 0));
	float s11 = DeformationMap.Load(int3(t1.x, t1.y, 0));

	return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

// B-spline bicubic sample of the deformation map, built from four bilinear
// taps at fractional offsets. Smooth VALUE and smooth GRADIENT: plain
// bilinear leaves texel-rate creases in trench walls, and value-only
// smoothing terraces the gradient the normals consume.
// Returns 0 outside the deformation window.
float SampleDeformation(float2 gridLocal)
{
	float2 uv = (GridToDeformOffset + gridLocal) * DeformInvWorldSize;
	if (any(uv < 0.0) || any(uv > 1.0))
		return 0.0;

	float2 dims;
	DeformationMap.GetDimensions(dims.x, dims.y);
	float2 t = uv * dims - 0.5;
	float2 i = floor(t);
	float2 f = t - i;

	float2 f2 = f * f;
	float2 f3 = f2 * f;
	float2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
	float2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
	float2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
	float2 w3 = f3 / 6.0;

	float2 g0 = w0 + w1;
	float2 g1 = w2 + w3;
	float2 h0 = i - 1.0 + w1 / g0;
	float2 h1 = i + 1.0 + w3 / g1;

	float v00 = SampleDeformationBilinear(float2(h0.x, h0.y), dims);
	float v10 = SampleDeformationBilinear(float2(h1.x, h0.y), dims);
	float v01 = SampleDeformationBilinear(float2(h0.x, h1.y), dims);
	float v11 = SampleDeformationBilinear(float2(h1.x, h1.y), dims);

	return g0.y * (g0.x * v00 + g1.x * v10) + g1.y * (g0.x * v01 + g1.x * v11);
}

// The shell surface: per-texture-class snow depth carved by deformation.
// Class depths blend by their baked weights on the CPU (window rebuild), so
// boundaries between differently-deep snows are geometric depth ramps —
// negative depths (roads) submerge the shell below the surface.
// Shared by the VS (geometry) and PS (per-pixel normals) so both agree.
float ShellSurfaceZ(float2 gridLocal, out float coverage, out float terrainHeight)
{
	float3 terrain = SampleTerrain(gridLocal);
	terrainHeight = terrain.x;
	float rampDepth = terrain.y;
	coverage = saturate(terrain.z);

	// Distant de-noising: out here one terrain-window texel spans 100+
	// world units, and the bilinear height/coverage under-resolve — on
	// slopes the interpolated height dips BELOW the real landscape mesh
	// and single bare texels punch pinholes into continuous snowfields,
	// holes that crawl as the window re-rasterizes with camera motion. A
	// snow-biased neighborhood MAX over one texel radius lifts the shell
	// clear of the mesh and fills the pinholes; the blend-in leaves the
	// near field untouched.
	float camDist = length(gridLocal - WarpedHalfSpan);
	[branch] if (camDist > 4000.0)
	{
		float farBlend = smoothstep(4000.0, 10000.0, camDist);
		float3 n0 = SampleTerrain(gridLocal + float2(TerrainTexelSize, 0.0));
		float3 n1 = SampleTerrain(gridLocal - float2(TerrainTexelSize, 0.0));
		float3 n2 = SampleTerrain(gridLocal + float2(0.0, TerrainTexelSize));
		float3 n3 = SampleTerrain(gridLocal - float2(0.0, TerrainTexelSize));
		float3 n4 = SampleTerrain(gridLocal + float2(TerrainTexelSize, TerrainTexelSize));
		float3 n5 = SampleTerrain(gridLocal - float2(TerrainTexelSize, TerrainTexelSize));
		float3 n6 = SampleTerrain(gridLocal + float2(TerrainTexelSize, -TerrainTexelSize));
		float3 n7 = SampleTerrain(gridLocal + float2(-TerrainTexelSize, TerrainTexelSize));
		// Never-rasterized window texels at the data edge hold sentinel
		// heights: ONE such neighbor tap would explode the ridge pad into a
		// kilometer-tall white pillar. Trust only plausible taps (fall back
		// to the center height) and cap the pad at a sane bound.
		float h0 = n0.x > -50000.0 ? n0.x : terrainHeight;
		float h1 = n1.x > -50000.0 ? n1.x : terrainHeight;
		float h2 = n2.x > -50000.0 ? n2.x : terrainHeight;
		float h3 = n3.x > -50000.0 ? n3.x : terrainHeight;
		float h4 = n4.x > -50000.0 ? n4.x : terrainHeight;
		float h5 = n5.x > -50000.0 ? n5.x : terrainHeight;
		float h6 = n6.x > -50000.0 ? n6.x : terrainHeight;
		float h7 = n7.x > -50000.0 ? n7.x : terrainHeight;
		float maxHeight = max(max(max(h0, h1), max(h2, h3)), max(max(h4, h5), max(h6, h7)));
		float c0 = n0.x > -50000.0 ? n0.z : 0.0;
		float c1 = n1.x > -50000.0 ? n1.z : 0.0;
		float c2 = n2.x > -50000.0 ? n2.z : 0.0;
		float c3 = n3.x > -50000.0 ? n3.z : 0.0;
		float c4 = n4.x > -50000.0 ? n4.z : 0.0;
		float c5 = n5.x > -50000.0 ? n5.z : 0.0;
		float c6 = n6.x > -50000.0 ? n6.z : 0.0;
		float c7 = n7.x > -50000.0 ? n7.z : 0.0;
		float maxCoverage = saturate(max(max(max(c0, c1), max(c2, c3)), max(max(c4, c5), max(c6, c7))));
		// Within-texel ridge error: one height sample per 100+ world units
		// cannot see a crest between texel centers — on a slope the real
		// mesh can top the interpolated field by half the local gradient,
		// which is exactly the crawling holes that survive the plain max.
		// Pad by that bound so the shell always clears the mesh.
		float ridgePad = min(0.25 * max(abs(h0 - h1), abs(h2 - h3)), 150.0);
		terrainHeight = lerp(terrainHeight, max(terrainHeight, maxHeight) + ridgePad, farBlend);
		coverage = lerp(coverage, max(coverage, maxCoverage), farBlend);
		// Decisive separation: wherever the pad still lands the far shell
		// within a few units of the landscape mesh, the two interleave at
		// depth precision and the winner flips with the view angle — holes
		// that appear and vanish as the camera turns. A fixed covered-only
		// margin makes the shell win at every angle.
		terrainHeight += farBlend * 8.0 * saturate(coverage);
	}

	// Fade toward the grid boundary so the shell melts into the terrain.
	float2 delta = abs(gridLocal - WarpedHalfSpan);
	float edgeFade = saturate((WarpedHalfSpan - max(delta.x, delta.y)) / 2048.0);

	// Bare ground contributes negative depth so the shell submerges toward
	// uncovered terrain as well.
	float bare = saturate(1.0 - coverage);
	float depth = rampDepth + (-8.0) * bare;
	depth = lerp(-8.0, depth, edgeFade);

	// Deformation carves only where the layer is actually raised; the
	// negative-depth submerge at class edges is untouched.
	[flatten] if (depth > 0.0)
	{
		float deformation = saturate(SampleDeformation(gridLocal));
		depth *= 1.0 - deformation;
	}

	return terrainHeight + depth;
}

#ifdef VSHADER
VS_OUTPUT main(uint vertexID : SV_VertexID)
{
	static const float2 kCorners[6] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
	static const float2 kCornersFlipped[6] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, { 0, 1 } };

	uint quadIndex = vertexID / 6;
	uint2 quadXY = uint2(quadIndex % GridDim, quadIndex / GridDim);
	// Union-jack triangulation: alternate the quad diagonal on a checkerboard
	// so trench walls crossing the grid alias half as hard — one uniform
	// diagonal turns every wall into a same-handed sawtooth. The parity is
	// anchored to WORLD position (GridOrigin folded in), not grid indices:
	// index parity re-phases every camera step and makes walls visibly shift
	// with camera movement.
	int2 parityBase = int2(floor(GridOrigin / GridSpacing));
	uint parity = uint(parityBase.x + int(quadXY.x)) ^ uint(parityBase.y + int(quadXY.y));
	float2 corner = ((parity & 1) != 0) ? kCornersFlipped[vertexID % 6] : kCorners[vertexID % 6];
	float2 gridPos = float2(quadXY) + corner;
	// Warped placement: gridLocal stays a world-unit offset from GridOrigin
	// (the warped grid's min corner), so all field sampling is unchanged.
	float2 u = gridPos - (float)GridDim * 0.5;
	float2 gridLocal = float2(WarpAxis(u.x), WarpAxis(u.y)) + WarpedHalfSpan;
	float2 absXY = GridOrigin + gridLocal;

	// World-anchored outer rings: warped-zone vertex offsets are camera-
	// relative, so distant geometry MORPHS as the camera moves — terrain
	// resampled at ever-shifting positions reads as breathing shapes,
	// landscape peeking through, and blotchy far shading. Snapping each
	// vertex to its OWN ring's step in WORLD space pins distant vertices in
	// place; a vertex hops one ring-step only when the camera crosses one,
	// which distance and TAA absorb. The inner linear region is already
	// stable through GridOrigin's whole-texel snapping, so the snap blends
	// in smoothly where the rings start stretching.
	float2 ringStep = GridSpacing * pow(kWarpGrowth, max(abs(u) - kWarpInnerVerts, 0.0));
	float2 snapT = saturate(ringStep / GridSpacing - 1.0);
	float2 snapped = floor(absXY / ringStep + 0.5) * ringStep;
	absXY = lerp(absXY, snapped, snapT);
	gridLocal = absXY - GridOrigin;

	float coverage;
	float terrainHeight;
	float z = ShellSurfaceZ(gridLocal, coverage, terrainHeight);

	// Smooth terrain normal per-vertex (wide 32-unit differences bridge the
	// 128-unit data texels — interpolation then removes the faceting the
	// per-pixel piecewise-constant gradient produced).
	float hxp = SampleTerrain(gridLocal + float2(32.0, 0.0)).x;
	float hxn = SampleTerrain(gridLocal - float2(32.0, 0.0)).x;
	float hyp = SampleTerrain(gridLocal + float2(0.0, 32.0)).x;
	float hyn = SampleTerrain(gridLocal - float2(0.0, 32.0)).x;
	float3 terrainNormal = normalize(float3(-(hxp - hxn) / 64.0, -(hyp - hyn) / 64.0, 1.0));

	// Coverage alpha drives both geometry taper and edge dithering in the PS.
	float taper = smoothstep(0.0, 0.6, coverage);
	float2 edgeDelta = abs(gridLocal - WarpedHalfSpan);
	float edgeFade = saturate((WarpedHalfSpan - max(edgeDelta.x, edgeDelta.y)) / 2048.0);
	float coverageAlpha = taper * edgeFade;

	// Data debug: conforming plane well above the sampled terrain height,
	// colored by the sampled values.
	if (ShellDebugData != 0)
		z = terrainHeight + 200.0;

	float3 absolutePos = float3(absXY, z);

	// Shader world space is camera-relative; previous frame uses its own adjust.
	float3 rel = absolutePos - ShellCameraPosAdjust.xyz;
	float3 prevRel = absolutePos - ShellCameraPreviousPosAdjust.xyz;

	VS_OUTPUT vsout;
	vsout.Position = mul(CameraViewProj, float4(rel, 1.0));
	vsout.CurrentClip = mul(CameraViewProjUnjittered, float4(rel, 1.0));
	vsout.PreviousClip = mul(CameraPreviousViewProjUnjittered, float4(prevRel, 1.0));
	vsout.WorldPos = rel;
	vsout.GridLocal = gridLocal;
	vsout.Snowness = coverage;
	vsout.DebugHeight = terrainHeight;
	vsout.TerrainNormalAlpha = float4(terrainNormal, coverageAlpha);
	return vsout;
}
#endif

#ifdef PSHADER
struct PS_OUTPUT
{
	float4 Diffuse : SV_Target0;
	float4 MotionVectors : SV_Target1;
	float4 NormalGlossiness : SV_Target2;
	float4 Albedo : SV_Target3;
	float4 Specular : SV_Target4;
	float4 Reflectance : SV_Target5;
	float4 Masks : SV_Target6;
	float4 Masks2 : SV_Target7;
};

PS_OUTPUT main(VS_OUTPUT input)
{
	// Same convention as MotionBlur::GetSSMotionVector.
	float2 motionVector = float2(-0.5, 0.5) * (input.CurrentClip.xy / input.CurrentClip.w - input.PreviousClip.xy / input.PreviousClip.w);

	// Coverage alpha recomputed PER PIXEL from the terrain field (Terrain
	// Blending-style): smooth at texture resolution, independent of vertex
	// interpolation. The temporally-varying stochastic test then dithers the
	// boundary and TAA resolves it into a true cross-fade.
	float2 gridLocal = input.GridLocal;
	float3 pixelTerrain = SampleTerrain(gridLocal);
	float pixelCoverage = saturate(pixelTerrain.z);
	float2 psEdgeDelta = abs(gridLocal - WarpedHalfSpan);
	float psEdgeFade = saturate((WarpedHalfSpan - max(psEdgeDelta.x, psEdgeDelta.y)) / 2048.0);

	// Un-carved class depth ramp at this pixel. Where it goes negative the
	// shell is submerged (depth-rejected anyway). The dither rides the ramp:
	// alpha fades over the tail of positive depth, so a boundary toward
	// shallower/negative classes (where coverage stays ~1 and the coverage
	// term can't blend) dissolves stochastically as the shell thins, instead
	// of presenting a bare geometric plunge with a thin z-fight strip.
	float pixelBare = saturate(1.0 - pixelCoverage);
	float pixelRampDepth = pixelTerrain.y + (-8.0) * pixelBare;
	pixelRampDepth = lerp(-8.0, pixelRampDepth, psEdgeFade);

	// Depth reads shared by the edge dissolve and the proximity fade below.
	float sceneZ = SharedData::GetScreenDepth(SceneDepth.Load(int3(input.Position.xy, 0)));
	float shellZ = input.CurrentClip.w;

	float rampFadeBand = max(2.0, pixelRampDepth * 0.2);
	float coverageAlpha = smoothstep(0.0, 0.6, pixelCoverage) * psEdgeFade * smoothstep(0.0, rampFadeBand, pixelRampDepth);

	// Object blending (Terrain Blending-style depth proximity): where the
	// shell hovers within a few units in front of ANY geometry behind it —
	// walkway planks, mesh roads, rocks — dissolve it into the dither. The
	// shell only knows terrain heights, so this is what makes it meet
	// statics softly instead of slicing across them at the depth test.
	// The gap is measured along the view ray, so it shifts with the camera;
	// widening the band with distance turns that parallax wobble into a
	// broad soft fade instead of a visible shape-hunting shimmer around
	// distant uncovered objects.
	// Mild distance scaling only: a steep view-Z-proportional band makes the
	// blend wobble with camera tilt even up close.
	float objectFadeBand = 10.0 + shellZ * 0.004;
	float proximityFade = saturate((sceneZ - shellZ) / objectFadeBand);
	// Carved trench floors DELIBERATELY hug the geometry behind them
	// (terrain, actor feet standing in the trench) and must override the
	// fade — without the override they get view-dependently dithered away,
	// which reads as "snow moving with the camera".
	float pixelCarve = saturate(SampleDeformation(gridLocal));
	float carveOverride = smoothstep(0.1, 0.5, pixelCarve);
	coverageAlpha *= max(proximityFade, carveOverride);

	// Stochastic discard dither: proven to blend in this pipeline (the wide
	// distance fade). Writing alpha without discarding blends nothing in our
	// pass — TB's alpha path runs through depth-prepass machinery we do not
	// replicate.
	float screenNoise = Random::InterleavedGradientNoise(input.Position.xy, SharedData::FrameCount);
	[branch] if (ShellDebugData == 0)
	{
		if (screenNoise * screenNoise >= coverageAlpha)
			discard;
	}

	// Normal = smooth interpolated terrain normal + per-pixel deformation
	// gradient (central differences at the deformation map's resolution).
	const float step = 4.0;
	float dXP = SampleDeformation(gridLocal + float2(step, 0.0));
	float dXN = SampleDeformation(gridLocal - float2(step, 0.0));
	float dYP = SampleDeformation(gridLocal + float2(0.0, step));
	float dYN = SampleDeformation(gridLocal - float2(0.0, step));
	float2 deformGradient = float2(dXP - dXN, dYP - dYN) / (2.0 * step);

	float3 terrainNormal = normalize(input.TerrainNormalAlpha.xyz);
	// z = T + depth*(1-D)  =>  grad z = grad T - depth*grad D
	float pixelDepth = max(pixelRampDepth, 0.0);
	float2 gradZ = -terrainNormal.xy / max(terrainNormal.z, 0.1) - pixelDepth * deformGradient;

	float3 normalWS = normalize(float3(gradZ * -1.0, 1.0));
	float3 viewNormal = normalize(mul((float3x3)CameraView, normalWS));

	// PBR snow material: GGX microfacet specular with Fresnel and
	// energy-conserving lobes. Light and ambient stay in the frame's units
	// (raw DirLightColor / Color::Ambient — DirLightColor is already
	// π-scaled by pipeline convention, so no Lambert 1/π on diffuse); the
	// material RESPONSE is physically based, and the indirect specular lobe
	// goes to the Reflectance RT where the composite applies cubemap and
	// ambient specular like any TruePBR surface.
	static const float3 kSnowAlbedo = float3(0.82, 0.84, 0.88);
	static const float kSnowRoughness = 0.6;
	static const float3 kSnowF0 = float3(0.028, 0.028, 0.028);

	float3 V = -normalize(input.WorldPos);
	float3 L = SharedData::DirLightDirection.xyz;
	float3 H = normalize(V + L);
	float satNdotL = saturate(dot(normalWS, L));
	float satNdotV = saturate(abs(dot(normalWS, V)) + 1e-5);
	float satNdotH = saturate(dot(normalWS, H));
	float satVdotH = saturate(dot(V, H));

	// Unshadowed sun for now: shadow sampling on the shell lands with the
	// shadow layers.
	float3 sunLight = SharedData::DirLightColor.xyz;

	float3 F = BRDF::F_Schlick(kSnowF0, satVdotH);
	float specD = BRDF::D_GGX(kSnowRoughness, satNdotH);
	float specV = BRDF::Vis_SmithJointApprox(kSnowRoughness, satNdotV, satNdotL);

	// Indirect lobes: the specular weight is what the environment reflects,
	// diffuse receives only what specular does not (energy conservation).
	float2 envBRDF = BRDF::EnvBRDF(kSnowRoughness, satNdotV);
	float3 specularLobe = kSnowF0 * envBRDF.x + envBRDF.y;
	float3 diffuseLobe = kSnowAlbedo * (1.0 - specularLobe);

	float3 directDiffuse = sunLight * satNdotL * (1.0 - F) * kSnowAlbedo;
	float3 directSpecular = specD * specV * F * sunLight * satNdotL;

	float3 ambientColor = Color::Ambient(max(0, SharedData::GetAmbient(normalWS)));
	float3 ambientPart = ambientColor * diffuseLobe;
	float3 preLit = ambientPart + directDiffuse;

	[branch] if (ShellDebugData != 0)
	{
		// R = height, G = per-pixel coverage, B = ramp depth (40 units =
		// full blue) — depth-class boundaries show as blue-intensity steps.
		float heightNorm = saturate((input.DebugHeight + 4000.0) / 8000.0);
		bool isSentinel = input.DebugHeight < -50000.0;
		preLit = isSentinel ? float3(0.0, 0.0, 0.5) : float3(heightNorm * 0.25, pixelCoverage * 0.5, saturate(pixelTerrain.y / 40.0));
	}

	// Terrain Blending-style output: alpha rides every .w and the stochastic
	// blend mask goes to NormalGlossiness.w, exactly as Lighting.hlsl's
	// deferred tail encodes it for the temporal resolve.
	float alpha = ShellDebugData != 0 ? 1.0 : coverageAlpha;
	float stochasticBlend = (screenNoise * screenNoise) < alpha ? 1.0 : 0.0;

	PS_OUTPUT psout;
	psout.Diffuse = float4(preLit, alpha);
	psout.MotionVectors = float4(motionVector, 0.0, alpha);
	psout.NormalGlossiness = float4(GBuffer::EncodeNormal(viewNormal), 1.0 - kSnowRoughness, stochasticBlend);
	// Albedo carries the diffuse lobe (Lighting's PBR tail writes the same),
	// Specular the direct GGX lobe, Reflectance the environment lobe weight.
	psout.Albedo = float4(diffuseLobe, alpha);
	psout.Specular = float4(directSpecular, alpha);
	psout.Reflectance = float4(specularLobe, alpha);
	// Masks.z carries the raw directional ambient luma for the composite
	// (un-multiplied by albedo, matching Lighting's masksZ convention).
	psout.Masks = float4(0.0, 0.0, Color::RGBToYCoCg(ambientColor).x, alpha);
	psout.Masks2 = float4(0.0, 0.0, 0.0, alpha);
	return psout;
}
#endif
