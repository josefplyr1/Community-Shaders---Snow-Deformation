// Snow shell renderer.
//
// S4: lit, materialized snow. The shell conforms to terrain (S2) and is
// carved by the deformation map (S3); this stage adds real lighting from
// CS's shared data (SH ambient + directional sun), per-pixel normals
// computed from the height/deformation fields (smooth trench walls even on
// coarse geometry), and edge feathering: snow tapers to the ground at
// coverage borders and at the grid boundary.
//
// Vertex-buffer-less: the grid is generated from SV_VertexID, 6 vertices per
// quad. Camera matrices arrive via the private ShellCB (b0), copied from the
// same per-frame data the game uploads to b12. SharedData (b5) is bound by
// the CPU side for lighting.

#include "Common/BRDF.hlsli"
#include "Common/Color.hlsli"
#include "Common/GBuffer.hlsli"
#include "Common/Random.hlsli"
#include "Common/SharedData.hlsli"

#ifdef PSHADER
// TruePBR's procedural glint NDF (Deliot & Chermain 2023) for snow sparkle.
// Needs only the shared 128px noise texture at t20, which the CPU side binds
// for this pass (EnableGlints gates the path when it is unavailable).
#	include "Common/Glints/Glints2023.hlsli"
// Shadow sampling for the shell surface: terrain/cloud shadows via
// GetWorldShadow and dynamic (actor) shadows via the VolumetricShadows
// shared shadow map — softer than the game's cascades but world-anchored.
// (The screen-space shadow mask was tried and rejected: it holds values for
// the terrain BEHIND the shell along the view ray, so shadows slide with
// camera movement. Direct cascade sampling is the queued proper upgrade.)
#	define TERRAIN_SHADOWS
#	define CLOUD_SHADOWS
#	define VOLUMETRIC_SHADOWS
SamplerState ShellLinearSampler : register(s1);
#	define LinearSampler ShellLinearSampler
#	include "Common/ShadowSampling.hlsli"
#	include "ScreenSpaceShadows/ScreenSpaceShadows.hlsli"
#	include "SnowDeformation/SnowShadow.hlsli"
#endif

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
	float SnowDepth;

	// Grid-local sampling offsets, precomputed on CPU: absolute world XY at
	// ~1e5 magnitude destroys float32 finite differences (shimmer).
	float2 GridToTerrainOffset;
	float TerrainTexelSize;
	float WarpedHalfSpan;

	uint GridDim;
	uint TerrainDim;
	uint ShellDebugData;
	float DeformInvWorldSize;

	float2 GridToDeformOffset;
	float2 SnowUVOffset;

	uint HasSnowTexture;
	float RoadSnowDepth;
	float SnowTextureIsLinear;
	float ObjectLiftCap;

	float2 ObjectHeightCenter;
	float ObjectHeightHalfExtent;
	float EnableGlints;

	float CrispShadows;
	float HasSnowNormal;
	float HasSnowRmaos;
	float SnowRoughnessScale;

	float4 SnowGlintParams;  // x logDensity, y microfacetRoughness, z densityRandomization, w screenSpaceScale

	float SnowSpecularLevel;
	float BorderNoise;   // world-unit domain-warp jitter of class-depth borders
	float BorderSmooth;  // world-unit ramp-widening radius between classes
	float BorderTrampledFade;  // depth window: trench-floor taper toward borders

	float BorderUntrampledFade;  // depth band: untrampled edge dissolve
	float SnowSnowFade;          // object-skin <-> landscape-shell cross-fade band
	float SkinFadeStart;         // statics-skin distance dissolve band (units)
	float SkinFadeEnd;

	column_major float4x4 LodShadowProj;  // slice-2 LOD shadow cascade (see SnowShadow.hlsli)
	float LodShadowEnd;
	float LodShadowActive;
	float ScreenSpaceShadowsActive;  // t45 bound: long-range depth-marched shadows
	float padLod;
}

Texture2D<float4> TerrainWindow : register(t0);
Texture2D<float> DeformationMap : register(t1);
Texture2D<float4> SnowDiffuse : register(t2);
// Full-scene depth copy (Terrain Blending's blended depth when available) —
// never the bound DSV, so sampling during the shell draw is legal.
Texture2D<float> SceneDepth : register(t3);
// TruePBR snow companion maps (auto-resolved from the Textures\PBR\ variant
// of the snow path): tangent-space normals (_n) and roughness/metal/AO/spec
// (_rmaos). Gated by HasSnowNormal / HasSnowRmaos.
Texture2D<float4> SnowNormalMap : register(t6);
Texture2D<float4> SnowRmaosMap : register(t7);
// S7 unified blanket: filtered top-down object maps. Tops (world Z, empty
// -100000) and bottoms (world Z, empty +100000) — together they say whether
// an object is grounded (lift the blanket over it) or floating (no lift,
// shelter the ground beneath it).
Texture2D<float> ObjectHeights : register(t4);
Texture2D<float> ObjectBottoms : register(t5);
SamplerState SnowSampler : register(s0);

static const float kSnowUVTile = 256.0;

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
// smoothing terraces the gradient the normals consume (the M3 lesson).
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
// S7 blanket: bilinear samples of the object top/bottom maps at world XY.
// The raster pass maps +worldY to +ndcY = texture v0 (top), so v mirrors.
float2 ObjectMapTexel(float2 worldXY, out bool valid)
{
	float2 local = (worldXY - ObjectHeightCenter) / ObjectHeightHalfExtent;
	valid = all(abs(local) < 0.98);
	float2 dims;
	ObjectHeights.GetDimensions(dims.x, dims.y);
	float2 uv = float2(local.x * 0.5 + 0.5, 0.5 - local.y * 0.5);
	return clamp(uv * dims - 0.5, 0.0, dims.x - 1.001);
}

float SampleObjectHeight(float2 worldXY)
{
	bool valid;
	float2 t = ObjectMapTexel(worldXY, valid);
	if (!valid)
		return -100000.0;
	int2 t0 = (int2)t;
	float2 f = t - t0;
	int2 t1 = min(t0 + 1, int2(512 - 1, 512 - 1));

	float s00 = ObjectHeights.Load(int3(t0.x, t0.y, 0));
	float s10 = ObjectHeights.Load(int3(t1.x, t0.y, 0));
	float s01 = ObjectHeights.Load(int3(t0.x, t1.y, 0));
	float s11 = ObjectHeights.Load(int3(t1.x, t1.y, 0));

	return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

float SampleObjectBottom(float2 worldXY)
{
	bool valid;
	float2 t = ObjectMapTexel(worldXY, valid);
	if (!valid)
		return 100000.0;
	int2 t0 = (int2)t;
	float2 f = t - t0;
	int2 t1 = min(t0 + 1, int2(512 - 1, 512 - 1));

	float s00 = ObjectBottoms.Load(int3(t0.x, t0.y, 0));
	float s10 = ObjectBottoms.Load(int3(t1.x, t0.y, 0));
	float s01 = ObjectBottoms.Load(int3(t0.x, t1.y, 0));
	float s11 = ObjectBottoms.Load(int3(t1.x, t1.y, 0));

	return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

// ---- Surface undulation: wind-settled dunes ----
// Deep snow is never a mathematically smooth sheet: wind works it into broad,
// gentle waves. Two octaves of world-anchored value noise, added as REAL
// geometry (via ShellSurfaceZ, so the VS displaces and the self-shadow march
// sees it) and shaded per-pixel through its analytic-ish gradient. Amplitude
// scales with local depth so thin snow, class boundaries and carved floors
// stay flat.
float UndulationHash(float2 cell)
{
	float3 p3 = frac(float3(cell.x, cell.y, cell.x) * float3(0.1031, 0.1030, 0.0973));
	p3 += dot(p3, p3.yzx + 33.33);
	return frac((p3.x + p3.y) * p3.z);
}

float UndulationNoise(float2 p)
{
	float2 i = floor(p);
	float2 f = frac(p);
	f = f * f * (3.0 - 2.0 * f);
	return lerp(lerp(UndulationHash(i), UndulationHash(i + float2(1, 0)), f.x),
		lerp(UndulationHash(i + float2(0, 1)), UndulationHash(i + float2(1, 1)), f.x), f.y);
}

static const float kUndulationAmp = 3.5;
// Minimum snow cover on carved trench floors (world units). Covers the
// terrain window's bilinear approximation error so the real landscape mesh
// never pokes through a floor.
static const float kTrenchFloor = 5.0;

float Undulation(float2 worldXY)
{
	float n = UndulationNoise(worldXY / 340.0) * 0.72 + UndulationNoise(worldXY / 110.0) * 0.28;
	return (n - 0.5) * 2.0 * kUndulationAmp;
}

// Class-border shaping. Landscape texture borders are hard edges in the
// baked depth/coverage data, so a +30 snow class meeting a -5 mud class
// produced a ravine wall exactly along the texture seam. Two live controls:
// BorderNoise domain-warps WHERE the border falls (depth/coverage sampled at
// a noise-jittered position — real snow edges never trace a seam), and
// BorderSmooth widens the ramp with a tap cross so the two depths meet in a
// slope. Terrain HEIGHT is always sampled at the true position — the shell
// keeps conforming exactly.
float3 SampleTerrainShaped(float2 gridLocal)
{
	float3 result = SampleTerrain(gridLocal);
	[branch] if (BorderNoise >= 0.01 || BorderSmooth >= 0.01)
	{
		float2 worldXY = GridOrigin + gridLocal;
		float2 jitter = float2(
			UndulationNoise(worldXY / 37.0) - 0.5,
			UndulationNoise(worldXY / 37.0 + 111.7) - 0.5) * (2.0 * BorderNoise);
		float2 shapedLocal = gridLocal + jitter;

		float2 depthCoverage = SampleTerrain(shapedLocal).yz;
		[branch] if (BorderSmooth >= 0.01)
		{
			float r = BorderSmooth;
			depthCoverage += SampleTerrain(shapedLocal + float2(r, 0.0)).yz;
			depthCoverage += SampleTerrain(shapedLocal - float2(r, 0.0)).yz;
			depthCoverage += SampleTerrain(shapedLocal + float2(0.0, r)).yz;
			depthCoverage += SampleTerrain(shapedLocal - float2(0.0, r)).yz;
			depthCoverage *= 0.2;
		}
		result.yz = depthCoverage;
	}
	return result;
}

float ShellSurfaceZ(float2 gridLocal, out float coverage, out float terrainHeight)
{
	float3 terrain = SampleTerrainShaped(gridLocal);
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
		// heights: ONE such neighbor tap exploded the ridge pad into a
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
		// which is exactly the crawling holes that survived the plain max.
		// Pad by that bound so the shell always clears the mesh.
		float ridgePad = min(0.25 * max(abs(h0 - h1), abs(h2 - h3)), 150.0);
		terrainHeight = lerp(terrainHeight, max(terrainHeight, maxHeight) + ridgePad, farBlend);
		coverage = lerp(coverage, max(coverage, maxCoverage), farBlend);
		// Decisive separation: wherever the pad still lands the far shell
		// within a few units of the landscape mesh, the two interleave at
		// depth precision and the winner flips with the view angle — the
		// holes that appear and vanish as the camera turns. A fixed
		// covered-only margin makes the shell win at every angle.
		terrainHeight += farBlend * 8.0 * saturate(coverage);
	}

	// S7 height field: t4 holds the SLOPE-LIMITED field — terrain lifted
	// only by corpse burial mounds (the object-top blanket lift was removed),
	// run through the angle-of-repose cone transform; t5 holds the shelter
	// mask (floating structures 60+ units up ⇒ no snow beneath) plus the
	// door/campfire clearings.
	[branch] if (ObjectLiftCap > 0.0)
	{
		float2 worldXY = GridOrigin + gridLocal;
		float field = SampleObjectHeight(worldXY);
		[flatten] if (field > -50000.0)
			terrainHeight = max(terrainHeight, field);
		// Suppression mask: 1 under floating structures (shelter) and near
		// doors — smooth, so clearings fade instead of cutting.
		coverage *= saturate(1.0 - SampleObjectBottom(worldXY));
	}

	// Fade toward the grid boundary so the shell melts into the terrain.
	float2 delta = abs(gridLocal - WarpedHalfSpan);
	float edgeFade = saturate((WarpedHalfSpan - max(delta.x, delta.y)) / 2048.0);

	// Bare ground contributes negative depth so the shell submerges toward
	// uncovered terrain as well.
	float bare = saturate(1.0 - coverage);
	float depth = rampDepth + (-8.0) * bare;
	depth = lerp(-8.0, depth, edgeFade);

	// Deformation carves only where the layer is actually raised. The carved
	// floor never drops below kTrenchFloor units (or the un-carved depth when
	// thinner) so trench bottoms are always shell snow and the landscape
	// texture underneath never shows through — 5 units, not 1: the terrain
	// window is bilinear-approximate and can undershoot the real mesh by a
	// couple of units, which poked bare landscape through wide trench
	// floors. The negative-depth submerge at category edges is untouched.
	[flatten] if (depth > 0.0)
	{
		float deformation = saturate(SampleDeformation(gridLocal));
		// The floor scales away near class borders: the UNCARVED depth ramp
		// is the "between two textures" signal — deep interior snow keeps
		// the full anti-peek floor, but a thin ramp edge stays strippable,
		// or trampled borders hold a hard-edged slab over bare ground
		// instead of blending out.
		// Fixed taper: full anti-peek floor by 8 units of natural depth —
		// slider-driven widening thinned floors in DEEP snow too (exposed
		// landscape); border fade is an ALPHA matter, handled in the PS.
		float floorDepth = min(depth, kTrenchFloor * smoothstep(0.5, 8.0, depth));
		depth = max(depth * (1.0 - deformation), floorDepth);
		depth += Undulation(GridOrigin + gridLocal) * saturate(depth / 8.0);
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
	// index parity re-phases every camera step and made walls visibly shift
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
	// relative, so distant geometry MORPHED as the camera moved — terrain
	// resampled at ever-shifting positions read as breathing shapes,
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

#ifdef SNOW_SHADOW_CAST
	// Shadow-caster variant: only the EXCESS height above the ambient snow
	// depth casts. Casting the full shell shadowed every receiver inside or
	// beneath the layer — the terrain it visually replaces, actor legs
	// wading in it, grass — which read as the whole landscape darkening.
	//
	// The base is not merely flattened but SUNK far below the terrain: our
	// terrain window is bilinear-approximate, and writing it at ground level
	// out-depthed the game's true terrain mesh wherever the approximation
	// overshot by a few units — false shadow blotches on open ground. The
	// caster also requires solid snow coverage, so field lifts whose visual
	// blanket is dithered away never cast from invisible snow. Only clearly
	// raised, clearly covered mounds/drifts/lifts emerge above ground as
	// casters.
	float3 rawTerrainCast = SampleTerrain(gridLocal);
	float castBase = rawTerrainCast.x + max(rawTerrainCast.y, 0.0);
	float castExcess = max(0.0, z - castBase);
	float castGate = smoothstep(3.0, 8.0, castExcess) * smoothstep(0.2, 0.5, coverage);
	z = rawTerrainCast.x + lerp(-64.0, castExcess, castGate);
#endif

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
// Cheap 2D cell hash for stochastic tiling offsets.
float2 StochasticHash(float2 cell)
{
	float3 p3 = frac(float3(cell.x, cell.y, cell.x) * float3(0.1031, 0.1030, 0.0973));
	p3 += dot(p3, p3.yzx + 33.33);
	return frac(float2((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

// Anti-tiling snow fetch (Terrain Variation-style): blend 3 taps of the
// texture at random per-cell UV offsets over a triangular lattice, so the
// 256-unit repeat never lines up. Weight sharpening keeps the cross-fade
// zones from reading as ghosted double-images.
// Shared anti-tiling taps: computed once, applied to every snow map (albedo,
// normal, RMAOS) so all channels agree on the same stochastic offsets.
struct SnowTaps
{
	float2 uv0, uv1, uv2;
	float3 weights;
	float2 duvdx, duvdy;
};

SnowTaps ComputeSnowTaps(float2 uv, float2 worldXY)
{
	// World-anchored lattice (~427 units per cell): the snow uv rebases by
	// tile multiples as the camera-following grid moves, and a tile is not a
	// whole number of lattice cells — a uv-derived lattice made the pattern
	// jump with the camera. Absolute-coordinate precision is fine here: hash
	// cell selection tolerates far more error than float32 carries at world
	// magnitudes (unlike height-field finite differences).
	float2 lattice = mul(float2x2(1.0, -0.57735027, 0.0, 1.15470054), worldXY * (0.6 / 256.0));
	float2 cellBase = floor(lattice);
	float2 f = frac(lattice);

	float2 v0, v1, v2;
	float3 bary;
	if (f.x + f.y < 1.0) {
		v0 = cellBase;
		v1 = cellBase + float2(1, 0);
		v2 = cellBase + float2(0, 1);
		bary = float3(1.0 - f.x - f.y, f.x, f.y);
	} else {
		v0 = cellBase + float2(1, 1);
		v1 = cellBase + float2(0, 1);
		v2 = cellBase + float2(1, 0);
		bary = float3(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y);
	}

	bary = pow(bary, 4.0);
	bary /= dot(bary, 1.0);

	SnowTaps taps;
	taps.uv0 = uv + StochasticHash(v0);
	taps.uv1 = uv + StochasticHash(v1);
	taps.uv2 = uv + StochasticHash(v2);
	taps.weights = bary;
	// All taps share the CONTINUOUS base uv's derivatives: the per-cell
	// offsets are constant within a triangle but jump at lattice seams, and
	// letting the sampler derive gradients there makes anisotropic filtering
	// fetch the deepest mips — visible as discolored (beige) streaks.
	taps.duvdx = ddx(uv);
	taps.duvdy = ddy(uv);
	return taps;
}

float4 SampleSnowMap(Texture2D<float4> tex, SnowTaps taps)
{
	return taps.weights.x * tex.SampleGrad(SnowSampler, taps.uv0, taps.duvdx, taps.duvdy) +
	       taps.weights.y * tex.SampleGrad(SnowSampler, taps.uv1, taps.duvdx, taps.duvdy) +
	       taps.weights.z * tex.SampleGrad(SnowSampler, taps.uv2, taps.duvdx, taps.duvdy);
}

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
	float2 gridLocalPS = input.GridLocal;
	// Shaped (border-noised/smoothed) so the per-pixel coverage and ramp
	// dither agree with the shaped geometry.
	float3 pixelTerrain = SampleTerrainShaped(gridLocalPS);
	float pixelCoverage = saturate(pixelTerrain.z);
	float2 psEdgeDelta = abs(gridLocalPS - WarpedHalfSpan);
	float psEdgeFade = saturate((WarpedHalfSpan - max(psEdgeDelta.x, psEdgeDelta.y)) / 2048.0);

	// Un-carved class depth ramp at this pixel. Where it goes negative the
	// shell is submerged (depth-rejected anyway). The dither rides the ramp:
	// alpha fades over the last ~40% of positive depth, so a boundary toward
	// shallower/negative classes (where coverage stays ~1 and the coverage
	// term can't blend) dissolves stochastically as the shell thins, instead
	// of presenting a bare geometric plunge with a thin z-fight strip.
	float pixelBare = saturate(1.0 - pixelCoverage);
	float pixelRampDepth = pixelTerrain.y + (-8.0) * pixelBare;
	pixelRampDepth = lerp(-8.0, pixelRampDepth, psEdgeFade);

	// Depth reads shared by the edge dissolve and the proximity fade below.
	float sceneZ = SharedData::GetScreenDepth(SceneDepth.Load(int3(input.Position.xy, 0)));
	float shellZ = input.CurrentClip.w;

	// User-tunable dissolve band (depth units): how much of the ramp's tail
	// the untrampled edge dithers across before committing. (A TB-style
	// view-ray variant was A/B tested and scrapped — visually near-identical
	// here, because the proximity fade below already supplies the view-ray
	// component of the edge blend.)
	float rampFadeBand = max(2.0, BorderUntrampledFade);
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
	// Mild distance scaling only: a steep view-Z-proportional band made the
	// blend wobble with camera tilt even up close.
	float objectFadeBand = 10.0 + shellZ * 0.004;
	float proximityFade = saturate((sceneZ - shellZ) / objectFadeBand);
	// Two situations DELIBERATELY hug geometry and must override the fade:
	// carved trench floors (1-unit floor over terrain/actor feet), and the
	// blanket riding a lifted object (mound a few units above the rock or
	// drift it swallowed). Without the override, both get view-dependently
	// dithered away — which read as "snow moving with the camera" and
	// partially-covered objects.
	float pixelCarve = saturate(SampleDeformation(gridLocalPS));
	float pixelLift = 0.0;
	[branch] if (ObjectLiftCap > 0.0)
	{
		float fieldHeight = SampleObjectHeight(GridOrigin + gridLocalPS);
		[flatten] if (fieldHeight > -50000.0)
			pixelLift = fieldHeight - pixelTerrain.x;
	}
	// The carve override near class borders is what made trenches END HARD
	// while untrampled snow dissolved softly: the proximity dissolve is a
	// large part of the border blend, and carved pixels were exempt from it.
	// Trampled Border Fade scales the override away as the uncarved ramp
	// thins, so walked snow rejoins the same soft dissolve at borders while
	// deep-field trench floors keep their guaranteed visibility.
	float carveOverride = smoothstep(0.1, 0.5, pixelCarve) * smoothstep(0.5, max(BorderTrampledFade, 1.0), pixelTerrain.y);
	coverageAlpha *= max(proximityFade, saturate(carveOverride + smoothstep(2.0, 10.0, pixelLift)));

	// Stochastic discard dither: proven to blend in this pipeline (the wide
	// distance fade). Writing alpha without discarding was tried and blends
	// nothing in our pass — TB's alpha path runs through depth-prepass
	// machinery we do not replicate.
	float screenNoise = Random::InterleavedGradientNoise(input.Position.xy, SharedData::FrameCount);
	[branch] if (ShellDebugData == 0)
	{
		if (screenNoise * screenNoise >= coverageAlpha)
			discard;
	}

	// Normal = smooth interpolated terrain normal + per-pixel deformation
	// gradient (central differences at the deformation map's resolution).
	float2 gridLocal = input.GridLocal;
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

	// Undulation gradient (same field the VS displaced by) shades the dunes.
	float2 worldXYPS = GridOrigin + gridLocal;
	float undScale = saturate(pixelDepth / 8.0);
	[branch] if (undScale > 0.001)
	{
		const float uStep = 12.0;
		float uXP = Undulation(worldXYPS + float2(uStep, 0.0));
		float uXN = Undulation(worldXYPS - float2(uStep, 0.0));
		float uYP = Undulation(worldXYPS + float2(0.0, uStep));
		float uYN = Undulation(worldXYPS - float2(0.0, uStep));
		gradZ += float2(uXP - uXN, uYP - uYN) / (2.0 * uStep) * undScale;
	}

	float3 normalWS = normalize(float3(gradZ * -1.0, 1.0));

	// Snow texture taps — shared by albedo, normal and RMAOS so every map
	// agrees on the same anti-tiling offsets.
	float2 snowUV = (SnowUVOffset + gridLocal) / kSnowUVTile;
	SnowTaps snowTaps = ComputeSnowTaps(snowUV, worldXYPS);

	// Micro-relief. PBR sets carry a REAL tangent-space normal map — the
	// same dimples, clumps and crust the landscape shows; legacy sets fall
	// back to treating the diffuse luminance as a height proxy. Both fade
	// with distance, where the grain frequency aliases instead of detailing.
	float bumpFade = 1.0 - smoothstep(600.0, 2200.0, shellZ);
	[branch] if (HasSnowNormal > 0.5 && bumpFade > 0.001)
	{
		float3 texN = SampleSnowMap(SnowNormalMap, snowTaps).xyz * 2.0 - 1.0;
		texN.z = sqrt(saturate(1.0 - dot(texN.xy, texN.xy)));
		texN.y = -texN.y;  // DDS v grows down; our uv v grows with world +Y
		float3 bumpT = normalize(cross(float3(0.0, 1.0, 0.0), normalWS) + float3(1e-5, 0.0, 0.0));
		float3 bumpB = cross(normalWS, bumpT);
		normalWS = normalize(normalWS + (bumpT * texN.x + bumpB * texN.y) * bumpFade);
	}
	else if (HasSnowTexture != 0 && bumpFade > 0.001)
	{
		const float kBumpTile = 64.0;
		const float kBumpHeight = 0.55;
		float2 texDims;
		SnowDiffuse.GetDimensions(texDims.x, texDims.y);
		float e = 1.5 / texDims.x;
		float2 detailUV = worldXYPS / kBumpTile;
		const float3 kLum = float3(0.30, 0.45, 0.25);
		float h0 = dot(SnowDiffuse.Sample(SnowSampler, detailUV).rgb, kLum);
		float hx = dot(SnowDiffuse.Sample(SnowSampler, detailUV + float2(e, 0.0)).rgb, kLum);
		float hy = dot(SnowDiffuse.Sample(SnowSampler, detailUV + float2(0.0, e)).rgb, kLum);
		float2 bumpGrad = float2(hx - h0, hy - h0) * (kBumpHeight / (e * kBumpTile));
		normalWS = normalize(normalWS + float3(-bumpGrad * bumpFade, 0.0));
	}

	float3 viewNormal = normalize(mul((float3x3)CameraView, normalWS));

	// Snow material: the modlist's projected snow diffuse when available,
	// otherwise a bright, slightly blue constant.
	float3 kSnowAlbedo = float3(0.82, 0.84, 0.88);
	[branch] if (HasSnowTexture != 0)
	{
		kSnowAlbedo = SampleSnowMap(SnowDiffuse, snowTaps).rgb;
		// PBR-authored textures store linear color; the rest of this path
		// works in the pipeline's gamma space. Auto-enabled when the PBR set
		// was resolved — no manual checkbox needed.
		[flatten] if (SnowTextureIsLinear != 0.0)
			kSnowAlbedo = Color::LinearToSrgb(kSnowAlbedo);
	}
	// PBR snow material: GGX microfacet specular with Fresnel and
	// energy-conserving lobes. Light and ambient stay in the frame's units
	// (raw DirLightColor / Color::Ambient — DirLightColor is already
	// π-scaled by pipeline convention, so no Lambert 1/π on diffuse); the
	// material RESPONSE is physically based, and the indirect specular lobe
	// goes to the Reflectance RT where the composite applies cubemap and
	// ambient specular like any TruePBR surface.
	static const float kSnowRoughness = 0.6;
	static const float3 kSnowF0 = float3(0.028, 0.028, 0.028);

	// Per-pixel PBR response from the RMAOS map (TruePBR channel layout:
	// roughness / metallic / AO / specular level), with the landscape
	// config's authored scales (roughness ×0.7, specularLevel 0.02) — the
	// darker-and-brighter patchiness real PBR snow shows.
	float snowRoughness = kSnowRoughness;
	float3 snowF0 = kSnowF0;
	float snowAO = 1.0;
	[branch] if (HasSnowRmaos > 0.5)
	{
		float4 rmaos = SampleSnowMap(SnowRmaosMap, snowTaps);
		snowRoughness = clamp(rmaos.x * SnowRoughnessScale, 0.05, 1.0);
		snowAO = rmaos.z;
		snowF0 = rmaos.w * SnowSpecularLevel;
	}

	float3 V = -normalize(input.WorldPos);
	float3 L = SharedData::DirLightDirection.xyz;
	float3 H = normalize(V + L);
	float satNdotL = saturate(dot(normalWS, L));
	float satNdotV = saturate(abs(dot(normalWS, V)) + 1e-5);
	float satNdotH = saturate(dot(normalWS, H));
	float satVdotH = saturate(dot(V, H));

	float worldShadow = ShadowSampling::GetWorldShadow(input.WorldPos, ShellCameraPosAdjust.xyz);
	// Distant shadow softening: the far cascade's texels and the self-shadow
	// march's terrain-window texels (100+ units out there) both quantize
	// into hard blocky patches on distant snow. The cascades must NOT be
	// faded out — LOD trees cast into them and bare ground keeps their
	// shadows at range — so the crisp path instead WIDENS its PCF ring
	// with distance: same shadows, soft penumbra blobs instead of blocks.
	float farShadowT = smoothstep(6000.0, 15000.0, length(input.WorldPos));
	float sunShadow;
	[branch] if (CrispShadows > 0.5)
	{
		// Full-resolution comparison PCF against the game's raw cascade
		// atlas: the same crisp tree/actor shadows bare ground receives.
		sunShadow = worldShadow * SnowShadow::GetCascadeShadow(input.WorldPos, normalWS, lerp(1.0, 6.0, farShadowT), LodShadowProj, LodShadowEnd, LodShadowActive);
	}
	else
	{
		// Fallback: the Volumetric Shadows 512px VSM moments copy (blurry).
		float detailedShadow;
		float dynamicShadow = ShadowSampling::GetLightingShadow(input.WorldPos, detailedShadow);
		sunShadow = worldShadow * min(dynamicShadow, detailedShadow);
	}

	// Heightfield self-shadowing: the shell IS a heightfield, so march it
	// toward the sun and find the horizon this pixel must clear. Hills,
	// mounds, drift lifts and the dune undulation all cast soft shadows onto
	// the snow behind them — contact detail the game's cascades cannot hold.
	// Geometric growth in the tap distances gives sharp close shadows and
	// long soft ones at low sun angles.
	[branch] if (sunShadow > 0.01 && satNdotL > 0.001 && L.z > 0.01)
	{
		static const float kMarchDist[5] = { 28.0, 70.0, 170.0, 420.0, 1000.0 };
		float sunLen2D = max(length(L.xy), 1e-4);
		float sunTan = L.z / sunLen2D;
		float2 stepDir = L.xy / sunLen2D;
		float surfZ = input.WorldPos.z + ShellCameraPosAdjust.z;
		float horizonTan = -10.0;
		[unroll] for (uint marchI = 0; marchI < 5; marchI++)
		{
			float d = kMarchDist[marchI];
			float2 sampleLocal = gridLocal + stepDir * d;
			float3 st = SampleTerrain(sampleLocal);
			float sampleDepth = max(st.y, 0.0);
			// The march must see the CARVED surface: without the carve, a
			// wide trench reads as ringed by full-height snow and sits in
			// permanent shadow even facing the sun.
			float sampleDeform = saturate(SampleDeformation(sampleLocal));
			sampleDepth = max(sampleDepth * (1.0 - sampleDeform), min(sampleDepth, kTrenchFloor * smoothstep(0.5, 8.0, sampleDepth)));
			float sh = st.x + sampleDepth + Undulation(GridOrigin + sampleLocal) * saturate(sampleDepth / 8.0);
			[branch] if (ObjectLiftCap > 0.0)
			{
				float sf = SampleObjectHeight(GridOrigin + sampleLocal);
				[flatten] if (sf > -50000.0)
					sh = max(sh, sf + sampleDepth);
			}
			horizonTan = max(horizonTan, (sh - surfZ) / d);
		}
		// Near: the original crisp band. Far: a much wider penumbra plus
		// attenuated strength — the march's per-texel horizon steps stop
		// reading as hard-edged blocks.
		float soft = lerp(0.06, 0.35, farShadowT);
		sunShadow *= lerp(smoothstep(-0.12 - (soft - 0.06) * 2.0, soft, sunTan - horizonTan), 1.0, 0.7 * farShadowT);
	}

	// Screen-Space Shadows (the integrated long-range depth march): these
	// carry the distant LOD tree shadows far beyond the two cascades — bare
	// ground multiplies them into its lighting (Lighting.hlsl), so the
	// shell does the same or distant snow reads shadowless.
	[branch] if (ScreenSpaceShadowsActive > 0.5)
		sunShadow *= ScreenSpaceShadows::GetScreenSpaceShadow(input.Position.xyz, float2(0.0, 0.0), 0.0);

	float3 sunLight = SharedData::DirLightColor.xyz * sunShadow;

	float3 F = BRDF::F_Schlick(snowF0, satVdotH);
	float specD = BRDF::D_GGX(snowRoughness, satNdotH);
	// Sparkle: replace the smooth GGX NDF with TruePBR's discrete glint NDF —
	// individual microfacets flash in and out as view/sun angles change, which
	// is the single strongest "real snow in sunlight" cue. Parameters match
	// the landscape's authored PBR config (LandscapeSnow01.json: logDensity 6,
	// densityRandomization 5) so shell sparkle equals ground sparkle, and the
	// uv is the albedo uv so the sparkle field rides the same tiling.
	[branch] if (EnableGlints > 0.5 && SnowGlintParams.x > 1.1)
	{
		float3 glintT = normalize(cross(float3(0.0, 1.0, 0.0), normalWS) + float3(1e-5, 0.0, 0.0));
		float3 glintB = cross(normalWS, glintT);
		float3 glintH = float3(dot(H, glintT), dot(H, glintB), saturate(dot(H, normalWS)));
		float glintNoise = Random::R1Modified(float(SharedData::FrameCount), (Random::pcg2d(uint2(input.Position.xy)) / 4294967296.0).x);
		Glints::GlintCachedVars glintCache;
		Glints::PrecomputeGlints(glintNoise, snowUV, snowTaps.duvdx, snowTaps.duvdy, SnowGlintParams.w, glintCache);
		float dMax = BRDF::D_GGX(snowRoughness, 1.0);
		specD = Glints::SampleGlints2023NDF(glintNoise, SnowGlintParams.x, SnowGlintParams.y, SnowGlintParams.z, glintCache, glintH, specD, dMax).x;
	}
	float specV = BRDF::Vis_SmithJointApprox(snowRoughness, satNdotV, satNdotL);

	// Indirect lobes: the specular weight is what the environment reflects,
	// diffuse receives only what specular does not (energy conservation).
	float2 envBRDF = BRDF::EnvBRDF(snowRoughness, satNdotV);
	float3 specularLobe = snowF0 * envBRDF.x + envBRDF.y;
	float3 diffuseLobe = kSnowAlbedo * (1.0 - specularLobe);

	float3 directDiffuse = sunLight * satNdotL * (1.0 - F) * kSnowAlbedo;
	float3 directSpecular = specD * specV * F * sunLight * satNdotL;

	float3 ambientColor = Color::Ambient(max(0, SharedData::GetAmbient(normalWS))) * snowAO;
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
	psout.NormalGlossiness = float4(GBuffer::EncodeNormal(viewNormal), 1.0 - snowRoughness, stochasticBlend);
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
