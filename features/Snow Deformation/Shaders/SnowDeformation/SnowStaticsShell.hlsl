// S6 statics shell: re-draws captured projected-snow statics (cliffs, rocks,
// drifts, roofs, logs) inflated along their vertex normals, with the same
// snow material as the terrain shell so the two read as one blanket.
//
// Drawn inside SnowDeformation::DrawShell right after the terrain shell, so
// it inherits that pass's bindings: ShellCB (b0), deformation map (t1), snow
// diffuse (t2), scene depth (t3), samplers s0/s1 and the b4-b6 shared data.
// Only the input layout, vertex/index buffers, shaders and StaticCB (b1)
// change per object.
//
// The VS consumes only POSITION and NORMAL — D3D11 accepts input layouts
// carrying more elements than the shader reads, so one layout per vertex
// descriptor covers every static mesh format.
//
// cbuffer ShellCB MUST stay layout-identical to SnowShell.hlsl (and ShellCB
// in SnowDeformation.h). Includes resolve from Data\Shaders root, matching
// the CompileShader include convention.

#include "Common/BRDF.hlsli"
#include "Common/Color.hlsli"
#include "Common/GBuffer.hlsli"
#include "Common/Random.hlsli"
#include "Common/SharedData.hlsli"

#ifdef PSHADER
// TruePBR's procedural glint NDF for snow sparkle (noise texture at t20,
// bound by the CPU side for the whole shell pass; EnableGlints gates it).
#	include "Common/Glints/Glints2023.hlsli"
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
	row_major float4x4 CameraViewProj;
	row_major float4x4 CameraViewProjUnjittered;
	row_major float4x4 CameraPreviousViewProjUnjittered;
	row_major float4x4 CameraView;

	float4 ShellCameraPosAdjust;
	float4 ShellCameraPreviousPosAdjust;

	float2 GridOrigin;
	float GridSpacing;
	float SnowDepth;

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
	float BorderNoise;   // shared dials: also shape the ground-shell hand-off below
	float BorderSmooth;
	float BorderTrampledFade;

	float BorderUntrampledFade;
	float SnowSnowFade;  // object-skin <-> landscape-shell cross-fade band
	float SkinFadeStart;  // statics-skin distance dissolve band (units)
	float SkinFadeEnd;

	column_major float4x4 LodShadowProj;  // slice-2 LOD shadow cascade (see SnowShadow.hlsli)
	float LodShadowEnd;
	float LodShadowActive;
	float ScreenSpaceShadowsActive;  // t45 bound: long-range depth-marched shadows
	float padLod;
}

cbuffer StaticCB : register(b1)
{
	// Object world transform rows (rotation*scale in xyz, translation in w,
	// absolute world coordinates).
	float4 WorldRow0;
	float4 WorldRow1;
	float4 WorldRow2;

	float ObjectsDepth;  // FLAT-class depth (walkways, roofs, planks)
	float2 HeightWindowCenter;
	float HeightHalfExtent;

	// >0.5: SmoothedNormals (VS t10) holds position-averaged normals for
	// this object — pillow inflation for flat split-normal meshes.
	float HasSmoothedNormals;
	float RoundedDepth;   // ROUNDED-class depth (rocks, drifts, logs)
	float VertexCountF;   // index of the flatness-stats element in SmoothedNormals
	float padStat;
}

#ifdef PSHADER
Texture2D<float4> TerrainWindow : register(t0);
#endif
Texture2D<float> DeformationMap : register(t1);
Texture2D<float4> SnowDiffuse : register(t2);
#ifdef PSHADER
Texture2D<float> SceneDepth : register(t3);
// TruePBR snow companion maps (see SnowShell.hlsl) — inherited bindings.
Texture2D<float4> SnowNormalMap : register(t6);
Texture2D<float4> SnowRmaosMap : register(t7);
// Depth AFTER the terrain shell drew (its surface included) — the skin's
// view-ray reference for cross-fading into the landscape shell.
Texture2D<float> ShellDepthCopy : register(t9);
#endif
SamplerState SnowSampler : register(s0);

static const float kSnowUVTile = 256.0;

// Bilinear deformation sample from grid-local XY (world - GridOrigin);
// matches the terrain shell's window math. Returns 0 outside the window.
float SampleDeformation(float2 gridLocal)
{
	float2 uv = (GridToDeformOffset + gridLocal) * DeformInvWorldSize;
	if (any(uv < 0.0) || any(uv > 1.0))
		return 0.0;

	float2 dims;
	DeformationMap.GetDimensions(dims.x, dims.y);
	float2 t = clamp(uv * dims - 0.5, 0.0, dims.x - 1.001);
	int2 t0 = (int2)t;
	float2 f = t - t0;
	int2 t1 = min(t0 + 1, int2(dims) - 1);

	float s00 = DeformationMap.Load(int3(t0.x, t0.y, 0));
	float s10 = DeformationMap.Load(int3(t1.x, t0.y, 0));
	float s01 = DeformationMap.Load(int3(t0.x, t1.y, 0));
	float s11 = DeformationMap.Load(int3(t1.x, t1.y, 0));

	return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

struct VS_INPUT
{
	float4 Position : POSITION0;
	float4 Normal : NORMAL0;
	uint VertexID : SV_VertexID;
};

#ifdef VSHADER
// Position-averaged normals (model space) built by SmoothNormalsCS, indexed
// by vertex id. w=0 entries are unresolved — fall back to the raw normal.
StructuredBuffer<float4> SmoothedNormals : register(t10);
#endif

struct VS_OUTPUT
{
	float4 Position : SV_POSITION;
	float4 CurrentClip : TEXCOORD0;
	float4 PreviousClip : TEXCOORD1;
	float3 WorldPos : TEXCOORD2;
	float3 NormalWS : TEXCOORD3;
	float2 GridLocal : TEXCOORD4;
	float Coverage : TEXCOORD5;
	float Flat : TEXCOORD6;
};

#ifdef VSHADER
#	ifdef PATCH
// TRENCH PATCH: the landscape shell's recipe applied to objects — a dense
// 8-unit grid (256x256 quads, +-1024 units around the camera) draped over
// the top-down object height raster and carved per VERTEX by the
// deformation map. REAL geometry: real silhouettes, floors that hold at
// every camera angle, no parallax. Grid cells with no object top, no
// trampling, or a painted-flat (0-depth) class collapse to NaN and cull;
// the skin dithers itself away over trails to hand off (see the PS).
Texture2D<float> ObjectTopRaw : register(t11);
Texture2D<float> ObjectSkinDepth : register(t12);

// Max-of-4 texel sample: bilinear would poison against sentinel texels at
// object edges; MAX both ignores them and keeps the patch on the highest
// (safest) surface.
float2 PatchTexel(float2 worldXY, float2 dims)
{
	float2 local = (worldXY - HeightWindowCenter) / HeightHalfExtent;
	float2 uv = float2(local.x * 0.5 + 0.5, 0.5 - local.y * 0.5);
	return clamp(uv * dims - 0.5, 0.0, dims.x - 1.001);
}

float PatchTop(float2 worldXY)
{
	float2 dims;
	ObjectTopRaw.GetDimensions(dims.x, dims.y);
	float2 t = PatchTexel(worldXY, dims);
	int2 t0 = (int2)t;
	int2 t1 = min(t0 + 1, int2(dims) - 1);
	return max(max(ObjectTopRaw.Load(int3(t0.x, t0.y, 0)), ObjectTopRaw.Load(int3(t1.x, t0.y, 0))),
		max(ObjectTopRaw.Load(int3(t0.x, t1.y, 0)), ObjectTopRaw.Load(int3(t1.x, t1.y, 0))));
}

float PatchSkinDepth(float2 worldXY)
{
	float2 dims;
	ObjectSkinDepth.GetDimensions(dims.x, dims.y);
	float2 t = PatchTexel(worldXY, dims);
	int2 t0 = (int2)t;
	int2 t1 = min(t0 + 1, int2(dims) - 1);
	return max(max(ObjectSkinDepth.Load(int3(t0.x, t0.y, 0)), ObjectSkinDepth.Load(int3(t1.x, t0.y, 0))),
		max(ObjectSkinDepth.Load(int3(t0.x, t1.y, 0)), ObjectSkinDepth.Load(int3(t1.x, t1.y, 0))));
}

VS_OUTPUT main(uint vertexID : SV_VertexID)
{
	static const float2 kCorners[6] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
	uint quadIndex = vertexID / 6;
	float2 gridXY = float2(quadIndex % 256, quadIndex / 256) + kCorners[vertexID % 6];
	// WorldRow0.xy carries the snapped patch origin (see the CPU fill).
	float2 worldXY = WorldRow0.xy + gridXY * 8.0;

	float top = PatchTop(worldXY);
	float skinDepth = PatchSkinDepth(worldXY);
	float2 gridLocal = worldXY - GridOrigin;
	float deform = saturate(SampleDeformation(gridLocal));

	VS_OUTPUT vsout;
	vsout.CurrentClip = float4(0.0, 0.0, 0.0, 1.0);
	vsout.PreviousClip = float4(0.0, 0.0, 0.0, 1.0);
	vsout.WorldPos = float3(0.0, 0.0, 0.0);
	vsout.NormalWS = float3(0.0, 0.0, 1.0);
	vsout.GridLocal = gridLocal;
	vsout.Coverage = 1.0;
	vsout.Flat = 0.0;
	[branch] if (top < -50000.0 || skinDepth < 1.0 || deform < 0.02)
	{
		float nan = asfloat(0x7fc00000);
		vsout.Position = float4(nan, nan, nan, nan);
		return vsout;
	}

	// The landscape shell's carve, verbatim: floored so the mesh beneath
	// never shows, at any trample level.
	float floorDepth = min(skinDepth, 5.0 * smoothstep(0.5, 8.0, skinDepth));
	float depth = max(skinDepth * (1.0 - deform), floorDepth);
	float3 worldAbs = float3(worldXY, top + depth);

	float3 rel = worldAbs - ShellCameraPosAdjust.xyz;
	float3 prevRel = worldAbs - ShellCameraPreviousPosAdjust.xyz;
	vsout.Position = mul(CameraViewProj, float4(rel, 1.0));
	vsout.CurrentClip = mul(CameraViewProjUnjittered, float4(rel, 1.0));
	vsout.PreviousClip = mul(CameraPreviousViewProjUnjittered, float4(prevRel, 1.0));
	vsout.WorldPos = rel;
	// Carved-surface shading normal from the deformation gradient — the
	// 8-unit geometry carries the shape, this smooths the facets.
	float2 grad = float2(
		SampleDeformation(gridLocal + float2(4.0, 0.0)) - SampleDeformation(gridLocal - float2(4.0, 0.0)),
		SampleDeformation(gridLocal + float2(0.0, 4.0)) - SampleDeformation(gridLocal - float2(0.0, 4.0))) / 8.0;
	vsout.NormalWS = normalize(float3(grad * skinDepth * 0.6, 1.0));
	return vsout;
}
#	else
VS_OUTPUT main(VS_INPUT input)
{
	float3 posMS = input.Position.xyz;
	float3 nrmMS = input.Normal.xyz * 2.0 - 1.0;

	// Pillow inflation: displace along POSITION-AVERAGED normals where
	// available. Split-normal flat meshes (planks, roofs, pole caps) get the
	// smooth normals they lack — shared-position twins displace identically
	// (rim cracks sealed by construction), plank edges mushroom outward like
	// pole caps. Already-smooth meshes are unchanged (average == raw).
	float3 inflateMS = nrmMS;
	float isFlat = 0.0;
	[branch] if (HasSmoothedNormals > 0.5)
	{
		// Mesh-level flatness stats (element appended past the last vertex):
		// split-normal plates — walkways, roofs, planks — score a high
		// divergent fraction and get COMPLETELY FLAT snow (straight-up
		// offset, raw shading normal, separate depth slider). Organically
		// smooth meshes keep the pillow.
		// FLAT = split-normal box (high divergence: walkways, planks) — the
		// ORIGINAL divergence-only rule. Two rounds of alignment-based
		// extensions (planar sheets, then sloped-only sheets) both
		// misclassified real-world meshes worse than they helped and were
		// reverted at Josef's call; roofs classifying rounded is the known,
		// accepted cost.
		float4 flatStats = SmoothedNormals[(uint)VertexCountF];
		[flatten] if (flatStats.w > 0.5 && flatStats.x > 0.5)
			isFlat = 1.0;
		float4 smoothEntry = SmoothedNormals[input.VertexID];
		[flatten] if (smoothEntry.w > 0.5)
			inflateMS = smoothEntry.xyz;
	}

	float3 worldAbs = float3(
		dot(WorldRow0.xyz, posMS) + WorldRow0.w,
		dot(WorldRow1.xyz, posMS) + WorldRow1.w,
		dot(WorldRow2.xyz, posMS) + WorldRow2.w);
	float3 nrmWS = normalize(float3(
		dot(WorldRow0.xyz, nrmMS),
		dot(WorldRow1.xyz, nrmMS),
		dot(WorldRow2.xyz, nrmMS)));
	float3 inflateWS = normalize(float3(
		dot(WorldRow0.xyz, inflateMS),
		dot(WorldRow1.xyz, inflateMS),
		dot(WorldRow2.xyz, inflateMS)));
	// FLAT class: displacement goes straight up — no pillow, no mushroom
	// rims, no per-plank shading gradient. The snow on a walkway is a
	// featureless flat sheet.
	[flatten] if (isFlat > 0.5)
		inflateWS = float3(0.0, 0.0, 1.0);
	float depthBase = lerp(RoundedDepth, ObjectsDepth, isFlat);

	float2 gridLocal = worldAbs.xy - GridOrigin;

	// Snow accumulates on up-facing surfaces. Threshold tightened toward the
	// vanilla projection's extent (steep shingles stay bare); actors trample
	// it off via the live deformation map. Geometry carve is per-vertex —
	// low-poly meshes get their trenches from the pixel-stage carve instead.
	// Gated on the INFLATION normal: plank-edge and pole-cap-rim vertices
	// (averaged ~45 degrees) keep partial coverage so the mushroomed rim is
	// snow, not a bald lip. (An earlier bound-sphere pseudo-normal guess at
	// this was reverted — real position-averaged normals replace it.)
	// Flat meshes gate accumulation on the RAW normal (their straight-up
	// inflate direction would paint the sides white).
	// ROUNDED meshes use the DRAPE ramp: with the old hard 0.4-0.7 gate a
	// deep layer became a full-height cap hovering over bare sides, joined
	// by near-vertical skirts (the culled gaps onto the log mesh). Ramping
	// depth over almost the whole up-facing range pins the shell's edges
	// to the mesh — it puffs at the top and meets the surface in a sloped
	// snow lip.
	float upFacing = isFlat > 0.5 ? smoothstep(0.4, 0.7, nrmWS.z) : smoothstep(0.05, 0.85, inflateWS.z);
	// NO vertex carve: on low-poly meshes a carved vertex dragged whole
	// 100+-unit triangles down with it — jagged sawtooth walls into the
	// surrounding untrampled snow, and knife ridges between parallel
	// trails. The layer stays geometrically UNCARVED; ALL trench relief is
	// traced per pixel in the PS (parallax + real SV_Depth), floored like
	// the landscape shell so the mesh beneath never shows.
	float depth = depthBase * upFacing;

	worldAbs += inflateWS * depth;

	float3 rel = worldAbs - ShellCameraPosAdjust.xyz;
	float3 prevRel = worldAbs - ShellCameraPreviousPosAdjust.xyz;

	VS_OUTPUT vsout;
	vsout.Position = mul(CameraViewProj, float4(rel, 1.0));
	vsout.CurrentClip = mul(CameraViewProjUnjittered, float4(rel, 1.0));
	vsout.PreviousClip = mul(CameraPreviousViewProjUnjittered, float4(prevRel, 1.0));
	vsout.WorldPos = rel;
	// Displaced snow shades by the smooth surface it forms, not the flat
	// face beneath; undisplaced vertices keep the raw normal. FLAT meshes
	// always shade by the raw normal — the smoothed-normal lerp is what
	// stamped an identical shading gradient onto every plank instance.
	vsout.NormalWS = isFlat > 0.5 ? nrmWS : normalize(lerp(nrmWS, inflateWS, saturate(depth / max(depthBase, 0.01)) * 0.85));
	// Coverage gates on the RAW normal: displacement direction may be the
	// smoothed one (twins must move together to seal rims), but ACCUMULATION
	// is physics — a vertical face does not collect snow just because its
	// top edge is welded to a horizontal one. Gating on the smoothed normal
	// painted the upper halves of box poles white.
	// RAW normal Z, interpolated — the PS runs the up-facing smoothstep
	// PER PIXEL. Thresholding in the VS made low-poly rocks flip whole
	// FACES between snowed and bare (blocky patches); thresholding the
	// interpolated normal instead varies smoothly across faces.
	vsout.Coverage = nrmWS.z;
	vsout.GridLocal = gridLocal;
	vsout.Flat = isFlat;
	return vsout;
}
#	endif
#endif

#ifdef PSHADER
// Cheap 2D cell hash for stochastic tiling offsets (matches SnowShell.hlsl).
float2 StochasticHash(float2 cell)
{
	float3 p3 = frac(float3(cell.x, cell.y, cell.x) * float3(0.1031, 0.1030, 0.0973));
	p3 += dot(p3, p3.yzx + 33.33);
	return frac(float2((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

// Anti-tiling snow taps — identical to the terrain shell's, so the texture
// pattern continues seamlessly from ground onto objects, and every snow map
// (albedo, normal, RMAOS) agrees on the same stochastic offsets.
struct SnowTaps
{
	float2 uv0, uv1, uv2;
	float3 weights;
	float2 duvdx, duvdy;
};

SnowTaps ComputeSnowTaps(float2 uv, float2 worldXY)
{
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
	// Written so the parallax trench relief is REAL to the depth buffer:
	// the carved floor's projected depth replaces the flat top's, so feet
	// and props z-test against the trench instead of vanishing under it,
	// and camera motion sees a geometrically consistent surface.
	float Depth : SV_Depth;
};

// Smooth value noise (~24-unit cells) modulating the coverage edge, standing
// in for the projection's noise texture so snow extent looks organic rather
// than a hard slope threshold.
float CoverageNoise(float2 worldXY)
{
	float2 c = worldXY / 24.0;
	float2 i = floor(c);
	float2 f = frac(c);
	f = f * f * (3.0 - 2.0 * f);
	float n00 = StochasticHash(i).x;
	float n10 = StochasticHash(i + float2(1, 0)).x;
	float n01 = StochasticHash(i + float2(0, 1)).x;
	float n11 = StochasticHash(i + float2(1, 1)).x;
	return lerp(lerp(n00, n10, f.x), lerp(n01, n11, f.x), f.y);
}

// Terrain window sample (height, rampDepth, coverage), matching the terrain
// shell's math, for blending the object shell into the ground shell.
float3 SampleTerrainStatics(float2 gridLocal)
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

PS_OUTPUT main(VS_OUTPUT input)
{
	float2 motionVector = float2(-0.5, 0.5) * (input.CurrentClip.xy / input.CurrentClip.w - input.PreviousClip.xy / input.PreviousClip.w);

	float3 normalWS = normalize(input.NormalWS);
	float2 worldXY = GridOrigin + input.GridLocal;
	// Per-pixel up-facing gate from the interpolated RAW normal (see the VS
	// note on Coverage): smooth accumulation edges on low-poly meshes.
	// Rounded shells use the wide DRAPE gate so the pinned snow lip stays
	// visible down to the mesh (a tight gate faded the skirt out and
	// re-opened the very gaps the drape closes); truly vertical faces
	// still shed. Flat paint keeps the tops-only gate.
	float pixelCoverage = input.Flat > 0.5 ? smoothstep(0.4, 0.7, input.Coverage) : smoothstep(0.15, 0.6, input.Coverage);

	// Per-pixel trench RELIEF: the vertex carve cannot dent low-poly meshes
	// (cliff edges measure 20-160 units — no vertex ever lands inside a
	// trail), so the trench is traced per PIXEL instead: parallax-march the
	// view ray down into the deformation heightfield and shade from where
	// the ray actually hits the carved surface. Real depth from any angle,
	// mesh density irrelevant, cost = a few bilinear taps in trench pixels
	// only. Gated to solidly covered pixels: on rim triangles (up-facing
	// verts stretching to side-facing ones) the shading produced flat
	// flake artifacts.
	float pixelDeform = saturate(SampleDeformation(input.GridLocal));
	float2 trenchGridLocal = input.GridLocal;
	float3 viewDirWS = normalize(input.WorldPos);
	// Ray parameter (world units along the view ray) to the parallax hit —
	// 0 means no carve; drives the SV_Depth push at the end.
	float trenchHitS = 0.0;
#	ifndef PATCH
	// Hand-off to the trench patch: trampled ROUNDED pixels near the camera
	// dissolve out so the patch's REAL carved geometry beneath shows
	// through. Three gates keep the hand-off airtight:
	// - RoundedDepth > 1: the patch culls sub-1-unit layers, so the skin
	//   must not discard into nothing (the vanished floors at depth 1).
	// - GEOMETRICALLY UP-FACING pixels only: the top-down patch can never
	//   replace a flank — discarding the log's side pixels (whose map
	//   column carries the trail) punched see-through holes.
	// Far trails and painted-flat trails keep the parallax relief instead.
	[branch] if (input.Flat < 0.5 && RoundedDepth > 1.0 && pixelDeform > 0.02 && length(input.WorldPos.xy) < 950.0)
	{
		float3 geoUp = normalize(cross(ddy(input.WorldPos), ddx(input.WorldPos)));
		if (abs(geoUp.z) > 0.55 && Random::InterleavedGradientNoise(input.Position.xy, SharedData::FrameCount) < smoothstep(0.03, 0.15, pixelDeform))
			discard;
	}
#	endif
	// PATCH pixels have REAL carved geometry and a VS gradient normal —
	// neither the parallax march nor the pixel tilt applies. FLAT-class
	// pixels never parallax either (Josef's ruling: parallax trenches are
	// banned — flat trails keep only the compression darkening).
#	ifndef PATCH
	[branch] if (input.Flat < 0.5 && pixelDeform > 0.001 && pixelCoverage > 0.35)
	{
		float pomDepth = min(lerp(RoundedDepth, ObjectsDepth, input.Flat), 25.0);
		[branch] if (pomDepth > 0.5)
		{
			// Trench floor — the landscape shell's kTrenchFloor rule, ported:
			// the carve can never reach the object mesh. Without the cap a
			// full-depth carve pushed SV_Depth to (and past) the underlying
			// surface, which both exposed its texture and z-fought it into
			// floors that flickered with the camera angle.
			float pomFloor = min(pomDepth, 5.0 * smoothstep(0.5, 8.0, pomDepth));
			float carveCap = pomDepth - pomFloor;
			// At depth t below the snow top the ray has drifted by
			// view.xy/-view.z * t in world XY. In air while t < carve(xy);
			// the hit is refined linearly between the straddling samples.
			// The -view.z clamp keeps grazing rays from smearing.
			float invRayZ = 1.0 / max(-viewDirWS.z, 0.25);
			float2 stepXY = viewDirWS.xy * invRayZ;
			float tPrev = 0.0;
			float carvePrev = min(pomDepth * pixelDeform, carveCap);
			[loop] for (uint pomI = 1; pomI <= 8; pomI++) {
				float t = carveCap * float(pomI) / 8.0;
				float2 xy = input.GridLocal + stepXY * t;
				float carve = min(pomDepth * saturate(SampleDeformation(xy)), carveCap);
				[branch] if (t >= carve)
				{
					float w = saturate((carvePrev - tPrev) / max((carvePrev - tPrev) + (t - carve), 1e-4));
					float tHit = lerp(tPrev, t, w);
					trenchGridLocal = input.GridLocal + stepXY * tHit;
					trenchHitS = tHit * invRayZ;
					break;
				}
				tPrev = t;
				carvePrev = carve;
			}
			// Fallback: ray stayed under the (capped) surface through every
			// sample — land it on the floor.
			[flatten] if (trenchHitS == 0.0 && carvePrev > 0.0)
			{
				trenchGridLocal = input.GridLocal + stepXY * carveCap;
				trenchHitS = carveCap * invRayZ;
			}
			pixelDeform = saturate(SampleDeformation(trenchGridLocal));
		}

		const float step = 4.0;
		float dXP = SampleDeformation(trenchGridLocal + float2(step, 0.0));
		float dXN = SampleDeformation(trenchGridLocal - float2(step, 0.0));
		float dYP = SampleDeformation(trenchGridLocal + float2(0.0, step));
		float dYN = SampleDeformation(trenchGridLocal - float2(0.0, step));
		float2 deformGradient = float2(dXP - dXN, dYP - dYN) / (2.0 * step);

		// Surface drops by depth*deform toward the trench: tilt the normal
		// up the slope so walls shade like real dents. Tilt is capped and
		// softened — full-depth gradients carved black gashes on deep-snow
		// cliffs.
		float pixelDepth = min(lerp(RoundedDepth, ObjectsDepth, input.Flat), 12.0) * pixelCoverage;
		normalWS = normalize(normalWS + float3(deformGradient * pixelDepth * 0.6, 0.0));
	}
#	endif

	// Per-pixel coverage: noisy up-facing gate (vanilla-projection-like
	// extent). NO deformation carve in alpha: cutting holes revealed the
	// bright projected-diffuse beneath — trampling only DENTS the geometry
	// (vertex-stage carve, floored so cover always remains).
	// RAW-normal gate (interpolated per-vertex): the pillow's shading normal
	// is smoothed, but accumulation must follow the real surface facing.
	// The noise only MODULATES existing coverage — it must never create
	// snow from nothing, or undersides and walls pick up dithered dabs.
	float coverageGate = saturate(pixelCoverage + (CoverageNoise(worldXY) - 0.5) * 0.3 * saturate(pixelCoverage * 4.0));
	float coverageAlpha = smoothstep(0.05, 0.35, coverageGate);
	// Hard down-facing kill: snow accumulates on TOPS only. The interpolated
	// raw normal is negative on every underside pixel, whatever the noise,
	// seam blends or floor-hold below decide.
	coverageAlpha *= smoothstep(-0.05, 0.1, input.Coverage);

	// Blend into the ground shell: where this pixel sits at or below the
	// terrain shell's snow surface, dissolve so the two shells dither into
	// one blanket instead of meeting at a hard seam. The SAME dials that
	// shape class borders shape this hand-off: Border Smoothness widens the
	// dissolve band (a taller, softer rise of ground snow up the object) and
	// Border Noise jitters WHERE the meeting line sits, so the seam wanders
	// organically around a rock's base instead of tracing a level line.
	float3 groundData = SampleTerrainStatics(input.GridLocal);
	[flatten] if (groundData.x > -50000.0)
	{
		float groundShellZ = groundData.x + max(groundData.y, 0.0);
		float pixelAbsZ = input.WorldPos.z + ShellCameraPosAdjust.z;
		float seamNoise = (CoverageNoise(worldXY * 0.5) - 0.5) * BorderNoise * 0.5;
		float bandLow = -(4.0 + BorderSmooth * 0.5);
		float bandHigh = 2.0 + BorderSmooth * 0.125;
		coverageAlpha *= smoothstep(bandLow, bandHigh, pixelAbsZ - (groundShellZ + seamNoise));
	}

	// Snow<->Snow Fade — Terrain Blending's technique adapted: the fade is
	// measured along the VIEW RAY against the landscape shell's actually-
	// rendered surface (post-shell depth copy), and ONLY where the thing
	// behind this pixel IS the shell (pre-vs-post depth divergence). A
	// height-based band could dissolve the skin over its own mesh and expose
	// the bare road beneath (the beige-smudge failure); this construction
	// can only ever fade white snow into white snow.
	[branch] if (SnowSnowFade > 0.01)
	{
		float postShellZ = SharedData::GetScreenDepth(ShellDepthCopy.Load(int3(input.Position.xy, 0)));
		float preShellZ = SharedData::GetScreenDepth(SceneDepth.Load(int3(input.Position.xy, 0)));
		[flatten] if (preShellZ - postShellZ > 1.0)  // the landscape shell is behind this pixel
		{
			float skinZ = input.CurrentClip.w;
			coverageAlpha *= smoothstep(0.0, max(SnowSnowFade, 1.0), postShellZ - skinZ);
		}
	}

	// Guaranteed snow floor in object trenches — the statics-skin mirror of
	// the landscape shell's kTrenchFloor: a carved, solidly-covered pixel
	// must never dissolve to the object's own texture (the road-mesh
	// texture mismatch), whatever the seam blends above decided.
	coverageAlpha = max(coverageAlpha, smoothstep(0.15, 0.5, pixelDeform) * smoothstep(0.35, 0.6, pixelCoverage));

#	ifndef PATCH
	// Vertical cull, MIDDLE strength: with the drape ramp the connective
	// snow lips are SLOPED geometry (z well above 0.2) and survive, while
	// truly vertical surfaces die — the interpolation-smeared white veils
	// down house walls and pole sides that the widened coverage gate
	// exposed. (Fully relaxing this cull was what let those through; the
	// old harder cull was what cut gap windows into pre-drape skirts —
	// the drape makes both safe.) The PATCH is exempt: its trench walls
	// are steep real geometry.
	float3 geoNormal = normalize(cross(ddy(input.WorldPos), ddx(input.WorldPos)));
	coverageAlpha *= smoothstep(0.06, 0.18, abs(geoNormal.z));
#	endif

	// Distance dissolve: from SkinFadeStart the skin stochastically thins
	// back into the object's own material, fully gone by SkinFadeEnd (the
	// capture range) — distant objects keep their real look instead of
	// turning blank white.
	float pixelDist = length(input.WorldPos);
	coverageAlpha *= 1.0 - smoothstep(SkinFadeStart, SkinFadeEnd, pixelDist);

	float screenNoise = Random::InterleavedGradientNoise(input.Position.xy, SharedData::FrameCount);
	if (screenNoise * screenNoise >= coverageAlpha)
		discard;

	// Snow texture taps — shared by albedo, normal and RMAOS. Sampled at
	// the parallax-corrected position so the texture rides the relief.
	// Steep drape sides re-project along the facing wall plane: the
	// top-down projection stretches down a puffed shell's flanks, and the
	// stochastic cells follow the same plane so the anti-tiling stays
	// coherent instead of smearing.
	float2 snowUV = (SnowUVOffset + trenchGridLocal) / kSnowUVTile;
	float2 snowCellXY = worldXY;
	float snowSteepness = smoothstep(0.55, 0.25, abs(normalWS.z));
	[branch] if (snowSteepness > 0.001)
	{
		float worldZAbs = input.WorldPos.z + ShellCameraPosAdjust.z;
		float2 sidePlane = abs(normalWS.x) > abs(normalWS.y) ? float2(worldXY.y, worldZAbs) : float2(worldXY.x, worldZAbs);
		snowUV = lerp(snowUV, (SnowUVOffset + sidePlane) / kSnowUVTile, snowSteepness);
		snowCellXY = lerp(worldXY, sidePlane, snowSteepness);
	}
	SnowTaps snowTaps = ComputeSnowTaps(snowUV, snowCellXY);

	// Micro-relief — identical recipe to the terrain shell so ground and
	// object snow carry the same grain: real PBR normal map when available,
	// luminance height-proxy fallback otherwise. Applied after the coverage
	// gate: bending the normal first would jitter the up-facing test into
	// speckled edges.
	float bumpFade = 1.0 - smoothstep(600.0, 2200.0, pixelDist);
	[branch] if (HasSnowNormal > 0.5 && bumpFade > 0.001)
	{
		float3 texN = SampleSnowMap(SnowNormalMap, snowTaps).xyz * 2.0 - 1.0;
		texN.z = sqrt(saturate(1.0 - dot(texN.xy, texN.xy)));
		texN.y = -texN.y;
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
		float2 detailUV = worldXY / kBumpTile;
		const float3 kLum = float3(0.30, 0.45, 0.25);
		float h0 = dot(SnowDiffuse.Sample(SnowSampler, detailUV).rgb, kLum);
		float hx = dot(SnowDiffuse.Sample(SnowSampler, detailUV + float2(e, 0.0)).rgb, kLum);
		float hy = dot(SnowDiffuse.Sample(SnowSampler, detailUV + float2(0.0, e)).rgb, kLum);
		float2 bumpGrad = float2(hx - h0, hy - h0) * (kBumpHeight / (e * kBumpTile));
		normalWS = normalize(normalWS + float3(-bumpGrad * bumpFade, 0.0));
	}

	float3 viewNormal = normalize(mul((float3x3)CameraView, normalWS));

	// Snow material — same albedo path as the terrain shell.
	float3 kSnowAlbedo = float3(0.82, 0.84, 0.88);
	[branch] if (HasSnowTexture != 0)
	{
		kSnowAlbedo = SampleSnowMap(SnowDiffuse, snowTaps).rgb;
		[flatten] if (SnowTextureIsLinear != 0.0)
			kSnowAlbedo = Color::LinearToSrgb(kSnowAlbedo);
	}
	// Compressed snow reads slightly darker and bluer than powder.
	kSnowAlbedo *= 1.0 - pixelDeform * float3(0.13, 0.12, 0.08);

	// PBR response — identical constants to the terrain shell.
	static const float kSnowRoughness = 0.6;
	static const float3 kSnowF0 = float3(0.028, 0.028, 0.028);

	// Per-pixel PBR response from the RMAOS map — same scales as the
	// terrain shell (roughness ×0.7, specularLevel 0.02).
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
	float sunShadow;
	[branch] if (CrispShadows > 0.5)
	{
		// Full-resolution comparison PCF — same path as the terrain shell.
		sunShadow = worldShadow * SnowShadow::GetCascadeShadow(input.WorldPos, normalWS, 1.0, LodShadowProj, LodShadowEnd, LodShadowActive);
	}
	else
	{
		float detailedShadow;
		float dynamicShadow = ShadowSampling::GetLightingShadow(input.WorldPos, detailedShadow);
		sunShadow = worldShadow * min(dynamicShadow, detailedShadow);
	}
	// Screen-Space Shadows: same long-range term bare ground multiplies in,
	// distance-blended past the cascades like the landscape shell (the SSS
	// march ran on the PREPASS depth — near, it belongs to the surface
	// UNDER the skin, and the crisp cascades already cover the skin).
	[branch] if (ScreenSpaceShadowsActive > 0.5)
	{
		float sssBlend = smoothstep(4000.0, 9000.0, pixelDist);
		sunShadow *= lerp(1.0, ScreenSpaceShadows::GetScreenSpaceShadow(input.Position.xyz, float2(0.0, 0.0), 0.0), sssBlend);
	}
	float3 sunLight = SharedData::DirLightColor.xyz * sunShadow;

	float3 F = BRDF::F_Schlick(snowF0, satVdotH);
	float specD = BRDF::D_GGX(snowRoughness, satNdotH);
	// Sparkle — same glint NDF and authored parameters as the terrain shell.
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

	float2 envBRDF = BRDF::EnvBRDF(snowRoughness, satNdotV);
	float3 specularLobe = snowF0 * envBRDF.x + envBRDF.y;
	float3 diffuseLobe = kSnowAlbedo * (1.0 - specularLobe);

	float3 directDiffuse = sunLight * satNdotL * (1.0 - F) * kSnowAlbedo;
	float3 directSpecular = specD * specV * F * sunLight * satNdotL;

	float3 ambientColor = Color::Ambient(max(0, SharedData::GetAmbient(normalWS))) * snowAO;
	float3 preLit = ambientColor * diffuseLobe + directDiffuse;

	float stochasticBlend = (screenNoise * screenNoise) < coverageAlpha ? 1.0 : 0.0;

	PS_OUTPUT psout;
	// Depth: unchanged pixels echo the rasterized depth; carved pixels
	// project the parallax hit point through the SAME (jittered) matrix
	// the VS used, so the trench floor is real to the z-buffer.
	psout.Depth = input.Position.z;
	[branch] if (trenchHitS > 0.0)
	{
		float4 hitClip = mul(CameraViewProj, float4(input.WorldPos + viewDirWS * trenchHitS, 1.0));
		psout.Depth = hitClip.z / max(hitClip.w, 1e-4);
	}
	psout.Diffuse = float4(preLit, coverageAlpha);
	psout.MotionVectors = float4(motionVector, 0.0, coverageAlpha);
	psout.NormalGlossiness = float4(GBuffer::EncodeNormal(viewNormal), 1.0 - snowRoughness, stochasticBlend);
	psout.Albedo = float4(diffuseLobe, coverageAlpha);
	psout.Specular = float4(directSpecular, coverageAlpha);
	psout.Reflectance = float4(specularLobe, coverageAlpha);
	psout.Masks = float4(0.0, 0.0, Color::RGBToYCoCg(ambientColor).x, coverageAlpha);
	psout.Masks2 = float4(0.0, 0.0, 0.0, coverageAlpha);
	return psout;
}
#endif
