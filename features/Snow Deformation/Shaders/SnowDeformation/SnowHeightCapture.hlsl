// Top-down height capture of snow statics.
//
// Captured statics are rasterized from above into R32F world-height maps
// with MIN/MAX blending (no depth buffer needed; the highest/lowest surface
// wins in any draw order): object TOPS, object BOTTOMS (together they tell
// whether a texel's object is grounded or floating), and the snow-layer
// depth the texel's model class wears; the surface description later snow
// systems (the trench patch, sheltering, burial mounds) build on.
//
// StaticCB layout must match StaticsCB in SnowDeformation.h and StaticCB in
// SnowStaticsShell.hlsl.

cbuffer StaticCB : register(b1)
{
	float4 WorldRow0;
	float4 WorldRow1;
	float4 WorldRow2;

	float ObjectsDepth;
	float2 HeightWindowCenter;
	float HeightHalfExtent;

	float HasSmoothedNormals;  // layout sync with SnowStaticsShell; stats-only here
	float RoundedDepth;
	float VertexCountF;
	// >0: discard fragments this far above the terrain. The fine (patch)
	// capture gates out roofs, eaves and beams: the max-blend otherwise
	// records them over the floor beneath, and the patch's kill heuristics
	// carve their outlines into the top sheet as house-shaped holes. The
	// coarse capture passes 0: shelter detection needs the roofs.
	float CaptureAltitudeCap;

	float2 TerrainOriginS;
	float TerrainTexelS;
	float TerrainDimS;
}

struct VS_INPUT
{
	float4 Position : POSITION0;
	float4 Normal : NORMAL0;
};

struct VS_OUTPUT
{
	float4 Position : SV_POSITION;
	float WorldZ : TEXCOORD0;
	float SkinDepth : TEXCOORD1;
	float2 WorldXY : TEXCOORD2;
};

#ifdef VSHADER
// Flatness stats (element past the last vertex); the same GPU
// classification the skin uses, so the skin-depth raster (RT2) reports the
// class layer depth per texel.
StructuredBuffer<float4> SmoothedNormals : register(t10);

VS_OUTPUT main(VS_INPUT input)
{
	float3 posMS = input.Position.xyz;
	float3 worldAbs = float3(
		dot(WorldRow0.xyz, posMS) + WorldRow0.w,
		dot(WorldRow1.xyz, posMS) + WorldRow1.w,
		dot(WorldRow2.xyz, posMS) + WorldRow2.w);

	// Ortho top-down: world XY window to NDC. +worldY maps to +ndcY, which
	// rasterizes to texture v=0 at the top; the samplers mirror this.
	float2 ndc = (worldAbs.xy - HeightWindowCenter) / HeightHalfExtent;

	float skinDepth = RoundedDepth;
	[branch] if (HasSmoothedNormals > 0.5)
	{
		// Same flat condition as the skin VS (divergence-only).
		float4 flatStats = SmoothedNormals[(uint)VertexCountF];
		[flatten] if (flatStats.w > 0.5 && flatStats.x > 0.5)
			skinDepth = ObjectsDepth;
	}

	VS_OUTPUT vsout;
	vsout.Position = float4(ndc.x, ndc.y, 0.5, 1.0);
	vsout.WorldZ = worldAbs.z;
	vsout.SkinDepth = skinDepth;
	vsout.WorldXY = worldAbs.xy;
	return vsout;
}
#endif

#ifdef PSHADER
// Terrain window for the altitude gate (bound only when the gate is on).
Texture2D<float4> TerrainWindow : register(t0);

struct PS_OUTPUT
{
	// RT0 blends MAX (object top surface), RT1 blends MIN (object bottom),
	// RT2 blends MAX (the snow-layer depth this texel's class wears).
	float Top : SV_Target0;
	float Bottom : SV_Target1;
	float SkinDepth : SV_Target2;
};

PS_OUTPUT main(VS_OUTPUT input)
{
	[branch] if (CaptureAltitudeCap > 0.5)
	{
		float2 t = (input.WorldXY - TerrainOriginS) / TerrainTexelS;
		t = clamp(t, 0.0, TerrainDimS - 1.001);
		int2 t0 = (int2)t;
		float2 f = t - t0;
		int2 t1 = min(t0 + 1, int2((int)TerrainDimS - 1, (int)TerrainDimS - 1));
		float s00 = TerrainWindow.Load(int3(t0.x, t0.y, 0)).x;
		float s10 = TerrainWindow.Load(int3(t1.x, t0.y, 0)).x;
		float s01 = TerrainWindow.Load(int3(t0.x, t1.y, 0)).x;
		float s11 = TerrainWindow.Load(int3(t1.x, t1.y, 0)).x;
		float terrain = lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
		if (input.WorldZ - terrain > CaptureAltitudeCap)
			discard;
	}

	PS_OUTPUT psout;
	psout.Top = input.WorldZ;
	psout.Bottom = input.WorldZ;
	psout.SkinDepth = input.SkinDepth;
	return psout;
}
#endif
