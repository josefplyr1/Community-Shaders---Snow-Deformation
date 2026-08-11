// S7 unified blanket: top-down height capture of snow statics.
//
// Captured statics are rasterized from above into an R32F world-height map
// with MAX blending (no depth buffer needed — the highest surface wins in
// any draw order). The terrain shell samples this map and treats object tops
// as ground: objects LIFT the one blanket instead of wearing a second shell,
// so seams between ground snow and object snow cease to exist.
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

	float HasSmoothedNormals;  // layout sync with SnowStaticsShell; unused here
	float RoundedDepth;
	float VertexCountF;
	float padStat;
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
};

#ifdef VSHADER
VS_OUTPUT main(VS_INPUT input)
{
	float3 posMS = input.Position.xyz;
	float3 worldAbs = float3(
		dot(WorldRow0.xyz, posMS) + WorldRow0.w,
		dot(WorldRow1.xyz, posMS) + WorldRow1.w,
		dot(WorldRow2.xyz, posMS) + WorldRow2.w);

	// Ortho top-down: world XY window to NDC. +worldY maps to +ndcY, which
	// rasterizes to texture v=0 at the top — the sampler mirrors this.
	float2 ndc = (worldAbs.xy - HeightWindowCenter) / HeightHalfExtent;

	VS_OUTPUT vsout;
	vsout.Position = float4(ndc.x, ndc.y, 0.5, 1.0);
	vsout.WorldZ = worldAbs.z;
	return vsout;
}
#endif

#ifdef PSHADER
struct PS_OUTPUT
{
	// RT0 blends MAX (object top surface), RT1 blends MIN (object bottom) —
	// together they tell the shell whether a texel's object is grounded
	// (lift the blanket) or floating (no lift; shelter the ground beneath).
	float Top : SV_Target0;
	float Bottom : SV_Target1;
};

PS_OUTPUT main(VS_OUTPUT input)
{
	PS_OUTPUT psout;
	psout.Top = input.WorldZ;
	psout.Bottom = input.WorldZ;
	return psout;
}
#endif
