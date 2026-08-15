// Terrain-window far fill.
//
// Runs once per window rebuild, after the CPU upload of baked cell data:
// texels that no baked cell covers (sentinel height) get their height from
// the xLODGen worldspace heightmap (the Terrain Shadows data source) and
// their snow coverage from a north-shifted snow line, so distant snow
// exists everywhere instead of only where the player has walked.
// Texel .w marks the provenance (1 = heightmap-sourced) for the debug view.

RWTexture2D<float4> TerrainWindow : register(u0);
Texture2D<float> HeightMap : register(t0);
SamplerState LinearSampler : register(s0);

cbuffer WindowFillCB : register(b0)
{
	// World XY of window texel (0,0); texels are vertex-aligned (no half-texel).
	float2 WindowOriginWorld;
	float TexelSize;
	uint WindowDim;

	// world.xy * Scale + Offset = heightmap UV (Terrain Shadows convention).
	float2 HeightMapScale;
	float2 HeightMapOffset;

	// Decoded height = lerp(HeightRange.x, HeightRange.y, sample), where
	// HeightRange is the metadata's pos0.z/pos1.z — the range the file's
	// values are normalized over (+-32767*8 for xLODGen exports). This is
	// the convention ShadowUpdate.cs.hlsl decodes with; the metadata's
	// zRange is the actual min/max CONTENT height and is a much narrower
	// band, so decoding with it flattens the world toward its midpoint.
	float2 HeightRange;
	// Worldspace south/north Y bounds for the latitude term.
	float2 WorldYRange;

	float SnowLineZ;
	float SnowNorthDrop;
	float SnowLineFade;
	// Snow depth (world units) a fully covered heightmap texel carries.
	float SnowDepthUnits;
}

[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {
	if (id.x >= WindowDim || id.y >= WindowDim)
		return;
	// Baked cell data always wins; only sentinel texels are filled.
	if (TerrainWindow[id.xy].x > -50000.0)
		return;

	float2 worldXY = WindowOriginWorld + float2(id.xy) * TexelSize;
	float2 uv = worldXY * HeightMapScale + HeightMapOffset;
	// Outside the worldspace: stay sentinel (the shell edge-fades there).
	if (any(uv < 0.0) || any(uv > 1.0))
		return;

	float z = lerp(HeightRange.x, HeightRange.y, HeightMap.SampleLevel(LinearSampler, uv, 0));

	// Snow line sinks toward the north edge over the top ~third of the map,
	// so the northern coast reads snowy at sea level while the mid-map
	// plains stay bare at the same elevation.
	float northness = saturate((worldXY.y - WorldYRange.x) / max(WorldYRange.y - WorldYRange.x, 1.0));
	float snowLine = SnowLineZ - SnowNorthDrop * smoothstep(0.55, 0.9, northness);
	float coverage = smoothstep(snowLine - SnowLineFade, snowLine + SnowLineFade, z);

	TerrainWindow[id.xy] = float4(z, coverage * SnowDepthUnits, coverage, 1.0);
}
