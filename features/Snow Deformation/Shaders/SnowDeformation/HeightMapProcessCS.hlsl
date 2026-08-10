// S7 blanket height-field processing.
//
// ScrollCS  — persistence: carries the accumulated raw top/bottom maps into
//             the current window position (whole-texel offsets).
// CombineCS — builds the blanket base field: terrain height everywhere,
//             max'd with grounded object tops (bottom clearance < 40); also
//             writes the shelter mask (floating structures 60+ units up ⇒
//             no snow beneath: dynamic "No Snow Under Roofs").
// ConeCS    — angle of repose: iterative min-plus cone transform. No point
//             of the field may rise steeper than SlopePerUnit from its
//             neighbors, so thin/tall features (posts, walls) barely lift
//             the blanket while broad objects build natural mounds. This is
//             the "angled camera": snow exists only where the cone of view
//             from surrounding ground reaches.
//
// Sentinels: top empty = -100000, bottom empty = +100000.

cbuffer HeightProcessCB : register(b0)
{
	int2 ScrollDelta;
	uint ClearAll;
	uint ConeStep;  // texel step for this cone iteration

	float2 HeightWindowCenter;
	float HeightHalfExtent;
	float SlopePerUnit;  // max rise per world unit (1.0 = 45 degrees)

	float2 TerrainWindowOrigin;  // world XY of terrain window texel (0,0)
	float TerrainTexelSize;
	uint TerrainDim;

	float GhostDecay;  // units/frame the accumulated maps drift toward empty
	float3 padH;
}

#define MAX_EXCLUSIONS 96

cbuffer DoorCB : register(b1)
{
	float4 ExclusionPosRadius[MAX_EXCLUSIONS];  // xyz = position, w = radius
	float4 ExclusionDirExtType[MAX_EXCLUSIONS];  // xy = facing, z = forward extent, w = type (0 door, 1 fire)
	uint ExclusionCount;
	float3 exclusionPad;
}

// Cheap value noise for organic fire-clearing edges.
float ExclusionNoise(float2 worldXY)
{
	float2 c = worldXY / 24.0;
	float2 i = floor(c);
	float2 f = frac(c);
	f = f * f * (3.0 - 2.0 * f);
	float4 h;
	h.x = frac(sin(dot(i, float2(127.1, 311.7))) * 43758.5453);
	h.y = frac(sin(dot(i + float2(1, 0), float2(127.1, 311.7))) * 43758.5453);
	h.z = frac(sin(dot(i + float2(0, 1), float2(127.1, 311.7))) * 43758.5453);
	h.w = frac(sin(dot(i + float2(1, 1), float2(127.1, 311.7))) * 43758.5453);
	return lerp(lerp(h.x, h.y, f.x), lerp(h.z, h.w, f.x), f.y);
}

Texture2D<float> InA : register(t0);
Texture2D<float> InB : register(t1);
Texture2D<float4> TerrainWindow : register(t2);
RWTexture2D<float> OutA : register(u0);
RWTexture2D<float> OutB : register(u1);

// World XY of a height-map texel (v axis mirrors world +Y).
float2 TexelWorldXY(uint2 p, uint2 dims)
{
	float texel = HeightHalfExtent * 2.0 / dims.x;
	return float2(
		HeightWindowCenter.x + (float(p.x) - dims.x * 0.5 + 0.5) * texel,
		HeightWindowCenter.y + (dims.y * 0.5 - float(p.y) - 0.5) * texel);
}

float SampleTerrainHeight(float2 worldXY)
{
	float2 t = (worldXY - TerrainWindowOrigin) / TerrainTexelSize;
	t = clamp(t, 0.0, (float)(TerrainDim - 1) - 0.001);
	int2 t0 = (int2)t;
	float2 f = t - t0;
	int2 t1 = min(t0 + 1, int2(TerrainDim - 1, TerrainDim - 1));

	float s00 = TerrainWindow.Load(int3(t0.x, t0.y, 0)).x;
	float s10 = TerrainWindow.Load(int3(t1.x, t0.y, 0)).x;
	float s01 = TerrainWindow.Load(int3(t0.x, t1.y, 0)).x;
	float s11 = TerrainWindow.Load(int3(t1.x, t1.y, 0)).x;

	return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

[numthreads(8, 8, 1)] void ScrollCS(uint3 dtid
									: SV_DispatchThreadID) {
	uint2 dims;
	OutA.GetDimensions(dims.x, dims.y);
	if (any(dtid.xy >= dims))
		return;

	float top = -100000.0;
	float bottom = 100000.0;

	if (!ClearAll) {
		int2 src = int2(dtid.xy) + ScrollDelta;
		if (all(src >= 0) && all(src < int2(dims))) {
			// Ghost decay: accumulated heights fade unless re-rasterized this
			// frame — live objects re-assert themselves every frame, but
			// stale imprints (disabled/harvested/moved objects) melt away
			// instead of persisting until the window scrolls past them.
			top = InA[uint2(src)] - GhostDecay;
			bottom = InB[uint2(src)] + GhostDecay;
		}
	}

	OutA[dtid.xy] = top;
	OutB[dtid.xy] = bottom;
}

// InA = raw tops, InB = raw bottoms. OutA = base field, OutB = shelter mask.
[numthreads(8, 8, 1)] void CombineCS(uint3 dtid
									 : SV_DispatchThreadID) {
	uint2 dims;
	OutA.GetDimensions(dims.x, dims.y);
	if (any(dtid.xy >= dims))
		return;

	float2 worldXY = TexelWorldXY(dtid.xy, dims);
	float terrain = SampleTerrainHeight(worldXY);
	float field = terrain;
	float suppress = 0.0;

	float top = InA[dtid.xy];
	[branch] if (top > -50000.0)
	{
		float bottom = InB[dtid.xy];
		float clearance = bottom - terrain;
		float lift = top - terrain;
		// Grounded lift is CAPPED: the raster keeps one top and one bottom
		// per cell, so a rock under a walkway (or posts under a roof) yields
		// bottom = ground and top = deck/roof — without the cap the deck
		// becomes "ground" and the blanket mounds up on top of it. Anything
		// taller than the cap keeps the field at terrain; snow drifts against
		// its base via the cone transform instead.
		if (clearance < 40.0 && lift > 0.0 && lift <= 150.0)
			field = max(field, top);  // grounded: object top becomes ground
		else if (clearance >= 40.0 && lift > 60.0)
			suppress = 1.0;  // floating structure: bare ground beneath
	}

	// Exclusion zones: pull the field back to terrain and suppress snow.
	// Doors use an ELLIPSE stretched along their facing axis (both ways —
	// the recess and the doorstep); campfires use a noisy-edged circle for
	// an organic melt ring. Running BEFORE the cone transform means
	// surrounding mounds re-slope into every clearing at the angle of
	// repose — no ravine walls, by construction. Z-gated (300) so
	// upper-floor doors do not clear ground snow far below, while sunken
	// cave entrances still qualify.
	for (uint exclusionI = 0; exclusionI < ExclusionCount; exclusionI++) {
		float3 center = ExclusionPosRadius[exclusionI].xyz;
		float radius = ExclusionPosRadius[exclusionI].w;
		float4 dirExtType = ExclusionDirExtType[exclusionI];
		[branch] if (abs(center.z - terrain) < 300.0)
		{
			float2 d = worldXY - center.xy;
			float influence;
			[branch] if (dirExtType.w < 0.5)
			{
				// Door: symmetric ellipse, long axis along the facing, edge
				// perturbed by the same noise as fire clearings so no two
				// doorway hollows read as identical stamped shapes.
				float u = dot(d, dirExtType.xy);
				float v = dot(d, float2(-dirExtType.y, dirExtType.x));
				float a = radius + dirExtType.z;
				float b = radius * 0.85;
				float e = sqrt((u * u) / (a * a) + (v * v) / (b * b));
				e /= 0.8 + 0.4 * ExclusionNoise(worldXY);
				influence = 1.0 - smoothstep(0.45, 1.0, e);
			}
			else
			{
				// Campfire: melt ring with a noise-perturbed edge.
				float noisyRadius = radius * (0.8 + 0.5 * ExclusionNoise(worldXY));
				float dist = length(d);
				influence = 1.0 - smoothstep(noisyRadius * 0.4, noisyRadius, dist);
			}
			field = lerp(field, terrain, influence);
			suppress = max(suppress, influence);
		}
	}

	OutA[dtid.xy] = field;
	OutB[dtid.xy] = suppress;
}

// InA = field. OutA = slope-limited field (one iteration at ConeStep).
[numthreads(8, 8, 1)] void ConeCS(uint3 dtid
								  : SV_DispatchThreadID) {
	uint2 dims;
	OutA.GetDimensions(dims.x, dims.y);
	if (any(dtid.xy >= dims))
		return;

	float texel = HeightHalfExtent * 2.0 / dims.x;
	float h = InA[dtid.xy];

	[unroll] for (int dy = -1; dy <= 1; dy++)
	{
		[unroll] for (int dx = -1; dx <= 1; dx++)
		{
			if (dx == 0 && dy == 0)
				continue;
			int2 p = int2(dtid.xy) + int2(dx, dy) * int(ConeStep);
			if (any(p < 0) || any(p >= int2(dims)))
				continue;
			float dist = length(float2(dx, dy)) * ConeStep * texel;
			h = min(h, InA[uint2(p)] + SlopePerUnit * dist);
		}
	}

	// The blanket can never sink below the actual terrain.
	float terrain = SampleTerrainHeight(TexelWorldXY(dtid.xy, dims));
	OutA[dtid.xy] = max(h, terrain);
}
