// Object height-window processing.
//
// ScrollCS   persistence: carries the accumulated raw top/bottom maps into
//            the current window position (whole-texel offsets). The maps
//            must not depend on what the camera renders this frame; the
//            capture list is frustum-culled, and rebuilding from it alone
//            makes object heights vanish behind the camera.
// CombineCS  builds the base snow-height field (terrain, lifted only by
//            corpse burial mounds) and the shelter mask: where the raw maps
//            show a structure floating well above the ground (walkways,
//            roofs, bridges, tents), the ground beneath is sheltered from
//            snowfall - a soft melt down to a light dusting, never a
//            coverage kill.
//            Exclusion zones: doors clear the field and add to the mask
//            (coverage fades to bare ground); fires write NEGATIVE mask
//            values instead - a melt fraction that thins the shell's depth
//            toward a floor, so fire pits keep a thin snow floor that never
//            vanishes or sinks below terrain.
// ConeCS     angle of repose: iterative min-plus cone transform. No point of
//            the field may rise steeper than SlopePerUnit from its
//            neighbors, so thin or tall features barely lift the field while
//            broad raises settle into natural mounds.
//
// Sentinels: top empty = -100000, bottom empty = +100000.

#define MAX_OBSTRUCTIONS 48

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
	float2 DeformWindowOriginH;  // deformation-map addressing (refill gate)
	float DeformInvWorldSizeH;

	uint CorpseSphereCount;  // resting dead actors' collision spheres
	float CorpseMoundCap;    // max mound height above terrain
	float2 WindBiasH;        // unit wind direction (blowing toward) x strength 0-1
	float4 CorpseSpheres[64];  // xyz world center, w radius

	float DriftHeight;  // peak wall-drift bank height (0 disables)
	uint ObstructionCount;
	float2 padObs;
	float4 ObstructionPosExt[MAX_OBSTRUCTIONS];  // xy world center, zw half extents (local XY)
	float4 ObstructionRot[MAX_OBSTRUCTIONS];     // xy = sin/cos of Z rotation, z = foundation height
}

#define MAX_EXCLUSIONS 256

// Wall drifts: band width past the wall, baseline bank fraction in calm
// weather, extra fraction earned by windward alignment x wind strength.
#define DRIFT_BAND 140.0
#define DRIFT_BASE 0.3
#define DRIFT_WIND 0.7

// Shelter melt strength: snow under roofs/tents/walkways thins to a light
// dusting (the shell keeps covering the ground - bare ground would expose
// the mismatched projected snow diffuse beneath).
#define SHELTER_MELT 0.9
// Soft-shelter ring radius in texels (4 world units each): widens the
// per-texel roofline test into a gradual transition band. 10 texels =
// a ~40-unit band, so a full-depth sink slopes at ~20 degrees instead
// of presenting a snow cliff at the roofline.
#define SHELTER_RING_TEXELS 10

cbuffer DoorCB : register(b1)
{
	float4 ExclusionPosRadius[MAX_EXCLUSIONS];   // xyz = position, w = radius
	float4 ExclusionDirExtType[MAX_EXCLUSIONS];  // doors (w=0): xy = facing, z = forward extent. Fires (w=1 noisy, w=2 smooth): xy = elongation axis x (aspect-1), z = melt strength
	uint ExclusionCount;
	float3 exclusionPad;
}

Texture2D<float> InA : register(t0);
Texture2D<float> InB : register(t1);
Texture2D<float4> TerrainWindow : register(t2);
Texture2D<float> DeformMap : register(t3);  // CombineCS: corpse-mound refill gate
RWTexture2D<float> OutA : register(u0);
RWTexture2D<float> OutB : register(u1);

// Cheap value noise for organic clearing edges.
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
			// frame; live objects re-assert themselves every frame, but
			// stale imprints (disabled/harvested/moved objects) melt away
			// instead of persisting until the window scrolls past them.
			top = InA[uint2(src)] - GhostDecay;
			bottom = InB[uint2(src)] + GhostDecay;
		}
	}

	OutA[dtid.xy] = top;
	OutB[dtid.xy] = bottom;
}

// How strongly the raster at p reads as a floating structure above the
// given terrain height (walkway, roof, bridge, tent canvas). CONTINUOUS:
// a binary test quantized the ring average into visible melt terraces
// under eaves - stairs descending toward the wall.
float ShelterTap(int2 p, int2 dims, float terrain)
{
	p = clamp(p, int2(0, 0), dims - 1);
	float result = 0.0;
	float top = InA[p];
	[branch] if (top > -50000.0)
	{
		float bottom = InB[p];
		result = smoothstep(20.0, 60.0, bottom - terrain) * smoothstep(40.0, 80.0, top - terrain);
	}
	return result;
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
	float melt = 0.0;

	// Shelter only: grounded object tops deliberately do NOT raise the field
	// (an object-top "blanket" lift was tried and removed; it produced seams
	// against the landscape shell and 45-degree spike cones at range). The
	// raster feeds just the floating-structure test: a bottom well clear of
	// the ground with a top high above it is a walkway/roof/bridge, and the
	// ground beneath it is sheltered from snowfall. Consumed as MELT (thin
	// shell floor, snow texture kept - a coverage kill exposed the
	// mismatched projected snow beneath and cut a cliff at the roofline);
	// the per-texel test is binary, so a center + 8-tap ring fraction turns
	// the cut into a smooth sink under the eaves. Taps reuse the center
	// terrain height: terrain varies slowly at ring scale.
	{
		int2 texel = int2(dtid.xy);
		int2 dimsI = int2(dims);
		static const int2 kShelterRing[8] = {
			int2(SHELTER_RING_TEXELS, 0), int2(-SHELTER_RING_TEXELS, 0),
			int2(0, SHELTER_RING_TEXELS), int2(0, -SHELTER_RING_TEXELS),
			int2(7, 7), int2(7, -7), int2(-7, 7), int2(-7, -7)
		};
		static const int2 kShelterRingInner[8] = {
			int2(5, 0), int2(-5, 0), int2(0, 5), int2(0, -5),
			int2(4, 4), int2(4, -4), int2(-4, 4), int2(-4, -4)
		};
		float shelterFrac = ShelterTap(texel, dimsI, terrain) * 2.0;
		[unroll] for (uint ringI = 0; ringI < 8; ringI++)
			shelterFrac += ShelterTap(texel + kShelterRing[ringI], dimsI, terrain) +
			               ShelterTap(texel + kShelterRingInner[ringI], dimsI, terrain);
		shelterFrac /= 18.0;
		// Deliberately no edge noise: roofline sinks read best smooth (fire
		// bowls keep their noisy rims; sheltered snow follows the structure).
		melt = max(melt, SHELTER_MELT * saturate(shelterFrac));
	}

	// Wall drifts: wind piles snow into banks against large statics
	// (buildings, towers, boulders), passed as OBB footprints. Windward
	// walls (outward normal facing INTO the wind) bank toward full
	// DriftHeight; calm weather keeps a modest all-around bank; leeward
	// walls simply never earn the windward bonus. The cone transform
	// downstream rounds every bank into a natural slope; exclusions run
	// AFTER this, so doorways stay swept through the banks.
	[branch] if (DriftHeight > 0.01)
	{
		float windStrength = length(WindBiasH);
		float2 windDir = windStrength > 0.001 ? WindBiasH / windStrength : float2(0.0, 0.0);
		for (uint obsI = 0; obsI < ObstructionCount; obsI++) {
			float4 posExt = ObstructionPosExt[obsI];
			float4 obsRot = ObstructionRot[obsI];
			[branch] if (abs(obsRot.z - terrain) < 400.0)
			{
				float2 rel = worldXY - posExt.xy;
				float2 local = float2(obsRot.y * rel.x - obsRot.x * rel.y, obsRot.x * rel.x + obsRot.y * rel.y);
				float2 q = abs(local) - posExt.zw;
				float outside = length(max(q, 0.0));
				[branch] if (outside < DRIFT_BAND)
				{
					// Windwardness from the RADIAL direction around the object,
					// not the nearest-face normal: face normals flip instantly
					// at box corners, seaming a full windward bank against a
					// baseline leeward one - a cliff at every slider value. The
					// radial direction varies continuously around walls,
					// corners and the interior alike, so bank height glides
					// from windward maximum to leeward baseline. No leeward
					// scour: leeward simply never earns the windward bonus.
					float relLen = length(rel);
					float windward = relLen > 0.001 ? saturate(-dot(rel / relLen, windDir)) : 0.0;
					float amp = DRIFT_BASE + DRIFT_WIND * windward * windStrength;
					// Inside the footprint (outside == 0) the profile is 1: a
					// hidden plateau, continuous with the wall banks. Without
					// it the un-lifted interior is the cone transform's lowest
					// neighbor and every bank gets cut down INTO the wall at
					// the repose slope, terraced by the sparse cone steps.
					float profile = 1.0 - smoothstep(0.0, DRIFT_BAND, outside);
					field = max(field, terrain + DriftHeight * amp * profile);
				}
			}
		}
	}

	// Exclusion zones: pull the field back to terrain, then either suppress
	// snow (doors: coverage fades to bare ground) or melt it (fires: depth
	// thins toward a floor). Doors use an ELLIPSE stretched along their
	// facing axis (both ways; the recess and the doorstep); fires use a
	// noisy-edged circle for an organic melt basin. Z-gated (300) so
	// upper-floor doors do not clear ground snow far below, while sunken
	// cave entrances still qualify.
	for (uint exclusionI = 0; exclusionI < ExclusionCount; exclusionI++) {
		float3 center = ExclusionPosRadius[exclusionI].xyz;
		float radius = ExclusionPosRadius[exclusionI].w;
		float4 dirExtType = ExclusionDirExtType[exclusionI];
		[branch] if (abs(center.z - terrain) < 300.0)
		{
			float2 d = worldXY - center.xy;
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
				float influence = 1.0 - smoothstep(0.45, 1.0, e);
				field = lerp(field, terrain, influence);
				suppress = max(suppress, influence);
			}
			else
			{
				// Melt bowl - full melt in the core, then a long gradual
				// rise. Type 1 (fires): two noise octaves, fine wobble plus
				// large-scale shape irregularity, so no two bowls read as
				// stamped circles. Type 2 (bedding): smooth edge, so a
				// bedroll's bowl joins a tent's shelter sink cleanly.
				// dirExtType.xy = elongation axis x (aspect-1): compressing
				// the along-axis distance stretches the bowl into an oval
				// centered on the object. dirExtType.z = melt strength.
				float2 dEff = d;
				float axisLen = length(dirExtType.xy);
				[branch] if (axisLen > 0.001)
				{
					float2 axisDir = dirExtType.xy / axisLen;
					dEff -= axisDir * dot(d, axisDir) * (axisLen / (1.0 + axisLen));
				}
				float noisy = 1.0;
				[branch] if (dirExtType.w < 1.5)
				{
					noisy = 0.7 + 0.35 * ExclusionNoise(worldXY) + 0.35 * ExclusionNoise(worldXY * 0.3);
				}
				float noisyRadius = radius * noisy;
				float dist = length(dEff);
				float influence = (1.0 - smoothstep(noisyRadius * 0.35, noisyRadius, dist)) * dirExtType.z;
				field = lerp(field, terrain, influence);
				melt = max(melt, influence);
			}
		}
	}

	// Corpse burial mounds: resting dead actors inject their collision-
	// sphere caps as field tops; CAPPED above terrain so a mammoth makes a
	// bump, not a hill; gated by the LOCAL REFILL state so the story reads
	// fall -> imprint -> snow closes -> mound swells over the buried body.
	// Stateless: loot or move the corpse and the mound is gone next frame.
	// The cone transform downstream rounds every mound into a natural lump.
	for (uint corpseI = 0; corpseI < CorpseSphereCount; corpseI++) {
		float4 sphere = CorpseSpheres[corpseI];
		float2 dc = worldXY - sphere.xy;
		float sqDist = dot(dc, dc);
		[branch] if (sqDist < sphere.w * sphere.w)
		{
			// Grounded corpses only: a body on a high ledge must not mound
			// the snow far below it.
			[branch] if (sphere.z - sphere.w - terrain < 40.0)
			{
				float capZ = sphere.z + sqrt(sphere.w * sphere.w - sqDist);
				float mound = min(capZ, terrain + CorpseMoundCap);

				float deform = 0.0;
				float2 deformUV = (worldXY - DeformWindowOriginH) * DeformInvWorldSizeH;
				[branch] if (all(deformUV >= 0.0) && all(deformUV <= 1.0))
				{
					float2 deformDims;
					DeformMap.GetDimensions(deformDims.x, deformDims.y);
					deform = DeformMap.Load(int3(int2(deformUV * deformDims), 0));
				}
				// Refill first, then the mound: no mound while the death
				// imprint is still carved open.
				float refillGate = 1.0 - saturate(deform / 0.3);
				field = max(field, lerp(terrain, mound, refillGate));
			}
		}
	}

	OutA[dtid.xy] = field;
	// One mask channel, two signals: positive = door suppression (coverage
	// kill to bare ground), negative = melt fraction (fires, workspace
	// clearings, sheltered ground). Doors win where both apply.
	OutB[dtid.xy] = suppress > 0.001 ? suppress : -melt;
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

	// The field can never sink below the actual terrain.
	float terrain = SampleTerrainHeight(TexelWorldXY(dtid.xy, dims));
	OutA[dtid.xy] = max(h, terrain);
}
