// Persistent snow deformation map update.
//
// The map is a square world-space window following the camera in whole-texel
// steps. Each frame the previous map is re-read at a scrolled offset, snow
// refill is applied, and this frame's actor stamps are max-blended in.
// Texel value = normalized depression depth (0 = untouched, 1 = ground).

#define MAX_STAMPS 256
#define MAX_TRAMPLE_ZONES 64

// Upwind supply sample distance for wind-biased refill, in texels.
#define DRIFT_FETCH_TEXELS 3.0
// Refill multiplier at full supply and full wind; interior texels with a
// carved upwind neighbor stall, so the average fill rate stays near uniform.
#define DRIFT_GAIN 2.0
// Compression held by a trample zone: partially trodden, visibly shallower
// than a real trail (those carve to 1.0 and max over this).
#define TRAMPLE_DEPTH 0.55

cbuffer PerFrame : register(b0)
{
	float2 WindowOrigin;
	int2 ScrollDelta;

	float TexelSize;
	uint StampCount;
	float RefillAmount;
	uint ClearMap;

	// Lower smoothstep edge of the stamp falloff (fraction of radius):
	// higher = steeper trench walls.
	float StampFalloffStart;
	// Fraction-of-radius noise wobbling each stamp's edge.
	float StampNoiseAmp;
	// Unit wind direction (world XY, blowing toward) times wind strength
	// 0-1; zero = uniform refill.
	float2 WindBias;

	float4 Stamps[MAX_STAMPS];     // xy: world pos, z: depth, w: radius
	float4 StampEnds[MAX_STAMPS];  // xy: previous world pos (capsule segment start)
}

cbuffer TrampleCB : register(b1)
{
	float4 TramplePosRadius[MAX_TRAMPLE_ZONES];  // xyz = position, w = radius
	float4 TrampleDirBias[MAX_TRAMPLE_ZONES];    // xy = facing dir, z = forward bias, w = unused
	uint TrampleCount;
	float3 padTrample;
}

Texture2D<float> PreviousDeformation : register(t0);
RWTexture2D<float> CurrentDeformation : register(u0);

// World-anchored value noise (8-unit cells at the call site) wobbling each
// stamp's falloff distance, so trail edges read as churned snow instead of
// swept circles.
float StampNoiseHash(float2 cell)
{
	// Wrap the lattice so the hash's frac() math stays well inside float
	// precision at world-scale inputs; the 512-cell repeat is invisible in
	// edge wobble.
	cell -= 512.0 * floor(cell / 512.0);
	float3 p3 = frac(float3(cell.x, cell.y, cell.x) * float3(0.1031, 0.1030, 0.0973));
	p3 += dot(p3, p3.yzx + 33.33);
	return frac((p3.x + p3.y) * p3.z);
}

float StampNoise(float2 p)
{
	float2 i = floor(p);
	float2 f = frac(p);
	f = f * f * (3.0 - 2.0 * f);
	return lerp(lerp(StampNoiseHash(i), StampNoiseHash(i + float2(1, 0)), f.x),
		lerp(StampNoiseHash(i + float2(0, 1)), StampNoiseHash(i + float2(1, 1)), f.x), f.y);
}

[numthreads(8, 8, 1)] void main(uint3 DTid
								: SV_DispatchThreadID) {
	uint2 pixel = DTid.xy;

	float deformation = 0.0;

	if (!ClearMap) {
		int2 sourcePixel = int2(pixel) + ScrollDelta;

		uint2 dims;
		PreviousDeformation.GetDimensions(dims.x, dims.y);

		[branch] if (all(sourcePixel >= 0) && all(sourcePixel < int2(dims)))
		{
			deformation = PreviousDeformation[uint2(sourcePixel)];
		}

		// Wind-biased refill: recovery scales with the intact snow a few
		// texels upwind (the drift supply), so carved areas fill from their
		// upwind edge and the fill front marches downwind. Calm weather
		// falls back to uniform refill.
		float refill = RefillAmount;
		float windStrength = length(WindBias);
		[branch] if (refill > 0.0 && windStrength > 0.001)
		{
			int2 upwindPixel = sourcePixel - int2(round(WindBias / windStrength * DRIFT_FETCH_TEXELS));
			float upwindDeformation = deformation;
			[branch] if (all(upwindPixel >= 0) && all(upwindPixel < int2(dims)))
			{
				upwindDeformation = PreviousDeformation[uint2(upwindPixel)];
			}
			refill *= lerp(1.0, (1.0 - upwindDeformation) * DRIFT_GAIN, windStrength);
		}
		deformation = max(deformation - refill, 0.0);
	}

	float2 worldPos = WindowOrigin + (float2(pixel) + 0.5) * TexelSize;

	for (uint i = 0; i < StampCount; i++) {
		// Capsule stamp: distance to the segment from the actor's previous
		// position, so trails are continuous regardless of movement speed.
		float2 p0 = StampEnds[i].xy;
		float2 p1 = Stamps[i].xy;
		float2 seg = p1 - p0;
		float segLenSq = dot(seg, seg);
		float t = segLenSq > 1e-4 ? saturate(dot(worldPos - p0, seg) / segLenSq) : 0.0;
		float2 delta = worldPos - (p0 + seg * t);
		float distSq = dot(delta, delta);
		float radius = Stamps[i].w;

		// Edge noise can push the falloff outward, so the gate widens with it.
		float gateRadius = radius * (1.0 + 0.5 * StampNoiseAmp);
		[branch] if (distSq < gateRadius * gateRadius)
		{
			float edgeDist = sqrt(distSq) / radius;
			[branch] if (StampNoiseAmp > 0.001)
			{
				edgeDist += (StampNoise(worldPos * 0.125) - 0.5) * StampNoiseAmp;
			}
			// Falloff from StampFalloffStart of the radius: low values keep a
			// wide edge band coarser consumers of the map can still represent,
			// high values hold full depth almost to the edge.
			float falloff = 1.0 - smoothstep(StampFalloffStart, 1.0, edgeDist);
			deformation = max(deformation, Stamps[i].z * falloff);
		}
	}

	// Trample zones: regularly worked ground (workstations, wood yards,
	// shrines) holds a partial compression floor, re-asserted every frame.
	// Actor trails max over it; refill pulls it down each frame and this
	// puts it back, so the spot stays trodden while its object is present.
	// The patch center rides forward along the object's facing (where the
	// user stands); noisy edges keep it from reading as a stamped disc.
	for (uint zoneI = 0; zoneI < TrampleCount; zoneI++) {
		float2 center = TramplePosRadius[zoneI].xy + TrampleDirBias[zoneI].xy * TrampleDirBias[zoneI].z;
		float radius = TramplePosRadius[zoneI].w;
		float2 dz = worldPos - center;
		float distSq = dot(dz, dz);
		float zoneGate = radius * 1.3;
		[branch] if (distSq < zoneGate * zoneGate)
		{
			float noisyRadius = radius * (0.75 + 0.5 * StampNoise(worldPos * 0.04));
			float influence = 1.0 - smoothstep(noisyRadius * 0.35, noisyRadius, sqrt(distSq));
			deformation = max(deformation, TRAMPLE_DEPTH * influence);
		}
	}

	CurrentDeformation[pixel] = deformation;
}
