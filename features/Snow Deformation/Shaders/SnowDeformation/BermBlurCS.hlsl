// Wide Gaussian blur of the deformation map into the berm field. The berm
// field has ~16-world-unit texels regardless of the deformation map's
// resolution (the CPU sizes it that way), so the fixed 3-texel kernel
// radius is a fixed ~48-unit world reach. The berm's shape input must be
// genuinely smooth: sparse ring taps at shading time leave a piecewise-
// constant staircase that prints plateaus on the hill.
Texture2D<float> DeformationMap : register(t0);
RWTexture2D<float> BermField : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	uint2 outDims;
	BermField.GetDimensions(outDims.x, outDims.y);
	if (any(id.xy >= outDims))
		return;
	uint2 inDims;
	DeformationMap.GetDimensions(inDims.x, inDims.y);

	float2 stride = float2(inDims) / float2(outDims);
	float2 center = (float2(id.xy) + 0.5) * stride - 0.5;
	// sigma = 1.5 berm texels (~24 units); nearest-texel rounding of the
	// tap positions is well under the kernel's resolving power.
	const float invTwoSigma2 = 1.0 / (2.0 * 1.5 * 1.5);
	float sum = 0.0;
	float wsum = 0.0;
	[unroll] for (int dy = -3; dy <= 3; dy++)
	{
		[unroll] for (int dx = -3; dx <= 3; dx++)
		{
			float w = exp(-(float)(dx * dx + dy * dy) * invTwoSigma2);
			int2 p = clamp(int2(round(center + float2(dx, dy) * stride)), int2(0, 0), int2(inDims) - 1);
			sum += DeformationMap.Load(int3(p, 0)) * w;
			wsum += w;
		}
	}
	BermField[id.xy] = saturate(sum / wsum);
}
