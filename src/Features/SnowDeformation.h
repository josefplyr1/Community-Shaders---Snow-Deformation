#pragma once

#include "Buffer.h"

struct SnowDeformation : Feature
{
public:
	virtual inline std::string GetName() override { return "Snow Deformation"; }
	virtual std::string GetDisplayName() override { return T("feature.snow_deformation.name", "Snow Deformation"); }
	virtual inline std::string GetShortName() override { return "SnowDeformation"; }
	virtual inline std::string_view GetShaderDefineName() override { return "SNOW_DEFORMATION"; }
	virtual std::string_view GetCategory() const override { return FeatureCategories::kLandscapeAndTextures; }

	/** @brief Lighting samples the deformation map; Utility carries the statics-shell displacement so depth prepasses stay in sync with the displaced color pass. */
	virtual bool HasShaderDefine(RE::BSShader::Type shaderType) override
	{
		return shaderType == RE::BSShader::Type::Lighting || shaderType == RE::BSShader::Type::Utility;
	}

	virtual std::pair<std::string, std::vector<std::string>> GetFeatureSummary() override
	{
		return { T("feature.snow_deformation.description", "Maintains a persistent deformation map around the player so snow can be visibly compressed by actors moving through it, leaving lasting trails."),
			{ T("feature.snow_deformation.key_feature_1", "Persistent world-space deformation map following the player"),
				T("feature.snow_deformation.key_feature_2", "Trails carved by the player, NPCs and creatures"),
				T("feature.snow_deformation.key_feature_3", "Configurable snow refill over time"),
				T("feature.snow_deformation.key_feature_4", "Compute-shader based, low performance impact") } };
	};

	// The deformation map is a square world-space window that follows the camera
	// in whole-texel steps. Values are normalized depression depth: 0 = untouched
	// snow, 1 = compressed to the ground.
	static constexpr uint kTextureDim = 2048;
	static constexpr float kWorldSize = 8192.0f;
	static constexpr float kTexelSize = kWorldSize / kTextureDim;
	static constexpr uint kMaxStamps = 128;

	// Per-texture-class depth: landscape mods retexture the same vanilla LTEX
	// files, so classes match on diffuse filename substrings — FIRST match
	// wins, so more specific names must precede their substrings (grasssnow
	// before snow01: "grasssnow01" contains both). EVERY texture classifies:
	// unmatched snow-material textures fall to the "Snow 01" class, anything
	// else to "Other" (default -5 = submerged, giving users full control).
	static constexpr uint kSnowClassCount = 12;
	// Classes below this index count as snow for the terrain-shader mask bits.
	static constexpr uint kSnowOnlyClassCount = 5;
	struct SnowClassDef
	{
		const char* label;
		const char* match;
		float defaultDepth;
	};
	static constexpr SnowClassDef kSnowClasses[kSnowClassCount] = {
		{ "Grass Snow", "grasssnow", 14.0f },
		{ "Trodden Path", "snowpath", 18.0f },
		{ "Snowy Rocks", "snowrocks", 26.0f },
		{ "Snow 01", "snow01", 30.0f },
		{ "Snow 02", "snow02", 30.0f },
		{ "Roads", "road", -5.0f },
		{ "Dirt", "dirt", -5.0f },
		{ "Grass & Fields", "grass", -5.0f },
		{ "Rocks & Cliffs", "rock", -5.0f },
		{ "Coast & Beach", "coast", -5.0f },
		{ "Mud & Rivers", "mud", -5.0f },
		{ "Other", "", -5.0f },
	};

	struct Settings
	{
		bool EnableSnowDeformation = true;
		/** @brief Scale on Havok collision-shape radii (20 = 1.0x = the shapes' actual size, Josef-tuned default). */
		float StampRadius = 20.0f;
		float RefillTime = 700.0f;
		/** @brief When set (default), compressed snow only recovers while the current weather is actually snowing — trails persist through clear spells and interiors. */
		bool RefillOnlyWhenSnowing = true;
		bool ShowDebugTexture = false;

		float DeformationDepth = 30.0f;
		float RoadSnowDepth = -5.0f;
		/** @brief Per-class shell depths, indexed like kSnowClasses (defaults duplicated from the table). */
		std::array<float, kSnowClassCount> SnowClassDepths = { 14.0f, 18.0f, 26.0f, 30.0f, 30.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f };
		/** @brief S6 statics skin, FLAT class: layer height on flat split-normal meshes (walkways, roofs, planks) — classified per mesh on the GPU by smoothed-vs-raw normal divergence. These get COMPLETELY FLAT snow (straight-up offset, raw shading normal, no pillow). Default 0: painted directly onto the surface — even 1 unit reads as a tiny hover (Josef-tuned). */
		float ObjectsSnowDepth = 0.0f;
		/** @brief S6 statics skin, ROUNDED class: layer height on organically smooth meshes (rocks, drifts, logs), where pillow inflation reads correctly. */
		float SnowMeshesDepth = 5.0f;
		/** @brief Model-class override: ROAD MESHES (matched deterministically by road/bridge texture paths, not geometry stats) wear their own deep layer — carved by the trench patch into the deep wagon-rut look. */
		float RoadMeshesDepth = 25.0f;
		/** @brief Shell albedo texture, loaded through the VFS. User-editable so the shell can be matched to the modlist's snow by eye. The loader resolves PBR companion maps and falls back to the legacy path when the PBR set is absent. */
		std::string SnowTexturePath = "Textures\\PBR\\Landscape\\snow01.dds";
		/** @brief Set when the texture stores linear (PBR) color; the shader converts to the pipeline's gamma space. Auto-enabled when a PBR set is resolved — only matters for legacy textures. */
		bool SnowTextureLinear = false;
		/** @brief Angle-of-repose slope for blanket mounds (rise per world unit; 1.0 = 45°). Steeper = lifted snow clings tighter: narrow banks against cliffs instead of broad aprons, juttier rock mounds. */
		float SnowMoundSteepness = 1.0f;
		/** @brief World-unit jitter of WHERE class-depth borders fall (domain warp), so snow edges never trace the texture seam. */
		float SnowBorderNoise = 32.0f;
		/** @brief World-unit radius widening the depth ramp between neighboring classes — deep snow meets shallow ground in a slope, not a ravine wall. */
		float SnowBorderSmoothness = 32.0f;
		/** @brief How far from a class border trampled snow keeps its visibility override — beyond ~20 the landscape under trenches becomes too visible (Josef-tuned). */
		float SnowBorderTrampledFade = 20.0f;
		/** @brief Depth band (units) over which untrampled snow's edge dissolves at class borders. */
		float SnowBorderUntrampledFade = 5.0f;
		/** @brief View-ray band (units) over which the OBJECT snow skin cross-fades into the LANDSCAPE shell behind it — kills the hard seam between the two snow kinds. 15 = Josef-tuned sweet spot. */
		float SnowSnowFade = 15.0f;
		/** @brief Render distances in METERS (converted via kUnitsPerMeter). Shell scales the warped grid spacing (CB-only, live); trenches resize the deformation window (map clears on change); skins is a plain capture cutoff. */
		float RangeShellM = 375.0f;
		float RangeTrenchesM = 100.0f;
		float RangeSkinsM = 750.0f;
		/** @brief Distance (m) where the object-snow skin STARTS dissolving; fully gone at the Object Snow range end. Cures distant blank-white objects by blending them back to their own material. */
		float RangeSkinsFadeM = 100.0f;
		float TrailDarkening = 0.6f;
		float TrailAOStrength = 0.5f;
		float NormalStrength = 1.0f;
		// Fake parallax trail depth on the landscape shader — obsolete now
		// that the shell carves real geometry; kept as an opt-in for users
		// running without the shell.
		float ParallaxDepth = 0.0f;
	};

	/** @brief GPU-side settings, appended to the shared FeatureData cbuffer (b6). Layout must match SnowDeformationSettings in SharedData.hlsli. */
	struct alignas(16) SettingsGPU
	{
		float2 WindowOrigin;
		float InvWorldSize;
		float DeformationDepth;

		uint EnableSnowDeformation;
		float ObjectsSnowDepth;
		float TrailAOStrength;
		float NormalStrength;

		uint DebugTerrainOverlay;
		float ParallaxDepth;
		float2 WindowOriginRelCam;
	};
	STATIC_ASSERT_ALIGNAS_16(SettingsGPU);

	/**
	 * @brief Returns this frame's GPU settings for the shared FeatureData buffer.
	 *
	 * Also advances the deformation window when a_inWorld is true: the origin
	 * MUST be computed here (during State::UpdateSharedData, before Prepass)
	 * so the constant buffer and the scrolled texture agree within a frame —
	 * computing it in Prepass made trails swim on window-scroll frames.
	 */
	SettingsGPU GetCommonBufferData(bool a_inWorld);

	struct alignas(16) PerFrame
	{
		float2 WindowOrigin;
		DirectX::XMINT2 ScrollDelta;

		float TexelSize;
		uint StampCount;
		float RefillAmount;
		uint ClearMap;

		float4 Stamps[kMaxStamps];
		/** @brief Segment start per stamp (the actor's previous stamped position) — stamps are capsules, so trails stay continuous regardless of actor speed. */
		float4 StampEnds[kMaxStamps];
	};
	STATIC_ASSERT_ALIGNAS_16(PerFrame);

	Settings settings;

	ConstantBuffer* perFrame = nullptr;
	Texture2D* deformationTextures[2] = { nullptr, nullptr };
	uint currentTexture = 0;

	/** @brief SRV of the most recently written deformation map, for shader sampling and debug UI. */
	ID3D11ShaderResourceView* GetDeformationSRV() const { return deformationTextures[currentTexture]->srv.get(); }
	/** @brief World XY of the corner of texel (0,0) of the current deformation window. */
	float2 GetWindowOrigin() const { return windowOrigin; }

	/** @brief Returns the deformation update compute shader, compiling it on first use. */
	ID3D11ComputeShader* GetDeformationUpdateCS();
	ID3D11ComputeShader* deformationUpdateCS = nullptr;
	/** @brief Returns the depth sync compute shader (shell depth -> Terrain Blending's blended depth copies), compiling it on first use. */
	ID3D11ComputeShader* GetDepthSyncCS();
	ID3D11ComputeShader* depthSyncCS = nullptr;
	virtual void ClearShaderCache() override;

	// ---- Snow shell (S2: terrain-conforming) ----

	static constexpr uint kShellGridDim = 640;
	// 8-unit inner spacing: trench walls are ~10-unit features carved from
	// 4-unit deformation texels — 16-unit vertices undersampled them into
	// blocky silhouettes no amount of data smoothing could fix.
	static constexpr float kShellGridSpacing = 8.0f;

	// Distance warp: the inner kShellWarpInnerVerts vertices per side keep
	// linear kShellGridSpacing; beyond them each ring's spacing grows by
	// kShellWarpGrowth, stretching the grid to ~26k units half-span so the
	// shell continues into the distance. Constants MUST match the WarpAxis()
	// constants in SnowShell.hlsl.
	static constexpr float kShellWarpInnerVerts = 256.0f;
	static constexpr float kShellWarpGrowth = 1.0902f;

	/** @brief World half-span of the warped shell grid (center to edge) at a given inner spacing. Linear in spacing: the range slider scales spacing, the warp shape is unchanged. */
	static float ShellWarpedHalfSpan(float a_spacing = kShellGridSpacing)
	{
		const float outerVerts = kShellGridDim * 0.5f - kShellWarpInnerVerts;
		const float outer = kShellWarpGrowth * (std::pow(kShellWarpGrowth, outerVerts) - 1.0f) / (kShellWarpGrowth - 1.0f);
		return (kShellWarpInnerVerts + outer) * a_spacing;
	}

	/** @brief Skyrim world units per meter (1 unit ≈ 1.43 cm). Range sliders are in meters. */
	static constexpr float kUnitsPerMeter = 70.0f;

	// Terrain data window: 16x16 cells at land-vertex resolution (128 units)
	// — sized so the warped grid never samples past the window even with the
	// camera at a cell edge.
	static constexpr int kShellWindowDim = 1024;
	static constexpr float kShellVertexSpacing = 128.0f;
	static constexpr int kShellTexelsPerCell = 32;
	static constexpr int kShellWindowCells = kShellWindowDim / kShellTexelsPerCell;

	/** @brief Baked per-cell terrain data: 33x33 vertex heights (absolute Z) plus per-class coverage weights (0-255). Depths are applied at window-rebuild time so class sliders retune without a game re-bake. */
	struct ShellCellData
	{
		std::array<float, 33 * 33> height;
		std::array<std::array<uint8_t, 33 * 33>, kSnowClassCount> classWeights;
	};

	/** @brief Per-draw constants for the shell pass. Layout must match ShellCB in SnowShell.hlsl. */
	struct alignas(16) ShellCB
	{
		Matrix CameraViewProj;
		Matrix CameraViewProjUnjittered;
		Matrix CameraPreviousViewProjUnjittered;
		Matrix CameraView;

		float4 CameraPosAdjust;
		float4 CameraPreviousPosAdjust;

		float2 GridOrigin;
		float GridSpacing;
		float SnowDepth;

		// Precomputed on CPU so all shader-side field sampling happens in
		// small grid-local coordinates (absolute world XY at ~1e5 magnitude
		// destroys float32 finite differences → shimmering normals).
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

		float4 SnowGlintParams;

		float SnowSpecularLevel;
		float BorderNoise;
		float BorderSmooth;
		float BorderTrampledFade;

		float BorderUntrampledFade;
		float SnowSnowFade;
		/** @brief Camera-distance band (world units) over which the statics skin dissolves back to the object's own material — start of the fade and the hard end (the skins capture range). */
		float SkinFadeStart;
		float SkinFadeEnd;

		/** @brief Slice-2 LOD shadow cascade: distant LOD trees/objects render into it, but the shared 2-cascade DirectionalShadowLights buffer omits it — bare ground kept their shadows and the shell lost them. Stored column-major (XMStoreFloat4x4), matching Deferred's convention. */
		DirectX::XMFLOAT4X4 LodShadowProj;
		float LodShadowEnd;
		float LodShadowActive;
		/** @brief Screen-Space Shadows texture (t45) is bound and valid: the depth-marched long-range shadows that carry distant LOD tree shadows beyond the two cascades. Bare ground multiplies them into its lighting, so the shell must too. */
		float ScreenSpaceShadowsActive;
		float padLod;
	};
	STATIC_ASSERT_ALIGNAS_16(ShellCB);

	/**
	 * @brief Draws the snow shell into the deferred G-buffer.
	 *
	 * Called from Deferred::DeferredPasses before the composite, while the
	 * frame's G-buffer contents and main depth are complete. Binds its own
	 * targets/state and restores the previous pipeline state afterwards.
	 */
	void DrawShell();

	ID3D11VertexShader* GetShellVS();
	ID3D11PixelShader* GetShellPS();
	/** @brief SNOW_SHADOW_CAST variant: flattens the base layer so only excess height (mounds, drifts, lifts) casts. */
	ID3D11VertexShader* GetShellShadowVS();
	ID3D11VertexShader* shellVS = nullptr;
	ID3D11VertexShader* shellShadowVS = nullptr;
	ID3D11PixelShader* shellPS = nullptr;

	ConstantBuffer* shellCB = nullptr;
	winrt::com_ptr<ID3D11RasterizerState> shellRasterState;
	winrt::com_ptr<ID3D11DepthStencilState> shellDepthState;

	/** @brief Diagnostic: lifts EVERY (non-excluded) object by ObjectsSnowDepth, bypassing the snow gates — separates a broken VS data path from a wrong gate. */
	bool debugStaticsLift = false;
	/** @brief Renders the shell as an always-visible plane colored by the sampled terrain data (red=height, green=snowness). */
	bool shellDataDebug = false;

	/** @brief The terrain data window texture (R32G32: height, snowness). */
	Texture2D* shellTerrainTexture = nullptr;

	/** @brief The landscape snow diffuse, loaded from the modlist via the VFS ("textures/landscape/snow01.dds"). Null when unavailable (constant-albedo fallback). */
	winrt::com_ptr<ID3D11ShaderResourceView> shellSnowDiffuseSRV;
	/** @brief TruePBR companion maps, auto-resolved by probing the Textures\PBR\ variant of the snow path: tangent normals (_n) and roughness/metal/AO/spec (_rmaos). Their presence also auto-selects linear color — the manual Linear checkbox only matters for legacy textures. */
	winrt::com_ptr<ID3D11ShaderResourceView> shellSnowNormalSRV;
	winrt::com_ptr<ID3D11ShaderResourceView> shellSnowRmaosSRV;
	bool shellSnowTextureIsPBR = false;
	bool shellSnowTextureAttempted = false;

	/** @brief Material parameters fetched from the modlist's own TruePBR config JSON (PBRTextureSets\, matched by texture basename) so the shell's sparkle and response follow whatever texture set the user runs. Defaults mirror common authored snow values. */
	float snowGlintLogDensity = 6.0f;
	float snowGlintMicroRoughness = 0.3f;
	float snowGlintDensityRandomization = 5.0f;
	float snowGlintScreenSpaceScale = 1.0f;
	float snowRoughnessScale = 0.7f;
	float snowSpecularLevel = 0.02f;
	winrt::com_ptr<ID3D11SamplerState> shellSnowSampler;
	winrt::com_ptr<ID3D11SamplerState> shellLinearSampler;

	/** @brief Full-resolution COPIES of the game's raw sun-shadow cascade atlas and its ESRAM partner, taken during the shadow-mask pass. Copies are mandatory: by deferred time the engine has reused the live targets (ESRAM is aliased scratch memory), and sampling them live produced garbage flicker. Taken BEFORE the shell is injected as a caster, so the shell's receiver path never sees itself (no self-shadow acne; the heightfield march owns shell-on-shell). */
	winrt::com_ptr<ID3D11Texture2D> shadowAtlasCopyTex;
	winrt::com_ptr<ID3D11ShaderResourceView> shadowAtlasCopySRV;
	winrt::com_ptr<ID3D11Texture2D> shadowEsramCopyTex;
	winrt::com_ptr<ID3D11ShaderResourceView> shadowEsramCopySRV;
	winrt::com_ptr<ID3D11SamplerState> shadowCmpSampler;

	/** @brief Depth copy taken AFTER the terrain shell draw (shell surface included), so the statics skin can measure its view-ray gap to the landscape shell — Terrain Blending's technique adapted to the two snow kinds. */
	winrt::com_ptr<ID3D11Texture2D> shellDepthCopyTex;
	winrt::com_ptr<ID3D11ShaderResourceView> shellDepthCopySRV;


	// ---- S6 statics shell: capture & redraw ----

	/** @brief One snow-flagged draw captured this frame, for re-rendering inflated in DeferredPasses. NiPointer keeps the geometry alive across the frame even if its cell detaches mid-frame. */
	struct CapturedSnowStatic
	{
		RE::NiPointer<RE::BSGeometry> geometry;
		RE::NiTransform world;
		/** @brief Road/bridge texture-path match: this capture uses RoadMeshesDepth for BOTH class depths, so the GPU class pick cannot override it. */
		bool road;
	};

	/** @brief Render-thread only: filled during opaque rendering by the SetupGeometry hook, consumed and cleared each frame. */
	std::vector<CapturedSnowStatic> capturedStatics;
	std::unordered_set<void*> capturedStaticsSet;
	std::atomic<uint32_t> statCapturedStatics{ 0 };

	// ---- LOD-shadow diagnostics (kept per Josef — negligible cost, useful history) ----
	/** @brief Cascade descriptor count the engine reports, the three end-split distances, whether the (retired) slice-2 path activated, and the copied atlas's slice count. */
	uint32_t dbgLodDescriptorCount = 0;
	float dbgLodEndSplits[3] = { 0.0f, 0.0f, 0.0f };
	float dbgLodActive = 0.0f;
	uint32_t dbgLodAtlasSlices = 0;

	// ---- Skin-trench design diagnostics: vertex density of captured meshes ----
	/** @brief When set, every UNIQUE captured skin mesh logs its name, vertex/triangle counts, world bound radius and estimated edge length once — the data that decides how object skins can carve REAL trench geometry (subdivision factor, tessellation budget). */
	bool debugLogSkinStats = false;
	std::unordered_set<const void*> skinStatsLogged;
	/** @brief Total skin vertices drawn last frame (the budget a subdivision approach would multiply). */
	uint32_t statSkinVertsDrawn = 0;

	// ---- Shell as shadow CASTER: injected into the live cascade atlas ----

	/** @brief Last frame's fully-computed ShellCB (heap-held: ShellCB is over-aligned and embedding it pads the class). The injection runs at the shadow-mask pass, before this frame's DrawShell recomputes the windows — one-frame-stale grid placement is invisible in a shadow. Null until the first DrawShell. */
	std::unique_ptr<ShellCB> lastShellCBData;

	// ---- Smoothed normals for the statics skin (pillow inflation) ----

	/** @brief GPU-side per-mesh CB for SmoothNormalsCS. Layout must match SmoothCB in SmoothNormalsCS.hlsl. */
	struct alignas(16) SmoothCB
	{
		uint32_t VertexCount;
		uint32_t StrideBytes;
		uint32_t NormalOffsetBytes;
		uint32_t PosIsFloat32;
		uint32_t TableMask;
		uint32_t padSm[3];
	};
	STATIC_ASSERT_ALIGNAS_16(SmoothCB);

	/** @brief Per unique geometry (keyed by vertex buffer pointer): position-averaged normals, built once by SmoothNormalsCS. Split-normal flat meshes (planks, roofs, pole caps) inflate along these so their snow drapes as a sealed pillow instead of a hovering parallel sheet. */
	struct SmoothedNormalsEntry
	{
		winrt::com_ptr<ID3D11Buffer> buffer;
		winrt::com_ptr<ID3D11ShaderResourceView> srv;
		bool ready = false;
	};
	std::unordered_map<void*, SmoothedNormalsEntry> smoothedNormalsCache;
	ID3D11ComputeShader* smoothAccumulateCS = nullptr;
	ID3D11ComputeShader* smoothResolveCS = nullptr;
	/** @brief Third pass: one-group reduction writing the mesh's flat/rounded classification (fraction of smoothed-vs-raw divergent vertices) into the stats element appended at SmoothedNormals[vertexCount]. */
	ID3D11ComputeShader* smoothFlatStatsCS = nullptr;
	ConstantBuffer* smoothCB = nullptr;

	/** @brief Builds (or returns) the smoothed-normal buffer for a captured geometry. Dispatches the two SmoothNormalsCS passes on first sight; cached thereafter. Returns null while unavailable (VS falls back to raw normals). */
	ID3D11ShaderResourceView* EnsureSmoothedNormals(RE::BSGeometry* a_geometry);

	/** @brief Per-cascade DSVs created on the LIVE atlas texture, cached by texture pointer (not owned — key only). */
	winrt::com_ptr<ID3D11DepthStencilView> shadowAtlasDSV[2];
	ID3D11Texture2D* shadowAtlasDSVTexture = nullptr;
	winrt::com_ptr<ID3D11RasterizerState> shadowCastRS;
	winrt::com_ptr<ID3D11DepthStencilState> shadowCastDSS;

	/** @brief Depth-renders the terrain shell and the statics skins into both live cascade slices so the world receives snow shadows. Called from CaptureShadowAtlas after the receiver copies are taken. */
	void InjectShellShadowCasters(ID3D11ShaderResourceView* a_atlasSRV);

	/** @brief Records projected-snow (kProjectedUV + kSnow) lighting draws for the statics shell. Called from the BSLightingShader::SetupGeometry hook. */
	void BSLightingShader_SetupGeometry(RE::BSRenderPass* a_pass);

	/** @brief Only statics within this range of the camera are captured (matches the deformation window's reach with margin). */
	static constexpr float kStaticsCaptureRange = 12288.0f;

	/** @brief Per-object constants for the statics shell and height capture. Layout must match StaticCB in SnowStaticsShell.hlsl AND SnowHeightCapture.hlsl. */
	struct alignas(16) StaticsCB
	{
		float4 WorldRow0;
		float4 WorldRow1;
		float4 WorldRow2;
		float ObjectsDepth;
		float2 HeightWindowCenter;
		float HeightHalfExtent;
		/** @brief >0.5: a smoothed-normal buffer is bound at VS t10 for this object (pillow inflation for flat meshes). */
		float HasSmoothedNormals;
		/** @brief ROUNDED-class depth (ObjectsDepth carries the FLAT-class depth); the VS picks per mesh from the GPU flatness stats. */
		float RoundedDepth;
		/** @brief Vertex count = index of the flatness-stats element appended to the SmoothedNormals buffer. */
		float VertexCountF;
		float padStat;
	};
	STATIC_ASSERT_ALIGNAS_16(StaticsCB);

	// ---- S7 unified blanket: object height map ----

	// Top-down height window: 4096 units at 8-unit texels, following the
	// camera — matches the shell's inner grid density.
	static constexpr uint kHeightMapDim = 512;
	static constexpr float kHeightMapHalfExtent = 2048.0f;
	/** @brief Object tops more than this far above the terrain do not lift the blanket (buildings must not become snow tents). */
	static constexpr float kObjectLiftCap = 150.0f;

	/** @brief Per-frame skin-depth raster (R16F, cleared each frame, MAX-blended): each captured mesh writes its class layer depth (flat 0-ish, rounded ~25) so the trench patch knows how thick the snow above any object top is. No scroll persistence — a missed frame degenerates patch verts for one frame, invisible. */
	Texture2D* heightSkinDepth = nullptr;

	/** @brief Trench patch: the landscape shell's dense-grid carve applied to OBJECT tops — real geometry where parallax could not notch silhouettes or hold floors angle-stably. */
	ID3D11VertexShader* patchVS = nullptr;
	ID3D11PixelShader* patchPS = nullptr;

	/** @brief Ping-pong accumulated raw maps (scrolled each frame, captures rasterized on top), the final slope-limited field + shelter mask the shell samples, and a cone-iteration scratch. Top empty = -100000, bottom empty = +100000. */
	Texture2D* heightTopRaw[2] = { nullptr, nullptr };
	Texture2D* heightBottomRaw[2] = { nullptr, nullptr };
	Texture2D* heightTopFiltered = nullptr;
	Texture2D* heightBottomFiltered = nullptr;
	Texture2D* heightScratch = nullptr;
	uint heightCurrent = 0;
	bool heightMapValid = false;

	/** @brief RT0 MAX (tops) + RT1 MIN (bottoms) in one raster pass. */
	winrt::com_ptr<ID3D11BlendState> heightMaxBlendState;
	ID3D11VertexShader* heightVS = nullptr;
	ID3D11PixelShader* heightPS = nullptr;
	ID3D11ComputeShader* heightScrollCS = nullptr;
	ID3D11ComputeShader* heightCombineCS = nullptr;
	ID3D11ComputeShader* heightConeCS = nullptr;

	struct alignas(16) HeightProcessCB
	{
		DirectX::XMINT2 ScrollDelta;
		uint ClearAll;
		uint ConeStep;

		float2 HeightWindowCenter;
		float HeightHalfExtent;
		float SlopePerUnit;

		float2 TerrainWindowOrigin;
		float TerrainTexelSize;
		uint TerrainDim;

		/** @brief Units/frame the accumulated tops/bottoms drift toward empty — stale object imprints (disabled/moved/harvested) melt instead of persisting until scrolled out. Always on (0.5). */
		float GhostDecay;
		/** @brief Deformation-map addressing for the corpse-mound refill gate (same mapping SampleDeformation uses). */
		float2 DeformWindowOriginH;
		float DeformInvWorldSizeH;

		/** @brief Dead actors at rest, as collision spheres (xyz world center, w radius): CombineCS raises capped snow mounds over them, gated by local refill. */
		uint32_t CorpseSphereCount;
		float CorpseMoundCap;
		float2 padH;
		float4 CorpseSpheres[32];
	};
	STATIC_ASSERT_ALIGNAS_16(HeightProcessCB);
	ConstantBuffer* heightProcessCB = nullptr;

	/** @brief Exclusion zones: doors (elliptical, stretched along facing; load doors get larger zones — cave/mine entrances) and campfires (noisy-edged full clears). The cone transform re-slopes surrounding mounds into every clearing at the angle of repose (no ravine walls). */
	static constexpr uint kMaxExclusions = 96;
	static constexpr float kDoorClearRadius = 110.0f;
	static constexpr float kDoorForwardExtent = 70.0f;
	static constexpr float kLoadDoorClearRadius = 150.0f;
	static constexpr float kLoadDoorForwardExtent = 150.0f;
	static constexpr float kFireClearRadius = 70.0f;
	struct alignas(16) ExclusionsCB
	{
		float4 PosRadius[kMaxExclusions];    // xyz = position, w = radius
		float4 DirExtType[kMaxExclusions];   // xy = facing dir, z = forward extent, w = type (0 door, 1 fire)
		uint ExclusionCount;
		float pad[3];
	};
	STATIC_ASSERT_ALIGNAS_16(ExclusionsCB);
	ConstantBuffer* doorsCB = nullptr;
	uint32_t doorRefreshCounter = 0;

	float2 heightWindowCenter = { 0, 0 };

	/** @brief Scrolls the accumulated height maps to the new window, rasterizes this frame's captured statics into them (MAX/MIN), and erodes thin features into the filtered maps the shell samples. */
	void RenderObjectHeightMap();

	ID3D11VertexShader* staticsVS = nullptr;
	ID3D11PixelShader* staticsPS = nullptr;
	/** @brief Retained VS bytecode: input layouts are created against it, one per vertex descriptor. */
	winrt::com_ptr<ID3DBlob> staticsVSBlob;
	bool staticsShadersFailed = false;
	ConstantBuffer* staticsCB = nullptr;
	std::unordered_map<uint64_t, winrt::com_ptr<ID3D11InputLayout>> staticsILCache;

	/** @brief Compiles the statics shell VS (keeping bytecode) and PS on first use. */
	bool EnsureStaticsShaders();
	/** @brief Re-draws this frame's captured projected-snow statics inflated, inside DrawShell's bound state. */
	void DrawCapturedStatics();

	/** @brief Bakes one cell's heights and per-vertex snow coverage from LoadedLandData. Called from the TESObjectLAND hook. */
	void BakeShellCell(RE::TESObjectLAND* land);

	/** @brief Recenters and re-uploads the terrain data window when the camera crosses cells or new cells were baked. Called from Prepass. */
	void UpdateShellTerrainWindow();

private:
	std::unordered_map<uint64_t, ShellCellData> shellCells;
	std::shared_mutex shellCellMutex;
	std::atomic<bool> shellDataDirty{ true };
	int shellWindowCellX = INT_MIN;
	int shellWindowCellY = INT_MIN;
	std::vector<float> shellUploadScratch;

	// Diagnostics for the settings UI (written on window rebuild).
	uint32_t shellStatCellsInWindow = 0;
	uint32_t shellStatSnowTexels = 0;
	float shellStatMinHeight = 0.0f;
	float shellStatMaxHeight = 0.0f;

public:
	/** @brief Thread-safe count of baked cells, for the settings UI. */
	size_t ShellCellCountForUI()
	{
		const std::shared_lock lock(shellCellMutex);
		return shellCells.size();
	}

	/** @brief Creates the ping-pong deformation textures and the per-frame constant buffer. */
	virtual void SetupResources() override;

	/** @brief Draws the ImGui settings UI, including the debug view of the deformation map. */
	virtual void DrawSettings() override;

	/**
	 * @brief Per-frame update: gathers actor stamp positions, scrolls the window
	 * to follow the camera, and dispatches the deformation update compute shader.
	 */
	virtual void Prepass() override;

	/** @brief Called from State::UpdateSharedData while the game renders the shadow MASK (Utility shader, RenderShadowmask) — the only point where PS t4 genuinely holds the sun cascade atlas. Copies it (and the ESRAM partner) for crisp shell shadows. Same trigger VolumetricShadows and Skylighting use. */
	void CaptureShadowAtlas();

	virtual void LoadSettings(json& o_json) override;
	virtual void SaveSettings(json& o_json) override;
	virtual void RestoreDefaultSettings() override;

	/** @brief Installs both hooks. The TESObjectLAND detour is installed here (after TruePBR's, which runs first in the feature list) so ours is OUTER and sees the final — possibly TruePBR-replaced — material on each quad. */
	virtual void PostPostLoad() override;

	/** @brief Caches a 6-bit "tile is snow material" mask per landscape quad, keyed by the vanilla material hashKey. */
	void TESObjectLAND_SetupMaterial(RE::TESObjectLAND* land);
	/** @brief Publishes the cached snow mask for the material about to be drawn via ExtraFeatureDescriptor bits 10-15. */
	void BSLightingShader_SetupMaterial(RE::BSLightingShaderMaterialBase const* material);

private:
	/** @brief Fills perFrameData.Stamps from the player and nearby loaded actors. */
	void GatherStamps(PerFrame& perFrameData);

	float2 windowOrigin = { 0, 0 };
	DirectX::XMINT2 pendingScrollDelta = { 0, 0 };
	bool clearRequested = true;

	/** @brief Trail history per collision shape: key = (formID << 16) | traversal index. */
	std::unordered_map<uint64_t, float2> stampPrevPositions;
	/** @brief Rebuilt each frame in GatherStamps: resting dead actors' collision spheres, consumed by CombineCS as capped snow mounds (buried-corpse bumps). */
	std::vector<float4> corpseMoundSpheres;

	/** @brief Stillness latch per corpse (formID). Ragdoll micro-drift accumulates against the FROZEN resting anchors and would eventually cross the 3-unit movement gate, firing a one-frame trench pulse under an already-buried corpse. Once a corpse has been still long enough it settles: only a large accumulated displacement (real dragging, explosions) wakes it again. Erased when the actor is seen alive (reanimation). */
	struct CorpseRest
	{
		uint16_t stillFrames = 0;
		bool settled = false;
	};
	std::unordered_map<uint32_t, CorpseRest> corpseRestStates;

	/** @brief Last reference position per loose inanimate object (formID), rebuilt every frame from the in-range scan. Props carve only while their REFERENCE moves — the anchor refreshes every frame, so collision traversal never runs for resting clutter and micro-creep can't accumulate into stamp pulses. */
	std::unordered_map<uint32_t, RE::NiPoint3> propPrevPositions;

	// ---- Runtime render-distance state (driven by the Range* settings) ----
	/** @brief Deformation window world size (2x trench range). Changing it invalidates the map (content is scale-relative), so the trench slider clears on apply. */
	float deformWorldSize = 14000.0f;
	/** @brief Height-field dimensions/extent (fixed near-camera window: shelter, exclusions, corpse mounds — the object-blanket lift is gone, so no range setting drives these anymore). */
	uint32_t heightMapDim = 512;
	float heightHalfExtent = 2048.0f;
	bool trenchRangeDirty = false;
	bool rangeInitApplied = false;

	/** @brief (Re)creates the 7 height-field textures at heightMapDim. Safe to call at runtime (Prepass, before the height pipeline runs). */
	void CreateHeightFieldResources();
	/** @brief Applies pending range-setting changes: trench window resize (+map clear) and height-field recreation. Called at Prepass start; first call applies loaded settings. */
	void ApplyRangeSettings();

	std::unordered_map<uintptr_t, uint8_t> snowMasks;
	std::shared_mutex snowMaskMutex;

public:
	/** @brief Diagnostics shown in the settings UI: landscape materials that hit/missed the snow-mask cache at draw time. */
	std::atomic<uint64_t> landMaskHits{ 0 };
	std::atomic<uint64_t> landMaskMisses{ 0 };

	/** @brief Thread-safe size read for the settings UI. */
	size_t snowMasksSizeForUI()
	{
		const std::shared_lock lock(snowMaskMutex);
		return snowMasks.size();
	}

public:
	/** @brief Runtime-only diagnostic toggle; not persisted in settings JSON. */
	bool debugTerrainOverlay = false;
};
