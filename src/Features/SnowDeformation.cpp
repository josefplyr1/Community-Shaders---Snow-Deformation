#include "SnowDeformation.h"

#include <DDSTextureLoader.h>
#include <d3dcompiler.h>
#include <imgui_stdlib.h>

#include "Deferred.h"
#include "Features/ScreenSpaceShadows.h"
#include "Features/TerrainBlending.h"
#include "Globals.h"
#include "I18n/I18n.h"
#include "State.h"
#include "TruePBR.h"
#include "Utils/ActorUtils.h"
#include "Utils/D3D.h"
#include "Utils/Game.h"
#include "Utils/UI.h"

#define I18N_KEY_PREFIX "feature.snow_deformation."

static constexpr float MAX_ACTOR_DISTANCE = 4096.0f;
static constexpr float MAX_ACTOR_SQ_DISTANCE = MAX_ACTOR_DISTANCE * MAX_ACTOR_DISTANCE;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
	SnowDeformation::Settings,
	EnableSnowDeformation,
	StampRadius,
	RefillTime,
	RefillOnlyWhenSnowing,
	DeformationDepth,
	RoadSnowDepth,
	SnowClassDepths,
	ObjectsSnowDepth,
	SnowMeshesDepth,
	SnowTexturePath,
	SnowTextureLinear,
	SnowMoundSteepness,
	SnowBorderNoise,
	SnowBorderSmoothness,
	SnowBorderTrampledFade,
	SnowBorderUntrampledFade,
	SnowSnowFade,
	RangeShellM,
	RangeTrenchesM,
	RangeSkinsM,
	RangeSkinsFadeM,
	TrailDarkening,
	TrailAOStrength,
	NormalStrength,
	ParallaxDepth)

void SnowDeformation::SetupResources()
{
	perFrame = new ConstantBuffer(ConstantBufferDesc<PerFrame>());

	D3D11_TEXTURE2D_DESC texDesc = {
		.Width = kTextureDim,
		.Height = kTextureDim,
		.MipLevels = 1,
		.ArraySize = 1,
		.Format = DXGI_FORMAT_R16_FLOAT,
		.SampleDesc = { .Count = 1 },
		.Usage = D3D11_USAGE_DEFAULT,
		.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS
	};

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {
		.Format = texDesc.Format,
		.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
		.Texture2D = {
			.MostDetailedMip = 0,
			.MipLevels = 1 }
	};

	D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {
		.Format = texDesc.Format,
		.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D,
		.Texture2D = { .MipSlice = 0 }
	};

	for (uint i = 0; i < 2; i++) {
		deformationTextures[i] = new Texture2D(texDesc);
		deformationTextures[i]->CreateSRV(srvDesc);
		deformationTextures[i]->CreateUAV(uavDesc);
	}

	shellCB = new ConstantBuffer(ConstantBufferDesc<ShellCB>());

	auto device = globals::d3d::device;

	D3D11_RASTERIZER_DESC rasterDesc{};
	rasterDesc.FillMode = D3D11_FILL_SOLID;
	rasterDesc.CullMode = D3D11_CULL_NONE;
	rasterDesc.DepthClipEnable = TRUE;
	DX::ThrowIfFailed(device->CreateRasterizerState(&rasterDesc, shellRasterState.put()));
	Util::SetResourceName(shellRasterState.get(), "SnowDeformation::ShellRasterState");

	D3D11_DEPTH_STENCIL_DESC depthDesc{};
	depthDesc.DepthEnable = TRUE;
	depthDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	depthDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
	DX::ThrowIfFailed(device->CreateDepthStencilState(&depthDesc, shellDepthState.put()));
	Util::SetResourceName(shellDepthState.get(), "SnowDeformation::ShellDepthState");

	D3D11_SAMPLER_DESC samplerDesc{};
	samplerDesc.Filter = D3D11_FILTER_ANISOTROPIC;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.MaxAnisotropy = 8;
	samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
	DX::ThrowIfFailed(device->CreateSamplerState(&samplerDesc, shellSnowSampler.put()));
	Util::SetResourceName(shellSnowSampler.get(), "SnowDeformation::ShellSnowSampler");

	D3D11_SAMPLER_DESC linearDesc{};
	linearDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	linearDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	linearDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	linearDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	linearDesc.MaxLOD = D3D11_FLOAT32_MAX;
	DX::ThrowIfFailed(device->CreateSamplerState(&linearDesc, shellLinearSampler.put()));
	Util::SetResourceName(shellLinearSampler.get(), "SnowDeformation::ShellLinearSampler");

	{
		D3D11_TEXTURE2D_DESC terrainDesc = {
			.Width = kShellWindowDim,
			.Height = kShellWindowDim,
			.MipLevels = 1,
			.ArraySize = 1,
			.Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
			.SampleDesc = { .Count = 1 },
			.Usage = D3D11_USAGE_DEFAULT,
			.BindFlags = D3D11_BIND_SHADER_RESOURCE
		};

		D3D11_SHADER_RESOURCE_VIEW_DESC terrainSrvDesc = {
			.Format = terrainDesc.Format,
			.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
			.Texture2D = {
				.MostDetailedMip = 0,
				.MipLevels = 1 }
		};

		shellTerrainTexture = new Texture2D(terrainDesc);
		shellTerrainTexture->CreateSRV(terrainSrvDesc);
	}

	staticsCB = new ConstantBuffer(ConstantBufferDesc<StaticsCB>());

	CreateHeightFieldResources();

	{
		// RT0 MAX (tops) + RT1 MIN (bottoms): highest and lowest surfaces
		// win per texel in any draw order — no depth buffer needed.
		D3D11_BLEND_DESC minmaxBlendDesc{};
		minmaxBlendDesc.IndependentBlendEnable = TRUE;
		for (int i = 0; i < 2; i++) {
			minmaxBlendDesc.RenderTarget[i].BlendEnable = TRUE;
			minmaxBlendDesc.RenderTarget[i].SrcBlend = D3D11_BLEND_ONE;
			minmaxBlendDesc.RenderTarget[i].DestBlend = D3D11_BLEND_ONE;
			minmaxBlendDesc.RenderTarget[i].BlendOp = i == 0 ? D3D11_BLEND_OP_MAX : D3D11_BLEND_OP_MIN;
			minmaxBlendDesc.RenderTarget[i].SrcBlendAlpha = D3D11_BLEND_ONE;
			minmaxBlendDesc.RenderTarget[i].DestBlendAlpha = D3D11_BLEND_ONE;
			minmaxBlendDesc.RenderTarget[i].BlendOpAlpha = D3D11_BLEND_OP_MAX;
			minmaxBlendDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED;
		}
		DX::ThrowIfFailed(globals::d3d::device->CreateBlendState(&minmaxBlendDesc, heightMaxBlendState.put()));
		Util::SetResourceName(heightMaxBlendState.get(), "SnowDeformation::HeightMinMaxBlend");

		heightProcessCB = new ConstantBuffer(ConstantBufferDesc<HeightProcessCB>());
		doorsCB = new ConstantBuffer(ConstantBufferDesc<ExclusionsCB>());
		smoothCB = new ConstantBuffer(ConstantBufferDesc<SmoothCB>());
	}
}

void SnowDeformation::DrawSettings()
{
	if (ImGui::TreeNodeEx(T(TKEY("snow_deformation"), "Snow Deformation"), ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox(T(TKEY("enable"), "Enable Snow Deformation"), &settings.EnableSnowDeformation);

		ImGui::InputText(T(TKEY("snow_texture_path"), "Shell Snow Texture"), &settings.SnowTexturePath);
		if (auto _ttTex = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("snow_texture_path_tooltip"), "DDS path (relative to Data) for the shell's snow diffuse. Point it at the modlist's snow texture, then press Reload."));
		ImGui::SameLine();
		if (ImGui::Button(T(TKEY("reload_texture"), "Reload"))) {
			shellSnowDiffuseSRV = nullptr;
			shellSnowNormalSRV = nullptr;
			shellSnowRmaosSRV = nullptr;
			shellSnowTextureIsPBR = false;
			shellSnowTextureAttempted = false;
		}
		ImGui::Checkbox(T(TKEY("snow_texture_linear"), "Linear (PBR) Texture"), &settings.SnowTextureLinear);
		if (auto _ttLin = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("snow_texture_linear_tooltip"), "Legacy override: enable when a NON-PBR texture stores linear color. When a PBR set is auto-resolved (Textures\\PBR\\...), linear color is detected automatically and this checkbox is ignored."));

		ImGui::SliderFloat(T(TKEY("stamp_radius"), "Stamp Radius"), &settings.StampRadius, 4.0f, 128.0f, "%.0f");
		if (auto _ttStamp = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("stamp_radius_tooltip"), "Scales the Havok collision-shape radii used for stamping (20 = the shapes' actual size). Stamps come from actors' real collision shapes — feet and legs carve individually."));
		ImGui::Checkbox(T(TKEY("refill_only_snowing"), "Refill Only While Snowing"), &settings.RefillOnlyWhenSnowing);
		if (auto _ttRefillSnow = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("refill_only_snowing_tooltip"), "Compressed snow only recovers while the current weather is snowing. Trails and trenches persist through clear weather."));
		ImGui::SliderFloat(T(TKEY("refill_time"), "Snow Refill Time"), &settings.RefillTime, 0.0f, 3600.0f, "%.0f s");
		if (auto _ttRefill = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("refill_time_tooltip"), "Time for compressed snow to fully recover. 0 disables refilling."));

		if (ImGui::TreeNodeEx(T(TKEY("class_depths"), "Snow Depth by Texture Class"), ImGuiTreeNodeFlags_Framed)) {
			if (auto _ttClasses = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("class_depths_tooltip"), "Shell height per snow texture family (classified by the vanilla LTEX filenames every retexture mod overrides). Negative values submerge the shell below the surface. Retunes live from cached data."));
			ImGui::SliderFloat(T(TKEY("objects_snow_depth"), "Objects (Flat Snow)"), &settings.ObjectsSnowDepth, 0.0f, 25.0f, "%.0f units");
			if (auto _ttObj = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("objects_snow_depth_tooltip"), "Snow layer on flat hard-edged meshes (walkways, roofs, planks) — these get a completely flat overlay, no fake 3D. Classified automatically per mesh."));
			ImGui::SliderFloat(T(TKEY("snow_meshes_depth"), "Objects (Rounded Snow)"), &settings.SnowMeshesDepth, 0.0f, 25.0f, "%.0f units");
			if (auto _ttMesh = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("snow_meshes_depth_tooltip"), "Snow layer on organically smooth meshes (rocks, drifts, logs), where the puffed pillow layer reads correctly in 3D."));
			bool classDepthsChanged = false;
			for (uint32_t classI = 0; classI < kSnowClassCount; ++classI)
				classDepthsChanged |= ImGui::SliderFloat(kSnowClasses[classI].label, &settings.SnowClassDepths[classI], -20.0f, 64.0f, "%.0f units");
			if (classDepthsChanged)
				shellDataDirty.store(true, std::memory_order_release);
			ImGui::TreePop();
		}

		if (ImGui::TreeNodeEx(T(TKEY("snow_borders"), "Snow Borders"), ImGuiTreeNodeFlags_Framed)) {
			if (auto _ttBorders = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("snow_borders_tooltip"), "How the shell behaves where two texture classes with different snow depths meet (deep snow next to mud, roads, coast...)."));
			ImGui::SliderFloat(T(TKEY("border_noise"), "Border Noise"), &settings.SnowBorderNoise, 0.0f, 64.0f, "%.0f units");
			if (auto _ttBn = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("border_noise_tooltip"), "Wiggles WHERE the depth border between neighboring texture classes falls, so snow edges wander organically instead of tracing the texture seam (same noise idea as door clearings)."));
			ImGui::SliderFloat(T(TKEY("border_smoothness"), "Border Smoothness"), &settings.SnowBorderSmoothness, 0.0f, 64.0f, "%.0f units");
			if (auto _ttBs = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("border_smoothness_tooltip"), "Widens the depth ramp between neighboring classes so deep snow meets shallow ground in a slope instead of a ravine wall."));
			ImGui::SliderFloat(T(TKEY("border_trampled_fade"), "Trampled Border Fade"), &settings.SnowBorderTrampledFade, 0.0f, 64.0f, "%.0f units");
			if (auto _ttTf = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("border_trampled_fade_tooltip"), "How gradually TRAMPLED snow (trench floors) blends out toward a class border, letting the ground beneath show through faintly. Too high and the landscape becomes too visible under trenches."));
			ImGui::SliderFloat(T(TKEY("border_untrampled_fade"), "Untrampled Border Fade"), &settings.SnowBorderUntrampledFade, 0.0f, 64.0f, "%.0f units");
			if (auto _ttUf = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("border_untrampled_fade_tooltip"), "How gradually UNTRAMPLED snow dissolves at a class border. Shorter = the pristine snow edge commits sooner."));
			ImGui::SliderFloat(T(TKEY("snow_snow_fade"), "Snow <-> Snow Fade"), &settings.SnowSnowFade, 0.0f, 64.0f, "%.0f units");
			if (auto _ttSs = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("snow_snow_fade_tooltip"), "Cross-fade between OBJECT snow and LANDSCAPE snow where their surfaces run close in height (road meshes, low platforms). Wider = the two snow kinds dither into each other instead of meeting at a hard seam."));
			ImGui::TreePop();
		}

		if (ImGui::TreeNodeEx(T(TKEY("snow_mounds"), "Snow Mounds"), ImGuiTreeNodeFlags_Framed)) {
			if (auto _ttMounds = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("snow_mounds_tooltip"), "How the blanket mounds over objects and buried corpses."));
			ImGui::SliderFloat(T(TKEY("mound_steepness"), "Mound Steepness"), &settings.SnowMoundSteepness, 0.5f, 3.0f, "%.1f");
			if (auto _ttSteep = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("mound_steepness_tooltip"), "Angle of repose for snow mounds over objects (1.0 = 45 degrees). Steeper = lifted snow clings tighter: narrow banks against cliffs instead of broad aprons, and juttier, less smoothed-over rocks."));
			ImGui::TreePop();
		}

		if (ImGui::TreeNodeEx(T(TKEY("render_distance"), "Render Distance"), ImGuiTreeNodeFlags_Framed)) {
			if (auto _ttRd = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("render_distance_tooltip"), "How far each snow system reaches. Higher = more VRAM and GPU cost — stress-test territory."));
			ImGui::SliderFloat(T(TKEY("range_shell"), "Snow Shell"), &settings.RangeShellM, 94.0f, 750.0f, "%.0f m");
			if (auto _ttRs = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("range_shell_tooltip"), "Warped-grid span. Applies live; near-field vertex density scales with range (8-unit spacing at 375 m)."));
			ImGui::SliderFloat(T(TKEY("range_trenches"), "Trenches"), &settings.RangeTrenchesM, 29.0f, 200.0f, "%.0f m");
			if (ImGui::IsItemDeactivatedAfterEdit())
				trenchRangeDirty = true;
			if (auto _ttRt = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("range_trenches_tooltip"), "Deformation window radius (also the actor stamping cutoff). Applying a change CLEARS existing trenches; trench detail coarsens with range."));
			ImGui::SliderFloat(T(TKEY("range_skins"), "Object Snow"), &settings.RangeSkinsM, 29.0f, 750.0f, "%.0f m");
			if (auto _ttRk = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("range_skins_tooltip"), "Capture radius for snow skins on objects (rocks, cliffs, roofs). Applies live."));
			ImGui::SliderFloat(T(TKEY("range_skins_fade"), "Distant Snow Blend"), &settings.RangeSkinsFadeM, 29.0f, 750.0f, "%.0f m");
			if (auto _ttRkf = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("range_skins_fade_tooltip"), "Distance where object snow starts dissolving back into the object's own appearance; fully faded by the Object Snow range end. Cures distant blank-white objects."));
			ImGui::TreePop();
		}
		if (ImGui::TreeNodeEx(T(TKEY("debug_options"), "Debugging Options"), ImGuiTreeNodeFlags_Framed)) {
			ImGui::Checkbox(T(TKEY("show_debug"), "Show Deformation Map"), &settings.ShowDebugTexture);
			if (settings.ShowDebugTexture) {
				ImGui::Text("%s", T(TKEY("debug_hint"), "White = compressed snow. The map follows the camera."));
				ImGui::Image(GetDeformationSRV(), { 512.0f, 512.0f });
			}

			if (ImGui::Button(T(TKEY("clear"), "Clear Deformation Map")))
				clearRequested = true;

			ImGui::Checkbox(T(TKEY("shell_data_debug"), "Shell: Data Debug Plane"), &shellDataDebug);
			if (auto _tt4 = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("shell_data_debug_tooltip"), "Renders the shell as a flat plane under the camera, colored by the terrain data the shell samples: red = height, green = snow coverage. Black = no data reaches the shader."));

			ImGui::Checkbox(T(TKEY("debug_overlay"), "Debug Terrain Overlay"), &debugTerrainOverlay);
			if (auto _tt = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("debug_overlay_tooltip"), "Paints diagnostics on terrain: red = outside deformation window, green = deformation, blue = detected snow."));

			ImGui::Checkbox(T(TKEY("debug_statics_lift"), "Debug: Lift All Statics"), &debugStaticsLift);
			if (auto _ttLift = Util::HoverTooltipWrapper())
				ImGui::Text("%s", T(TKEY("debug_statics_lift_tooltip"), "Diagnostic: raises EVERY object by the Objects depth, ignoring the snow gates. If the world visibly lifts, the statics-shell data path works and only the snow detection is wrong."));

			ImGui::TreePop();
		}


		ImGui::Text("Snow statics captured: %u", statCapturedStatics.load(std::memory_order_relaxed));
		ImGui::Text("Snow mask cache: %zu entries, %llu hits, %llu misses",
			snowMasksSizeForUI(),
			(unsigned long long)landMaskHits.load(std::memory_order_relaxed),
			(unsigned long long)landMaskMisses.load(std::memory_order_relaxed));
		ImGui::Text("Shell data: %zu cells baked, %u in window, %u snow texels, height [%.0f, %.0f]",
			ShellCellCountForUI(), shellStatCellsInWindow, shellStatSnowTexels, shellStatMinHeight, shellStatMaxHeight);
		// TEMPORARY: LOD-shadow diagnostics (remove once distant LOD shadows work).
		ImGui::Text("LOD shadow: descriptors=%u endSplits=%.0f/%.0f/%.0f active=%.0f atlasSlices=%u",
			dbgLodDescriptorCount, dbgLodEndSplits[0], dbgLodEndSplits[1], dbgLodEndSplits[2], dbgLodActive, dbgLodAtlasSlices);

		ImGui::TreePop();
	}
}

// Settled-latch tuning: accumulated displacement that wakes a settled corpse
// (real dragging/explosions), and how long a corpse must be still to settle.
static constexpr float kCorpseWakeDistance = 50.0f;
static constexpr uint16_t kCorpseSettleFrames = 90;

void SnowDeformation::GatherStamps(PerFrame& perFrameData)
{
	uint stampCount = 0;
	RE::NiPoint3 cameraPosition = Util::GetEyePosition();
	std::unordered_map<uint64_t, float2> currentPositions;
	corpseMoundSpheres.clear();

	// Stamps come from the actors' actual Havok collision shapes — the same
	// per-shape extraction Grass Collision uses (Util::GetShapeBound over
	// TraverseScenegraphCollision) instead of one scaled circle at the actor
	// center. Feet and lower-leg capsules carve individually (trails gain
	// real footfall structure), ragdolls carve where their limbs lie, and
	// horses or giants get wide tracks from their genuinely larger shapes
	// with no per-race tuning.
	auto addStamps = [&](RE::ActorHandle a_handle) {
		if (stampCount >= kMaxStamps)
			return;
		auto actor = a_handle.get();
		if (!actor || !actor->Is3DLoaded())
			return;
		auto position = actor->GetPosition();
		if (cameraPosition.GetSquaredDistance(position) > 0.25f * deformWorldSize * deformWorldSize)
			return;
		auto root = actor->Get3D(false);
		if (!root)
			return;

		const uint32_t formID = actor->formID;
		// The living keep their trenches open just by being there; the dead
		// carve only WHILE MOVING (the ragdoll fall stamps its imprint),
		// then go quiet at rest — and the refill slowly buries them. A
		// corpse already at rest when first seen never stamps at all.
		// (A "fresh kill waives first-sight" mechanism for decapitation's
		// 3D swap was tried and reverted: it produced re-trench pulses on
		// buried corpses.)
		const bool isDead = actor->IsDead();

		// Settled latch: resting anchors are FROZEN (so micro-jitter cannot
		// hold the trench open), which lets slow ragdoll drift accumulate
		// toward the 3-unit gate and fire a one-frame trench pulse under a
		// buried corpse. Once still for kCorpseSettleFrames, a corpse only
		// re-stamps after kCorpseWakeDistance of ACCUMULATED displacement —
		// real dragging or explosions, not creep.
		CorpseRest* rest = nullptr;
		if (isDead) {
			if (corpseRestStates.size() > 512 && !corpseRestStates.contains(formID))
				corpseRestStates.clear();
			rest = &corpseRestStates[formID];
		} else {
			// Seen alive (including reanimation): back to living rules.
			corpseRestStates.erase(formID);
		}
		bool anyShapeMoved = false;
		bool anyShapeWoken = false;

		// Airborne LIVING actors do not touch the snow: jumping, levitating
		// or falling carves nothing until contact. Dead ragdolls are exempt:
		// their controllers freeze in stale states (often kInAir), which
		// would suppress normal corpse imprints; their movement gate
		// governs them instead.
		if (!isDead)
			if (auto* charController = actor->GetCharController(); charController && charController->context.currentState == RE::hkpCharacterStateType::kInAir)
				return;

		const float groundZ = position.z;
		uint32_t shapeIndex = 0;
		RE::BSVisit::TraverseScenegraphCollision(root, [&](RE::bhkNiCollisionObject* a_object) -> RE::BSVisit::BSVisitControl {
			RE::NiPoint3 centerPos;
			float radius;
			if (Util::GetShapeBound(a_object, centerPos, radius)) {
				// Stable per-skeleton traversal order keys the trail history.
				const uint32_t thisIndex = shapeIndex++;
				if (stampCount >= kMaxStamps)
					return RE::BSVisit::BSVisitControl::kStop;
				// Only shapes near the snow surface carve: feet, calves, a
				// sneaking torso, ragdoll limbs — not heads walking past.
				if (centerPos.z - radius > groundZ + 40.0f)
					return RE::BSVisit::BSVisitControl::kContinue;
				if (radius < 4.0f || radius > 128.0f)
					return RE::BSVisit::BSVisitControl::kContinue;

				// Capsule stamping: the segment runs from this shape's
				// previous position, so fast movers carve continuous trails
				// instead of chains of spaced circles. Large jumps
				// (teleports, cell loads) collapse to a point.
				float2 current = { centerPos.x, centerPos.y };
				float2 previous = current;
				const uint64_t key = (uint64_t(formID) << 16) | uint64_t(thisIndex & 0xFFFF);
				auto it = stampPrevPositions.find(key);
				bool moved = true;
				float sqDelta = 0.0f;
				if (it != stampPrevPositions.end()) {
					float2 delta = { current.x - it->second.x, current.y - it->second.y };
					sqDelta = delta.x * delta.x + delta.y * delta.y;
					if (sqDelta < 256.0f * 256.0f)
						previous = it->second;
					moved = sqDelta > 3.0f * 3.0f;
				}
				[[maybe_unused]] bool firstSight = (it == stampPrevPositions.end());
				// Measured against the frozen resting anchor, so dragging
				// accumulates past the wake distance within a few frames
				// while jitter oscillating around a point never does.
				const bool woken = !firstSight && sqDelta > kCorpseWakeDistance * kCorpseWakeDistance;
				if (isDead) {
					anyShapeMoved |= !firstSight && moved;
					anyShapeWoken |= woken;
				}
				if (isDead && (firstSight || !moved || (rest->settled && !woken))) {
					// Corpse at rest: no stamp. Keep the OLD anchor so
					// ragdoll micro-jitter cannot hold the trench open, but
					// real movement (dragging, explosions) re-triggers
					// against it. First-seen corpses store a baseline. The
					// resting shapes feed the burial mounds instead.
					currentPositions[key] = firstSight ? current : it->second;
					if (corpseMoundSpheres.size() < 32)
						corpseMoundSpheres.push_back({ centerPos.x, centerPos.y, centerPos.z, radius });
					return RE::BSVisit::BSVisitControl::kContinue;
				}
				currentPositions[key] = current;

				float4 stamp{};
				stamp.x = current.x;
				stamp.y = current.y;
				stamp.z = 1.0f;
				// StampRadius acts as a scale on the shape's own radius
				// (default 20 = 1.0x the Havok shape size).
				stamp.w = radius * settings.StampRadius * 0.05f;
				perFrameData.Stamps[stampCount] = stamp;
				perFrameData.StampEnds[stampCount] = { previous.x, previous.y, 0.0f, 0.0f };
				stampCount++;
			}
			return RE::BSVisit::BSVisitControl::kContinue;
		});

		if (rest) {
			if (anyShapeWoken) {
				rest->settled = false;
				rest->stillFrames = 0;
			} else if (anyShapeMoved) {
				rest->stillFrames = 0;
			} else if (!rest->settled && ++rest->stillFrames >= kCorpseSettleFrames) {
				rest->settled = true;
			}
		}
	};

	if (auto player = RE::PlayerCharacter::GetSingleton())
		addStamps(player->GetHandle());

	if (const auto processLists = RE::ProcessLists::GetSingleton()) {
		for (auto& actorHandle : processLists->highActorHandles)
			addStamps(actorHandle);
	}

	// Loose inanimate objects (dropped weapons, kicked clutter, tumbling
	// barrels, thrown ingredients) carve while they MOVE; at rest they go
	// quiet and the refill buries their imprint — same story as corpses.
	// The gate runs on the cheap REFERENCE position first, so collision
	// traversal only ever happens for props actually in motion: resting
	// world clutter costs one hash lookup per frame. Anchors refresh every
	// frame (unlike the frozen corpse anchors), so micro-creep can never
	// accumulate into a stamp pulse.
	std::unordered_map<uint32_t, RE::NiPoint3> currentPropPositions;
	const auto tes = RE::TES::GetSingleton();
	auto* playerRef = RE::PlayerCharacter::GetSingleton();
	if (tes && playerRef) {
		tes->ForEachReferenceInRange(playerRef, 0.5f * deformWorldSize, [&](RE::TESObjectREFR* a_ref) {
			if (!a_ref || a_ref->As<RE::Actor>())
				return RE::BSContainer::ForEachResult::kContinue;
			auto* base = a_ref->GetBaseObject();
			if (!base)
				return RE::BSContainer::ForEachResult::kContinue;
			// Havok-movable base types only — statics, trees, furniture and
			// containers never travel, and flying projectiles must not carve
			// the snow under their flight path.
			switch (base->GetFormType()) {
			case RE::FormType::Misc:
			case RE::FormType::Weapon:
			case RE::FormType::Armor:
			case RE::FormType::Ammo:
			case RE::FormType::Book:
			case RE::FormType::Ingredient:
			case RE::FormType::AlchemyItem:
			case RE::FormType::SoulGem:
			case RE::FormType::KeyMaster:
			case RE::FormType::Light:
			case RE::FormType::MovableStatic:
				break;
			default:
				return RE::BSContainer::ForEachResult::kContinue;
			}
			if (!a_ref->Is3DLoaded())
				return RE::BSContainer::ForEachResult::kContinue;
			auto root = a_ref->Get3D(false);
			if (!root)
				return RE::BSContainer::ForEachResult::kContinue;

			// Movement gate on the 3D root's WORLD transform — never the
			// reference position: Havok-simulated clutter moves its scene
			// graph every frame while the REFERENCE's stored position lags
			// until the body settles (the rolled-Sigil-stone-left-no-tracks
			// bug: the ref position sat frozen through the whole roll).
			const auto position = root->world.translate;
			const uint32_t formID = a_ref->formID;
			auto prevIt = propPrevPositions.find(formID);
			if (prevIt == propPrevPositions.end()) {
				currentPropPositions[formID] = position;
				return RE::BSContainer::ForEachResult::kContinue;  // first sight: baseline only
			}
			// FROZEN anchor: slow motion accumulates toward the gate instead
			// of being reset every frame. Sleeping Havok props are truly
			// frozen, so drift pulses cannot happen.
			const bool propMoved = position.GetSquaredDistance(prevIt->second) >= 3.0f * 3.0f;
			currentPropPositions[formID] = propMoved ? position : prevIt->second;
			if (!propMoved)
				return RE::BSContainer::ForEachResult::kContinue;  // at rest: the refill buries it
			if (stampCount >= kMaxStamps)
				return RE::BSContainer::ForEachResult::kContinue;  // keep collecting anchors

			// Ground reference is the LAND height, not the object's own
			// origin — a thrown or carried item rides high above it, so the
			// near-surface gate keeps mid-air flight paths from carving.
			float groundZ = position.z;
			tes->GetLandHeight(position, groundZ);

			uint32_t shapeIndex = 0;
			RE::BSVisit::TraverseScenegraphCollision(root, [&](RE::bhkNiCollisionObject* a_object) -> RE::BSVisit::BSVisitControl {
				RE::NiPoint3 centerPos;
				float radius;
				if (Util::GetShapeBound(a_object, centerPos, radius)) {
					const uint32_t thisIndex = shapeIndex++;
					if (stampCount >= kMaxStamps)
						return RE::BSVisit::BSVisitControl::kStop;
					if (centerPos.z - radius > groundZ + 40.0f)
						return RE::BSVisit::BSVisitControl::kContinue;
					if (radius < 4.0f || radius > 128.0f)
						return RE::BSVisit::BSVisitControl::kContinue;

					// Same capsule-trail scheme as actors: props share the
					// (formID << 16 | shape) keyspace, which cannot collide
					// with actor keys because formIDs are unique.
					float2 current = { centerPos.x, centerPos.y };
					float2 previous = current;
					const uint64_t key = (uint64_t(formID) << 16) | uint64_t(thisIndex & 0xFFFF);
					auto it = stampPrevPositions.find(key);
					if (it != stampPrevPositions.end()) {
						float2 delta = { current.x - it->second.x, current.y - it->second.y };
						if (delta.x * delta.x + delta.y * delta.y < 256.0f * 256.0f)
							previous = it->second;
					}
					currentPositions[key] = current;

					float4 stamp{};
					stamp.x = current.x;
					stamp.y = current.y;
					stamp.z = 1.0f;
					stamp.w = radius * settings.StampRadius * 0.05f;
					perFrameData.Stamps[stampCount] = stamp;
					perFrameData.StampEnds[stampCount] = { previous.x, previous.y, 0.0f, 0.0f };
					stampCount++;
				}
				return RE::BSVisit::BSVisitControl::kContinue;
			});

			// Fallback for props whose collision shape type has no bound
			// extractor (MOPP/list clutter): one stamp from the 3D root's
			// bound sphere, so anything that rolls always carves SOMETHING.
			if (shapeIndex == 0 && stampCount < kMaxStamps) {
				const auto& bound = root->worldBound;
				float radius = std::clamp(bound.radius, 4.0f, 128.0f);
				if (bound.center.z - radius <= groundZ + 40.0f) {
					float2 current = { bound.center.x, bound.center.y };
					float2 previous = current;
					const uint64_t key = (uint64_t(formID) << 16) | 0xFFFFull;
					auto it = stampPrevPositions.find(key);
					if (it != stampPrevPositions.end()) {
						float2 delta = { current.x - it->second.x, current.y - it->second.y };
						if (delta.x * delta.x + delta.y * delta.y < 256.0f * 256.0f)
							previous = it->second;
					}
					currentPositions[key] = current;

					float4 stamp{};
					stamp.x = current.x;
					stamp.y = current.y;
					stamp.z = 1.0f;
					stamp.w = radius * settings.StampRadius * 0.05f;
					perFrameData.Stamps[stampCount] = stamp;
					perFrameData.StampEnds[stampCount] = { previous.x, previous.y, 0.0f, 0.0f };
					stampCount++;
				}
			}
			return RE::BSContainer::ForEachResult::kContinue;
		});
	}
	propPrevPositions = std::move(currentPropPositions);

	stampPrevPositions = std::move(currentPositions);
	perFrameData.StampCount = stampCount;
}

// Copies a shadow depth target into an owned shader-resource texture,
// recreating the copy when the source dimensions/format change. The source
// SRV's view desc is reused so typeless depth formats resolve identically.
static void SD_CopyShadowTarget(ID3D11ShaderResourceView* a_srcSRV, const char* a_name,
	winrt::com_ptr<ID3D11Texture2D>& a_tex, winrt::com_ptr<ID3D11ShaderResourceView>& a_srv)
{
	winrt::com_ptr<ID3D11Resource> srcRes;
	a_srcSRV->GetResource(srcRes.put());
	auto srcTex = srcRes.try_as<ID3D11Texture2D>();
	if (!srcTex)
		return;

	D3D11_TEXTURE2D_DESC srcDesc;
	srcTex->GetDesc(&srcDesc);

	// The SRV is nulled on invalid frames as the validity signal, so it must
	// be rebuilt even when the texture itself is still current.
	bool recreate = !a_tex || !a_srv;
	if (a_tex) {
		D3D11_TEXTURE2D_DESC haveDesc;
		a_tex->GetDesc(&haveDesc);
		recreate |= haveDesc.Width != srcDesc.Width || haveDesc.Height != srcDesc.Height ||
		            haveDesc.ArraySize != srcDesc.ArraySize || haveDesc.Format != srcDesc.Format;
	}
	if (recreate) {
		a_srv = nullptr;
		a_tex = nullptr;

		D3D11_TEXTURE2D_DESC copyDesc = srcDesc;
		copyDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		copyDesc.MiscFlags = 0;
		copyDesc.Usage = D3D11_USAGE_DEFAULT;
		copyDesc.CPUAccessFlags = 0;
		if (FAILED(globals::d3d::device->CreateTexture2D(&copyDesc, nullptr, a_tex.put())))
			return;
		Util::SetResourceName(a_tex.get(), a_name);

		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		a_srcSRV->GetDesc(&srvDesc);
		if (FAILED(globals::d3d::device->CreateShaderResourceView(a_tex.get(), &srvDesc, a_srv.put()))) {
			a_tex = nullptr;
			return;
		}
	}

	globals::d3d::context->CopyResource(a_tex.get(), srcTex.get());
}

void SnowDeformation::CreateHeightFieldResources()
{
	for (int i = 0; i < 2; i++) {
		delete heightTopRaw[i];
		heightTopRaw[i] = nullptr;
		delete heightBottomRaw[i];
		heightBottomRaw[i] = nullptr;
	}
	delete heightTopFiltered;
	heightTopFiltered = nullptr;
	delete heightBottomFiltered;
	heightBottomFiltered = nullptr;
	delete heightScratch;
	heightScratch = nullptr;

	D3D11_TEXTURE2D_DESC heightDesc = {
		.Width = heightMapDim,
		.Height = heightMapDim,
		.MipLevels = 1,
		.ArraySize = 1,
		.Format = DXGI_FORMAT_R32_FLOAT,
		.SampleDesc = { .Count = 1 },
		.Usage = D3D11_USAGE_DEFAULT,
		.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS
	};
	D3D11_SHADER_RESOURCE_VIEW_DESC heightSrvDesc = {
		.Format = heightDesc.Format,
		.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
		.Texture2D = { .MostDetailedMip = 0, .MipLevels = 1 }
	};
	D3D11_RENDER_TARGET_VIEW_DESC heightRtvDesc = {
		.Format = heightDesc.Format,
		.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D,
		.Texture2D = { .MipSlice = 0 }
	};
	D3D11_UNORDERED_ACCESS_VIEW_DESC heightUavDesc = {
		.Format = heightDesc.Format,
		.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D,
		.Texture2D = { .MipSlice = 0 }
	};

	auto makeHeightTexture = [&]() {
		auto* texture = new Texture2D(heightDesc);
		texture->CreateSRV(heightSrvDesc);
		texture->CreateRTV(heightRtvDesc);
		texture->CreateUAV(heightUavDesc);
		return texture;
	};
	for (int i = 0; i < 2; i++) {
		heightTopRaw[i] = makeHeightTexture();
		heightBottomRaw[i] = makeHeightTexture();
	}
	heightTopFiltered = makeHeightTexture();
	heightBottomFiltered = makeHeightTexture();
	heightScratch = makeHeightTexture();

	heightMapValid = false;
}

void SnowDeformation::ApplyRangeSettings()
{
	// Trench window: content is scale-relative, so a resize clears the map.
	if (trenchRangeDirty || !rangeInitApplied) {
		float newWorldSize = std::clamp(settings.RangeTrenchesM, 29.0f, 200.0f) * 2.0f * kUnitsPerMeter;
		if (std::abs(newWorldSize - deformWorldSize) > 1.0f) {
			deformWorldSize = newWorldSize;
			clearRequested = true;
		}
		trenchRangeDirty = false;
	}

	// The height field (shelter, exclusions, corpse mounds) runs at a fixed
	// near-camera window since the object-blanket lift was removed — no
	// range setting drives it anymore.

	rangeInitApplied = true;
}

void SnowDeformation::CaptureShadowAtlas()
{
	// Called mid-frame while the game renders the shadow MASK: PS t4 holds
	// the sun cascade atlas RIGHT NOW (at any other time it holds whatever
	// material texture the last draw bound — the r56/57 bug: an EarlyPrepass
	// grab captured a 128px BC7 diffuse). COPY the atlas and its ESRAM
	// partner immediately: by deferred time the engine has reused the live
	// targets (ESRAM is aliased scratch).
	//
	// When a frame skips this pass, the previous copies are KEPT — one-frame-
	// stale cascades are invisible, but flapping between the crisp and VSM
	// paths reads as full-surface flicker.
	if (!globals::state->HasDirectionalShadows()) {
		shadowAtlasCopySRV = nullptr;
		shadowEsramCopySRV = nullptr;
		return;
	}

	winrt::com_ptr<ID3D11ShaderResourceView> liveAtlasSRV;
	globals::d3d::context->PSGetShaderResources(4, 1, liveAtlasSRV.put());

	ID3D11ShaderResourceView* liveEsramSRV = nullptr;
	if (auto* gameRenderer = globals::game::renderer)
		liveEsramSRV = gameRenderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kVOLUMETRIC_LIGHTING_SHADOWMAPS_ESRAM].depthSRV;

	if (liveAtlasSRV && liveEsramSRV) {
		SD_CopyShadowTarget(liveAtlasSRV.get(), "SnowDeformation::ShadowAtlasCopy", shadowAtlasCopyTex, shadowAtlasCopySRV);
		SD_CopyShadowTarget(liveEsramSRV, "SnowDeformation::ShadowEsramCopy", shadowEsramCopyTex, shadowEsramCopySRV);
		// TEMPORARY LOD-shadow diagnostic: how many slices the copy carries.
		if (shadowAtlasCopyTex) {
			D3D11_TEXTURE2D_DESC atlasDesc;
			shadowAtlasCopyTex->GetDesc(&atlasDesc);
			dbgLodAtlasSlices = atlasDesc.ArraySize;
		}

		// With the receiver copies safely taken (pre-shell), render the shell
		// INTO the live atlas so the world receives snow shadows: the game's
		// shadow mask is drawn right after this and picks the shell up like
		// any other caster.
		InjectShellShadowCasters(liveAtlasSRV.get());
	}

	// One-shot + transition diagnostics: what IS the shadow source?
	static uint32_t shadowLogCount = 0;
	static int lastValidState = -1;
	int validState = (shadowAtlasCopySRV && shadowEsramCopySRV) ? 1 : 0;
	if ((shadowLogCount < 6 || validState != lastValidState) && shadowLogCount < 40) {
		shadowLogCount++;
		lastValidState = validState;
		D3D11_TEXTURE2D_DESC atlasDesc{};
		if (shadowAtlasCopyTex)
			shadowAtlasCopyTex->GetDesc(&atlasDesc);
		D3D11_SHADER_RESOURCE_VIEW_DESC liveViewDesc{};
		if (liveAtlasSRV)
			liveAtlasSRV->GetDesc(&liveViewDesc);
		logger::info("[SNOW DEFORMATION] ShadowCopy: atlasLive={} esramLive={} valid={} atlas={}x{} slices={} fmt={} samples={} liveViewDim={} liveViewFmt={}",
			liveAtlasSRV != nullptr, liveEsramSRV != nullptr, validState,
			atlasDesc.Width, atlasDesc.Height, atlasDesc.ArraySize, uint32_t(atlasDesc.Format), atlasDesc.SampleDesc.Count,
			uint32_t(liveViewDesc.ViewDimension), uint32_t(liveViewDesc.Format));
	}
}

void SnowDeformation::InjectShellShadowCasters(ID3D11ShaderResourceView* a_atlasSRV)
{
	// One-shot diagnostics: name the exit taken, so a missing shadow points
	// straight at its gate.
	static std::unordered_set<std::string> injectLogged;
	auto logOnce = [](const char* a_msg) {
		if (injectLogged.size() < 12 && injectLogged.insert(a_msg).second)
			logger::info("[SNOW DEFORMATION] ShadowInject: {}", a_msg);
	};

	if (!settings.EnableSnowDeformation || !lastShellCBData) {
		logOnce("skip: disabled or no ShellCB snapshot yet");
		return;
	}

	auto* vs = GetShellShadowVS();
	if (!vs) {
		logOnce("skip: shell shadow VS unavailable");
		return;
	}

	auto* shadowSceneNode = globals::game::smState->shadowSceneNode[0];
	auto* sunShadowLight = shadowSceneNode ? shadowSceneNode->GetRuntimeData().sunShadowDirLight : nullptr;
	if (!sunShadowLight) {
		logOnce("skip: no sun shadow light");
		return;
	}
	auto& lightRuntime = sunShadowLight->GetRuntimeData();
	const uint32_t cascadeCount = std::min<uint32_t>((uint32_t)lightRuntime.shadowmapDescriptors.size(), 2u);
	if (cascadeCount == 0) {
		logOnce("skip: zero cascades");
		return;
	}

	auto context = globals::d3d::context;
	auto device = globals::d3d::device;

	// Per-slice DSVs on the LIVE atlas texture (D16_UNORM over R16_TYPELESS),
	// cached by texture pointer.
	winrt::com_ptr<ID3D11Resource> atlasRes;
	a_atlasSRV->GetResource(atlasRes.put());
	auto atlasTex = atlasRes.try_as<ID3D11Texture2D>();
	if (!atlasTex) {
		logOnce("skip: atlas SRV resource is not a Texture2D");
		return;
	}
	D3D11_TEXTURE2D_DESC atlasDesc;
	atlasTex->GetDesc(&atlasDesc);
	if (!(atlasDesc.BindFlags & D3D11_BIND_DEPTH_STENCIL) || atlasDesc.ArraySize < cascadeCount) {
		logOnce("skip: atlas lacks DEPTH_STENCIL bind or slices");
		return;
	}
	if (shadowAtlasDSVTexture != atlasTex.get()) {
		DXGI_FORMAT dsvFormat;
		switch (atlasDesc.Format) {
		case DXGI_FORMAT_R16_TYPELESS:
		case DXGI_FORMAT_D16_UNORM:
			dsvFormat = DXGI_FORMAT_D16_UNORM;
			break;
		case DXGI_FORMAT_R24G8_TYPELESS:
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
			dsvFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
			break;
		case DXGI_FORMAT_R32_TYPELESS:
		case DXGI_FORMAT_D32_FLOAT:
			dsvFormat = DXGI_FORMAT_D32_FLOAT;
			break;
		default:
			return;
		}
		for (uint32_t i = 0; i < 2; i++) {
			shadowAtlasDSV[i] = nullptr;
			D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
			dsvDesc.Format = dsvFormat;
			dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
			dsvDesc.Texture2DArray.MipSlice = 0;
			dsvDesc.Texture2DArray.FirstArraySlice = i;
			dsvDesc.Texture2DArray.ArraySize = 1;
			if (i < atlasDesc.ArraySize)
				device->CreateDepthStencilView(atlasTex.get(), &dsvDesc, shadowAtlasDSV[i].put());
			if (shadowAtlasDSV[i])
				Util::SetResourceName(shadowAtlasDSV[i].get(), "SnowDeformation::ShadowAtlas DSV");
		}
		shadowAtlasDSVTexture = atlasTex.get();
	}
	if (!shadowAtlasDSV[0]) {
		logOnce("skip: DSV creation failed");
		return;
	}

	if (!shadowCastRS) {
		D3D11_RASTERIZER_DESC rsDesc{};
		rsDesc.FillMode = D3D11_FILL_SOLID;
		rsDesc.CullMode = D3D11_CULL_NONE;
		rsDesc.DepthClipEnable = TRUE;
		device->CreateRasterizerState(&rsDesc, shadowCastRS.put());
		Util::SetResourceName(shadowCastRS.get(), "SnowDeformation::ShadowCastRS");
	}
	if (!shadowCastDSS) {
		D3D11_DEPTH_STENCIL_DESC dsDesc{};
		dsDesc.DepthEnable = TRUE;
		dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
		dsDesc.DepthFunc = D3D11_COMPARISON_LESS;
		device->CreateDepthStencilState(&dsDesc, shadowCastDSS.put());
		Util::SetResourceName(shadowCastDSS.get(), "SnowDeformation::ShadowCastDSS");
	}
	if (!shadowCastRS || !shadowCastDSS) {
		logOnce("skip: RS/DSS creation failed");
		return;
	}
	logOnce("injecting: shell grid + statics into both cascades");

	// ---- Save every piece of state we touch: we are INSIDE the game's setup
	// for the shadow-mask draw and must hand it back byte-identical.
	winrt::com_ptr<ID3D11RenderTargetView> prevRTVs[8];
	winrt::com_ptr<ID3D11DepthStencilView> prevDSV;
	{
		ID3D11RenderTargetView* rtvs[8] = {};
		ID3D11DepthStencilView* dsv = nullptr;
		context->OMGetRenderTargets(8, rtvs, &dsv);
		for (uint32_t i = 0; i < 8; i++)
			prevRTVs[i].attach(rtvs[i]);
		prevDSV.attach(dsv);
	}
	UINT prevViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	D3D11_VIEWPORT prevViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]{};
	context->RSGetViewports(&prevViewportCount, prevViewports);
	winrt::com_ptr<ID3D11RasterizerState> prevRS;
	context->RSGetState(prevRS.put());
	winrt::com_ptr<ID3D11DepthStencilState> prevDSS;
	UINT prevStencilRef = 0;
	context->OMGetDepthStencilState(prevDSS.put(), &prevStencilRef);
	winrt::com_ptr<ID3D11VertexShader> prevVS;
	context->VSGetShader(prevVS.put(), nullptr, nullptr);
	winrt::com_ptr<ID3D11PixelShader> prevPS;
	context->PSGetShader(prevPS.put(), nullptr, nullptr);
	D3D11_PRIMITIVE_TOPOLOGY prevTopology;
	context->IAGetPrimitiveTopology(&prevTopology);
	winrt::com_ptr<ID3D11InputLayout> prevLayout;
	context->IAGetInputLayout(prevLayout.put());
	winrt::com_ptr<ID3D11Buffer> prevVSCBs[2];
	{
		ID3D11Buffer* cbs[2] = {};
		context->VSGetConstantBuffers(0, 2, cbs);
		prevVSCBs[0].attach(cbs[0]);
		prevVSCBs[1].attach(cbs[1]);
	}
	winrt::com_ptr<ID3D11ShaderResourceView> prevVSSRVs[6];
	{
		ID3D11ShaderResourceView* srvs[6] = {};
		context->VSGetShaderResources(0, 6, srvs);
		for (uint32_t i = 0; i < 6; i++)
			prevVSSRVs[i].attach(srvs[i]);
	}
	winrt::com_ptr<ID3D11Buffer> prevVB;
	UINT prevVBStride = 0, prevVBOffset = 0;
	{
		ID3D11Buffer* vb = nullptr;
		context->IAGetVertexBuffers(0, 1, &vb, &prevVBStride, &prevVBOffset);
		prevVB.attach(vb);
	}
	winrt::com_ptr<ID3D11Buffer> prevIB;
	DXGI_FORMAT prevIBFormat = DXGI_FORMAT_UNKNOWN;
	UINT prevIBOffset = 0;
	{
		ID3D11Buffer* ib = nullptr;
		context->IAGetIndexBuffer(&ib, &prevIBFormat, &prevIBOffset);
		prevIB.attach(ib);
	}
	winrt::com_ptr<ID3D11ShaderResourceView> prevPSSRV4;
	{
		ID3D11ShaderResourceView* srv = nullptr;
		context->PSGetShaderResources(4, 1, &srv);
		prevPSSRV4.attach(srv);
	}

	// ---- Common caster state.
	context->PSSetShader(nullptr, nullptr, 0);  // depth-only
	context->RSSetState(shadowCastRS.get());
	context->OMSetDepthStencilState(shadowCastDSS.get(), 0);
	D3D11_VIEWPORT atlasViewport{ 0.0f, 0.0f, float(atlasDesc.Width), float(atlasDesc.Height), 0.0f, 1.0f };
	context->RSSetViewports(1, &atlasViewport);
	// Binding the atlas as DSV force-nulls its SRV binding at PS t4; we
	// restore it at the end.
	ID3D11ShaderResourceView* nullPSSRV = nullptr;
	context->PSSetShaderResources(4, 1, &nullPSSRV);

	// Shell field textures for the VS (ShellSurfaceZ): terrain window,
	// deformation, blanket top/bottom. Snow diffuse / scene depth are PS-only.
	ID3D11ShaderResourceView* vsSRVs[6] = { shellTerrainTexture ? shellTerrainTexture->srv.get() : nullptr,
		GetDeformationSRV(), nullptr, nullptr,
		heightTopFiltered ? heightTopFiltered->srv.get() : nullptr,
		heightBottomFiltered ? heightBottomFiltered->srv.get() : nullptr };
	context->VSSetShaderResources(0, 6, vsSRVs);
	ID3D11Buffer* cb0 = shellCB->CB();
	context->VSSetConstantBuffers(0, 1, &cb0);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	globals::profiler->BeginPass("SnowDeformation::ShellShadowCast");
	for (uint32_t cascade = 0; cascade < cascadeCount; cascade++) {
		if (!shadowAtlasDSV[cascade])
			continue;

		// lightTransform maps absolute world -> per-slice [0,1] UV + depth
		// (row-vector XM convention, proven by GetVSMShadow2D). Append the
		// UV->clip mapping and a small caster push AWAY from the light so
		// receivers exactly at the shell surface (fence tops under their own
		// snow skin) do not acne.
		DirectX::XMMATRIX uvProj = DirectX::XMLoadFloat4x4(
			reinterpret_cast<const DirectX::XMFLOAT4X4*>(&lightRuntime.shadowmapDescriptors[cascade].lightTransform));
		constexpr float kCasterDepthPush = 0.0008f;
		DirectX::XMMATRIX uvToClip = DirectX::XMMatrixSet(
			2.0f, 0.0f, 0.0f, 0.0f,
			0.0f, -2.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			-1.0f, 1.0f, kCasterDepthPush, 1.0f);
		DirectX::XMMATRIX clip = DirectX::XMMatrixMultiply(uvProj, uvToClip);

		// ShellCB matrices are row_major with mul(M, v): store the transpose.
		ShellCB shadowCB = *lastShellCBData;
		shadowCB.CameraViewProj = DirectX::XMMatrixTranspose(clip);
		// Absolute-world rendering: zero the camera-relative adjust.
		shadowCB.CameraPosAdjust = { 0.0f, 0.0f, 0.0f, 0.0f };
		shadowCB.CameraPreviousPosAdjust = { 0.0f, 0.0f, 0.0f, 0.0f };
		shadowCB.ShellDebugData = 0;
		shellCB->Update(shadowCB);

		context->OMSetRenderTargets(0, nullptr, shadowAtlasDSV[cascade].get());

		// Terrain shell only: vertex-buffer-less grid with the excess-height
		// caster VS. The statics SKINS deliberately do NOT cast: a skin
		// hovers a few units above the object's own surface, so the object
		// beneath always reads as shadowed by its own snow cap (offset rock-
		// shaped shadow patches) — and the skins add almost nothing to the
		// silhouette of objects that already cast their own shadows.
		context->IASetInputLayout(nullptr);
		ID3D11Buffer* nullVB = nullptr;
		UINT zero = 0;
		context->IASetVertexBuffers(0, 1, &nullVB, &zero, &zero);
		context->VSSetShader(vs, nullptr, 0);
		context->Draw(kShellGridDim * kShellGridDim * 6, 0);
	}
	globals::profiler->EndPass();

	// ---- Restore everything.
	{
		ID3D11RenderTargetView* rtvs[8];
		for (uint32_t i = 0; i < 8; i++)
			rtvs[i] = prevRTVs[i].get();
		context->OMSetRenderTargets(8, rtvs, prevDSV.get());
	}
	if (prevViewportCount)
		context->RSSetViewports(prevViewportCount, prevViewports);
	context->RSSetState(prevRS.get());
	context->OMSetDepthStencilState(prevDSS.get(), prevStencilRef);
	context->VSSetShader(prevVS.get(), nullptr, 0);
	context->PSSetShader(prevPS.get(), nullptr, 0);
	context->IASetPrimitiveTopology(prevTopology);
	context->IASetInputLayout(prevLayout.get());
	{
		ID3D11Buffer* cbs[2] = { prevVSCBs[0].get(), prevVSCBs[1].get() };
		context->VSSetConstantBuffers(0, 2, cbs);
	}
	{
		ID3D11ShaderResourceView* srvs[6];
		for (uint32_t i = 0; i < 6; i++)
			srvs[i] = prevVSSRVs[i].get();
		context->VSSetShaderResources(0, 6, srvs);
	}
	{
		ID3D11Buffer* vb = prevVB.get();
		context->IASetVertexBuffers(0, 1, &vb, &prevVBStride, &prevVBOffset);
	}
	context->IASetIndexBuffer(prevIB.get(), prevIBFormat, prevIBOffset);
	{
		ID3D11ShaderResourceView* srv = prevPSSRV4.get();
		context->PSSetShaderResources(4, 1, &srv);
	}
}

void SnowDeformation::Prepass()
{
	ApplyRangeSettings();

	auto context = globals::d3d::context;

	// The lighting shader samples t101 whenever the feature is compiled in, so
	// keep the SRV bound even while paused or disabled (the shader also checks
	// EnableSnowDeformation from FeatureData before using it).
	ID3D11ShaderResourceView* deformationSRV = GetDeformationSRV();
	context->PSSetShaderResources(101, 1, &deformationSRV);
	// The statics-shell displacement samples the map in the VERTEX stage.
	context->VSSetShaderResources(101, 1, &deformationSRV);

	// New frame: publish last frame's statics-capture count and reset the
	// list before this frame's opaque rendering fills it again.
	statCapturedStatics.store((uint32_t)capturedStatics.size(), std::memory_order_relaxed);
	capturedStatics.clear();
	capturedStaticsSet.clear();

	if (settings.EnableSnowDeformation && globals::state->inWorld)
		UpdateShellTerrainWindow();

	if (!settings.EnableSnowDeformation)
		return;

	auto ui = globals::game::ui;
	if (ui && ui->GameIsPaused())
		return;

	PerFrame perFrameData{};

	// The window origin was advanced in GetCommonBufferData (during
	// UpdateSharedData) so the FeatureData constant buffer and the texture we
	// scroll here describe the same frame. We only consume the stored state.
	perFrameData.ScrollDelta = pendingScrollDelta;
	pendingScrollDelta = { 0, 0 };

	perFrameData.WindowOrigin = windowOrigin;
	perFrameData.TexelSize = deformWorldSize / kTextureDim;

	float deltaTime = *globals::game::deltaTime;
	// Refill gate: with RefillOnlyWhenSnowing (default), compressed snow only
	// recovers while the CURRENT weather actually carries snow — trails
	// persist through clear spells (and interiors, where there is no sky).
	bool refillActive = !settings.RefillOnlyWhenSnowing;
	if (!refillActive)
		if (auto* sky = RE::Sky::GetSingleton())
			if (auto* weather = sky->currentWeather)
				refillActive = weather->data.flags.any(RE::TESWeather::WeatherDataFlag::kSnow);
	perFrameData.RefillAmount = (refillActive && settings.RefillTime > 0.0f) ? deltaTime / settings.RefillTime : 0.0f;

	perFrameData.ClearMap = clearRequested;
	clearRequested = false;

	GatherStamps(perFrameData);

	perFrame->Update(perFrameData);

	uint previousTexture = currentTexture;
	currentTexture = 1 - currentTexture;

	{
		ID3D11Buffer* buffers[1] = { perFrame->CB() };
		context->CSSetConstantBuffers(0, 1, buffers);

		ID3D11ShaderResourceView* srvs[] = { deformationTextures[previousTexture]->srv.get() };
		context->CSSetShaderResources(0, ARRAYSIZE(srvs), srvs);

		ID3D11UnorderedAccessView* uavs[] = { deformationTextures[currentTexture]->uav.get() };
		context->CSSetUnorderedAccessViews(0, ARRAYSIZE(uavs), uavs, nullptr);

		context->CSSetShader(GetDeformationUpdateCS(), nullptr, 0);
		globals::profiler->BeginPass("SnowDeformation::DeformationUpdate");
		context->Dispatch(kTextureDim / 8, kTextureDim / 8, 1);
		globals::profiler->EndPass();
	}

	context->CSSetShader(nullptr, nullptr, 0);

	ID3D11Buffer* nullBuffer = nullptr;
	context->CSSetConstantBuffers(0, 1, &nullBuffer);

	ID3D11ShaderResourceView* nullSrvs[1] = { nullptr };
	context->CSSetShaderResources(0, 1, nullSrvs);

	ID3D11UnorderedAccessView* nullUavs[1] = { nullptr };
	context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);

	// Rebind: after the ping-pong flip this points at the freshly written map.
	deformationSRV = GetDeformationSRV();
	context->PSSetShaderResources(101, 1, &deformationSRV);
	context->VSSetShaderResources(101, 1, &deformationSRV);
}

SnowDeformation::SettingsGPU SnowDeformation::GetCommonBufferData(bool a_inWorld)
{
	// Advance the window once per frame, and only from the in-world upload
	// (StartDeferred) — the reflections/early uploads can carry probe cameras
	// whose position must not steer the deformation window.
	static Util::FrameChecker frameChecker;
	if (a_inWorld && frameChecker.IsNewFrame()) {
		// Snap to whole texels so scrolling never resamples the map. Use the
		// cached FrameBuffer camera position: it is exactly what the lighting
		// PS sees as CameraPosAdjust, so map and terrain never diverge.
		auto eyePosFB = globals::game::frameBufferCached.GetCameraPosAdjust();
		const float deformTexel = deformWorldSize / kTextureDim;
		float2 desiredOrigin = {
			std::floor((eyePosFB.x - deformWorldSize * 0.5f) / deformTexel) * deformTexel,
			std::floor((eyePosFB.y - deformWorldSize * 0.5f) / deformTexel) * deformTexel
		};

		pendingScrollDelta.x += (int)std::lround((desiredOrigin.x - windowOrigin.x) / deformTexel);
		pendingScrollDelta.y += (int)std::lround((desiredOrigin.y - windowOrigin.y) / deformTexel);
		windowOrigin = desiredOrigin;
	}

	SettingsGPU data{};
	data.WindowOrigin = windowOrigin;
	data.InvWorldSize = 1.0f / deformWorldSize;
	data.DeformationDepth = settings.DeformationDepth;
	data.EnableSnowDeformation = settings.EnableSnowDeformation;
	data.ObjectsSnowDepth = settings.ObjectsSnowDepth;
	data.TrailAOStrength = settings.TrailAOStrength;
	data.NormalStrength = settings.NormalStrength;
	// Bitfield: bit 0 = terrain overlay, bit 1 = statics-lift diagnostic.
	data.DebugTerrainOverlay = (debugTerrainOverlay ? 1u : 0u) | (debugStaticsLift ? 2u : 0u);
	data.ParallaxDepth = settings.ParallaxDepth;
	// The statics-shell VS works in camera-relative world space; hand it the
	// deformation window origin in the same space (per-view: probe uploads
	// carry the probe camera's adjust, keeping the carve consistent there).
	auto eyeForRel = globals::game::frameBufferCached.GetCameraPosAdjust();
	data.WindowOriginRelCam = { windowOrigin.x - eyeForRel.x, windowOrigin.y - eyeForRel.y };
	return data;
}

ID3D11ComputeShader* SnowDeformation::GetDeformationUpdateCS()
{
	if (!deformationUpdateCS) {
		logger::debug("Compiling DeformationUpdateCS");
		deformationUpdateCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\DeformationUpdateCS.hlsl", {}, "cs_5_0"));
	}
	return deformationUpdateCS;
}

ID3D11ComputeShader* SnowDeformation::GetDepthSyncCS()
{
	if (!depthSyncCS) {
		logger::debug("Compiling DepthSyncCS");
		depthSyncCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\DepthSyncCS.hlsl", {}, "cs_5_0"));
	}
	return depthSyncCS;
}

void SnowDeformation::ClearShaderCache()
{
	if (deformationUpdateCS)
		deformationUpdateCS->Release();
	deformationUpdateCS = nullptr;
	if (depthSyncCS)
		depthSyncCS->Release();
	depthSyncCS = nullptr;
	if (staticsVS)
		staticsVS->Release();
	staticsVS = nullptr;
	if (staticsPS)
		staticsPS->Release();
	staticsPS = nullptr;
	if (heightVS)
		heightVS->Release();
	heightVS = nullptr;
	if (heightPS)
		heightPS->Release();
	heightPS = nullptr;
	if (heightScrollCS)
		heightScrollCS->Release();
	heightScrollCS = nullptr;
	if (heightCombineCS)
		heightCombineCS->Release();
	heightCombineCS = nullptr;
	if (heightConeCS)
		heightConeCS->Release();
	heightConeCS = nullptr;
	staticsVSBlob = nullptr;
	staticsILCache.clear();
	staticsShadersFailed = false;

	if (shellVS)
		shellVS->Release();
	shellVS = nullptr;
	if (shellShadowVS)
		shellShadowVS->Release();
	shellShadowVS = nullptr;
	if (smoothAccumulateCS)
		smoothAccumulateCS->Release();
	smoothAccumulateCS = nullptr;
	if (smoothResolveCS)
		smoothResolveCS->Release();
	smoothResolveCS = nullptr;
	if (smoothFlatStatsCS)
		smoothFlatStatsCS->Release();
	smoothFlatStatsCS = nullptr;
	if (shellPS)
		shellPS->Release();
	shellPS = nullptr;
}

ID3D11VertexShader* SnowDeformation::GetShellVS()
{
	if (!shellVS) {
		logger::debug("Compiling SnowShell VS");
		shellVS = static_cast<ID3D11VertexShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SnowShell.hlsl", { { "VSHADER", "" } }, "vs_5_0"));
	}
	return shellVS;
}

ID3D11VertexShader* SnowDeformation::GetShellShadowVS()
{
	if (!shellShadowVS) {
		logger::debug("Compiling SnowShell shadow-cast VS");
		shellShadowVS = static_cast<ID3D11VertexShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SnowShell.hlsl", { { "VSHADER", "" }, { "SNOW_SHADOW_CAST", "" } }, "vs_5_0"));
	}
	return shellShadowVS;
}

ID3D11PixelShader* SnowDeformation::GetShellPS()
{
	if (!shellPS) {
		logger::debug("Compiling SnowShell PS");
		shellPS = static_cast<ID3D11PixelShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SnowShell.hlsl", { { "PSHADER", "" } }, "ps_5_0"));
	}
	return shellPS;
}

void SnowDeformation::DrawShell()
{
	if (!settings.EnableSnowDeformation)
		return;

	if (!globals::state->inWorld)
		return;

	auto vs = GetShellVS();
	auto ps = GetShellPS();
	if (!vs || !ps)
		return;

	auto context = globals::d3d::context;
	auto renderer = globals::game::renderer;
	auto& fb = globals::game::frameBufferCached;

	ShellCB cbData{};
	cbData.CameraViewProj = fb.GetCameraViewProj();
	cbData.CameraViewProjUnjittered = fb.GetCameraViewProjUnjittered();
	cbData.CameraPreviousViewProjUnjittered = fb.GetCameraPreviousViewProjUnjittered();
	cbData.CameraView = fb.GetCameraView();
	cbData.CameraPosAdjust = fb.GetCameraPosAdjust();
	cbData.CameraPreviousPosAdjust = fb.GetCameraPreviousPosAdjust();

	// Shell range: the slider scales the warped grid's inner spacing — the
	// warp shape is unchanged, so range costs no extra vertices, only
	// near-field density (8 units at the 375 m default).
	const float shellSpacing = kShellGridSpacing * std::clamp(settings.RangeShellM, 94.0f, 750.0f) * kUnitsPerMeter / ShellWarpedHalfSpan(kShellGridSpacing);
	cbData.GridSpacing = shellSpacing;
	cbData.GridDim = kShellGridDim;
	// The warped grid is camera-centered: snap the center to the grid step
	// so inner vertices stay texel-stable, then offset by the warped span.
	const float warpedHalfSpan = ShellWarpedHalfSpan(shellSpacing);
	cbData.WarpedHalfSpan = warpedHalfSpan;
	cbData.GridOrigin = {
		std::floor(cbData.CameraPosAdjust.x / kShellGridSpacing) * kShellGridSpacing - warpedHalfSpan,
		std::floor(cbData.CameraPosAdjust.y / kShellGridSpacing) * kShellGridSpacing - warpedHalfSpan
	};

	cbData.SnowDepth = settings.DeformationDepth;
	cbData.TerrainTexelSize = kShellVertexSpacing;
	cbData.TerrainDim = kShellWindowDim;
	cbData.ShellDebugData = shellDataDebug;
	cbData.DeformInvWorldSize = 1.0f / deformWorldSize;

	// Keep shader-side sampling math in small grid-local coordinates.
	cbData.GridToTerrainOffset = {
		cbData.GridOrigin.x - shellWindowCellX * 4096.0f,
		cbData.GridOrigin.y - shellWindowCellY * 4096.0f
	};
	cbData.GridToDeformOffset = {
		cbData.GridOrigin.x - windowOrigin.x,
		cbData.GridOrigin.y - windowOrigin.y
	};

	constexpr float kSnowUVTile = 256.0f;
	cbData.SnowUVOffset = {
		std::fmod(cbData.GridOrigin.x, kSnowUVTile),
		std::fmod(cbData.GridOrigin.y, kSnowUVTile)
	};

	// Lazy-load the shell snow diffuse from the user-configurable path (via
	// the MO2 VFS). The "correct" texture is the one the engine projects on
	// snow statics, but MATO records point at shader NIFs whose embedded
	// texture set holds the actual path (extraction queued) — until then the
	// path setting lets the shell be matched to the modlist by eye.
	if (!shellSnowTextureAttempted) {
		shellSnowTextureAttempted = true;
		shellSnowDiffuseSRV = nullptr;
		shellSnowNormalSRV = nullptr;
		shellSnowRmaosSRV = nullptr;
		shellSnowTextureIsPBR = false;

		auto tryLoadDDS = [](const std::string& a_path, winrt::com_ptr<ID3D11ShaderResourceView>& a_srv) {
			a_srv = nullptr;
			std::wstring wide = L"Data\\" + std::wstring(a_path.begin(), a_path.end());
			return SUCCEEDED(DirectX::CreateDDSTextureFromFile(globals::d3d::device, wide.c_str(), nullptr, a_srv.put()));
		};

		std::string chosenPath = settings.SnowTexturePath.empty() ? "Textures\\Landscape\\snow01.dds" : settings.SnowTexturePath;
		for (auto& pathChar : chosenPath)
			if (pathChar == '/')
				pathChar = '\\';
		std::string loweredPath = chosenPath;
		std::transform(loweredPath.begin(), loweredPath.end(), loweredPath.begin(),
			[](unsigned char c) { return (char)std::tolower(c); });

		// TruePBR sets live under Textures\PBR\... with _n / _rmaos companion
		// maps and linear-color albedo. Probe the PBR variant of the chosen
		// path FIRST — matching the landscape's actual material beats the
		// legacy diffuse, and finding it removes the manual Linear guesswork.
		std::string pbrPath;
		if (loweredPath.find("\\pbr\\") != std::string::npos || loweredPath.rfind("pbr\\", 0) == 0)
			pbrPath = chosenPath;
		else if (size_t texPos = loweredPath.find("textures\\"); texPos != std::string::npos)
			pbrPath = chosenPath.substr(0, texPos + 9) + "PBR\\" + chosenPath.substr(texPos + 9);

		if (!pbrPath.empty() && pbrPath.size() > 4 && tryLoadDDS(pbrPath, shellSnowDiffuseSRV)) {
			shellSnowTextureIsPBR = true;
			std::string base = pbrPath.substr(0, pbrPath.size() - 4);
			bool hasNormal = tryLoadDDS(base + "_n.dds", shellSnowNormalSRV);
			bool hasRmaos = tryLoadDDS(base + "_rmaos.dds", shellSnowRmaosSRV);
			logger::info("[SNOW DEFORMATION] PBR snow set: {} (normal={} rmaos={})", pbrPath, hasNormal, hasRmaos);

			// Authored material parameters from the modlist's OWN TruePBR
			// config, so sparkle/roughness/spec follow whatever texture set
			// the user runs: search Data\PBRTextureSets for a JSON whose name
			// contains the texture basename (snow01 → LandscapeSnow01.json).
			snowGlintLogDensity = 6.0f;
			snowGlintMicroRoughness = 0.3f;
			snowGlintDensityRandomization = 5.0f;
			snowGlintScreenSpaceScale = 1.0f;
			snowRoughnessScale = 0.7f;
			snowSpecularLevel = 0.02f;
			try {
				size_t slashPos = base.find_last_of('\\');
				std::string baseName = (slashPos == std::string::npos) ? base : base.substr(slashPos + 1);
				std::transform(baseName.begin(), baseName.end(), baseName.begin(),
					[](unsigned char c) { return (char)std::tolower(c); });
				for (const auto& entry : std::filesystem::directory_iterator("Data\\PBRTextureSets")) {
					if (!entry.is_regular_file())
						continue;
					std::string fname = entry.path().filename().string();
					std::string fnameLower = fname;
					std::transform(fnameLower.begin(), fnameLower.end(), fnameLower.begin(),
						[](unsigned char c) { return (char)std::tolower(c); });
					if (!fnameLower.ends_with(".json") || fnameLower.find(baseName) == std::string::npos)
						continue;
					std::ifstream file(entry.path());
					nlohmann::json cfg = nlohmann::json::parse(file, nullptr, false);
					if (cfg.is_discarded())
						continue;
					if (auto glintIt = cfg.find("glintParameters"); glintIt != cfg.end() && glintIt->is_object()) {
						snowGlintLogDensity = glintIt->value("logMicrofacetDensity", 6.0f);
						// Same clamps Lighting.hlsl applies (PBR::Constants).
						snowGlintMicroRoughness = std::clamp(glintIt->value("microfacetRoughness", 1.0f), 0.005f, 0.3f);
						snowGlintDensityRandomization = std::clamp(glintIt->value("densityRandomization", 5.0f), 0.0f, 5.0f);
						snowGlintScreenSpaceScale = std::max(1.0f, glintIt->value("screenSpaceScale", 1.0f));
						if (!glintIt->value("enabled", true))
							snowGlintLogDensity = 0.0f;  // below the shader's >1.1 gate
					}
					snowRoughnessScale = cfg.value("roughnessScale", 0.7f);
					snowSpecularLevel = cfg.value("specularLevel", 0.02f);
					logger::info("[SNOW DEFORMATION] PBR config matched: {} (glintDensity={:.1f} roughScale={:.2f} spec={:.3f})",
						fname, snowGlintLogDensity, snowRoughnessScale, snowSpecularLevel);
					break;
				}
			} catch (const std::exception& e) {
				logger::info("[SNOW DEFORMATION] PBR config scan failed: {}", e.what());
			}
		} else {
			bool ok = tryLoadDDS(chosenPath, shellSnowDiffuseSRV);
			if (!ok && chosenPath != "Textures\\Landscape\\snow01.dds") {
				logger::info("[SNOW DEFORMATION] Snow diffuse not loose-file loadable: {}", chosenPath);
				chosenPath = "Textures\\Landscape\\snow01.dds";
				ok = tryLoadDDS(chosenPath, shellSnowDiffuseSRV);
			}
			logger::info("[SNOW DEFORMATION] Snow diffuse load ({}): {}", ok ? "ok (legacy)" : "missing, using fallback color", chosenPath);
		}
	}
	cbData.HasSnowTexture = shellSnowDiffuseSRV != nullptr;
	cbData.SnowTextureIsLinear = (shellSnowTextureIsPBR || settings.SnowTextureLinear) ? 1.0f : 0.0f;
	cbData.HasSnowNormal = shellSnowNormalSRV ? 1.0f : 0.0f;
	cbData.HasSnowRmaos = shellSnowRmaosSRV ? 1.0f : 0.0f;
	cbData.SnowRoughnessScale = snowRoughnessScale;
	cbData.SnowGlintParams = { snowGlintLogDensity, snowGlintMicroRoughness, snowGlintDensityRandomization, snowGlintScreenSpaceScale };
	cbData.SnowSpecularLevel = snowSpecularLevel;
	cbData.BorderNoise = settings.SnowBorderNoise;
	cbData.BorderSmooth = settings.SnowBorderSmoothness;
	cbData.BorderTrampledFade = settings.SnowBorderTrampledFade;
	cbData.BorderUntrampledFade = settings.SnowBorderUntrampledFade;
	cbData.SnowSnowFade = settings.SnowSnowFade;
	// Statics-skin distance dissolve: starts at the blend slider, fully gone
	// at the skins capture range (floored one meter past the start so the
	// smoothstep never degenerates when the sliders cross).
	cbData.SkinFadeStart = settings.RangeSkinsFadeM * kUnitsPerMeter;
	cbData.SkinFadeEnd = std::max(settings.RangeSkinsM * kUnitsPerMeter, cbData.SkinFadeStart + kUnitsPerMeter);
	// Height field: the object-blanket lift is gone — the field now carries
	// only terrain + corpse mounds, with shelter/exclusions in the mask, so
	// the shell branch stays permanently enabled.
	cbData.ObjectLiftCap = kObjectLiftCap;

	// Slice-2 LOD shadow cascade: the shared DirectionalShadowLights buffer
	// carries only cascades 0/1, so distant LOD tree shadows reached bare
	// ground but never the shell. Mirror Deferred::CopyShadowLightData's
	// matrix convention (XMStoreFloat4x4 into a column_major HLSL field).
	cbData.LodShadowActive = 0.0f;
	dbgLodDescriptorCount = 0;
	if (auto* shadowSceneNode = globals::game::smState->shadowSceneNode[0]) {
		if (auto* sunShadowLight = shadowSceneNode->GetRuntimeData().sunShadowDirLight) {
			auto& dirLightData = sunShadowLight->GetShadowDirectionalLightRuntimeData();
			auto& shadowDescriptors = sunShadowLight->GetRuntimeData().shadowmapDescriptors;
			dbgLodDescriptorCount = shadowDescriptors.size();
			dbgLodEndSplits[0] = dirLightData.endSplitDistances[0];
			dbgLodEndSplits[1] = dirLightData.endSplitDistances[1];
			dbgLodEndSplits[2] = dirLightData.endSplitDistances[2];
			if (shadowDescriptors.size() >= 3) {
				auto lodProj = DirectX::XMLoadFloat4x4(reinterpret_cast<const DirectX::XMFLOAT4X4*>(&shadowDescriptors[2].lightTransform));
				DirectX::XMStoreFloat4x4(&cbData.LodShadowProj, lodProj);
				cbData.LodShadowEnd = dirLightData.endSplitDistances[2];
				cbData.LodShadowActive = 1.0f;
			}
		}
	}
	dbgLodActive = cbData.LodShadowActive;

	// Screen-Space Shadows: bind its output (t45) for the shell + statics
	// passes and flag it valid. The feature clears the texture to WHITE
	// every Prepass even when disabled, so multiplying is always safe once
	// the texture exists.
	cbData.ScreenSpaceShadowsActive = 0.0f;
	auto& screenSpaceShadowsFeature = globals::features::screenSpaceShadows;
	if (screenSpaceShadowsFeature.loaded && screenSpaceShadowsFeature.screenSpaceShadowsTexture) {
		ID3D11ShaderResourceView* sssSRV = screenSpaceShadowsFeature.screenSpaceShadowsTexture->srv.get();
		globals::d3d::context->PSSetShaderResources(45, 1, &sssSRV);
		cbData.ScreenSpaceShadowsActive = 1.0f;
	}
	// TEMPORARY: throttled log so the whole chain is visible in
	// CommunityShaders.log without menu screenshots.
	static uint32_t dbgLodLogCounter = 0;
	if (++dbgLodLogCounter % 600 == 1)
		logger::info("[SNOW DEFORMATION] LOD shadow debug: descriptors={} endSplits={:.0f}/{:.0f}/{:.0f} active={:.0f} atlasSlices={}",
			dbgLodDescriptorCount, dbgLodEndSplits[0], dbgLodEndSplits[1], dbgLodEndSplits[2], dbgLodActive, dbgLodAtlasSlices);
	cbData.ObjectHeightCenter = heightWindowCenter;
	cbData.ObjectHeightHalfExtent = heightHalfExtent;

	cbData.RoadSnowDepth = settings.RoadSnowDepth;

	// Sparkle: the shell's specular runs TruePBR's glint NDF when its shared
	// noise texture exists (bound to t20 below for the whole pass).
	cbData.EnableGlints = globals::features::truePBR.glintsNoiseTexture ? 1.0f : 0.0f;

	// Crisp shadows: full-resolution comparison PCF against the cascade-atlas
	// copies taken at EarlyPrepass. When the copies are missing this frame,
	// the shader falls back to the blurred VSM path.
	cbData.CrispShadows = (shadowAtlasCopySRV && shadowEsramCopySRV) ? 1.0f : 0.0f;

	// One-shot draw-state log for diagnosing an invisible shell.
	static bool loggedOnce = false;
	if (!loggedOnce) {
		loggedOnce = true;
		logger::info("[SNOW DEFORMATION] Shell draw: vs={} ps={} camera=({:.0f}, {:.0f}, {:.0f}) gridOrigin=({:.0f}, {:.0f})",
			(void*)vs, (void*)ps,
			cbData.CameraPosAdjust.x, cbData.CameraPosAdjust.y, cbData.CameraPosAdjust.z,
			cbData.GridOrigin.x, cbData.GridOrigin.y);
	}

	shellCB->Update(cbData);

	// Back up the pipeline state we touch so the composite and later game
	// passes see exactly what they expect.
	winrt::com_ptr<ID3D11RasterizerState> prevRaster;
	winrt::com_ptr<ID3D11DepthStencilState> prevDepth;
	winrt::com_ptr<ID3D11BlendState> prevBlend;
	UINT prevStencilRef = 0;
	FLOAT prevBlendFactor[4]{};
	UINT prevSampleMask = 0xFFFFFFFF;
	D3D11_PRIMITIVE_TOPOLOGY prevTopology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
	context->RSGetState(prevRaster.put());
	context->OMGetDepthStencilState(prevDepth.put(), &prevStencilRef);
	context->OMGetBlendState(prevBlend.put(), prevBlendFactor, &prevSampleMask);
	context->IAGetPrimitiveTopology(&prevTopology);
	D3D11_VIEWPORT prevViewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]{};
	UINT prevViewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
	context->RSGetViewports(&prevViewportCount, prevViewports);

	// S7: rasterize this frame's captured statics top-down into the object
	// height map — the blanket lift source — then restore the viewport for
	// the screen-space shell pass.
	if (EnsureStaticsShaders())
		RenderObjectHeightMap();
	if (prevViewportCount)
		context->RSSetViewports(prevViewportCount, prevViewports);

	// The height pass recentered the window AFTER the shell CB was filled —
	// sampling the scrolled map with last frame's center made the whole
	// blanket trail the camera by one frame of movement. Re-upload with the
	// current center.
	cbData.ObjectHeightCenter = heightWindowCenter;
	shellCB->Update(cbData);

	// Snapshot for next frame's shadow-caster injection (runs at the shadow
	// mask pass, before DrawShell recomputes these values).
	if (!lastShellCBData)
		lastShellCBData = std::make_unique<ShellCB>();
	*lastShellCBData = cbData;

	// Bind the deferred G-buffer exactly as StartDeferred configures it,
	// plus the main depth buffer for correct intersection with the world.
	auto& rtData = renderer->GetRuntimeData();
	ID3D11RenderTargetView* rtvs[8] = {
		rtData.renderTargets[RE::RENDER_TARGETS::kMAIN].RTV,
		rtData.renderTargets[RE::RENDER_TARGETS::kMOTION_VECTOR].RTV,
		rtData.renderTargets[NORMALROUGHNESS].RTV,
		rtData.renderTargets[ALBEDO].RTV,
		rtData.renderTargets[SPECULAR].RTV,
		rtData.renderTargets[REFLECTANCE].RTV,
		rtData.renderTargets[MASKS].RTV,
		rtData.renderTargets[MASKS2].RTV
	};
	auto dsv = renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kMAIN].views[0];
	context->OMSetRenderTargets(8, rtvs, dsv);

	context->IASetInputLayout(nullptr);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	context->RSSetState(shellRasterState.get());
	context->OMSetDepthStencilState(shellDepthState.get(), 0);
	context->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);

	ID3D11Buffer* cbs[1] = { shellCB->CB() };
	context->VSSetConstantBuffers(0, 1, cbs);
	context->PSSetConstantBuffers(0, 1, cbs);
	// SharedData (b5) supplies SH ambient + sun for the PS lighting; rebind
	// the b4-b6 triple exactly as Deferred does for its own passes.
	auto state = globals::state;
	ID3D11Buffer* sharedBuffers[3] = { state->permutationCB->CB(), state->sharedDataCB->CB(), state->featureDataCB->CB() };
	context->PSSetConstantBuffers(4, 3, sharedBuffers);
	// The PS evaluates ShellSurfaceZ for per-pixel normals, so the field
	// textures must be bound to BOTH stages; the snow diffuse and scene depth
	// are PS-only. The depth SRV is a copy (Terrain Blending's blended depth
	// when available), never the bound DSV, so sampling it here is legal;
	// the PS fades the shell where it hovers close in front of any geometry
	// so it dissolves into statics (walkways, mesh roads, rocks).
	ID3D11ShaderResourceView* shellSRVs[8] = { shellTerrainTexture->srv.get(), GetDeformationSRV(), shellSnowDiffuseSRV.get(), Util::GetCurrentSceneDepthSRV(false), heightTopFiltered->srv.get(), heightBottomFiltered->srv.get(), shellSnowNormalSRV.get(), shellSnowRmaosSRV.get() };
	context->VSSetShaderResources(0, 6, shellSRVs);
	context->PSSetShaderResources(0, 8, shellSRVs);
	// Glint noise (t20): TruePBR binds this each prepass, but slot 20's state
	// at deferred time is not guaranteed — bind explicitly for this pass (the
	// statics skin inherits it).
	if (globals::features::truePBR.glintsNoiseTexture) {
		ID3D11ShaderResourceView* glintSRV = globals::features::truePBR.glintsNoiseTexture->srv.get();
		context->PSSetShaderResources(20, 1, &glintSRV);
	}
	// Raw shadow atlases (t22/t23) + comparison sampler (s2) for crisp
	// cascade shadows; the statics skin inherits these too.
	if (cbData.CrispShadows > 0.5f) {
		if (!shadowCmpSampler) {
			D3D11_SAMPLER_DESC cmpDesc{};
			cmpDesc.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
			cmpDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
			cmpDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
			cmpDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
			cmpDesc.ComparisonFunc = D3D11_COMPARISON_LESS_EQUAL;
			cmpDesc.MaxLOD = D3D11_FLOAT32_MAX;
			globals::d3d::device->CreateSamplerState(&cmpDesc, shadowCmpSampler.put());
			Util::SetResourceName(shadowCmpSampler.get(), "SnowDeformation::ShadowCmpSampler");
		}
		ID3D11ShaderResourceView* shadowSRVs[2] = { shadowAtlasCopySRV.get(), shadowEsramCopySRV.get() };
		context->PSSetShaderResources(22, 2, shadowSRVs);
		ID3D11SamplerState* cmpSampler = shadowCmpSampler.get();
		context->PSSetSamplers(2, 1, &cmpSampler);
	}

	winrt::com_ptr<ID3D11SamplerState> prevSamplers[2];
	context->PSGetSamplers(0, 1, prevSamplers[0].put());
	context->PSGetSamplers(1, 1, prevSamplers[1].put());
	ID3D11SamplerState* shellSamplers[2] = { shellSnowSampler.get(), shellLinearSampler.get() };
	context->PSSetSamplers(0, 2, shellSamplers);

	context->VSSetShader(vs, nullptr, 0);
	context->PSSetShader(ps, nullptr, 0);

	globals::profiler->BeginPass("SnowDeformation::Shell");
	context->Draw(kShellGridDim * kShellGridDim * 6, 0);
	globals::profiler->EndPass();

	// Post-shell depth copy (Terrain Blending's technique adapted): the main
	// depth now contains the landscape shell's surface. The statics skin
	// samples this at t9 to measure its view-ray gap to the shell and cross-
	// fade into it — fading toward what is ACTUALLY behind the pixel, which
	// a height-based band cannot guarantee (it exposed the road beneath).
	// Targets must be unbound around CopyResource of a bound DSV.
	{
		auto& mainDepthDS = globals::game::renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kMAIN];
		if (mainDepthDS.depthSRV) {
			ID3D11RenderTargetView* boundRTVs[8] = {};
			ID3D11DepthStencilView* boundDSV = nullptr;
			context->OMGetRenderTargets(8, boundRTVs, &boundDSV);
			context->OMSetRenderTargets(0, nullptr, nullptr);
			SD_CopyShadowTarget(mainDepthDS.depthSRV, "SnowDeformation::ShellDepthCopy", shellDepthCopyTex, shellDepthCopySRV);
			context->OMSetRenderTargets(8, boundRTVs, boundDSV);
			for (auto* rtv : boundRTVs)
				if (rtv)
					rtv->Release();
			if (boundDSV)
				boundDSV->Release();
		}
		if (shellDepthCopySRV) {
			ID3D11ShaderResourceView* copySRV = shellDepthCopySRV.get();
			context->PSSetShaderResources(9, 1, &copySRV);
		}
	}

	// S6: captured projected-snow statics, inflated with the same material.
	// Inherits this pass's bindings (b0, t1-t3, s0/s1, b4-b6, RTs, depth).
	DrawCapturedStatics();

	// Restore everything we changed.
	context->VSSetShader(nullptr, nullptr, 0);
	context->PSSetShader(nullptr, nullptr, 0);
	ID3D11Buffer* nullCB = nullptr;
	context->VSSetConstantBuffers(0, 1, &nullCB);
	context->PSSetConstantBuffers(0, 1, &nullCB);
	ID3D11ShaderResourceView* nullSRVs[10] = {};
	context->VSSetShaderResources(0, 6, nullSRVs);
	context->VSSetShaderResources(10, 1, nullSRVs);
	context->PSSetShaderResources(0, 10, nullSRVs);
	// t22/t23 hold SRVs of the game's shadow depth targets — they MUST be
	// unbound before the next shadow render binds those targets as DSVs, or
	// D3D silently drops the binding with warning spam. t20 (glint noise)
	// cleared alongside for symmetry.
	ID3D11ShaderResourceView* nullShadowSRVs[4] = { nullptr, nullptr, nullptr, nullptr };
	context->PSSetShaderResources(20, 4, nullShadowSRVs);
	ID3D11SamplerState* restoreSamplers[2] = { prevSamplers[0].get(), prevSamplers[1].get() };
	context->PSSetSamplers(0, 2, restoreSamplers);
	ID3D11SamplerState* nullSampler = nullptr;
	context->PSSetSamplers(2, 1, &nullSampler);
	context->OMSetRenderTargets(0, nullptr, nullptr);
	context->RSSetState(prevRaster.get());
	context->OMSetDepthStencilState(prevDepth.get(), prevStencilRef);
	context->OMSetBlendState(prevBlend.get(), prevBlendFactor, prevSampleMask);
	context->IASetPrimitiveTopology(prevTopology);
	globals::game::stateUpdateFlags->set(RE::BSGraphics::ShaderFlags::DIRTY_RENDERTARGET);

	// Screen-space passes running after us (SSGI) read Terrain Blending's
	// blended depth, which was finalized during opaque rendering — without a
	// sync they see buried geometry poking up through the snow and paint
	// occlusion halos onto the shell. min() the shell's fresh depth writes
	// into both blended copies (DSV is unbound again at this point).
	auto& tb = globals::features::terrainBlending;
	if (tb.loaded && tb.settings.Enabled && tb.blendedDepthTexture && tb.blendedDepthTexture16) {
		if (auto cs = GetDepthSyncCS()) {
			auto mainDepthSRV = renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kMAIN].depthSRV;
			ID3D11UnorderedAccessView* syncUAVs[2] = { tb.blendedDepthTexture->uav.get(), tb.blendedDepthTexture16->uav.get() };
			context->CSSetShaderResources(0, 1, &mainDepthSRV);
			context->CSSetUnorderedAccessViews(0, 2, syncUAVs, nullptr);
			context->CSSetShader(cs, nullptr, 0);
			const auto& depthDesc = tb.blendedDepthTexture->desc;
			globals::profiler->BeginPass("SnowDeformation::DepthSync");
			context->Dispatch((depthDesc.Width + 7) / 8, (depthDesc.Height + 7) / 8, 1);
			globals::profiler->EndPass();

			ID3D11ShaderResourceView* nullSyncSRV = nullptr;
			ID3D11UnorderedAccessView* nullSyncUAVs[2] = { nullptr, nullptr };
			context->CSSetShaderResources(0, 1, &nullSyncSRV);
			context->CSSetUnorderedAccessViews(0, 2, nullSyncUAVs, nullptr);
			context->CSSetShader(nullptr, nullptr, 0);
		}
	}
}

void SnowDeformation::LoadSettings(json& o_json)
{
	settings = o_json;
}

void SnowDeformation::SaveSettings(json& o_json)
{
	o_json = settings;
}

void SnowDeformation::RestoreDefaultSettings()
{
	settings = {};
	clearRequested = true;
}

/** @brief Returns the kSnowClasses index for a land texture, or -1 when there is no texture at all. Every real texture classifies: name match first; unmatched snow-material textures land in "Snow 01", everything else in "Other". */
static int ClassifySnowClass(RE::TESLandTexture* a_landTexture)
{
	if (!a_landTexture || a_landTexture->formID == 0)
		return -1;

	if (auto textureSet = a_landTexture->textureSet) {
		if (auto path = textureSet->GetTexturePath(RE::BSTextureSet::Texture::kDiffuse)) {
			std::string lowered(path);
			std::transform(lowered.begin(), lowered.end(), lowered.begin(),
				[](unsigned char c) { return (char)std::tolower(c); });
			for (uint32_t classI = 0; classI < SnowDeformation::kSnowClassCount; ++classI) {
				const char* match = SnowDeformation::kSnowClasses[classI].match;
				if (match[0] != '\0' && lowered.find(match) != std::string::npos)
					return (int)classI;
			}
		}
	}

	bool snowMaterial = a_landTexture->materialType &&
	                    (a_landTexture->materialType->materialID == RE::MATERIAL_ID::kSnow ||
							a_landTexture->materialType->materialID == RE::MATERIAL_ID::kSnowStairs);
	return snowMaterial ? 3 /* Snow 01 */ : (int)SnowDeformation::kSnowClassCount - 1 /* Other */;
}

/** @brief True for classes that count as snow (terrain-shader mask bits). */
static bool IsSnowClass(int a_classIndex)
{
	return a_classIndex >= 0 && a_classIndex < (int)SnowDeformation::kSnowOnlyClassCount;
}

// One-shot diagnostic: log every distinct land texture the bake encounters
// with its classification, so misclassified modlist textures (e.g. static
// mesh roads that never appear here at all) can be identified from the log.
static void LogLandTextureOnce(RE::TESLandTexture* a_landTexture, int a_classIndex)
{
	static std::mutex logMutex;
	static std::unordered_set<uint32_t> loggedForms;

	if (!a_landTexture)
		return;

	const std::lock_guard lock(logMutex);
	if (loggedForms.size() > 200 || !loggedForms.insert(a_landTexture->formID).second)
		return;

	const char* path = "<no texture set>";
	if (a_landTexture->textureSet)
		if (auto p = a_landTexture->textureSet->GetTexturePath(RE::BSTextureSet::Texture::kDiffuse))
			path = p;

	const char* label = a_classIndex < 0 ? "BARE" : SnowDeformation::kSnowClasses[a_classIndex].label;
	logger::info("[SNOW DEFORMATION] LTEX {:08X} [{}] {}", a_landTexture->formID, label, path);
}

void SnowDeformation::TESObjectLAND_SetupMaterial(RE::TESObjectLAND* land)
{
	if (land == nullptr || land->loadedData == nullptr || land->loadedData->mesh[0] == nullptr)
		return;

	for (uint32_t quadI = 0; quadI < 4; ++quadI) {
		if (land->loadedData->mesh[quadI] == nullptr)
			continue;

		// This hook is OUTER relative to TruePBR's, so the shader property here
		// is the final one used for drawing — vanilla or TruePBR-replaced.
		const auto& children = land->loadedData->mesh[quadI]->GetChildren();
		auto geometry = children.empty() ? nullptr : static_cast<RE::BSGeometry*>(children[0].get());
		if (geometry == nullptr)
			continue;

		const auto shaderProp = static_cast<RE::BSLightingShaderProperty*>(geometry->GetGeometryRuntimeData().shaderProperty.get());
		if (shaderProp == nullptr || shaderProp->material == nullptr)
			continue;

		uint8_t mask = 0;
		if (IsSnowClass(ClassifySnowClass(land->loadedData->defQuadTextures[quadI])))
			mask |= 1;
		for (uint32_t textureI = 0; textureI < 5; ++textureI) {
			if (IsSnowClass(ClassifySnowClass(land->loadedData->quadTextures[quadI][textureI])))
				mask |= uint8_t(1 << (textureI + 1));
		}

		const std::unique_lock lock(snowMaskMutex);
		// Materials are freed on cell unload; bound the map so stale pointers
		// cannot accumulate over a long session.
		if (snowMasks.size() > 16384)
			snowMasks.clear();
		snowMasks[reinterpret_cast<uintptr_t>(shaderProp->material)] = mask;
	}

	BakeShellCell(land);
}

void SnowDeformation::BakeShellCell(RE::TESObjectLAND* land)
{
	auto cell = land->GetSaveParentCell();
	if (!cell)
		return;
	auto coords = cell->GetCoordinates();
	if (!coords)
		return;

	ShellCellData data{};
	data.height.fill(-100000.0f);
	for (auto& classArray : data.classWeights)
		classArray.fill(0);

	auto loadedData = land->loadedData;

	// One-shot exploration log: dump every candidate source of the absolute
	// height base for the first baked cell, so the correct one can be picked
	// from real data instead of guesses (world transforms are not composed
	// yet at setup time; translate-based placement collapsed the quads).
	static bool loggedBaseCandidates = false;
	if (!loggedBaseCandidates) {
		loggedBaseCandidates = true;
		float rawMin = FLT_MAX, rawMax = -FLT_MAX;
		for (uint32_t q = 0; q < 4; ++q)
			for (uint32_t v = 0; v < 289; ++v) {
				rawMin = std::min(rawMin, loadedData->heights[q][v]);
				rawMax = std::max(rawMax, loadedData->heights[q][v]);
			}
		logger::info("[SNOW DEFORMATION] Base candidates for cell ({}, {}): rawHeights=[{:.0f}, {:.0f}] heightExtents=({:.0f}, {:.0f})",
			coords->cellX, coords->cellY, rawMin, rawMax, loadedData->heightExtents.x, loadedData->heightExtents.y);
		for (uint32_t q = 0; q < 4; ++q) {
			auto mesh = loadedData->mesh[q];
			auto& geom = loadedData->geom[q];
			logger::info("[SNOW DEFORMATION]   quad {}: meshLocal=({:.0f}, {:.0f}, {:.0f}) meshWorld=({:.0f}, {:.0f}, {:.0f}) geomLocal=({:.0f}, {:.0f}, {:.0f}) geomWorld=({:.0f}, {:.0f}, {:.0f}) geomBoundZ={:.0f}",
				q,
				mesh ? mesh->local.translate.x : 0.0f, mesh ? mesh->local.translate.y : 0.0f, mesh ? mesh->local.translate.z : 0.0f,
				mesh ? mesh->world.translate.x : 0.0f, mesh ? mesh->world.translate.y : 0.0f, mesh ? mesh->world.translate.z : 0.0f,
				geom ? geom->local.translate.x : 0.0f, geom ? geom->local.translate.y : 0.0f, geom ? geom->local.translate.z : 0.0f,
				geom ? geom->world.translate.x : 0.0f, geom ? geom->world.translate.y : 0.0f, geom ? geom->world.translate.z : 0.0f,
				geom ? geom->worldBound.center.z : 0.0f);
		}
	}

	// heights[] are relative to the cell's mid-height; the absolute base is
	// the midpoint of heightExtents (verified: quad geometry local z equals
	// exactly (extents.x + extents.y) / 2).
	float cellBaseZ = (loadedData->heightExtents.x + loadedData->heightExtents.y) * 0.5f;

	for (uint32_t quadI = 0; quadI < 4; ++quadI) {
		// Index-based quad layout (0=SW, 1=SE, 2=NW, 3=NE) — confirmed by the
		// quad 3 geometry transform in the exploration log.
		float quadBaseZ = cellBaseZ;
		uint32_t quadX = quadI & 1;
		uint32_t quadY = quadI >> 1;

		int baseClass = ClassifySnowClass(loadedData->defQuadTextures[quadI]);
		LogLandTextureOnce(loadedData->defQuadTextures[quadI], baseClass);
		int layerClass[6];
		for (uint32_t layerI = 0; layerI < 6; ++layerI) {
			layerClass[layerI] = ClassifySnowClass(loadedData->quadTextures[quadI][layerI]);
			LogLandTextureOnce(loadedData->quadTextures[quadI][layerI], layerClass[layerI]);
		}

		for (uint32_t vertexI = 0; vertexI < 289; ++vertexI) {
			uint32_t vx = vertexI % 17;
			uint32_t vy = vertexI / 17;
			uint32_t cellX = quadX * 16 + vx;
			uint32_t cellY = quadY * 16 + vy;
			uint32_t cellIdx = cellY * 33 + cellX;

			data.height[cellIdx] = loadedData->heights[quadI][vertexI] + quadBaseZ;

			float layerSum = 0.0f;
			float classSum[kSnowClassCount] = {};
			for (uint32_t layerI = 0; layerI < 6; ++layerI) {
				// percents is declared std::int8_t but holds 0-255: a fully
				// painted layer reads as -1 without the unsigned cast, which
				// zeroes strong layers and hands their weight to the base.
				float weight = static_cast<uint8_t>(loadedData->percents[quadI][vertexI][layerI]) / 255.0f;
				layerSum += weight;
				if (layerClass[layerI] >= 0)
					classSum[layerClass[layerI]] += weight;
			}
			if (baseClass >= 0)
				classSum[baseClass] += std::max(0.0f, 1.0f - layerSum);
			for (uint32_t classI = 0; classI < kSnowClassCount; ++classI)
				data.classWeights[classI][cellIdx] = (uint8_t)std::clamp(classSum[classI] * 255.0f + 0.5f, 0.0f, 255.0f);
		}
	}

	uint64_t key = (uint64_t(uint32_t(coords->cellX)) << 32) | uint32_t(coords->cellY);
	{
		const std::unique_lock lock(shellCellMutex);
		if (shellCells.size() > 4096)
			shellCells.clear();
		// The engine re-runs land material setup frequently; only mark the
		// window dirty when the baked data actually changed, so the window
		// is not re-uploaded every frame.
		auto it = shellCells.find(key);
		if (it != shellCells.end() && it->second.height == data.height && it->second.classWeights == data.classWeights)
			return;
		shellCells[key] = data;
	}
	shellDataDirty.store(true, std::memory_order_release);
}

void SnowDeformation::UpdateShellTerrainWindow()
{
	auto eyeFB = globals::game::frameBufferCached.GetCameraPosAdjust();
	int camCellX = (int)std::floor(eyeFB.x / 4096.0f);
	int camCellY = (int)std::floor(eyeFB.y / 4096.0f);

	int desiredOriginX = camCellX - kShellWindowCells / 2;
	int desiredOriginY = camCellY - kShellWindowCells / 2;

	bool originChanged = desiredOriginX != shellWindowCellX || desiredOriginY != shellWindowCellY;
	if (!originChanged && !shellDataDirty.exchange(false, std::memory_order_acq_rel))
		return;

	shellWindowCellX = desiredOriginX;
	shellWindowCellY = desiredOriginY;
	shellDataDirty.store(false, std::memory_order_release);

	shellUploadScratch.resize(size_t(kShellWindowDim) * kShellWindowDim * 4);

	uint32_t statSnowTexels = 0;
	float statMinH = FLT_MAX;
	float statMaxH = -FLT_MAX;
	std::unordered_set<uint64_t> statCells;

	{
		const std::shared_lock lock(shellCellMutex);
		for (int ty = 0; ty < kShellWindowDim; ++ty) {
			int cellY = shellWindowCellY + ty / kShellTexelsPerCell;
			int vy = ty % kShellTexelsPerCell;
			const ShellCellData* rowCell = nullptr;
			uint64_t rowKey = ~0ull;
			for (int tx = 0; tx < kShellWindowDim; ++tx) {
				int cellX = shellWindowCellX + tx / kShellTexelsPerCell;
				int vx = tx % kShellTexelsPerCell;

				uint64_t key = (uint64_t(uint32_t(cellX)) << 32) | uint32_t(cellY);
				if (key != rowKey) {
					auto it = shellCells.find(key);
					rowCell = it != shellCells.end() ? &it->second : nullptr;
					rowKey = key;
				}

				float* texel = &shellUploadScratch[(size_t(ty) * kShellWindowDim + tx) * 4];
				if (rowCell) {
					uint32_t idx = uint32_t(vy) * 33 + uint32_t(vx);
					// Per-class depths are applied HERE, so the class sliders
					// retune the shell from cached weights without a re-bake.
					float rampDepth = 0.0f;
					float coverage = 0.0f;
					for (uint32_t classI = 0; classI < kSnowClassCount; ++classI) {
						float w = rowCell->classWeights[classI][idx] / 255.0f;
						rampDepth += w * settings.SnowClassDepths[classI];
						coverage += w;
					}
					texel[0] = rowCell->height[idx];
					texel[1] = rampDepth;
					texel[2] = std::min(coverage, 1.0f);
					texel[3] = 0.0f;

					statCells.insert(rowKey);
					statMinH = std::min(statMinH, texel[0]);
					statMaxH = std::max(statMaxH, texel[0]);
					if (texel[2] > 0.05f)
						statSnowTexels++;
				} else {
					texel[0] = -100000.0f;
					texel[1] = 0.0f;
					texel[2] = 0.0f;
					texel[3] = 0.0f;
				}
			}
		}
	}

	shellStatCellsInWindow = (uint32_t)statCells.size();
	shellStatSnowTexels = statSnowTexels;
	shellStatMinHeight = statMinH == FLT_MAX ? 0.0f : statMinH;
	shellStatMaxHeight = statMaxH == -FLT_MAX ? 0.0f : statMaxH;

	logger::info("[SNOW DEFORMATION] Shell window rebuilt: origin cell ({}, {}), {} cells in window, {} snow texels, height range [{:.0f}, {:.0f}]",
		shellWindowCellX, shellWindowCellY, shellStatCellsInWindow, shellStatSnowTexels, shellStatMinHeight, shellStatMaxHeight);

	globals::d3d::context->UpdateSubresource(shellTerrainTexture->resource.get(), 0, nullptr,
		shellUploadScratch.data(), kShellWindowDim * 4 * sizeof(float), 0);
}

void SnowDeformation::BSLightingShader_SetupMaterial(RE::BSLightingShaderMaterialBase const* material)
{
	auto state = globals::state;

	// Always clear first so bits never leak from the previous landscape draw.
	state->permutationData.ExtraFeatureDescriptor &= ~uint(State::ExtraFeatureDescriptors::SnowLandIsSnowMask);

	if (material == nullptr)
		return;

	uint8_t mask = 0;
	{
		const std::shared_lock lock(snowMaskMutex);
		auto it = snowMasks.find(reinterpret_cast<uintptr_t>(material));
		if (it == snowMasks.end()) {
			// Count misses only for landscape materials, where a miss is a bug.
			auto feature = material->GetFeature();
			if (feature == RE::BSShaderMaterial::Feature::kMultiTexLand || feature == static_cast<RE::BSShaderMaterial::Feature>(33))
				landMaskMisses.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		mask = it->second;
	}

	landMaskHits.fetch_add(1, std::memory_order_relaxed);
	state->permutationData.ExtraFeatureDescriptor |= uint32_t(mask) << 10;
}

struct SD_TESObjectLAND_SetupMaterial
{
	static bool thunk(RE::TESObjectLAND* land)
	{
		bool result = func(land);

		auto& snowDeformation = globals::features::snowDeformation;
		if (result && snowDeformation.loaded)
			snowDeformation.TESObjectLAND_SetupMaterial(land);

		return result;
	}
	static inline REL::Relocation<decltype(thunk)> func;
};

struct SD_BSLightingShader_SetupMaterial
{
	static void thunk(RE::BSLightingShader* shader, RE::BSLightingShaderMaterialBase const* material)
	{
		if (!material)
			return;

		func(shader, material);

		auto& snowDeformation = globals::features::snowDeformation;
		if (snowDeformation.loaded)
			snowDeformation.BSLightingShader_SetupMaterial(material);
	}
	static inline REL::Relocation<decltype(thunk)> func;
};

struct SD_BSLightingShader_SetupGeometry
{
	static void thunk(RE::BSLightingShader* shader, RE::BSRenderPass* pass, uint32_t flags)
	{
		func(shader, pass, flags);

		auto& snowDeformation = globals::features::snowDeformation;
		if (snowDeformation.loaded)
			snowDeformation.BSLightingShader_SetupGeometry(pass);
	}
	static inline REL::Relocation<decltype(thunk)> func;
};

void SnowDeformation::PostPostLoad()
{
	// Same detour target as TruePBR. TruePBR sits earlier in the feature list,
	// so its PostPostLoad detour is already installed; attaching now makes our
	// hook OUTER (detours are LIFO), i.e. we run after TruePBR has replaced
	// quad materials and can key the snow masks by the final material pointer.
	logger::info("[SNOW DEFORMATION] Hooking TESObjectLAND");
	stl::detour_thunk<SD_TESObjectLAND_SetupMaterial>(REL::RelocationID(18368, 18791));

	logger::info("[SNOW DEFORMATION] Hooking BSLightingShader::SetupMaterial");
	stl::write_vfunc<0x4, SD_BSLightingShader_SetupMaterial>(RE::VTABLE_BSLightingShader[0]);

	logger::info("[SNOW DEFORMATION] Hooking BSLightingShader::SetupGeometry");
	stl::write_vfunc<0x6, SD_BSLightingShader_SetupGeometry>(RE::VTABLE_BSLightingShader[0]);
}

void SnowDeformation::BSLightingShader_SetupGeometry(RE::BSRenderPass* a_pass)
{
	if (!a_pass || !a_pass->shaderProperty || !a_pass->geometry)
		return;
	if (!settings.EnableSnowDeformation || (settings.ObjectsSnowDepth <= 0.01f && settings.SnowMeshesDepth <= 0.01f))
		return;
	// Main world view only: probe/reflection passes must not fill the list.
	if (!globals::state->inWorld)
		return;

	// The clean gate (rounds 30-34): projected-UV + snow flags together —
	// covers rocks, roofs, logs, stumps and never flora, because foliage is
	// not snow-PROJECTED. Drifts (no flags at all) qualify via a NARROW
	// texture match on "drift" — the catch-all "snow" match is what dragged
	// frosted bushes in and caused the leaf-card shards.
	using Flag = RE::BSShaderProperty::EShaderPropertyFlag;
	const auto& flags = a_pass->shaderProperty->flags;
	// Animated flora never qualifies: card meshes shard under the statics
	// shell and must not contribute to the blanket height map either.
	if (flags.all(Flag::kTreeAnim))
		return;
	// (Flat-vs-rounded depth classes are decided per MESH on the GPU from
	// smoothed-vs-raw normal divergence — a flag-based split failed because
	// nearly every qualifying draw carries the projected pair.)
	if (!(flags.all(Flag::kProjectedUV) && flags.all(Flag::kSnow))) {
		auto* material = static_cast<RE::BSLightingShaderMaterialBase*>(a_pass->shaderProperty->material);
		if (!material)
			return;

		static std::unordered_map<const void*, bool> driftMaterialCache;
		if (driftMaterialCache.size() > 4096)
			driftMaterialCache.clear();
		auto [it, inserted] = driftMaterialCache.try_emplace(material, false);
		if (inserted) {
			if (auto textureSet = material->textureSet.get()) {
				if (auto path = textureSet->GetTexturePath(RE::BSTextureSet::Texture::kDiffuse)) {
					std::string lowered(path);
					std::transform(lowered.begin(), lowered.end(), lowered.begin(),
						[](unsigned char c) { return (char)std::tolower(c); });
					// Drifts wear plain LANDSCAPE snow textures (no "drift"
					// in the path) — requiring the landscape folder keeps
					// frosted plants (plant/tree folders) out.
					it->second = lowered.find("drift") != std::string::npos ||
					             (lowered.find("landscape") != std::string::npos && lowered.find("snow") != std::string::npos);
				}
			}
		}
		if (!it->second)
			return;
	}

	// Range cap: distant mountains are snow-projected everywhere in Skyrim;
	// the shell only matters where deformation can happen.
	auto eye = globals::game::frameBufferCached.GetCameraPosAdjust();
	const auto& translate = a_pass->geometry->world.translate;
	float dx = translate.x - eye.x;
	float dy = translate.y - eye.y;
	if (dx * dx + dy * dy > (settings.RangeSkinsM * kUnitsPerMeter) * (settings.RangeSkinsM * kUnitsPerMeter))
		return;

	// The same geometry renders through multiple passes; capture once.
	if (!capturedStaticsSet.insert(a_pass->geometry).second)
		return;

	capturedStatics.push_back({ RE::NiPointer<RE::BSGeometry>(a_pass->geometry), a_pass->geometry->world });
}

// Compiles one stage of the statics shell, RETURNING the bytecode blob —
// input layouts must be created against the VS bytecode, which
// Util::CompileShader discards. Include resolution matches CompileShader's
// convention (everything relative to Data\Shaders).
static ID3DBlob* SD_CompileShaderBlob(const wchar_t* a_path, const char* a_target, const char* a_stageDefine)
{
	struct ShaderInclude : public ID3DInclude
	{
		HRESULT Open(D3D_INCLUDE_TYPE, LPCSTR pFileName, LPCVOID, LPCVOID* ppData, UINT* pBytes) override
		{
			std::filesystem::path filePath = pFileName;
			filePath = L"Data\\Shaders" / filePath;
			std::ifstream file(filePath, std::ios::binary);
			if (!file.is_open()) {
				*ppData = nullptr;
				*pBytes = 0;
				return E_FAIL;
			}
			file.seekg(0, std::ios::end);
			UINT size = static_cast<UINT>(file.tellg());
			file.seekg(0, std::ios::beg);
			char* data = new char[size];
			file.read(data, size);
			*ppData = data;
			*pBytes = size;
			return S_OK;
		}
		HRESULT Close(LPCVOID pData) override
		{
			delete[] static_cast<const char*>(pData);
			return S_OK;
		}
	} includeHandler;

	D3D_SHADER_MACRO macros[] = {
		{ a_stageDefine, "" },
		{ "WINPC", "" },
		{ "DX11", "" },
		{ nullptr, nullptr }
	};

	ID3DBlob* blob = nullptr;
	ID3DBlob* errors = nullptr;
	if (FAILED(D3DCompileFromFile(a_path, macros, &includeHandler, "main", a_target,
			D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &errors))) {
		logger::warn("[SNOW DEFORMATION] Statics shell {} compile failed:\n{}", a_target,
			errors ? static_cast<char*>(errors->GetBufferPointer()) : "unknown error");
		if (errors)
			errors->Release();
		return nullptr;
	}
	if (errors)
		errors->Release();
	return blob;
}

bool SnowDeformation::EnsureStaticsShaders()
{
	if (staticsVS && staticsPS)
		return true;
	if (staticsShadersFailed)
		return false;

	constexpr auto path = L"Data\\Shaders\\SnowDeformation\\SnowStaticsShell.hlsl";

	if (!staticsVS) {
		winrt::com_ptr<ID3DBlob> blob;
		blob.attach(SD_CompileShaderBlob(path, "vs_5_0", "VSHADER"));
		if (blob) {
			if (SUCCEEDED(globals::d3d::device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &staticsVS))) {
				staticsVSBlob = blob;
				Util::SetResourceName(staticsVS, "SnowDeformation::StaticsShellVS");
			}
		}
	}
	if (!staticsPS) {
		winrt::com_ptr<ID3DBlob> blob;
		blob.attach(SD_CompileShaderBlob(path, "ps_5_0", "PSHADER"));
		if (blob) {
			if (SUCCEEDED(globals::d3d::device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &staticsPS)))
				Util::SetResourceName(staticsPS, "SnowDeformation::StaticsShellPS");
		}
	}

	constexpr auto heightPath = L"Data\\Shaders\\SnowDeformation\\SnowHeightCapture.hlsl";
	if (!heightVS) {
		winrt::com_ptr<ID3DBlob> blob;
		blob.attach(SD_CompileShaderBlob(heightPath, "vs_5_0", "VSHADER"));
		if (blob)
			globals::d3d::device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &heightVS);
	}
	if (!heightPS) {
		winrt::com_ptr<ID3DBlob> blob;
		blob.attach(SD_CompileShaderBlob(heightPath, "ps_5_0", "PSHADER"));
		if (blob)
			globals::d3d::device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &heightPS);
	}

	constexpr auto processPath = L"Data\\Shaders\\SnowDeformation\\HeightMapProcessCS.hlsl";
	if (!heightScrollCS)
		heightScrollCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(processPath, {}, "cs_5_0", "ScrollCS"));
	if (!heightCombineCS)
		heightCombineCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(processPath, {}, "cs_5_0", "CombineCS"));
	if (!heightConeCS)
		heightConeCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(processPath, {}, "cs_5_0", "ConeCS"));

	if (!staticsVS || !staticsPS || !heightVS || !heightPS || !heightScrollCS || !heightCombineCS || !heightConeCS) {
		staticsShadersFailed = true;
		logger::warn("[SNOW DEFORMATION] Statics shell disabled (shader compilation failed)");
		return false;
	}
	return true;
}

void SnowDeformation::RenderObjectHeightMap()
{
	auto context = globals::d3d::context;

	// Camera-following window, snapped to texel size for stability.
	auto eye = globals::game::frameBufferCached.GetCameraPosAdjust();
	const float texel = heightHalfExtent * 2.0f / heightMapDim;
	float2 newCenter = {
		std::floor(eye.x / texel) * texel,
		std::floor(eye.y / texel) * texel
	};

	// Scroll the ACCUMULATED maps into the new window position: the blanket
	// must not depend on what the camera renders this frame (frustum-culled
	// objects vanishing made the snow follow the camera).
	// +worldY maps to texture -v, so the v-axis delta is negated.
	DirectX::XMINT2 scrollDelta = {
		(int)std::lround((newCenter.x - heightWindowCenter.x) / texel),
		-(int)std::lround((newCenter.y - heightWindowCenter.y) / texel)
	};
	HeightProcessCB processData{};
	processData.ScrollDelta = scrollDelta;
	processData.ClearAll = heightMapValid ? 0u : 1u;
	processData.HeightWindowCenter = newCenter;
	processData.HeightHalfExtent = heightHalfExtent;
	processData.SlopePerUnit = std::clamp(settings.SnowMoundSteepness, 0.5f, 3.0f);  // 1.0 = 45 degrees
	processData.TerrainWindowOrigin = { shellWindowCellX * 4096.0f, shellWindowCellY * 4096.0f };
	processData.TerrainTexelSize = kShellVertexSpacing;
	processData.TerrainDim = kShellWindowDim;
	processData.GhostDecay = 0.5f;
	processData.DeformWindowOriginH = windowOrigin;
	processData.DeformInvWorldSizeH = 1.0f / deformWorldSize;
	processData.CorpseSphereCount = (uint32_t)corpseMoundSpheres.size();
	processData.CorpseMoundCap = 20.0f;
	for (size_t sphereI = 0; sphereI < corpseMoundSpheres.size(); sphereI++)
		processData.CorpseSpheres[sphereI] = corpseMoundSpheres[sphereI];
	heightProcessCB->Update(processData);
	heightWindowCenter = newCenter;
	heightMapValid = true;

	// Exclusion zones: refresh on window scroll and periodically. Doors get
	// elliptical clears along their facing (load doors — cave and building
	// entrances — larger); campfires get noisy-edged full clears.
	bool scrolled = scrollDelta.x != 0 || scrollDelta.y != 0;
	if (scrolled || (doorRefreshCounter++ % 60) == 0) {
		ExclusionsCB exclusionData{};
		uint32_t exclusionCount = 0;
		if (auto player = RE::PlayerCharacter::GetSingleton()) {
			if (auto tes = RE::TES::GetSingleton()) {
				tes->ForEachReferenceInRange(player, heightHalfExtent * 1.5f,
					[&](RE::TESObjectREFR* a_ref) {
						if (exclusionCount >= kMaxExclusions)
							return RE::BSContainer::ForEachResult::kStop;
						if (!a_ref || a_ref->IsDisabled() || !a_ref->Is3DLoaded())
							return RE::BSContainer::ForEachResult::kContinue;
						auto* base = a_ref->GetBaseObject();
						if (!base)
							return RE::BSContainer::ForEachResult::kContinue;

						if (base->Is(RE::FormType::Door)) {
							// Load doors (teleport data) are cave/building
							// entrances — deeper recesses, bigger clears.
							bool loadDoor = a_ref->extraList.HasType(RE::ExtraDataType::kTeleport);
							auto pos = a_ref->GetPosition();
							float angleZ = a_ref->GetAngleZ();
							exclusionData.PosRadius[exclusionCount] = { pos.x, pos.y, pos.z, loadDoor ? kLoadDoorClearRadius : kDoorClearRadius };
							exclusionData.DirExtType[exclusionCount] = { std::sin(angleZ), std::cos(angleZ), loadDoor ? kLoadDoorForwardExtent : kDoorForwardExtent, 0.0f };
							exclusionCount++;
						} else {
							// Explicit form-type chain: skyrim_cast to
							// TESModel silently returned null for activator
							// bases, which made campfires invisible to the
							// gather.
							const char* modelPath = nullptr;
							if (auto* acti = base->As<RE::TESObjectACTI>())
								modelPath = acti->GetModel();
							else if (auto* stat = base->As<RE::TESObjectSTAT>())
								modelPath = stat->GetModel();
							else if (auto* movable = base->As<RE::BGSMovableStatic>())
								modelPath = movable->GetModel();
							if (modelPath && modelPath[0]) {
								std::string lowered(modelPath);
								std::transform(lowered.begin(), lowered.end(), lowered.begin(),
									[](unsigned char c) { return (char)std::tolower(c); });
								if (lowered.find("campfire") != std::string::npos || lowered.find("firepit") != std::string::npos) {
									auto pos = a_ref->GetPosition();
									exclusionData.PosRadius[exclusionCount] = { pos.x, pos.y, pos.z, kFireClearRadius };
									exclusionData.DirExtType[exclusionCount] = { 0.0f, 1.0f, 0.0f, 1.0f };
									exclusionCount++;
								}
							}
						}
						return RE::BSContainer::ForEachResult::kContinue;
					});
			}
		}
		exclusionData.ExclusionCount = exclusionCount;
		doorsCB->Update(exclusionData);
	}

	uint previous = heightCurrent;
	heightCurrent ^= 1;

	ID3D11Buffer* processCB = heightProcessCB->CB();
	context->CSSetConstantBuffers(0, 1, &processCB);
	ID3D11ShaderResourceView* scrollSRVs[2] = { heightTopRaw[previous]->srv.get(), heightBottomRaw[previous]->srv.get() };
	ID3D11UnorderedAccessView* scrollUAVs[2] = { heightTopRaw[heightCurrent]->uav.get(), heightBottomRaw[heightCurrent]->uav.get() };
	context->CSSetShaderResources(0, 2, scrollSRVs);
	context->CSSetUnorderedAccessViews(0, 2, scrollUAVs, nullptr);
	context->CSSetShader(heightScrollCS, nullptr, 0);
	context->Dispatch((heightMapDim + 7) / 8, (heightMapDim + 7) / 8, 1);

	ID3D11ShaderResourceView* nullCsSRVs[2] = { nullptr, nullptr };
	ID3D11UnorderedAccessView* nullCsUAVs[2] = { nullptr, nullptr };
	context->CSSetShaderResources(0, 2, nullCsSRVs);
	context->CSSetUnorderedAccessViews(0, 2, nullCsUAVs, nullptr);

	ID3D11RenderTargetView* heightRTVs[2] = { heightTopRaw[heightCurrent]->rtv.get(), heightBottomRaw[heightCurrent]->rtv.get() };
	context->OMSetRenderTargets(2, heightRTVs, nullptr);
	context->OMSetBlendState(heightMaxBlendState.get(), nullptr, 0xFFFFFFFF);

	D3D11_VIEWPORT heightViewport{ 0.0f, 0.0f, float(heightMapDim), float(heightMapDim), 0.0f, 1.0f };
	context->RSSetViewports(1, &heightViewport);

	context->VSSetShader(heightVS, nullptr, 0);
	context->PSSetShader(heightPS, nullptr, 0);
	ID3D11Buffer* cb1 = staticsCB->CB();
	context->VSSetConstantBuffers(1, 1, &cb1);

	globals::profiler->BeginPass("SnowDeformation::ObjectHeightMap");
	for (const auto& cap : capturedStatics) {
		auto* geometry = cap.geometry.get();
		if (!geometry)
			continue;
		auto triShape = geometry->AsTriShape();
		if (!triShape)
			continue;
		auto rendererData = geometry->GetGeometryRuntimeData().rendererData;
		if (!rendererData || !rendererData->vertexBuffer || !rendererData->indexBuffer)
			continue;
		uint32_t indexCount = uint32_t(triShape->GetTrishapeRuntimeData().triangleCount) * 3;
		if (indexCount == 0)
			continue;

		auto desc = rendererData->vertexDesc;
		if (!desc.HasFlag(RE::BSGraphics::Vertex::VF_VERTEX) || !desc.HasFlag(RE::BSGraphics::Vertex::VF_NORMAL))
			continue;

		uint64_t descKey;
		memcpy(&descKey, &desc, sizeof(descKey));
		auto layoutIt = staticsILCache.find(descKey);
		if (layoutIt == staticsILCache.end() || !layoutIt->second)
			continue;  // layouts are created by the statics pass; reuse only
		context->IASetInputLayout(layoutIt->second.get());

		UINT stride = uint32_t(descKey & 0xF) * 4;
		if (stride == 0)
			continue;
		UINT offset = 0;
		auto* vb = reinterpret_cast<ID3D11Buffer*>(rendererData->vertexBuffer);
		auto* ib = reinterpret_cast<ID3D11Buffer*>(rendererData->indexBuffer);
		context->IASetVertexBuffers(0, 1, &vb, &stride, &offset);
		context->IASetIndexBuffer(ib, DXGI_FORMAT_R16_UINT, 0);

		StaticsCB scb{};
		const auto& rot = cap.world.rotate;
		const float scale = cap.world.scale;
		scb.WorldRow0 = { rot.entry[0][0] * scale, rot.entry[0][1] * scale, rot.entry[0][2] * scale, cap.world.translate.x };
		scb.WorldRow1 = { rot.entry[1][0] * scale, rot.entry[1][1] * scale, rot.entry[1][2] * scale, cap.world.translate.y };
		scb.WorldRow2 = { rot.entry[2][0] * scale, rot.entry[2][1] * scale, rot.entry[2][2] * scale, cap.world.translate.z };
		scb.ObjectsDepth = settings.ObjectsSnowDepth;
		scb.RoundedDepth = settings.SnowMeshesDepth;
		scb.VertexCountF = float(triShape->GetTrishapeRuntimeData().vertexCount);
		scb.HeightWindowCenter = heightWindowCenter;
		scb.HeightHalfExtent = heightHalfExtent;
		staticsCB->Update(scb);

		context->DrawIndexed(indexCount, 0, 0);
	}
	globals::profiler->EndPass();

	ID3D11RenderTargetView* nullRTVs[2] = { nullptr, nullptr };
	context->OMSetRenderTargets(2, nullRTVs, nullptr);

	const UINT dispatchDim = (heightMapDim + 7) / 8;
	ID3D11ShaderResourceView* terrainSRV = shellTerrainTexture->srv.get();
	context->CSSetConstantBuffers(0, 1, &processCB);
	context->CSSetShaderResources(2, 1, &terrainSRV);
	// Deformation map (t3): CombineCS gates corpse mounds on local refill.
	ID3D11ShaderResourceView* deformSRV = GetDeformationSRV();
	context->CSSetShaderResources(3, 1, &deformSRV);

	// Combine: terrain + grounded object tops -> base field (topFiltered),
	// plus the shelter mask for floating structures (bottomFiltered).
	{
		ID3D11ShaderResourceView* combineSRVs[2] = { heightTopRaw[heightCurrent]->srv.get(), heightBottomRaw[heightCurrent]->srv.get() };
		ID3D11UnorderedAccessView* combineUAVs[2] = { heightTopFiltered->uav.get(), heightBottomFiltered->uav.get() };
		ID3D11Buffer* doorCB = doorsCB->CB();
		context->CSSetConstantBuffers(1, 1, &doorCB);
		context->CSSetShaderResources(0, 2, combineSRVs);
		context->CSSetUnorderedAccessViews(0, 2, combineUAVs, nullptr);
		context->CSSetShader(heightCombineCS, nullptr, 0);
		context->Dispatch(dispatchDim, dispatchDim, 1);
		context->CSSetShaderResources(0, 2, nullCsSRVs);
		context->CSSetUnorderedAccessViews(0, 2, nullCsUAVs, nullptr);
		ID3D11Buffer* nullDoorCB = nullptr;
		context->CSSetConstantBuffers(1, 1, &nullDoorCB);
	}

	// Angle of repose: multi-scale min-plus cone passes (large steps first),
	// ping-ponging topFiltered <-> heightScratch and ENDING in topFiltered.
	static constexpr uint kConeSteps[] = { 32, 16, 8, 4, 2, 1 };
	context->CSSetShader(heightConeCS, nullptr, 0);
	Texture2D* coneIn = heightTopFiltered;
	Texture2D* coneOut = heightScratch;
	for (uint step : kConeSteps) {
		processData.ConeStep = step;
		heightProcessCB->Update(processData);
		ID3D11ShaderResourceView* coneSRV = coneIn->srv.get();
		ID3D11UnorderedAccessView* coneUAV = coneOut->uav.get();
		context->CSSetShaderResources(0, 1, &coneSRV);
		context->CSSetUnorderedAccessViews(0, 1, &coneUAV, nullptr);
		context->Dispatch(dispatchDim, dispatchDim, 1);
		context->CSSetShaderResources(0, 1, nullCsSRVs);
		context->CSSetUnorderedAccessViews(0, 1, nullCsUAVs, nullptr);
		std::swap(coneIn, coneOut);
	}
	// Six swaps: the final result lands back in heightTopFiltered.
	static_assert(std::size(kConeSteps) % 2 == 0);

	ID3D11ShaderResourceView* nullTailSRVs[2] = { nullptr, nullptr };
	context->CSSetShaderResources(2, 2, nullTailSRVs);
	context->CSSetShader(nullptr, nullptr, 0);
}

ID3D11ShaderResourceView* SnowDeformation::EnsureSmoothedNormals(RE::BSGeometry* a_geometry)
{
	auto triShape = a_geometry->AsTriShape();
	if (!triShape)
		return nullptr;
	auto rendererData = a_geometry->GetGeometryRuntimeData().rendererData;
	if (!rendererData || !rendererData->vertexBuffer)
		return nullptr;

	// Pointer reuse after cell unloads could serve stale normals to a new
	// mesh; the cap flushes the cache before that becomes likely.
	if (smoothedNormalsCache.size() > 1024)
		smoothedNormalsCache.clear();

	auto [it, inserted] = smoothedNormalsCache.try_emplace(rendererData->vertexBuffer);
	auto& entry = it->second;
	if (!inserted)
		return entry.ready ? entry.srv.get() : nullptr;

	// First sight of this geometry: build now (a buffer copy + two small
	// dispatches, once per unique mesh). Failures leave the permanent null
	// entry — no per-frame retries; the VS falls back to raw normals.
	if (!smoothAccumulateCS)
		smoothAccumulateCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SmoothNormalsCS.hlsl", { { "ACCUMULATE", "" } }, "cs_5_0"));
	if (!smoothResolveCS)
		smoothResolveCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SmoothNormalsCS.hlsl", { { "RESOLVE", "" } }, "cs_5_0"));
	if (!smoothFlatStatsCS)
		smoothFlatStatsCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SmoothNormalsCS.hlsl", { { "FLATSTATS", "" } }, "cs_5_0"));
	if (!smoothAccumulateCS || !smoothResolveCS || !smoothFlatStatsCS || !smoothCB)
		return nullptr;

	auto desc = rendererData->vertexDesc;
	if (!desc.HasFlag(RE::BSGraphics::Vertex::VF_VERTEX) || !desc.HasFlag(RE::BSGraphics::Vertex::VF_NORMAL))
		return nullptr;
	uint64_t descKey;
	memcpy(&descKey, &desc, sizeof(descKey));
	const uint32_t stride = uint32_t(descKey & 0xF) * 4;
	const uint32_t vertexCount = triShape->GetTrishapeRuntimeData().vertexCount;
	if (stride == 0 || vertexCount == 0)
		return nullptr;

	// Position size = distance to the first following attribute (same
	// offset-table logic the input layouts use; VF_FULLPREC is unreliable).
	uint32_t positionBytes = stride;
	static constexpr std::pair<RE::BSGraphics::Vertex::Flags, RE::BSGraphics::Vertex::Attribute> kSmoothAttrs[] = {
		{ RE::BSGraphics::Vertex::VF_UV, RE::BSGraphics::Vertex::VA_TEXCOORD0 },
		{ RE::BSGraphics::Vertex::VF_UV_2, RE::BSGraphics::Vertex::VA_TEXCOORD1 },
		{ RE::BSGraphics::Vertex::VF_NORMAL, RE::BSGraphics::Vertex::VA_NORMAL },
		{ RE::BSGraphics::Vertex::VF_TANGENT, RE::BSGraphics::Vertex::VA_BINORMAL },
		{ RE::BSGraphics::Vertex::VF_COLORS, RE::BSGraphics::Vertex::VA_COLOR },
		{ RE::BSGraphics::Vertex::VF_SKINNED, RE::BSGraphics::Vertex::VA_SKINNING },
		{ RE::BSGraphics::Vertex::VF_LANDDATA, RE::BSGraphics::Vertex::VA_LANDDATA },
		{ RE::BSGraphics::Vertex::VF_EYEDATA, RE::BSGraphics::Vertex::VA_EYEDATA },
	};
	for (auto [flag, attr] : kSmoothAttrs) {
		if (desc.HasFlag(flag)) {
			uint32_t attrOffset = desc.GetAttributeOffset(attr);
			if (attrOffset > 0 && attrOffset < positionBytes)
				positionBytes = attrOffset;
		}
	}
	const uint32_t normalOffset = desc.GetAttributeOffset(RE::BSGraphics::Vertex::VA_NORMAL);

	auto device = globals::d3d::device;
	auto context = globals::d3d::context;
	auto* gameVB = reinterpret_cast<ID3D11Buffer*>(rendererData->vertexBuffer);

	// SRV-capable copy of the game's vertex buffer (raw view).
	D3D11_BUFFER_DESC srcDesc{};
	gameVB->GetDesc(&srcDesc);
	D3D11_BUFFER_DESC copyDesc{};
	copyDesc.ByteWidth = srcDesc.ByteWidth;
	copyDesc.Usage = D3D11_USAGE_DEFAULT;
	copyDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	copyDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
	winrt::com_ptr<ID3D11Buffer> vbCopy;
	if (FAILED(device->CreateBuffer(&copyDesc, nullptr, vbCopy.put())))
		return nullptr;
	context->CopyResource(vbCopy.get(), gameVB);

	D3D11_SHADER_RESOURCE_VIEW_DESC rawSrvDesc{};
	rawSrvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	rawSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
	rawSrvDesc.BufferEx.NumElements = copyDesc.ByteWidth / 4;
	rawSrvDesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
	winrt::com_ptr<ID3D11ShaderResourceView> vbCopySRV;
	if (FAILED(device->CreateShaderResourceView(vbCopy.get(), &rawSrvDesc, vbCopySRV.put())))
		return nullptr;

	// Hash table (transient) and the persistent output buffer.
	uint32_t tableSlots = 64;
	while (tableSlots < vertexCount * 2)
		tableSlots <<= 1;
	D3D11_BUFFER_DESC tableDesc{};
	tableDesc.ByteWidth = tableSlots * 16;
	tableDesc.Usage = D3D11_USAGE_DEFAULT;
	tableDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
	tableDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
	winrt::com_ptr<ID3D11Buffer> tableBuffer;
	if (FAILED(device->CreateBuffer(&tableDesc, nullptr, tableBuffer.put())))
		return nullptr;
	D3D11_UNORDERED_ACCESS_VIEW_DESC rawUavDesc{};
	rawUavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	rawUavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	rawUavDesc.Buffer.NumElements = tableSlots * 4;
	rawUavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
	winrt::com_ptr<ID3D11UnorderedAccessView> tableUAV;
	if (FAILED(device->CreateUnorderedAccessView(tableBuffer.get(), &rawUavDesc, tableUAV.put())))
		return nullptr;

	// One extra element past the vertices: the mesh's flatness stats
	// (fraction of vertices whose smoothed normal diverges from the raw
	// one — high on split-normal plates like walkways and roofs), read by
	// the VS to pick the flat or rounded snow behavior without any CPU
	// readback.
	D3D11_BUFFER_DESC outDesc{};
	outDesc.ByteWidth = (vertexCount + 1) * 16;
	outDesc.Usage = D3D11_USAGE_DEFAULT;
	outDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	outDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	outDesc.StructureByteStride = 16;
	winrt::com_ptr<ID3D11Buffer> outBuffer;
	if (FAILED(device->CreateBuffer(&outDesc, nullptr, outBuffer.put())))
		return nullptr;
	Util::SetResourceName(outBuffer.get(), "SnowDeformation::SmoothedNormals");
	D3D11_SHADER_RESOURCE_VIEW_DESC outSrvDesc{};
	outSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
	outSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	outSrvDesc.Buffer.NumElements = vertexCount + 1;
	winrt::com_ptr<ID3D11ShaderResourceView> outSRV;
	if (FAILED(device->CreateShaderResourceView(outBuffer.get(), &outSrvDesc, outSRV.put())))
		return nullptr;
	D3D11_UNORDERED_ACCESS_VIEW_DESC outUavDesc{};
	outUavDesc.Format = DXGI_FORMAT_UNKNOWN;
	outUavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	outUavDesc.Buffer.NumElements = vertexCount + 1;
	winrt::com_ptr<ID3D11UnorderedAccessView> outUAV;
	if (FAILED(device->CreateUnorderedAccessView(outBuffer.get(), &outUavDesc, outUAV.put())))
		return nullptr;

	const UINT clearZero[4] = { 0, 0, 0, 0 };
	context->ClearUnorderedAccessViewUint(tableUAV.get(), clearZero);

	SmoothCB cb{};
	cb.VertexCount = vertexCount;
	cb.StrideBytes = stride;
	cb.NormalOffsetBytes = normalOffset;
	cb.PosIsFloat32 = positionBytes >= 16 ? 1u : 0u;
	cb.TableMask = tableSlots - 1;
	smoothCB->Update(cb);

	// Compute-only state: does not disturb the surrounding draw pipeline.
	ID3D11Buffer* cscb = smoothCB->CB();
	context->CSSetConstantBuffers(0, 1, &cscb);
	ID3D11ShaderResourceView* csSRV = vbCopySRV.get();
	context->CSSetShaderResources(0, 1, &csSRV);
	ID3D11UnorderedAccessView* csUAVs[2] = { tableUAV.get(), outUAV.get() };
	context->CSSetUnorderedAccessViews(0, 2, csUAVs, nullptr);
	const uint32_t groups = (vertexCount + 63) / 64;
	context->CSSetShader(smoothAccumulateCS, nullptr, 0);
	context->Dispatch(groups, 1, 1);
	context->CSSetShader(smoothResolveCS, nullptr, 0);
	context->Dispatch(groups, 1, 1);
	// Flatness stats: one group strided over the resolved normals.
	context->CSSetShader(smoothFlatStatsCS, nullptr, 0);
	context->Dispatch(1, 1, 1);

	ID3D11ShaderResourceView* nullCsSRV = nullptr;
	context->CSSetShaderResources(0, 1, &nullCsSRV);
	ID3D11UnorderedAccessView* nullCsUAVs[2] = { nullptr, nullptr };
	context->CSSetUnorderedAccessViews(0, 2, nullCsUAVs, nullptr);
	context->CSSetShader(nullptr, nullptr, 0);

	entry.buffer = outBuffer;
	entry.srv = outSRV;
	entry.ready = true;
	return entry.srv.get();
}

void SnowDeformation::DrawCapturedStatics()
{
	if (capturedStatics.empty() || (settings.ObjectsSnowDepth <= 0.01f && settings.SnowMeshesDepth <= 0.01f))
		return;
	if (!EnsureStaticsShaders())
		return;

	auto context = globals::d3d::context;
	auto device = globals::d3d::device;

	context->VSSetShader(staticsVS, nullptr, 0);
	context->PSSetShader(staticsPS, nullptr, 0);
	ID3D11Buffer* cb1 = staticsCB->CB();
	context->VSSetConstantBuffers(1, 1, &cb1);
	context->PSSetConstantBuffers(1, 1, &cb1);

	globals::profiler->BeginPass("SnowDeformation::StaticsShell");
	// One-shot skip diagnostics: geometries that capture but cannot draw are
	// the "why is THIS rock bare" cases — name the reason in the log.
	static std::unordered_set<std::string> loggedSkips;
	auto logSkip = [](RE::BSGeometry* a_geometry, const char* a_reason) {
		if (loggedSkips.size() < 24 && loggedSkips.insert(std::string(a_geometry->name.c_str()) + a_reason).second)
			logger::info("[SNOW DEFORMATION] Statics skip '{}': {}", a_geometry->name.c_str(), a_reason);
	};

	for (const auto& cap : capturedStatics) {
		auto* geometry = cap.geometry.get();
		if (!geometry)
			continue;
		auto triShape = geometry->AsTriShape();
		if (!triShape) {
			logSkip(geometry, "not a BSTriShape");
			continue;
		}
		auto rendererData = geometry->GetGeometryRuntimeData().rendererData;
		if (!rendererData || !rendererData->vertexBuffer || !rendererData->indexBuffer) {
			logSkip(geometry, "no renderer buffers");
			continue;
		}
		uint32_t indexCount = uint32_t(triShape->GetTrishapeRuntimeData().triangleCount) * 3;
		if (indexCount == 0) {
			logSkip(geometry, "zero triangles");
			continue;
		}

		auto desc = rendererData->vertexDesc;
		if (!desc.HasFlag(RE::BSGraphics::Vertex::VF_VERTEX) || !desc.HasFlag(RE::BSGraphics::Vertex::VF_NORMAL)) {
			logSkip(geometry, "vertex format lacks POSITION/NORMAL");
			continue;
		}

		// One-shot layout diagnostics: dump every distinct vertex descriptor
		// with both stride interpretations so the real buffer layout can be
		// decoded from the log instead of guessed.
		{
			static std::unordered_set<uint64_t> loggedDescs;
			uint64_t rawRenderer;
			memcpy(&rawRenderer, &rendererData->vertexDesc, sizeof(rawRenderer));
			if (loggedDescs.size() < 12 && loggedDescs.insert(rawRenderer).second) {
				uint64_t rawGeometry;
				auto geomDesc = geometry->GetGeometryRuntimeData().vertexDesc;
				memcpy(&rawGeometry, &geomDesc, sizeof(rawGeometry));
				logger::info("[SNOW DEFORMATION] Statics geom '{}': rendererDesc={:016X} geomDesc={:016X} nibbleStride={} GetSize={} normalOfs={} fullPrec={} tris={} verts={}",
					geometry->name.c_str(),
					rawRenderer, rawGeometry,
					uint32_t(rawRenderer & 0xF) * 4,
					desc.GetSize(),
					desc.GetAttributeOffset(RE::BSGraphics::Vertex::VA_NORMAL),
					desc.HasFlag(RE::BSGraphics::Vertex::VF_FULLPREC) ? 1 : 0,
					triShape->GetTrishapeRuntimeData().triangleCount,
					triShape->GetTrishapeRuntimeData().vertexCount);
			}
		}

		// One input layout per distinct vertex descriptor; layouts may carry
		// more elements than the VS consumes, so POSITION+NORMAL suffices.
		uint64_t descKey;
		memcpy(&descKey, &desc, sizeof(descKey));
		auto& layout = staticsILCache[descKey];
		if (!layout) {
			// Position size = distance to the first following attribute (the
			// descriptor's offset table is authoritative). The VF_FULLPREC
			// flag is NOT reliable: logged runtime buffers carry 16-byte
			// float4 positions with the flag clear, and reading them as
			// halfs shredded geometry into screen-wide streaks.
			uint32_t strideBytes = uint32_t(descKey & 0xF) * 4;
			uint32_t positionBytes = strideBytes;
			static constexpr std::pair<RE::BSGraphics::Vertex::Flags, RE::BSGraphics::Vertex::Attribute> kAttrs[] = {
				{ RE::BSGraphics::Vertex::VF_UV, RE::BSGraphics::Vertex::VA_TEXCOORD0 },
				{ RE::BSGraphics::Vertex::VF_UV_2, RE::BSGraphics::Vertex::VA_TEXCOORD1 },
				{ RE::BSGraphics::Vertex::VF_NORMAL, RE::BSGraphics::Vertex::VA_NORMAL },
				{ RE::BSGraphics::Vertex::VF_TANGENT, RE::BSGraphics::Vertex::VA_BINORMAL },
				{ RE::BSGraphics::Vertex::VF_COLORS, RE::BSGraphics::Vertex::VA_COLOR },
				{ RE::BSGraphics::Vertex::VF_SKINNED, RE::BSGraphics::Vertex::VA_SKINNING },
				{ RE::BSGraphics::Vertex::VF_LANDDATA, RE::BSGraphics::Vertex::VA_LANDDATA },
				{ RE::BSGraphics::Vertex::VF_EYEDATA, RE::BSGraphics::Vertex::VA_EYEDATA },
			};
			for (auto [flag, attr] : kAttrs) {
				if (desc.HasFlag(flag)) {
					uint32_t attrOffset = desc.GetAttributeOffset(attr);
					if (attrOffset > 0 && attrOffset < positionBytes)
						positionBytes = attrOffset;
				}
			}

			D3D11_INPUT_ELEMENT_DESC elements[2] = {
				{ "POSITION", 0, positionBytes >= 16 ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
				{ "NORMAL", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, desc.GetAttributeOffset(RE::BSGraphics::Vertex::VA_NORMAL), D3D11_INPUT_PER_VERTEX_DATA, 0 },
			};
			if (FAILED(device->CreateInputLayout(elements, 2, staticsVSBlob->GetBufferPointer(), staticsVSBlob->GetBufferSize(), layout.put())))
				continue;  // null stays cached: this descriptor is skipped from now on
		}
		if (!layout)
			continue;
		context->IASetInputLayout(layout.get());

		// Stride comes from the descriptor's low nibble (in dwords) — the
		// same field the game's renderer uses. VertexDesc::GetSize() is NOT
		// equivalent: it reconstructs from flags assuming 16-byte float
		// positions, but most SSE meshes store 8-byte half positions, and
		// the overshot stride shredded vertices into giant garbage triangles.
		UINT stride = uint32_t(descKey & 0xF) * 4;
		if (stride == 0 || desc.GetAttributeOffset(RE::BSGraphics::Vertex::VA_NORMAL) >= stride) {
			logSkip(geometry, "implausible stride/offset");
			continue;
		}
		UINT offset = 0;
		auto* vb = reinterpret_cast<ID3D11Buffer*>(rendererData->vertexBuffer);
		auto* ib = reinterpret_cast<ID3D11Buffer*>(rendererData->indexBuffer);
		context->IASetVertexBuffers(0, 1, &vb, &stride, &offset);
		context->IASetIndexBuffer(ib, DXGI_FORMAT_R16_UINT, 0);

		StaticsCB scb{};
		const auto& rot = cap.world.rotate;
		const float scale = cap.world.scale;
		scb.WorldRow0 = { rot.entry[0][0] * scale, rot.entry[0][1] * scale, rot.entry[0][2] * scale, cap.world.translate.x };
		scb.WorldRow1 = { rot.entry[1][0] * scale, rot.entry[1][1] * scale, rot.entry[1][2] * scale, cap.world.translate.y };
		scb.WorldRow2 = { rot.entry[2][0] * scale, rot.entry[2][1] * scale, rot.entry[2][2] * scale, cap.world.translate.z };
		scb.ObjectsDepth = settings.ObjectsSnowDepth;
		scb.RoundedDepth = settings.SnowMeshesDepth;
		scb.VertexCountF = float(triShape->GetTrishapeRuntimeData().vertexCount);
		scb.HeightWindowCenter = heightWindowCenter;
		scb.HeightHalfExtent = heightHalfExtent;
		// Smoothed normals (built once per unique mesh): pillow inflation
		// for flat split-normal surfaces — planks, roofs, pole caps.
		ID3D11ShaderResourceView* smoothSRV = EnsureSmoothedNormals(geometry);
		context->VSSetShaderResources(10, 1, &smoothSRV);
		scb.HasSmoothedNormals = smoothSRV ? 1.0f : 0.0f;
		staticsCB->Update(scb);

		context->DrawIndexed(indexCount, 0, 0);
	}
	globals::profiler->EndPass();

	// Leave IA clean, mirroring DrawShell's convention (game state manager
	// rebinds via DIRTY_RENDERTARGET).
	ID3D11Buffer* nullVB = nullptr;
	UINT zero = 0;
	context->IASetVertexBuffers(0, 1, &nullVB, &zero, &zero);
	context->IASetIndexBuffer(nullptr, DXGI_FORMAT_R16_UINT, 0);
	context->IASetInputLayout(nullptr);
	ID3D11Buffer* nullCB = nullptr;
	context->VSSetConstantBuffers(1, 1, &nullCB);
	context->PSSetConstantBuffers(1, 1, &nullCB);
}
