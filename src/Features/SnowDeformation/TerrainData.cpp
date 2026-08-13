#include "Features/SnowDeformation.h"

#include "Globals.h"
#include "State.h"

/** @brief True when a land texture is snow by material (the same classification the vanilla shader constants encode). */
static bool IsSnowLandTexture(RE::TESLandTexture* a_landTexture)
{
	if (!a_landTexture || a_landTexture->formID == 0)
		return false;

	return a_landTexture->materialType &&
	       (a_landTexture->materialType->materialID == RE::MATERIAL_ID::kSnow ||
			   a_landTexture->materialType->materialID == RE::MATERIAL_ID::kSnowStairs);
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
		// (TruePBR allocates a whole new property + material per quad, so the
		// vanilla material would be the wrong cache key.)
		const auto& children = land->loadedData->mesh[quadI]->GetChildren();
		auto geometry = children.empty() ? nullptr : static_cast<RE::BSGeometry*>(children[0].get());
		if (geometry == nullptr)
			continue;

		const auto shaderProp = static_cast<RE::BSLightingShaderProperty*>(geometry->GetGeometryRuntimeData().shaderProperty.get());
		if (shaderProp == nullptr || shaderProp->material == nullptr)
			continue;

		// Bit 0 = base texture, bits 1-5 = the quad's layer textures.
		uint8_t mask = 0;
		if (IsSnowLandTexture(land->loadedData->defQuadTextures[quadI]))
			mask |= 1;
		for (uint32_t textureI = 0; textureI < 5; ++textureI) {
			if (IsSnowLandTexture(land->loadedData->quadTextures[quadI][textureI]))
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
	data.height.fill(kShellMissingHeight);
	data.snowness.fill(0);

	auto loadedData = land->loadedData;

	// heights[] are relative to the cell's mid-height; the absolute base is
	// the midpoint of heightExtents (verified against the quad geometry local
	// transforms, which equal exactly (extents.x + extents.y) / 2). World
	// transforms are NOT composed yet when this hook runs, so they cannot be
	// used for placement.
	float cellBaseZ = (loadedData->heightExtents.x + loadedData->heightExtents.y) * 0.5f;

	for (uint32_t quadI = 0; quadI < 4; ++quadI) {
		// Index-based quad layout: 0=SW, 1=SE, 2=NW, 3=NE, vertices x-fastest.
		uint32_t quadX = quadI & 1;
		uint32_t quadY = quadI >> 1;

		bool baseSnow = IsSnowLandTexture(loadedData->defQuadTextures[quadI]);
		bool layerSnow[6];
		for (uint32_t layerI = 0; layerI < 6; ++layerI)
			layerSnow[layerI] = IsSnowLandTexture(loadedData->quadTextures[quadI][layerI]);

		for (uint32_t vertexI = 0; vertexI < 289; ++vertexI) {
			uint32_t vx = vertexI % 17;
			uint32_t vy = vertexI / 17;
			uint32_t cellX = quadX * 16 + vx;
			uint32_t cellY = quadY * 16 + vy;
			uint32_t cellIdx = cellY * 33 + cellX;

			data.height[cellIdx] = loadedData->heights[quadI][vertexI] + cellBaseZ;

			float layerSum = 0.0f;
			float snowSum = 0.0f;
			for (uint32_t layerI = 0; layerI < 6; ++layerI) {
				// percents is declared std::int8_t but holds 0-255: a fully
				// painted layer reads as -1 without the unsigned cast, which
				// zeroes strong layers and hands their weight to the base.
				float weight = static_cast<uint8_t>(loadedData->percents[quadI][vertexI][layerI]) / 255.0f;
				layerSum += weight;
				if (layerSnow[layerI])
					snowSum += weight;
			}
			if (baseSnow)
				snowSum += std::max(0.0f, 1.0f - layerSum);
			data.snowness[cellIdx] = (uint8_t)std::clamp(snowSum * 255.0f + 0.5f, 0.0f, 255.0f);
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
		if (it != shellCells.end() && it->second.height == data.height && it->second.snowness == data.snowness)
			return;
		shellCells[key] = data;
	}
	shellDataDirty.store(true, std::memory_order_release);
}

void SnowDeformation::UpdateShellTerrainWindow()
{
	auto eyeFB = globals::game::frameBufferCached.GetCameraPosAdjust();
	int camCellX = (int)std::floor(eyeFB.x / (kShellVertexSpacing * kShellTexelsPerCell));
	int camCellY = (int)std::floor(eyeFB.y / (kShellVertexSpacing * kShellTexelsPerCell));

	int desiredOriginX = camCellX - kShellWindowCells / 2;
	int desiredOriginY = camCellY - kShellWindowCells / 2;

	bool originChanged = desiredOriginX != shellWindowCellX || desiredOriginY != shellWindowCellY;
	if (!originChanged && !shellDataDirty.exchange(false, std::memory_order_acq_rel))
		return;

	shellWindowCellX = desiredOriginX;
	shellWindowCellY = desiredOriginY;
	shellDataDirty.store(false, std::memory_order_release);

	shellUploadScratch.resize(size_t(kShellWindowDim) * kShellWindowDim * 2);

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

				float* texel = &shellUploadScratch[(size_t(ty) * kShellWindowDim + tx) * 2];
				if (rowCell) {
					uint32_t idx = uint32_t(vy) * 33 + uint32_t(vx);
					texel[0] = rowCell->height[idx];
					texel[1] = rowCell->snowness[idx] / 255.0f;

					statCells.insert(rowKey);
					statMinH = std::min(statMinH, texel[0]);
					statMaxH = std::max(statMaxH, texel[0]);
					if (texel[1] > 0.05f)
						statSnowTexels++;
				} else {
					texel[0] = kShellMissingHeight;
					texel[1] = 0.0f;
				}
			}
		}
	}

	shellStatCellsInWindow = (uint32_t)statCells.size();
	shellStatSnowTexels = statSnowTexels;
	shellStatMinHeight = statMinH == FLT_MAX ? 0.0f : statMinH;
	shellStatMaxHeight = statMaxH == -FLT_MAX ? 0.0f : statMaxH;

	logger::debug("[SNOW DEFORMATION] Shell window rebuilt: origin cell ({}, {}), {} cells in window, {} snow texels, height range [{:.0f}, {:.0f}]",
		shellWindowCellX, shellWindowCellY, shellStatCellsInWindow, shellStatSnowTexels, shellStatMinHeight, shellStatMaxHeight);

	globals::d3d::context->UpdateSubresource(shellTerrainTexture->resource.get(), 0, nullptr,
		shellUploadScratch.data(), kShellWindowDim * 2 * sizeof(float), 0);
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
}
