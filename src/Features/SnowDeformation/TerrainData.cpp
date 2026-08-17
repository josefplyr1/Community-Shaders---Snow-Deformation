#include "Features/SnowDeformation.h"

#include <DDSTextureLoader.h>

#include "Features/TerrainShadows.h"
#include "Globals.h"
#include "State.h"
#include "Utils/D3D.h"

/** @brief Classifies a land texture into a kSnowClasses index by diffuse filename substring (first match wins), falling back on the snow material check. Returns -1 for absent textures. */
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

// One-shot diagnostic: log each distinct land texture with its
// classification, so misclassified modlist textures show up in the log.
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

		// Runs after TruePBR's detour, so this is the final material used for
		// drawing (TruePBR allocates a new property + material per quad).
		const auto& children = land->loadedData->mesh[quadI]->GetChildren();
		auto geometry = children.empty() ? nullptr : static_cast<RE::BSGeometry*>(children[0].get());
		if (geometry == nullptr)
			continue;

		const auto shaderProp = static_cast<RE::BSLightingShaderProperty*>(geometry->GetGeometryRuntimeData().shaderProperty.get());
		if (shaderProp == nullptr || shaderProp->material == nullptr)
			continue;

		// Bit 0 = base texture, bits 1-5 = the quad's layer textures.
		uint8_t mask = 0;
		if (IsSnowClass(ClassifySnowClass(land->loadedData->defQuadTextures[quadI])))
			mask |= 1;
		for (uint32_t textureI = 0; textureI < 5; ++textureI) {
			if (IsSnowClass(ClassifySnowClass(land->loadedData->quadTextures[quadI][textureI])))
				mask |= uint8_t(1 << (textureI + 1));
		}

		const std::unique_lock lock(snowMaskMutex);
		// Materials are freed on cell unload; bound the map against stale pointers.
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
	for (auto& classArray : data.classWeights)
		classArray.fill(0);
	if (auto* worldspace = cell->GetRuntimeData().worldSpace)
		data.worldspaceID = worldspace->GetFormID();

	auto loadedData = land->loadedData;

	// heights[] are relative to the cell's mid-height; the absolute base is
	// the midpoint of heightExtents. World transforms are not composed yet
	// when this hook runs and cannot be used for placement.
	float cellBaseZ = (loadedData->heightExtents.x + loadedData->heightExtents.y) * 0.5f;

	for (uint32_t quadI = 0; quadI < 4; ++quadI) {
		// Index-based quad layout: 0=SW, 1=SE, 2=NW, 3=NE, vertices x-fastest.
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

			data.height[cellIdx] = loadedData->heights[quadI][vertexI] + cellBaseZ;

			float layerSum = 0.0f;
			float classSum[kSnowClassCount] = {};
			for (uint32_t layerI = 0; layerI < 6; ++layerI) {
				// percents is declared std::int8_t but holds 0-255: a fully
				// painted layer reads as -1 without the unsigned cast.
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
		// Land material setup re-runs frequently; only mark the window dirty
		// when the baked data actually changed.
		auto it = shellCells.find(key);
		if (it != shellCells.end() && it->second.worldspaceID == data.worldspaceID &&
			it->second.height == data.height && it->second.classWeights == data.classWeights)
			return;
		shellCells[key] = data;
	}
	shellDataDirty.store(true, std::memory_order_release);
}

float SnowDeformation::GetNominalSnowDepthAt(float a_x, float a_y, float a_missing)
{
	// A cell is 33 vertices = 32 intervals of kShellVertexSpacing.
	constexpr float kCellSize = kShellVertexSpacing * 32.0f;
	const int cellX = (int)std::floor(a_x / kCellSize);
	const int cellY = (int)std::floor(a_y / kCellSize);
	const int vx = std::clamp((int)std::lround((a_x - cellX * kCellSize) / kShellVertexSpacing), 0, 32);
	const int vy = std::clamp((int)std::lround((a_y - cellY * kCellSize) / kShellVertexSpacing), 0, 32);

	const uint64_t key = (uint64_t(uint32_t(cellX)) << 32) | uint32_t(cellY);
	const std::shared_lock lock(shellCellMutex);
	const auto it = shellCells.find(key);
	if (it == shellCells.end() || it->second.worldspaceID != activeWorldspace.load(std::memory_order_acquire))
		return a_missing;

	// Same class-weight resolve the window rebuild uses, for one vertex.
	const uint32_t idx = uint32_t(vy) * 33 + uint32_t(vx);
	float depth = 0.0f;
	for (uint32_t classI = 0; classI < kSnowClassCount; ++classI)
		depth += it->second.classWeights[classI][idx] / 255.0f * settings.SnowClassDepths[classI];
	return depth;
}

void SnowDeformation::UpdateActiveWorldspace()
{
	auto* tes = RE::TES::GetSingleton();
	auto* worldspace = tes ? tes->GetRuntimeData2().worldSpace : nullptr;
	// Interiors keep the last exterior's state: a shop visit must not wipe the
	// tracks outside its door.
	if (!worldspace)
		return;

	const uint32_t id = worldspace->GetFormID();
	if (id == activeWorldspace.exchange(id, std::memory_order_acq_rel))
		return;

	// City worldspaces share their parent's cell coordinates AND world XY
	// (WindhelmWorld sits on the same 28-36 / 6-12 block as the Tamriel
	// terrain outside its gate), so every world-anchored cache now describes
	// the worldspace we just left. Baked cells carry their worldspace and are
	// filtered below; the object raster and the deformation map have no such
	// tag and have to be dropped.
	shellDataDirty.store(true, std::memory_order_release);
	heightMapValid = false;
	clearRequested = true;
	logger::debug("[SNOW DEFORMATION] Worldspace changed to {:08X}: dropping world-anchored caches", id);
}

void SnowDeformation::UpdateShellTerrainWindow()
{
	auto eyeFB = globals::game::frameBufferCached.GetCameraPosAdjust();
	int camCellX = (int)std::floor(eyeFB.x / (kShellVertexSpacing * kShellTexelsPerCell));
	int camCellY = (int)std::floor(eyeFB.y / (kShellVertexSpacing * kShellTexelsPerCell));

	int desiredOriginX = camCellX - kShellWindowCells / 2;
	int desiredOriginY = camCellY - kShellWindowCells / 2;

	bool originChanged = desiredOriginX != shellWindowCellX || desiredOriginY != shellWindowCellY;
	// A heightmap swap (worldspace change, or Terrain Shadows finishing its
	// load after this window was built) invalidates the far fill even when
	// the origin is unchanged.
	auto& terrainShadows = globals::features::terrainShadows;
	const std::string fillWorldspace = (terrainShadows.loaded && terrainShadows.IsHeightMapReady()) ? terrainShadows.cachedHeightmap->worldspace : std::string{};
	bool fillChanged = lastFillWorldspace != fillWorldspace;
	// A worldspace with no landscape of its own bakes nothing, so nothing
	// would ever mark the window dirty and the previous worldspace's texels
	// would stay resident.
	const uint32_t worldspace = activeWorldspace.load(std::memory_order_acquire);
	bool worldspaceChanged = worldspace != shellWindowWorldspace;
	if (!originChanged && !fillChanged && !worldspaceChanged && !shellDataDirty.exchange(false, std::memory_order_acq_rel))
		return;

	shellWindowCellX = desiredOriginX;
	shellWindowCellY = desiredOriginY;
	shellWindowWorldspace = worldspace;
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
					// A cell baked in another worldspace is not this window's
					// ground: city worldspaces reuse the coordinates of the
					// terrain outside them, so an unfiltered hit drapes the
					// countryside through the city at its own elevation.
					rowCell = it != shellCells.end() && it->second.worldspaceID == worldspace ? &it->second : nullptr;
					rowKey = key;
				}

				float* texel = &shellUploadScratch[(size_t(ty) * kShellWindowDim + tx) * 4];
				if (rowCell) {
					uint32_t idx = uint32_t(vy) * 33 + uint32_t(vx);
					// Class depths are applied here, so the sliders retune the
					// shell from cached weights without a re-bake.
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
					texel[0] = kShellMissingHeight;
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

	logger::debug("[SNOW DEFORMATION] Shell window rebuilt: origin cell ({}, {}), {} cells in window, {} snow texels, height range [{:.0f}, {:.0f}]",
		shellWindowCellX, shellWindowCellY, shellStatCellsInWindow, shellStatSnowTexels, shellStatMinHeight, shellStatMaxHeight);

	globals::d3d::context->UpdateSubresource(shellTerrainTexture->resource.get(), 0, nullptr,
		shellUploadScratch.data(), kShellWindowDim * 4 * sizeof(float), 0);

	FillShellWindowFromHeightmap();
}

ID3D11ComputeShader* SnowDeformation::GetWindowFillCS()
{
	if (!windowFillCS) {
		logger::debug("Compiling TerrainWindowFillCS");
		windowFillCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\TerrainWindowFillCS.hlsl", {}, "cs_5_0"));
	}
	return windowFillCS;
}

ID3D11ShaderResourceView* SnowDeformation::GetLODTile(const std::string& a_worldspace, int a_cellX, int a_cellY)
{
	const uint64_t key = (uint64_t(uint32_t(a_cellX)) << 32) | uint32_t(a_cellY);
	if (auto it = lodTileCache.find(key); it != lodTileCache.end())
		return it->second.get();
	if (lodTileMisses.contains(key))
		return nullptr;

	// sRGB ignored so classification runs on the stored gamma-space values
	// regardless of how the DDS declares itself.
	std::string path = std::format("Data\\textures\\terrain\\{}\\{}.32.{}.{}.dds", a_worldspace, a_worldspace, a_cellX, a_cellY);
	std::wstring widePath(path.begin(), path.end());
	winrt::com_ptr<ID3D11ShaderResourceView> srv;
	if (FAILED(DirectX::CreateDDSTextureFromFileEx(globals::d3d::device, widePath.c_str(), 0,
			D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0,
			DirectX::DDS_LOADER_IGNORE_SRGB, nullptr, srv.put()))) {
		logger::info("[SNOW DEFORMATION] LOD tile missing: {}", path);
		lodTileMisses.insert(key);
		return nullptr;
	}
	logger::debug("[SNOW DEFORMATION] LOD tile loaded: {}", path);
	lodTileCache[key] = srv;
	return srv.get();
}

void SnowDeformation::FillShellWindowFromHeightmap()
{
	auto& terrainShadows = globals::features::terrainShadows;
	if (!terrainShadows.loaded || !terrainShadows.texHeightMap || !terrainShadows.IsHeightMapReady()) {
		lastFillWorldspace.clear();
		return;
	}
	auto cs = GetWindowFillCS();
	if (!cs || !shellTerrainTexture->uav)
		return;

	const auto* heightmap = terrainShadows.cachedHeightmap;
	if (lastFillWorldspace != heightmap->worldspace) {
		lodTileCache.clear();
		lodTileMisses.clear();
	}

	WindowFillCB cbData{};
	constexpr float cellSize = kShellVertexSpacing * kShellTexelsPerCell;
	cbData.WindowOriginWorld = { shellWindowCellX * cellSize, shellWindowCellY * cellSize };
	cbData.TexelSize = kShellVertexSpacing;
	cbData.WindowDim = kShellWindowDim;
	cbData.HeightMapScale = { 1.0f / (heightmap->pos1.x - heightmap->pos0.x), 1.0f / (heightmap->pos1.y - heightmap->pos0.y) };
	cbData.HeightMapOffset = { -heightmap->pos0.x * cbData.HeightMapScale.x, -heightmap->pos0.y * cbData.HeightMapScale.y };
	// pos0.z/pos1.z = the file's normalization range (ShadowUpdate.cs.hlsl
	// decode convention), NOT zRange (the content min/max).
	cbData.HeightRange = { heightmap->pos0.z, heightmap->pos1.z };
	// TS convention: pos0 = left-TOP (north, larger Y), pos1 = right-bottom.
	cbData.WorldYRange = { heightmap->pos1.y, heightmap->pos0.y };
	cbData.SnowLineZ = settings.DistantSnowLineZ;
	cbData.SnowNorthDrop = settings.DistantSnowNorthDrop;
	cbData.SnowLineFade = std::max(settings.DistantSnowLineFade, 1.0f);
	cbData.SnowDepthUnits = std::max(settings.SnowClassDepths[3], 0.0f);  // "Snow 01"

	// 2x2 block of level-32 tiles (each spans exactly the window's 32 cells)
	// anchored at the window origin's tile; covers the window at any offset.
	constexpr int kTileCells = 32;
	const int tileX0 = (int)std::floor((float)shellWindowCellX / kTileCells) * kTileCells;
	const int tileY0 = (int)std::floor((float)shellWindowCellY / kTileCells) * kTileCells;
	cbData.LODTileBase = { tileX0 * cellSize, tileY0 * cellSize };
	cbData.LODTileSpan = kTileCells * cellSize;
	cbData.LODSnowSensitivity = std::clamp(settings.LODSnowSensitivity, 0.0f, 1.0f);
	ID3D11ShaderResourceView* tileSRVs[4] = {};
	for (int tileI = 0; tileI < 4; ++tileI) {
		tileSRVs[tileI] = GetLODTile(heightmap->worldspace, tileX0 + (tileI & 1) * kTileCells, tileY0 + (tileI >> 1) * kTileCells);
		(&cbData.LODTileValid.x)[tileI] = tileSRVs[tileI] ? 1.0f : 0.0f;
	}
	windowFillCB->Update(cbData);

	auto context = globals::d3d::context;
	ID3D11Buffer* cb = windowFillCB->CB();
	ID3D11ShaderResourceView* fillSRVs[5] = { terrainShadows.texHeightMap->srv.get(), tileSRVs[0], tileSRVs[1], tileSRVs[2], tileSRVs[3] };
	ID3D11UnorderedAccessView* windowUAV = shellTerrainTexture->uav.get();
	ID3D11SamplerState* sampler = shellLinearSampler.get();
	context->CSSetConstantBuffers(0, 1, &cb);
	context->CSSetShaderResources(0, 5, fillSRVs);
	context->CSSetSamplers(0, 1, &sampler);
	context->CSSetUnorderedAccessViews(0, 1, &windowUAV, nullptr);
	context->CSSetShader(cs, nullptr, 0);
	globals::profiler->BeginPass("SnowDeformation::WindowFill");
	context->Dispatch((kShellWindowDim + 7) / 8, (kShellWindowDim + 7) / 8, 1);
	globals::profiler->EndPass();

	ID3D11Buffer* nullCB = nullptr;
	ID3D11ShaderResourceView* nullSRVs[5] = {};
	ID3D11UnorderedAccessView* nullUAV = nullptr;
	ID3D11SamplerState* nullSampler = nullptr;
	context->CSSetConstantBuffers(0, 1, &nullCB);
	context->CSSetShaderResources(0, 5, nullSRVs);
	context->CSSetSamplers(0, 1, &nullSampler);
	context->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
	context->CSSetShader(nullptr, nullptr, 0);

	lastFillWorldspace = heightmap->worldspace;
}

void SnowDeformation::BSLightingShader_SetupMaterial(RE::BSLightingShaderMaterialBase const* material)
{
	auto state = globals::state;

	// Clear first so bits never leak from the previous landscape draw.
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
	// Same detour target as TruePBR, whose PostPostLoad runs earlier in the
	// feature list. Detours are LIFO, so attaching now makes this hook outer:
	// it sees the final, possibly TruePBR-replaced, quad materials.
	logger::info("[SNOW DEFORMATION] Hooking TESObjectLAND");
	stl::detour_thunk<SD_TESObjectLAND_SetupMaterial>(REL::RelocationID(18368, 18791));

	logger::info("[SNOW DEFORMATION] Hooking BSLightingShader::SetupMaterial");
	stl::write_vfunc<0x4, SD_BSLightingShader_SetupMaterial>(RE::VTABLE_BSLightingShader[0]);

	InstallStaticsCaptureHook();
}
