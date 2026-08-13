#include "Features/SnowDeformation.h"

#include "Deferred.h"
#include "Features/TerrainBlending.h"
#include "Globals.h"
#include "State.h"
#include "Utils/D3D.h"
#include "Utils/Game.h"

ID3D11VertexShader* SnowDeformation::GetShellVS()
{
	if (!shellVS) {
		logger::debug("Compiling SnowShell VS");
		shellVS = static_cast<ID3D11VertexShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SnowShell.hlsl", { { "VSHADER", "" } }, "vs_5_0"));
	}
	return shellVS;
}

ID3D11PixelShader* SnowDeformation::GetShellPS()
{
	if (!shellPS) {
		logger::debug("Compiling SnowShell PS");
		shellPS = static_cast<ID3D11PixelShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\SnowShell.hlsl", { { "PSHADER", "" } }, "ps_5_0"));
	}
	return shellPS;
}

ID3D11ComputeShader* SnowDeformation::GetDepthSyncCS()
{
	if (!depthSyncCS) {
		logger::debug("Compiling DepthSyncCS");
		depthSyncCS = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\SnowDeformation\\DepthSyncCS.hlsl", {}, "cs_5_0"));
	}
	return depthSyncCS;
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

	cbData.GridSpacing = kShellGridSpacing;
	cbData.GridDim = kShellGridDim;
	// The warped grid is camera-centered: snap the center to the grid step
	// so inner vertices stay texel-stable, then offset by the warped span.
	const float warpedHalfSpan = ShellWarpedHalfSpan();
	cbData.WarpedHalfSpan = warpedHalfSpan;
	cbData.GridOrigin = {
		std::floor(cbData.CameraPosAdjust.x / kShellGridSpacing) * kShellGridSpacing - warpedHalfSpan,
		std::floor(cbData.CameraPosAdjust.y / kShellGridSpacing) * kShellGridSpacing - warpedHalfSpan
	};

	cbData.TerrainTexelSize = kShellVertexSpacing;
	cbData.TerrainDim = kShellWindowDim;
	cbData.ShellDebugData = shellDataDebug;
	cbData.DeformInvWorldSize = 1.0f / kWorldSize;

	// Keep shader-side sampling math in small grid-local coordinates.
	constexpr float cellSize = kShellVertexSpacing * kShellTexelsPerCell;
	cbData.GridToTerrainOffset = {
		cbData.GridOrigin.x - shellWindowCellX * cellSize,
		cbData.GridOrigin.y - shellWindowCellY * cellSize
	};
	cbData.GridToDeformOffset = {
		cbData.GridOrigin.x - windowOrigin.x,
		cbData.GridOrigin.y - windowOrigin.y
	};

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
	// The PS evaluates the terrain/deformation fields for per-pixel coverage
	// and normals, so the field textures must be bound to BOTH stages. The
	// depth SRV is a copy (Terrain Blending's blended depth when available),
	// never the bound DSV, so sampling it here is legal; the PS fades the
	// shell where it hovers close in front of any geometry so it dissolves
	// into statics (walkways, mesh roads, rocks).
	ID3D11ShaderResourceView* shellSRVs[4] = { shellTerrainTexture->srv.get(), GetDeformationSRV(), nullptr, Util::GetCurrentSceneDepthSRV(false) };
	context->VSSetShaderResources(0, 4, shellSRVs);
	context->PSSetShaderResources(0, 4, shellSRVs);

	context->VSSetShader(vs, nullptr, 0);
	context->PSSetShader(ps, nullptr, 0);

	globals::profiler->BeginPass("SnowDeformation::Shell");
	context->Draw(kShellGridDim * kShellGridDim * 6, 0);
	globals::profiler->EndPass();

	// Restore everything we changed.
	context->VSSetShader(nullptr, nullptr, 0);
	context->PSSetShader(nullptr, nullptr, 0);
	ID3D11Buffer* nullCB = nullptr;
	context->VSSetConstantBuffers(0, 1, &nullCB);
	context->PSSetConstantBuffers(0, 1, &nullCB);
	ID3D11ShaderResourceView* nullSRVs[4] = {};
	context->VSSetShaderResources(0, 4, nullSRVs);
	context->PSSetShaderResources(0, 4, nullSRVs);
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
