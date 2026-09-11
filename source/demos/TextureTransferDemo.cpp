// Copyright (C), UNIGINE. All rights reserved.

#include "TextureTransferDemo.h"

#include <UnigineEngine.h>
#include <UnigineGame.h>
#include <UnigineLights.h>
#include <UnigineObjects.h>
#include <UnigineMaterials.h>
#include <UnigineMaterial.h>
#include <UnigineRender.h>
#include <UnigineProfiler.h>
#include <UnigineWindowManager.h>
#include <UnigineCallback.h>
#include <UnigineLog.h>
#include <UnigineMathLib.h>

using namespace Unigine;
using namespace Math;

// process texture on cuda, with changing size (rgba8 -> rgb8 in sample, it can be used to
// convert to YCbCr to reduce transfer size)
#define PROCESS_TEXTURE true

namespace
{
void cudaTestError(const char *message)
{
	if (cudaGetLastError() != cudaSuccess)
		Log::fatal(message);
}
} // namespace

// cuda kernel launcher
extern "C"
{
	void cuda_texture_transfer_process(cudaStream_t stream, void *dest, void *src, int width, int height);
}

void TextureTransferDemo::init()
{
	WindowManager::getMainWindow()->setResizable(false);

	// create player
	player = PlayerDummy::create();
	Game::setPlayer(player);

	// create lights
	LightOmniPtr light_0 = LightOmni::create(vec4(1.0f, 0.0f, 0.0f, 1.0f), 100.0f);
	light_0->setIntensity(50.0f);
	light_0->setTransform(translate(Vec3(20.0f, 0.0f, 20.0f)));

	LightOmniPtr light_1 = LightOmni::create(vec4(0.0f, 1.0f, 0.0f, 1.0f), 100.0f);
	light_1->setIntensity(50.0f);
	light_1->setTransform(translate(Vec3(0.0f, 20.0f, 20.0f)));

	LightOmniPtr light_2 = LightOmni::create(vec4(0.0f, 0.0f, 1.0f, 1.0f), 100.0f);
	light_2->setIntensity(50.0f);
	light_2->setTransform(translate(Vec3(-20.0f, 0.0f, 20.0f)));

	// load material
	MaterialPtr mesh_base = Materials::findManualMaterial("Unigine::mesh_base");

	// create meshes (asset shipped in the sample's data/)
	ObjectMeshStaticPtr mesh_0 = ObjectMeshStatic::create("cbox.mesh");
	mesh_0->setTransform(translate(Vec3(-2.0f, 0.0f, 0.0f)));
	mesh_0->setMaterial(mesh_base, "*");

	ObjectMeshStaticPtr mesh_1 = ObjectMeshStatic::create("cbox.mesh");
	mesh_1->setTransform(translate(Vec3(2.0f, 0.0f, 0.0f)));
	mesh_1->setMaterial(mesh_base, "*");

	ObjectMeshStaticPtr mesh_2 = ObjectMeshStatic::create("cbox.mesh");
	mesh_2->setTransform(translate(Vec3(0.0f, -2.0f, 0.0f)));
	mesh_2->setMaterial(mesh_base, "*");

	ObjectMeshStaticPtr mesh_3 = ObjectMeshStatic::create("cbox.mesh");
	mesh_3->setTransform(translate(Vec3(0.0f, 2.0f, 0.0f)));
	mesh_3->setMaterial(mesh_base, "*");

	image = Image::create();
	render_texture = Texture::create();

	Render::getEventEndScreen().connect(event_connection, this, &TextureTransferDemo::transfer_screen_color);

	Profiler::setValue("Engine Transfer", "ms", 0.0f, 30.0f, Math::vec4(1.0f, 1.0f, 1.0f, 1.0f));

	init_resources();
}

void TextureTransferDemo::update()
{
	float time = Game::getTime();
	float x = sinf(time * 1.0f) * 4.0f;
	float y = cosf(time * 1.0f) * 4.0f;
	float z = 2.0f + sinf(time * 3.0f) * 1.0f;
	player->setWorldTransform(setTo(Vec3(x, y, z), Vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)));
	player->setFov(60.0f + sinf(time * 2.0f) * 25.0f);
}

void TextureTransferDemo::transfer_screen_color()
{
	int profiler_micro_id = Profiler::beginMicro("CUDATextureTransfer", 1);

	// copy screen color
	RenderTargetPtr render_target = Render::getTemporaryRenderTarget();
	render_target->bindColorTexture(0, render_texture);
	render_target->enable();
	{
		Render::renderScreenMaterial("Unigine::render_copy_2d", Renderer::getTextureColor());
	}
	render_target->disable();
	render_target->unbindColorTextures();
	Render::releaseTemporaryRenderTarget(render_target);

	// Try transfering with engine default method
	int index = Engine::get()->getFrame() % NUM_TIME_FRAMES;
	frame_timer[index].begin();
	Render::transferTextureToImage(MakeCallback([this, index](ImagePtr img)
	{
		image->copy(img, 0, 0, 0, 0, img->getWidth(), img->getHeight());
		Profiler::setValue("Engine Transfer", "ms", frame_timer[index].endMilliseconds(), 30.0f, Math::vec4(1.0f, 1.0f, 1.0f, 1.0f));

	}), render_texture);

	Profiler::begin("CUDA Transfer", Math::vec4(0.7f, 0.9f, 0.0f, 1.0f));

	// get cuda array that points to engine texture
	cudaArray_t cuda_array = texture.getMipLevel(0);

	cudaTestError("Can't map CUDA array\n");

	size_t pitch = static_cast<size_t>(width) * pixel_size;
	if (PROCESS_TEXTURE)
	{
		// copy to gpu buffer
		cudaMemcpy2DFromArray(gpu_original_buffer, pitch, cuda_array, 0, 0, pitch, height, cudaMemcpyDeviceToDevice);
		cudaTestError("Can't copy data\n");

		// process
		cuda_texture_transfer_process(texture.getStream(), gpu_process_buffer, gpu_original_buffer, width, height);

		// copy to host
		cudaMemcpy(pinned_buffer, gpu_process_buffer, buffer_process_size, cudaMemcpyDeviceToHost);
		cudaTestError("Can't copy data to host\n");
	} else
	{
		cudaMemcpy2DFromArray(pinned_buffer, pitch, cuda_array, 0, 0, pitch, height, cudaMemcpyDeviceToHost);
	}

	// copy to destination image
	cudaMemcpy(image->getPixels(), pinned_buffer, image_size, cudaMemcpyHostToHost);

	Profiler::end();

	Profiler::endMicro(profiler_micro_id);
}

void TextureTransferDemo::init_resources()
{
	width = WindowManager::getMainWindow()->getSize().x;
	height = WindowManager::getMainWindow()->getSize().y;
	buffer_size = width * height * pixel_size;
	buffer_process_size = width * height * pixel_process_size;

	if (PROCESS_TEXTURE)
	{
		image_format = Image::FORMAT_RGB8;
		image_size = buffer_process_size;
	} else
	{
		image_format = Image::FORMAT_RGBA8;
		image_size = buffer_size;
	}

	render_texture->create2D(width, height, Texture::FORMAT_RGBA8, Texture::FORMAT_USAGE_RENDER | Texture::FORMAT_USAGE_SHARED);
	texture.exportToCuda(render_texture);

	// use special function to get OS-pinned memory
	cudaHostAlloc((void**)&pinned_buffer, image_size, cudaHostAllocMapped);
	memset(pinned_buffer, 0, image_size);
	cudaTestError("Can't allocate pinned memory\n");

	if (PROCESS_TEXTURE)
	{
		cudaMalloc(&gpu_original_buffer, buffer_size);
		cudaTestError("Can't alloc GPU buffer\n");
		cudaMemset(gpu_original_buffer, 0, buffer_size);

		cudaMalloc(&gpu_process_buffer, buffer_process_size);
		cudaTestError("Can't alloc GPU processed buffer\n");
		cudaMemset(gpu_process_buffer, 0, buffer_process_size);
	}

	image->create2D(width, height, image_format);
}

void TextureTransferDemo::release_resources()
{
	texture.clear();

	cudaFreeHost(pinned_buffer);
	cudaTestError("Can't free pinned memory\n");

	if (PROCESS_TEXTURE)
	{
		cudaFree(gpu_original_buffer);
		cudaTestError("Can't free GPU buffer\n");

		cudaFree(gpu_process_buffer);
		cudaTestError("Can't free GPU processed buffer\n");
	}
}

void TextureTransferDemo::shutdown()
{
	event_connection.disconnect();
	release_resources();
}
