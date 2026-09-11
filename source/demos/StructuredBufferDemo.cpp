// Copyright (C), UNIGINE. All rights reserved.

#include "StructuredBufferDemo.h"

#include <UnigineGame.h>
#include <UnigineLights.h>
#include <UnigineObjects.h>
#include <UnigineMaterials.h>
#include <UnigineRender.h>
#include <UnigineProfiler.h>
#include <UnigineMathLib.h>

using namespace Unigine;
using namespace Math;

namespace
{
struct Particle
{
	float x, y, z;
	float life;
	float vx, vy, vz;
	float align;
};

// Number of particles to draw per group.
// Must be kept in sync with ScreenSpaceParticles.basemat.
// The one in the CUDA kernel may be different.
const int NUM_GROUPS = 16;

const int NUM_PARTICLES = 3200;
} // namespace

// cuda kernel launcher
extern "C"
{
	void cuda_buffer_process(cudaStream_t stream, void *buffer, int num_particles, int frame);
}

void StructuredBufferDemo::init()
{
	// create player
	player = PlayerSpectator::create();
	Game::setPlayer(player);

	// create lights
	LightWorldPtr sun = LightWorld::create(vec4_one);

	// create shared structured buffer
	init_resources();

	// load material
	MaterialPtr mesh_base = Materials::findManualMaterial("Unigine::mesh_base");
	material = Materials::findManualMaterial("ScreenSpaceParticles");

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

	Render::getEventEndTAA().connect(event_connections, this, &StructuredBufferDemo::composite);
}

void StructuredBufferDemo::update()
{
	player->setFov(60.0f);
}

void StructuredBufferDemo::work()
{
	int profiler_micro_id = Profiler::beginMicro("CUDAStructuredBufferWrite");

	// process
	cuda_buffer_process(shared_buffer.getStream(), shared_buffer.getDevicePtr(), shared_buffer.getNumElements(), frame);
	frame++;

	Profiler::endMicro(profiler_micro_id);
}

void StructuredBufferDemo::composite()
{
	TexturePtr screen_color = RenderState::getScreenColorTexture();
	TexturePtr target = Render::getTemporaryTexture2D(screen_color->getWidth(), screen_color->getHeight(), screen_color->getFormat(),
													screen_color->getAllFlags() | Texture::FORMAT_USAGE_UNORDERED_ACCESS);

	target->copy(screen_color);

	RenderTargetPtr render_target = Render::getTemporaryRenderTarget();
	render_target->bindUnorderedAccessTexture(0, target, true);
	render_target->enableCompute();
	{
		RenderState::setStructuredBuffer(0, structured_buffer);

		Math::mat4 view = Math::mat4(Renderer::getModelview());
		material->setParameterFloat4("camera_view_row_x", view.getRow(0));
		material->setParameterFloat4("camera_view_row_y", view.getRow(1));
		material->setParameterFloat4("camera_view_row_z", view.getRow(2));
		material->setParameterFloat4("camera_view_row_w", view.getRow(3));

		material->setParameterInt("num_elements", structured_buffer->getNumElements());

		material->renderCompute(Render::PASS_POST, structured_buffer->getNumElements() / NUM_GROUPS);

		RenderState::setStructuredBuffer(0, nullptr);
	}
	render_target->disable();
	render_target->unbindAll();

	screen_color->copy(target);

	Render::releaseTemporaryRenderTarget(render_target);
	Render::releaseTemporaryTexture(target);
}

void StructuredBufferDemo::init_resources()
{
	frame = 0;

	structured_buffer = StructuredBuffer::create();
	structured_buffer->create(StructuredBuffer::USAGE_RENDER | StructuredBuffer::USAGE_SHARED, sizeof(Particle), NUM_PARTICLES);

	shared_buffer.exportToCuda(structured_buffer);
	shared_buffer.getEventWork().connect(event_connections, this, &StructuredBufferDemo::work);
}

void StructuredBufferDemo::release_resources()
{
	event_connections.disconnectAll();
	shared_buffer.clear();
	structured_buffer = nullptr;
}

void StructuredBufferDemo::shutdown()
{
	release_resources();
}
