// Copyright (C), UNIGINE. All rights reserved.

#include "TextureWriteDemo.h"

#include <UnigineGame.h>
#include <UnigineLights.h>
#include <UnigineObjects.h>
#include <UnigineMaterials.h>
#include <UnigineMaterial.h>
#include <UnigineProfiler.h>
#include <UnigineMathLib.h>

using namespace Unigine;
using namespace Math;

// cuda kernel launcher
extern "C"
{
	void cuda_texture_write_process(cudaStream_t stream, cudaSurfaceObject_t dest, int width, int height, int frame);
}

void TextureWriteDemo::init()
{
	// create player
	player = PlayerDummy::create();
	Game::setPlayer(player);

	// create lights
	LightWorldPtr sun = LightWorld::create(vec4_one);

	// create shared texture
	init_resources();

	// load material
	MaterialPtr mesh_base = Materials::findManualMaterial("Unigine::mesh_base");
	mesh_base = mesh_base->inherit();
	mesh_base->setTexture("albedo", render_texture);

	// create meshes (asset shipped in the sample's data/)
	ObjectMeshStaticPtr mesh_0 = ObjectMeshStatic::create("cbox.mesh");
	mesh_0->setTransform(translate(Vec3(-2.0f, 0.0f, 0.0f)));
	mesh_0->setMaterial(mesh_base, "*");

	ObjectMeshStaticPtr mesh_1 = ObjectMeshStatic::create("cbox.mesh");
	mesh_1->setTransform(translate(Vec3(2.0f, 0.0f, 0.0f)));
	mesh_1->setMaterial(mesh_base, "*");

	mesh_base = mesh_base->inherit();
	mesh_base->setTexture("albedo", render_texture2);

	ObjectMeshStaticPtr mesh_2 = ObjectMeshStatic::create("cbox.mesh");
	mesh_2->setTransform(translate(Vec3(0.0f, -2.0f, 0.0f)));
	mesh_2->setMaterial(mesh_base, "*");

	ObjectMeshStaticPtr mesh_3 = ObjectMeshStatic::create("cbox.mesh");
	mesh_3->setTransform(translate(Vec3(0.0f, 2.0f, 0.0f)));
	mesh_3->setMaterial(mesh_base, "*");
}

void TextureWriteDemo::update()
{
	float time = Game::getTime();
	float x = sinf(time * 1.0f) * 4.0f;
	float y = cosf(time * 1.0f) * 4.0f;
	float z = 2.0f + sinf(time * 3.0f) * 1.0f;
	player->setWorldTransform(setTo(Vec3(x, y, z), Vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)));
	player->setFov(60.0f + sinf(time * 2.0f) * 25.0f);
}

void TextureWriteDemo::work(TextureSharedCUDA *texture, int *frame)
{
	int profiler_micro_id = Profiler::beginMicro("CUDATextureWrite");
	Profiler::begin("Texture Write");

	// process
	cuda_texture_write_process(texture->getStream(), texture->getSurface(), texture->getWidth(), texture->getHeight(), *frame);
	++(*frame);

	Profiler::end();
	Profiler::endMicro(profiler_micro_id);
}

void TextureWriteDemo::init_resources()
{
	width = 512;
	height = 512;
	frame = 0;
	frame2 = 0;

	render_texture = Texture::create();
	render_texture->create2D(width, height, Texture::FORMAT_RGBA8, Texture::FORMAT_USAGE_RENDER | Texture::FORMAT_USAGE_SHARED);

	texture.exportToCuda(render_texture);
	texture.getEventWork().connect(event_connections, this, &TextureWriteDemo::work, &texture, &frame);

	render_texture2 = Texture::create();
	render_texture2->create2D(width, height, Texture::FORMAT_RGBA8, Texture::FORMAT_USAGE_RENDER | Texture::FORMAT_USAGE_SHARED);
	texture2.exportToCuda(render_texture2);
	texture2.getEventWork().connect(event_connections, this, &TextureWriteDemo::work, &texture2, &frame2);
}

void TextureWriteDemo::release_resources()
{
	event_connections.disconnectAll();
	texture.clear();
	texture2.clear();
	render_texture = nullptr;
	render_texture2 = nullptr;
}

void TextureWriteDemo::shutdown()
{
	release_resources();
}
