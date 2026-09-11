// Copyright (C), UNIGINE. All rights reserved.

#include "MeshDynamicDemo.h"

#include <UnigineGame.h>
#include <UnigineLights.h>
#include <UnigineMaterials.h>
#include <UnigineMaterial.h>
#include <UnigineMathLib.h>
#include <UnigineProfiler.h>

using namespace Unigine;
using namespace Math;

// cuda kernel launcher
extern "C"
{
	void cuda_mesh_process(cudaStream_t stream, void *dest, int num_vertices, int frame);
}

void MeshDynamicDemo::init()
{
	// create player
	player = PlayerDummy::create();
	Game::setPlayer(player);

	// create lights
	LightWorldPtr sun = LightWorld::create(vec4_one);

	// load material
	MaterialPtr mesh_base = Materials::findManualMaterial("Unigine::mesh_base");

	// create mesh (asset shipped in the sample's data/)
	mesh_dynamic = ObjectMeshDynamic::create("cbox.mesh", ObjectMeshDynamic::USAGE_MISC_SHARED | ObjectMeshDynamic::USAGE_IMMUTABLE_ALL);
	mesh_dynamic->setMaterial(mesh_base, "*");

	// create shared mesh
	init_resources();
}

void MeshDynamicDemo::update()
{
	player->setWorldTransform(setTo(Vec3(0.0f, 4.0f, 2.0f), Vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)));
	player->setFov(60.0f);
}

void MeshDynamicDemo::work()
{
	int profiler_micro_id = Profiler::beginMicro("CUDAMeshDynamic");
	Profiler::begin("Mesh Process");

	// process
	cuda_mesh_process(mesh.getStream(), mesh.getDevicePtr(), mesh.getNumVertices(), frame);
	frame++;

	Profiler::end();
	Profiler::endMicro(profiler_micro_id);
}

void MeshDynamicDemo::init_resources()
{
	frame = 0;
	mesh.exportToCuda(mesh_dynamic);
	mesh.getEventWork().connect(event_connection, this, &MeshDynamicDemo::work);
}

void MeshDynamicDemo::release_resources()
{
	event_connection.disconnect();
	mesh.clear();
	mesh_dynamic.deleteLater();
}

void MeshDynamicDemo::shutdown()
{
	release_resources();
}
