// Copyright (C), UNIGINE. All rights reserved.

#pragma once

#include <UniginePlayers.h>
#include <UnigineObjects.h>
#include <UnigineEvent.h>

#include "CudaDemo.h"
#include "../cuda/ObjectMeshDynamicSharedCUDA.h"

// [1] Mesh Dynamic: warps mesh vertices in a CUDA kernel through a shared vertex buffer,
// so the deformation happens entirely on the GPU with no CPU round-trip.
class MeshDynamicDemo : public CudaDemo
{
public:
	void init() override;
	void update() override;
	void shutdown() override;

private:
	void work();
	void init_resources();
	void release_resources();

	Unigine::PlayerDummyPtr player;
	Unigine::ObjectMeshDynamicPtr mesh_dynamic;

	ObjectMeshDynamicSharedCUDA mesh;
	Unigine::EventConnection event_connection;
	int frame = 0;
};
