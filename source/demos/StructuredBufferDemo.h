// Copyright (C), UNIGINE. All rights reserved.

#pragma once

#include <UniginePlayers.h>
#include <UnigineMaterial.h>
#include <UnigineEvent.h>

#include "CudaDemo.h"
#include "../cuda/StructuredBufferSharedCUDA.h"

// [2] Structured Buffer: runs a GPU particle simulation in a shared structured buffer,
// then composites the particles onto the screen with a compute material.
class StructuredBufferDemo : public CudaDemo
{
public:
	void init() override;
	void update() override;
	void shutdown() override;

private:
	void work();
	void composite();
	void init_resources();
	void release_resources();

	Unigine::MaterialPtr material;
	Unigine::PlayerSpectatorPtr player;
	Unigine::StructuredBufferPtr structured_buffer;

	StructuredBufferSharedCUDA shared_buffer;
	Unigine::EventConnections event_connections;
	int frame = 0;
};
