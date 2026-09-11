// Copyright (C), UNIGINE. All rights reserved.

#pragma once

#include <UniginePlayers.h>
#include <UnigineTextures.h>
#include <UnigineEvent.h>

#include "CudaDemo.h"
#include "../cuda/TextureSharedCUDA.h"

// [4] Texture Write: generates procedural texture content directly on the GPU with a CUDA
// kernel (surf2Dwrite) into a shared texture used as a mesh albedo.
class TextureWriteDemo : public CudaDemo
{
public:
	void init() override;
	void update() override;
	void shutdown() override;

private:
	void work(TextureSharedCUDA *texture, int *frame);
	void init_resources();
	void release_resources();

	Unigine::PlayerDummyPtr player;
	Unigine::TexturePtr render_texture;
	Unigine::TexturePtr render_texture2;

	TextureSharedCUDA texture;
	TextureSharedCUDA texture2;
	Unigine::EventConnections event_connections;

	int width{ 0 };
	int height{ 0 };
	int frame{ 0 };
	int frame2{ 0 };
};
