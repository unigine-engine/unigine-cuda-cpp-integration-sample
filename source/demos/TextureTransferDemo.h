// Copyright (C), UNIGINE. All rights reserved.

#pragma once

#include <UniginePlayers.h>
#include <UnigineImage.h>
#include <UnigineTextures.h>
#include <UnigineTimer.h>
#include <UnigineEvent.h>

#include "CudaDemo.h"
#include "../cuda/TextureSharedCUDA.h"

// [3] Texture Transfer: reads back a rendered texture through CUDA, converting rgba8 -> rgb8
// in a kernel (useful e.g. to reduce transfer size before copying to the host).
class TextureTransferDemo : public CudaDemo
{
public:
	void init() override;
	void update() override;
	void shutdown() override;

private:
	void transfer_screen_color();
	void init_resources();
	void release_resources();

	Unigine::PlayerDummyPtr player;
	Unigine::ImagePtr image;
	Unigine::TexturePtr render_texture;

	TextureSharedCUDA texture{};
	Unigine::EventConnection event_connection;

	// pinned memory
	unsigned char *pinned_buffer{ nullptr };

	static const int NUM_TIME_FRAMES = 100;
	Unigine::Timer frame_timer[NUM_TIME_FRAMES];

	int width{ 0 };
	int height{ 0 };
	const int pixel_size{ 4 }; // rgba
	int buffer_size{ 0 };

	const int pixel_process_size{ 3 }; // rgb
	int buffer_process_size{ 0 };
	void *gpu_original_buffer{ nullptr };
	void *gpu_process_buffer{ nullptr };

	int image_format{ 0 };
	int image_size{ 0 };
};
