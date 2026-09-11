// Copyright (C), UNIGINE. All rights reserved.

#include <stdio.h>

#include <cuda_runtime.h>

__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return  ((unsigned int)(rgba.w * 255.0f) << 24) |
			((unsigned int)(rgba.z * 255.0f) << 16) |
			((unsigned int)(rgba.y * 255.0f) << 8) |
			((unsigned int)(rgba.x * 255.0f));
}

__global__ void texwrite_kernel(cudaSurfaceObject_t dest, int width, int height, int frame)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height)
		return;

	float4 rgba{};
	rgba.x = ((x + frame) & 0xFF) / 255.0f;
	rgba.y = ((y + frame) & 0xFF) / 255.0f;
	rgba.z = 0.0f;
	rgba.w = 1.0f;
	surf2Dwrite(rgbaFloatToInt(rgba), dest, x * 4, y);
}

extern "C" void cuda_texture_write_process(cudaStream_t stream, cudaSurfaceObject_t dest, int width, int height, int frame)
{
	dim3 Db = dim3(16, 16); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	texwrite_kernel<<<Dg, Db, 0, stream>>>(dest, width, height, frame);

	cudaError_t ret = cudaGetLastError();
	if (ret != cudaSuccess)
		printf("cuda_texture_write_process() failed to launch error = %d\n", ret);
}
