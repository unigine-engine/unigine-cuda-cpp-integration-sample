// Copyright (C), UNIGINE. All rights reserved.

#include <stdio.h>

__global__ void textransfer_kernel(unsigned char *dest, unsigned char *src, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	static int src_pixel_width = 4; // rgba8
	static int dest_pixel_width = 3; // rgb8

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height)
		return;

	unsigned char *src_pixel = (unsigned char *)(src + y * width * src_pixel_width) + x * src_pixel_width;
	unsigned char *dest_pixel = (unsigned char *)(dest + y * width * dest_pixel_width) + x * dest_pixel_width;
	dest_pixel[0] = src_pixel[0];
	dest_pixel[1] = src_pixel[1];
	dest_pixel[2] = src_pixel[2];
}

extern "C" void cuda_texture_transfer_process(cudaStream_t stream, void *dest, void *src, int width, int height)
{
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	textransfer_kernel<<<Dg, Db, 0, stream>>>((unsigned char *)dest, (unsigned char *)src, width, height);

	cudaError_t ret = cudaGetLastError();
	if (ret != cudaSuccess)
		printf("cuda_texture_transfer_process() failed to launch error = %d\n", ret);
}
