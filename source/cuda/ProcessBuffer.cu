// Copyright (C), UNIGINE. All rights reserved.

#include <stdio.h>
#include <limits.h>

#include <cuda_runtime.h>

struct __builtin_align__(16) Particle
{
	float3 position;
	float life;
	float3 velocity;
	float align;
};

int __device__ hash(int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return (a % INT_MAX);
}

__global__ void buffer_kernel(Particle *buffer, int num_particles, int frame)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (index >= num_particles)
		return;

	Particle *particle = &buffer[index];

	if (particle->life > 3.0f || particle->life <= 0.0f)
	{
		// Reset particles
		particle->position.x = ((hash(particle->position.x + frame + 0 + index) % 100) / 100.0f * 2.0f - 1.0f);
		particle->position.y = ((hash(particle->position.y + frame + 1 + index) % 100) / 100.0f * 2.0f - 1.0f);
		particle->position.z = ((hash(particle->position.z + frame + 2 + index) % 100) / 100.0f * 2.0f - 1.0f);

		float inv_sqrt = rsqrtf(particle->position.x * particle->position.x + particle->position.y * particle->position.y + particle->position.z * particle->position.z);
		particle->velocity.x = ((hash(particle->position.x + frame + 0 + index) % 100) / 100.0f * 2.0f - 1.0f) * 0.02f;
		particle->velocity.y = ((hash(particle->position.y + frame + 1 + index) % 100) / 100.0f * 2.0f - 1.0f) * 0.02f;
		particle->velocity.z = ((hash(particle->position.z + frame + 2 + index) % 100) / 100.0f * 2.0f - 1.0f) * 0.02f;

		particle->life = 0.01f;
	} else
	{
		particle->position.x += particle->velocity.x;
		particle->position.y += particle->velocity.y;
		particle->position.z += particle->velocity.z;

		particle->velocity.x *= 0.997f;
		particle->velocity.y *= 0.997f;
		particle->velocity.z *= 0.997f;

		particle->life += 0.01f;
	}
}

extern "C" void cuda_buffer_process(cudaStream_t stream, void *buffer, int num_particles, int frame)
{
	dim3 Db = dim3(16);
	dim3 Dg = dim3((num_particles + Db.x - 1) / Db.x);

	buffer_kernel<<<Dg, Db, 0, stream>>>((Particle*)buffer, num_particles, frame);

	cudaError_t ret = cudaGetLastError();
	if (ret != cudaSuccess)
		printf("cuda_buffer_process() failed to launch error = %d\n", ret);
}
