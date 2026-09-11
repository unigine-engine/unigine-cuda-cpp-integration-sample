// Copyright (C), UNIGINE. All rights reserved.

#include <stdio.h>

#include <cuda_runtime.h>

struct Vertex
{
	float3 position;
	float2 uv_0;
	float2 uv_1;
	float4 basis;
	float4 color;
};

__device__ int sign(float x)
{
	int t = x < 0 ? -1 : 0;
	return x > 0 ? 1 : t;
}
__device__ void getTangentBasis(float4 basis, float3 &tangent, float3 &binormal, float3 &normal)
{
	float x2 = basis.x * 2.0f;
	float y2 = basis.y * 2.0f;
	float z2 = basis.z * 2.0f;
	float xx2 = basis.x * x2;
	float xy2 = basis.x * y2;
	float yy2 = basis.y * y2;
	float yz2 = basis.y * z2;
	float zz2 = basis.z * z2;
	float zx2 = basis.z * x2;
	float wx2 = basis.w * x2;
	float wy2 = basis.w * y2;
	float wz2 = basis.w * z2;
	normal = { zx2 + wy2, yz2 - wx2, 1.0f - xx2 - yy2 };
	tangent = { 1.0f - yy2 - zz2, xy2 + wz2, zx2 - wy2 };
	binormal = { xy2 - wz2, 1.0f - xx2 - zz2, yz2 + wx2 };

	binormal.x *= sign(basis.w);
	binormal.y *= sign(basis.w);
	binormal.z *= sign(basis.w);
}

__global__ void mesh_kernel(Vertex *dest, int num_vertices, int frame)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (index >= num_vertices)
		return;

	Vertex *v = &dest[index];

	float3 tangent, binormal, normal;
	getTangentBasis(v->basis, tangent, binormal, normal);

	float offset = sinf(frame / 60) * 0.01f;
	v->position.x += normal.x * offset;
	v->position.y += normal.y * offset;
	v->position.z += normal.z * offset;
}

extern "C" void cuda_mesh_process(cudaStream_t stream, void *dest, int num_vertices, int frame)
{
	dim3 Db = dim3(32); // block dimensions are fixed to be 32 threads
	dim3 Dg = dim3((num_vertices + Db.x - 1) / Db.x);

	mesh_kernel<<<Dg, Db, 0, stream>>>((Vertex*)dest, num_vertices, frame);

	cudaError_t ret = cudaGetLastError();
	if (ret != cudaSuccess)
		printf("cuda_mesh_process() failed to launch error = %d\n", ret);
}
