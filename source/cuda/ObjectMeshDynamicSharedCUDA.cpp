#include "ObjectMeshDynamicSharedCUDA.h"

#include <UnigineRender.h>

#include "CUDAUtils.h"

using namespace Unigine;

bool ObjectMeshDynamicSharedCUDA::exportToCuda(const ObjectMeshDynamicPtr &src)
{
	clear();

	if (src->isUsageShared() == false)
		return false;

	mesh_dynamic = src;

	// nullptr is returned if no SHARED flag
	rem_vertex = mesh_dynamic->getExternalMemoryVertexBuffer();
	if (rem_vertex == nullptr)
		return false;


	rem_vertex->getEventCreateHandle().connect(event_connections, this, &ObjectMeshDynamicSharedCUDA::create_handle);
	rem_vertex->getEventCloseHandle().connect(event_connections, [this]
	{
		if (external_memory)
			cudaDestroyExternalMemory(external_memory);
		external_memory = nullptr;
		rem_vertex = nullptr;
	});
	create_handle();

	fence = ResourceFence::create();
	importFenceExternal(fence, &external_semaphore);
	fence->closeHandle();


	Render::getEventEndFrameExecuteCommandLists().connect(event_connections, [this]()
	{
		if (fence.isValid() == false || fence->isEnabled() == false)
			return;

		if (event_work.empty() == false)
		{
			wait();
			event_work.run();
		}
		signal();
	});

	ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	ck(cudaStreamSynchronize(stream));

	initialized = true;
	return true;
}

void ObjectMeshDynamicSharedCUDA::clear()
{
	if (initialized == false)
		return;

	event_connections.disconnectAll();

	// NOTE:
	// This is not optimal for Mesh Dynamic re-creation between frames
	// as it's introduces high-cost GPU <-> CPU Sync point before we can destory old resources.
	//
	// What you want to do instead is to have double-buffered Mesh Dynamics
	// and to have a separate thread which will carry destruction of the current Mesh Dynamic
	// while the other one is used to render something.
	//
	// Also you can use a single fence for all of your shared resources so that you will have to wait
	// single time instead of waiting for multiple fences, this also may improve performance
	// as far fewver sync Wait/Signal pairs will be issued and waited on.
	cudaStreamSynchronize(stream);
	signal();
	fence->waitGPU();

	cudaFree(buffer_ptr);

	if (external_memory)
		cudaDestroyExternalMemory(external_memory);
	cudaDestroyExternalSemaphore(external_semaphore);

	mesh_dynamic = nullptr;
	rem_vertex = nullptr;
	fence = nullptr;

	initialized = false;
}

void ObjectMeshDynamicSharedCUDA::wait()
{
	cudaExternalSemaphoreWaitParams desc{};
	desc.params.fence.value = fence->getValue();
	ck(cudaWaitExternalSemaphoresAsync(&external_semaphore, &desc, 1, stream));
}
void ObjectMeshDynamicSharedCUDA::signal()
{
	fence->incrementValue();

	cudaExternalSemaphoreSignalParams desc{};
	desc.params.fence.value = fence->getValue();
	ck(cudaSignalExternalSemaphoresAsync(&external_semaphore, &desc, 1, stream));
}

void ObjectMeshDynamicSharedCUDA::create_handle()
{
	cudaExternalMemoryBufferDesc buffer_desc{};
	buffer_desc.size = rem_vertex->getSize();

	importResourceExternalMemory(rem_vertex, &external_memory);
	ck(cudaExternalMemoryGetMappedBuffer(&buffer_ptr, external_memory, &buffer_desc));
	rem_vertex->closeHandle();
}
