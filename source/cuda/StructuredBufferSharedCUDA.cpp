#include "StructuredBufferSharedCUDA.h"

#include <UnigineRender.h>

#include "CUDAUtils.h"

using namespace Unigine;

bool StructuredBufferSharedCUDA::exportToCuda(const StructuredBufferPtr &src)
{
	clear();

	if (src->isUsageShared() == false)
		return false;

	structured_buffer = src;

	// nullptr is returned if no SHARED flag
	resource_external_memory = structured_buffer->getResourceExternalMemory();
	if (resource_external_memory == nullptr)
		return false;

	cudaExternalMemoryBufferDesc buffer_desc{};
	buffer_desc.size = resource_external_memory->getSize();

	importResourceExternalMemory(resource_external_memory, &external_memory);
	ck(cudaExternalMemoryGetMappedBuffer(&buffer_ptr, external_memory, &buffer_desc));
	resource_external_memory->closeHandle();


	fence = ResourceFence::create();
	importFenceExternal(fence, &external_semaphore);
	fence->closeHandle();


	Render::getEventEndFrameExecuteCommandLists().connect(event_connection, [this]()
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

void StructuredBufferSharedCUDA::clear()
{
	if (initialized == false)
		return;

	event_connection.disconnect();

	// NOTE:
	// This is not optimal for Structured Buffer re-creation between frames
	// as it's introduces high-cost GPU <-> CPU Sync point before we can destory old resources.
	//
	// What you want to do instead is to have double-buffered Structured Buffers
	// and to have a separate thread which will carry destruction of the current Structured Buffer
	// while the other one is used to render something.
	//
	// Also you can use a single fence for all of your shared resources so that you will have to wait
	// single time instead of waiting for multiple fences, this also may improve performance
	// as far fewver sync Wait/Signal pairs will be issued and waited on.
	cudaStreamSynchronize(stream);
	signal();
	fence->waitGPU();

	cudaFree(buffer_ptr);

	cudaDestroyExternalMemory(external_memory);
	cudaDestroyExternalSemaphore(external_semaphore);

	structured_buffer = nullptr;
	resource_external_memory = nullptr;
	fence = nullptr;

	initialized = false;
}

void StructuredBufferSharedCUDA::wait()
{
	cudaExternalSemaphoreWaitParams desc{};
	desc.params.fence.value = fence->getValue();
	ck(cudaWaitExternalSemaphoresAsync(&external_semaphore, &desc, 1, stream));
}
void StructuredBufferSharedCUDA::signal()
{
	fence->incrementValue();

	cudaExternalSemaphoreSignalParams desc{};
	desc.params.fence.value = fence->getValue();
	ck(cudaSignalExternalSemaphoresAsync(&external_semaphore, &desc, 1, stream));
}
