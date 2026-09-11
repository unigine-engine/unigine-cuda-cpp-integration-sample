#pragma once

#include <cuda_runtime.h>
#include <UnigineTextures.h>
#include <UnigineResourceFence.h>
#include <UnigineResourceExternalMemory.h>
#include <UnigineEvent.h>

class StructuredBufferSharedCUDA
{
public:
	StructuredBufferSharedCUDA()
	{
	}
	~StructuredBufferSharedCUDA()
	{
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
		clear();
	}

	bool exportToCuda(const Unigine::StructuredBufferPtr &src);

	void clear();

	void *getDevicePtr() const { return buffer_ptr; }
	cudaExternalMemory_t getExternalMemory() const { return external_memory; }
	cudaStream_t getStream() const { return stream; }
	int getNumElements() const { return structured_buffer->getNumElements(); }

	Unigine::ResourceFencePtr &getFence() { return fence; }
	Unigine::Event<> &getEventWork() { return event_work; }

	void wait();
	void signal();

	operator const Unigine::StructuredBufferPtr&() const { return structured_buffer; }
	const Unigine::StructuredBufferPtr &getStructuredBuffer() const { return structured_buffer; }


	bool isInitialized() const { return initialized; }

	unsigned long long getSize() const { return resource_external_memory->getSize(); }

private:
	bool initialized{};

	Unigine::StructuredBufferPtr structured_buffer;

	Unigine::ResourceFencePtr fence{};
	Unigine::ResourceExternalMemoryPtr resource_external_memory{};

	Unigine::EventInvoker<> event_work;
	Unigine::EventConnection event_connection;

	cudaStream_t stream{};
	cudaExternalSemaphore_t external_semaphore{};
	cudaExternalMemory_t external_memory{};
	void *buffer_ptr;
};
