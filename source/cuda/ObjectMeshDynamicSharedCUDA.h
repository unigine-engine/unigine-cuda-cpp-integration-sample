#pragma once

#include <cuda_runtime.h>
#include <UnigineMeshDynamic.h>
#include <UnigineObjects.h>
#include <UnigineResourceFence.h>
#include <UnigineResourceExternalMemory.h>
#include <UnigineEvent.h>

class ObjectMeshDynamicSharedCUDA
{
public:
	ObjectMeshDynamicSharedCUDA()
	{
	}
	~ObjectMeshDynamicSharedCUDA()
	{
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
		clear();
	}

	bool exportToCuda(const Unigine::ObjectMeshDynamicPtr &src);

	void clear();

	void *getDevicePtr() const { return buffer_ptr; }
	cudaExternalMemory_t getExternalMemory() const { return external_memory; }
	cudaStream_t getStream() const { return stream; }

	int getNumVertices() const { return mesh_dynamic->getNumVertex(); }

	Unigine::ResourceFencePtr &getFence() { return fence; }
	Unigine::Event<> &getEventWork() { return event_work; }

	void wait();
	void signal();

	operator const Unigine::ObjectMeshDynamicPtr &() const { return mesh_dynamic; }
	const Unigine::ObjectMeshDynamicPtr &getMeshDynamic() const { return mesh_dynamic; }


	bool isInitialized() const { return initialized; }

private:
	void create_handle();

	bool initialized{};

	Unigine::ObjectMeshDynamicPtr mesh_dynamic;

	Unigine::ResourceFencePtr fence{};
	Unigine::ResourceExternalMemoryPtr rem_vertex{};
	// Samples use only vertex buffer for now, feel free to extend for index buffer
	//Unigine::ResourceExternalMemoryPtr rem_index{};

	Unigine::EventInvoker<> event_work;
	Unigine::EventConnections event_connections;

	cudaStream_t stream{};
	cudaExternalSemaphore_t external_semaphore{};
	cudaExternalMemory_t external_memory{};
	void *buffer_ptr;
};
