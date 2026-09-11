#pragma once

#include <cuda_runtime.h>
#include <UnigineTextures.h>
#include <UnigineResourceFence.h>
#include <UnigineResourceExternalMemory.h>
#include <UnigineEvent.h>

class TextureSharedCUDA
{
public:
	TextureSharedCUDA()
	{
	}
	~TextureSharedCUDA()
	{
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
		clear();
	}

	bool exportToCuda(const Unigine::TexturePtr &src);

	void clear();

	cudaSurfaceObject_t getSurface() const { return surface; }
	cudaExternalMemory_t getExternalMemory() const { return external_memory; }
	cudaStream_t getStream() const { return stream; }
	cudaMipmappedArray_t getMipArray() const { return mip_array; }
	cudaArray_t getMipLevel(int mip) const;

	Unigine::ResourceFencePtr &getFence() { return fence; }
	Unigine::Event<> &getEventWork() { return event_work; }

	void wait();
	void signal();

	operator const Unigine::TexturePtr&() const { return texture; }
	const Unigine::TexturePtr &getTexture() const { return texture; }


	bool isInitialized() const { return initialized; }

	int getWidth() const { return texture->getWidth(); }
	int getHeight() const { return texture->getHeight(); }

private:
	cudaChannelFormatDesc convert_texture_format() const;

	bool initialized{};

	Unigine::TexturePtr texture;

	Unigine::ResourceFencePtr fence{};
	Unigine::ResourceExternalMemoryPtr resource_external_memory{};

	Unigine::EventInvoker<> event_work;
	Unigine::EventConnection event_connection;

	cudaStream_t stream{};
	cudaExternalSemaphore_t external_semaphore{};
	cudaExternalMemory_t external_memory{};
	cudaSurfaceObject_t surface{};
	cudaMipmappedArray_t mip_array{};
};
