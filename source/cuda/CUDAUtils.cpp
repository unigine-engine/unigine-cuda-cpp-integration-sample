#include "CUDAUtils.h"

#include <UnigineSystemInfo.h>
#include <UnigineRender.h>


namespace Unigine
{

void importResourceExternalMemory(Unigine::ResourceExternalMemoryPtr resource_external_memory, cudaExternalMemory_t *external_memory)
{
	cudaExternalMemoryHandleDesc desc{};
	switch (Render::getAPI())
	{
		case Render::API_DIRECT3D12: desc.type = cudaExternalMemoryHandleTypeD3D12Resource; break;
		case Render::API_VULKAN:
			#ifdef _WIN32
				desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
			#else
				desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
			#endif
			break;
	}

	#ifdef _WIN32
		desc.handle.win32.handle = resource_external_memory->getWin32Handle();
	#else
		desc.handle.fd = resource_external_memory->getFdHandle();
	#endif

	desc.size = resource_external_memory->getSize();

	// Required since all Graphics APIs internally allocate dedicated memory for shared resources
	desc.flags = cudaExternalMemoryDedicated;

	ck(cudaImportExternalMemory(external_memory, &desc));
}

void importFenceExternal(Unigine::ResourceFencePtr fence, cudaExternalSemaphore_t *external_semaphore)
{
	cudaExternalSemaphoreHandleDesc desc{};
	#ifdef _WIN32
		switch (Render::getAPI())
		{
			case Render::API_DIRECT3D12: desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence; break;
			case Render::API_VULKAN: desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32; break;
		}

		desc.handle.win32.handle = fence->getWin32Handle();
	#else
		desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
		desc.handle.fd = fence->getFdHandle();
	#endif

	ck(cudaImportExternalSemaphore(external_semaphore, &desc));
}

#ifndef _WIN32
namespace
{

bool parse_uuid_string(const char *str, unsigned char (&bytes)[16])
{
	if (str == nullptr)
		return false;

	int nibbles = 0;
	for (const char *c = str; *c != 0; ++c)
	{
		// Group separators carry no data.
		if (*c == '-')
			continue;

		int digit = 0;
		if (*c >= '0' && *c <= '9')
			digit = *c - '0';
		else if (*c >= 'a' && *c <= 'f')
			digit = *c - 'a' + 10;
		else if (*c >= 'A' && *c <= 'F')
			digit = *c - 'A' + 10;
		else
			return false;

		if (nibbles >= 32)
			return false;

		if (nibbles % 2 == 0)
			bytes[nibbles / 2] = static_cast<unsigned char>(digit << 4);
		else
			bytes[nibbles / 2] |= static_cast<unsigned char>(digit);
		++nibbles;
	}

	return nibbles == 32;
}

} // namespace
#endif

void cudaSetDeviceToEngineGPU()
{
	int count = 0;
	ck(cudaGetDeviceCount(&count));

#ifdef _WIN32
	const unsigned long long luid = SystemInfo::getGPULuid();
#else
	const char *uuid_str = SystemInfo::getGPUUuidString();
	unsigned char uuid[16] = {};
	if (!parse_uuid_string(uuid_str, uuid))
	{
		Log::fatal("cudaSetDeviceToEngineGPU(): malformed engine GPU UUID '%s'\n",
			uuid_str ? uuid_str : "");
		return;
	}
#endif

	for (int i = 0; i < count; i++)
	{
		cudaDeviceProp prop{};
		if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
			continue;

#ifdef _WIN32
		if (memcmp(prop.luid, &luid, sizeof(luid)) == 0)
#else
		if (memcmp(prop.uuid.bytes, uuid, sizeof(uuid)) == 0)
#endif
		{
			ck(cudaSetDevice(i));
			return;
		}
	}
	Log::error("cudaSetDeviceToEngineGPU(): %d CUDA device(s) visible:\n", count);
	for (int i = 0; i < count; i++)
	{
		cudaDeviceProp prop{};
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
			Log::error("  CUDA device %d: '%s'\n", i, prop.name);
	}

#ifdef _WIN32
	Log::fatal("cudaSetDeviceToEngineGPU(): no CUDA device matches engine GPU LUID 0x%llx\n", luid);
#else
	Log::fatal("cudaSetDeviceToEngineGPU(): no CUDA device matches engine GPU UUID %s\n", uuid_str);
#endif
}

}
