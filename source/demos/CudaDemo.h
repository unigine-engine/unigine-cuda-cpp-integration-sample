// Copyright (C), UNIGINE. All rights reserved.

#pragma once

// Base interface for a single CUDA integration demo. The dispatcher (AppWorldLogic)
// owns exactly one instance at a time, chosen by the selected demo, and drives its
// lifecycle through the loaded world's init/update/shutdown.
class CudaDemo
{
public:
	virtual ~CudaDemo() {}

	virtual void init() {}
	virtual void update() {}
	virtual void shutdown() {}
};
