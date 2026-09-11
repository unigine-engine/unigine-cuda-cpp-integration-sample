// Copyright (C), UNIGINE. All rights reserved.

#include "AppWorldLogic.h"

#include "demos/CudaDemo.h"
#include "demos/DemoSelection.h"
#include "demos/MeshDynamicDemo.h"
#include "demos/StructuredBufferDemo.h"
#include "demos/TextureTransferDemo.h"
#include "demos/TextureWriteDemo.h"

// Selected demo, written by AppSystemLogic before the matching world is loaded.
DemoId g_selected_demo = DemoId::MeshDynamic;

AppWorldLogic::AppWorldLogic()
{}

AppWorldLogic::~AppWorldLogic()
{}

int AppWorldLogic::init()
{
	switch (g_selected_demo)
	{
		case DemoId::StructuredBuffer: demo = new StructuredBufferDemo(); break;
		case DemoId::TextureTransfer:  demo = new TextureTransferDemo(); break;
		case DemoId::TextureWrite:     demo = new TextureWriteDemo(); break;
		case DemoId::MeshDynamic:
		default:                       demo = new MeshDynamicDemo(); break;
	}

	demo->init();
	return 1;
}

int AppWorldLogic::update()
{
	if (demo)
		demo->update();
	return 1;
}

int AppWorldLogic::shutdown()
{
	if (demo)
	{
		demo->shutdown();
		delete demo;
		demo = nullptr;
	}
	return 1;
}
