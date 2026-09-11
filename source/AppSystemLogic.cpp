// Copyright (C), UNIGINE. All rights reserved.

#include "AppSystemLogic.h"

#include <UnigineEngine.h>
#include <UnigineConsole.h>
#include <UnigineInput.h>
#include <UnigineGui.h>
#include <UnigineWindowManager.h>

#include "demos/DemoSelection.h"

using namespace Unigine;

namespace
{
// Demo hotkey -> (selection id, world to load). Kept in sync with the legend below.
struct DemoEntry
{
	Input::KEY key;
	DemoId id;
	const char *world;
};

const DemoEntry DEMOS[] = {
	{ Input::KEY_DIGIT_1, DemoId::MeshDynamic,      "CUDA_mesh_dynamic" },
	{ Input::KEY_DIGIT_2, DemoId::StructuredBuffer, "CUDA_structured_buffer_write" },
	{ Input::KEY_DIGIT_3, DemoId::TextureTransfer,  "CUDA_texture_transfer" },
	{ Input::KEY_DIGIT_4, DemoId::TextureWrite,     "CUDA_texture_write" },
};

const char *LEGEND_TEXT =
	"CUDA C++ integration demos\n"
	"  [1] Mesh Dynamic      - warp mesh vertices in a CUDA kernel\n"
	"  [2] Structured Buffer - GPU particle simulation\n"
	"  [3] Texture Transfer  - fast texture readback (rgba8 -> rgb8)\n"
	"  [4] Texture Write     - procedural texture via surf2Dwrite";

void load_demo(const DemoEntry &entry)
{
	g_selected_demo = entry.id;
	Console::run(String::format("world_load %s", entry.world));
}
} // namespace

int AppSystemLogic::init()
{
	Engine::get()->setBackgroundUpdate(Engine::BACKGROUND_UPDATE_RENDER_NON_MINIMIZED);

	// on-screen legend, kept visible across demo switches
	legend = WidgetLabel::create(Gui::getCurrent(), LEGEND_TEXT);
	legend->setFontSize(16);
	Gui::getCurrent()->addChild(legend, Gui::ALIGN_LEFT | Gui::ALIGN_TOP | Gui::ALIGN_OVERLAP);

	// load the default demo world
	load_demo(DEMOS[0]);

	return 1;
}

int AppSystemLogic::update()
{
	EngineWindowViewportPtr main_window = WindowManager::getMainWindow();
	if (!main_window)
	{
		Engine::get()->quit();
		return 1;
	}

	for (const DemoEntry &entry : DEMOS)
	{
		if (Input::isKeyPressed(entry.key))
		{
			load_demo(entry);
			break;
		}
	}

	return 1;
}

int AppSystemLogic::shutdown()
{
	legend = nullptr;
	return 1;
}
