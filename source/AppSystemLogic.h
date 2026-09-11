// Copyright (C), UNIGINE. All rights reserved.

#ifndef __APP_SYSTEM_LOGIC_H__
#define __APP_SYSTEM_LOGIC_H__

#include <UnigineLogic.h>
#include <UnigineWidgets.h>

// System logic: owns the on-screen legend and the demo hotkeys (1-4). It persists across
// world switches, so it drives which CUDA demo world is loaded.
class AppSystemLogic : public Unigine::SystemLogic
{
public:
	AppSystemLogic() {}
	virtual ~AppSystemLogic() {}

	int init() override;
	int update() override;
	int shutdown() override;

private:
	Unigine::WidgetLabelPtr legend;
};

#endif // __APP_SYSTEM_LOGIC_H__
