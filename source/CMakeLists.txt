##==============================================================================
## General.
##==============================================================================
cmake_minimum_required(VERSION 3.19)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/UnigineCMakeVariables.cmake)

project(unigine-cuda-cpp-integration-sample LANGUAGES CXX)

##==============================================================================
## CUDA.
##==============================================================================
# CUDA is a hard requirement for this sample (external-memory / fence interop is
# implemented with the CUDA runtime). The centralized CI agents ship the CUDA
# Toolkit, so fail loudly rather than silently producing no binary.
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
else()
	message(FATAL_ERROR "CUDA Toolkit not found. Install CUDA Toolkit 12.x (https://developer.nvidia.com/cuda-downloads) to build unigine-cuda-cpp-integration-sample.")
endif()

# Link the static CUDA runtime so no cudart DLL has to be shipped alongside the app.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

##==============================================================================
## Environment.
##==============================================================================
set(UNIGINE_DOUBLE False CACHE BOOL "Double coords")
set(UNIGINE_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/../)
# Multi-sample repo fallback: if the per-sample SDK payload was not restored by
# SDK Browser (Repair), use the shared SDK at the repository root (two levels up).
# SDK Browser users (per-sample SDK present) are unaffected.
if(NOT EXISTS "${UNIGINE_SDK_PATH}include/UnigineEngine.h")
	get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
	set(UNIGINE_SDK_PATH "${_repo_root}/")
endif()
set(UNIGINE_BIN_DIR ${UNIGINE_SDK_PATH}bin)
set(UNIGINE_LIB_DIR ${UNIGINE_SDK_PATH}lib)
set(UNIGINE_INCLUDE_DIR ${UNIGINE_SDK_PATH}include)

##==============================================================================
## Dependencies.
##==============================================================================
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Engine.
find_package(Engine REQUIRED MODULE QUIET)

##==============================================================================
## Target.
##==============================================================================
set(target "unigine-cuda-cpp-integration-sample")

if (WIN32)
	# Console app in Debug; windowed in Release. The app entry point is main().
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()

add_executable(${target}
		${CMAKE_CURRENT_LIST_DIR}/main.cpp
		${CMAKE_CURRENT_LIST_DIR}/AppSystemLogic.cpp
		${CMAKE_CURRENT_LIST_DIR}/AppSystemLogic.h
		${CMAKE_CURRENT_LIST_DIR}/AppWorldLogic.cpp
		${CMAKE_CURRENT_LIST_DIR}/AppWorldLogic.h

		${CMAKE_CURRENT_LIST_DIR}/demos/CudaDemo.h
		${CMAKE_CURRENT_LIST_DIR}/demos/DemoSelection.h
		${CMAKE_CURRENT_LIST_DIR}/demos/MeshDynamicDemo.cpp
		${CMAKE_CURRENT_LIST_DIR}/demos/MeshDynamicDemo.h
		${CMAKE_CURRENT_LIST_DIR}/demos/StructuredBufferDemo.cpp
		${CMAKE_CURRENT_LIST_DIR}/demos/StructuredBufferDemo.h
		${CMAKE_CURRENT_LIST_DIR}/demos/TextureTransferDemo.cpp
		${CMAKE_CURRENT_LIST_DIR}/demos/TextureTransferDemo.h
		${CMAKE_CURRENT_LIST_DIR}/demos/TextureWriteDemo.cpp
		${CMAKE_CURRENT_LIST_DIR}/demos/TextureWriteDemo.h

		${CMAKE_CURRENT_LIST_DIR}/cuda/CUDAUtils.cpp
		${CMAKE_CURRENT_LIST_DIR}/cuda/CUDAUtils.h
		${CMAKE_CURRENT_LIST_DIR}/cuda/ObjectMeshDynamicSharedCUDA.cpp
		${CMAKE_CURRENT_LIST_DIR}/cuda/ObjectMeshDynamicSharedCUDA.h
		${CMAKE_CURRENT_LIST_DIR}/cuda/StructuredBufferSharedCUDA.cpp
		${CMAKE_CURRENT_LIST_DIR}/cuda/StructuredBufferSharedCUDA.h
		${CMAKE_CURRENT_LIST_DIR}/cuda/TextureSharedCUDA.cpp
		${CMAKE_CURRENT_LIST_DIR}/cuda/TextureSharedCUDA.h

		${CMAKE_CURRENT_LIST_DIR}/cuda/ProcessMesh.cu
		${CMAKE_CURRENT_LIST_DIR}/cuda/ProcessBuffer.cu
		${CMAKE_CURRENT_LIST_DIR}/cuda/ProcessTextureWrite.cu
		${CMAKE_CURRENT_LIST_DIR}/cuda/ProcessTextureTransfer.cu
)

target_include_directories(${target}
	PRIVATE
	${UNIGINE_INCLUDE_DIR}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	)

target_link_libraries(${target}
	PRIVATE
	Unigine::Engine
	)

target_compile_definitions(${target}
	PRIVATE
	$<$<BOOL:${UNIX}>:_LINUX>
	$<$<CONFIG:Debug>:DEBUG>
	$<$<NOT:$<CONFIG:Debug>>:NDEBUG>
	)

##==============================================================================
## Compiler constants.
##==============================================================================
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(UNIGINE_COMPILER_IS_MSVC TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(UNIGINE_COMPILER_IS_CLANG TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(UNIGINE_COMPILER_IS_GNU TRUE)
endif()

# NOTE: host compiler flags are guarded to $<COMPILE_LANGUAGE:CXX> so they are not
# forwarded to nvcc when compiling the .cu translation units.
if (UNIGINE_COMPILER_IS_MSVC)
    target_compile_definitions(${target}
	PRIVATE
	_CRT_SECURE_NO_DEPRECATE
	)
    include(ProcessorCount)
    ProcessorCount(proc_count)
    target_compile_options(${target}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:/TP>               # Specifies all source files are C++.
	$<$<COMPILE_LANGUAGE:CXX>:/FS>               # Serialize PDB writes through MSPDBSRV.EXE.
	$<$<COMPILE_LANGUAGE:CXX>:/fp:fast>          # Specifies floating-point behavior.
	$<$<COMPILE_LANGUAGE:CXX>:/bigobj>           # Increases addressable sections in an .obj file.
	$<$<COMPILE_LANGUAGE:CXX>:/EHsc>             # Specifies the kind of exception handling used.
	$<$<COMPILE_LANGUAGE:CXX>:/MP${proc_count}>  # Build with Multiple Processes.
	)
    unset(proc_count)
    target_link_options(${target} INTERFACE "/FIXED:NO")

elseif(UNIGINE_COMPILER_IS_GNU OR UNIGINE_COMPILER_IS_CLANG)

    target_compile_options(${target}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:-m64>
	$<$<COMPILE_LANGUAGE:CXX>:-march=athlon64>
	$<$<COMPILE_LANGUAGE:CXX>:-msse4.2>
	# Optimizations.
	$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
	$<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing>
	$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-fomit-frame-pointer>
	# Warnings.
	$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Wall>
	$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Wno-unknown-pragmas>
	$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Wno-unused-parameter>
	)
endif ()

set(binary_name ${target})
if (UNIGINE_DOUBLE)
	string(APPEND binary_name "_double")
endif()
string(APPEND binary_name "_x64")
set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(${target} PROPERTIES OUTPUT_NAME ${binary_name})

set_target_properties(${target}
	PROPERTIES
	# compute_75 PTX (JIT-forward-compatible to newer GPUs) + sm_75 SASS. sm_75 (Turing)
	# is the lowest architecture accepted by every CUDA Toolkit on the build agents
	# (12.3 / 12.4 / 13.x); CUDA 13 dropped Pascal/Volta, so the older compute_60 the
	# original engine samples used no longer compiles.
	CUDA_ARCHITECTURES "75-virtual;75-real"
	CUDA_SEPARABLE_COMPILATION ON
	RUNTIME_OUTPUT_DIRECTORY ${UNIGINE_BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${UNIGINE_BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${UNIGINE_BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${UNIGINE_BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${UNIGINE_BIN_DIR}
	)

unset(UNIGINE_DOUBLE CACHE)
