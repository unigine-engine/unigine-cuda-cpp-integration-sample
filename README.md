# CUDA Integration Sample

[**NVIDIA CUDA**](https://developer.nvidia.com/cuda-toolkit) lets you run general-purpose compute kernels on the
GPU. This sample shows how to share GPU resources between UNIGINE and CUDA **without a CPU round-trip**, using the
engine's external-memory / fence interop (Direct3D 12 shared resources or Vulkan external memory + timeline
semaphores). One executable bundles four demos; switch between them at runtime with keys **1–4**:

| Key | Demo | What it shows |
|---|---|---|
| **1** | **Mesh Dynamic** | warps the vertices of an `ObjectMeshDynamic` in a CUDA kernel through a shared vertex buffer |
| **2** | **Structured Buffer** | runs a GPU particle simulation in a shared `StructuredBuffer`, then composites it with a compute material |
| **3** | **Texture Transfer** | reads a rendered texture back through CUDA, converting `rgba8 -> rgb8` in a kernel |
| **4** | **Texture Write** | generates procedural texture content directly on the GPU with `surf2Dwrite` into a shared texture |

The shared-resource plumbing lives in `source/cuda/` (`CUDAUtils`, `ObjectMeshDynamicSharedCUDA`,
`StructuredBufferSharedCUDA`, `TextureSharedCUDA`); each demo's CUDA kernel is a separate `.cu` file.

## How to Run the Sample

### Prerequisites

- [**UNIGINE SDK Browser**](https://developer.unigine.com/en/docs/latest/start/installing_sdk) (latest version)
- **UNIGINE SDK Community** or **Engineering** edition (**Sim** upgrade supported)
- **Visual Studio 2022** (Windows) or GCC/Clang (Linux)
- **NVIDIA CUDA Toolkit 12.x** ([downloads](https://developer.nvidia.com/cuda-downloads)) with the Visual Studio
  build integration (Windows). CMake fails with a clear message if the CUDA compiler is not found.
- An **NVIDIA GPU** and a **Direct3D 12 (Windows)** or **Vulkan** rendering backend — the CUDA interop relies on
  shared external memory, which is unavailable on other backends.

### Third-party dependency

The CUDA runtime is linked **statically** (`cudart_static`), so no CUDA DLL is shipped next to the app. Only the
CUDA Toolkit (compiler + headers) is required at build time; an NVIDIA driver is required at run time. The kernels
are built for `compute_75` (PTX, JIT-forward-compatible to newer GPUs) plus `sm_75` SASS, so a **Turing (sm_75) or
newer** NVIDIA GPU is required (CUDA Toolkit 13 dropped Pascal/Volta support).

### Rendering backends

- **Windows:** Direct3D 12 or Vulkan. The Direct3D 12 backend loads the D3D12 Agility SDK redistributable from
  `bin/D3D12/` (relative to the executable).
- **Linux:** Vulkan.

### Step-by-Step Guide

1. **Clone or download** the sample.
2. **Open SDK Browser** and make sure you have the latest version.
3. **Add the sample project**: *My Projects* → *Add Existing* → select the `.project` file that matches your OS
   (`*_win_*` / `*_lin_*`), edition, and precision → *Import Project*.
4. **Repair** the project (only essential files are in Git; SDK Browser restores the rest), then *Configure Project*.
5. **Open** the project in your IDE: load the folder containing `source/CMakeLists.txt` (Visual Studio 2022 recommended).
6. **Build** and **Run**. Press **1–4** to switch demos.

> [!WARNING]
> Precision must match the `.project` you selected. The coordinate precision is set in `source/CMakeLists.txt`:
> ```diff
> - set(UNIGINE_DOUBLE False CACHE BOOL "Double coords")
> + set(UNIGINE_DOUBLE True  CACHE BOOL "Double coords")
> ```
> Use a `*_double.project` for double-precision builds and a `*_float.project` for float.

## What the Sample Contains

```
unigine-cuda-cpp-integration-sample/
  source/     — main.cpp, AppSystemLogic (legend + 1-4 hotkeys),
                AppWorldLogic (demo dispatcher),
                demos/ (MeshDynamic, StructuredBuffer, TextureTransfer, TextureWrite),
                cuda/  (shared interop wrappers + ProcessMesh/ProcessBuffer/
                        ProcessTextureWrite/ProcessTextureTransfer .cu kernels),
                CMakeLists.txt + cmake/ (Engine resolution + CUDA language)
  data/       — CUDA_mesh_dynamic / CUDA_structured_buffer_write / CUDA_texture_transfer /
                CUDA_texture_write worlds, ScreenSpaceParticles.basemat, cbox.mesh,
                root_mount.umount
  README.md   — this file
  *.project   — SDK Browser project files (per platform / edition / precision)
```

The whole repository is built by the shared scripts in the repo-root `ci/`
(`build_windows.bat` / `build_linux.sh`); samples do not carry their own `ci/` folder.

## If the Sample Fails to Run

- Re-check every setup step above.
- Ensure the **CUDA Toolkit 12.x** is installed and on `PATH` (Windows: the Visual Studio CUDA build integration
  must be present so the CUDA `.cu` files compile under the VS generator).
- Make sure the machine has an **NVIDIA GPU** and runs on the **Direct3D 12** or **Vulkan** backend; the shared
  external-memory interop is not available on other backends.
- On Windows with the Direct3D 12 backend, verify the D3D12 Agility SDK redistributable is present in `bin/D3D12/`.
- Ensure `UNIGINE_DOUBLE` matches the current build type (double/float) and the chosen `.project`.
- Use the `.project` file for your platform and SDK edition.
- Verify your SDK version is not older than the project's specified version.
- C++/CMake issues in Visual Studio: right-click the project → **Delete Cache and Reconfigure**, then rebuild.
