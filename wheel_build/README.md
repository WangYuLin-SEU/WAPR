# WAPR wheel build / WAPR wheel 构建

WAPR refines unseen-object 6D poses from RGB-D observations and a metric mesh. This directory builds the installable `wapr` package and its CUDA/EGL/OpenGL renderer. Source code and usage examples are in [WAPR](https://github.com/WangYuLin-SEU/WAPR); installation and input conventions are in the [Docs](https://wangyulin-seu.github.io/WAPR/docs/).

WAPR 根据 RGB-D 观测与米制网格修正未见物体的 6D 位姿。本目录构建可安装的 `wapr` 包与 CUDA/EGL/OpenGL 渲染器。源码和示例见 [WAPR](https://github.com/WangYuLin-SEU/WAPR)，安装步骤和输入约定见 [Docs](https://wangyulin-seu.github.io/WAPR/docs/?lang=zh)。

## Source wheel / 源码 wheel

For an existing Linux Python/PyTorch/CUDA environment, build a source-only wheel with the same build script. It includes WAPR's native sources and compiles the renderer on the target machine. It contains no compiled `.so`, PyTorch, model weights or TensorRT engines. The packaging host needs setuptools 77+ and wheel 0.45+; it does not need CUDA for this build.

面向已有 Linux Python/PyTorch/CUDA 环境，用同一构建脚本生成仅含源码的 wheel。它包含 WAPR 的本地模块源码，在目标机器编译渲染器，不包含预编译 `.so`、PyTorch、权重或 TensorRT 引擎。打包机器需 setuptools 77+ 与 wheel 0.45+；这一构建不需要 CUDA。

```bash
WAPR_WHEEL_SOURCE_ONLY=1 python wheel_build/build_wheel.py
python -m pip install --no-deps wheel_build/dist/wapr-0.0.3-py3-none-any.whl
python -m wapr.bootstrap
```

Run these installation commands with the target's existing Python. The package entry prepares missing dependencies and preserves the existing torch, torchvision, torchaudio and NumPy versions unless a displayed replacement plan is explicitly approved. It selects TensorRT 10.x for the existing CUDA 11, 12 or 13 family and ONNX for export, and installs missing EGL/OpenGL development prerequisites on root-owned Debian/Ubuntu systems. Other systems receive an explicit prerequisite error. CUDA `nvcc` must already be available. A toolkit too old for the GPU's SM uses a supported PTX target; actual inference must still verify that configuration.

安装时使用目标机器已有的 Python。包内入口补齐缺失依赖，默认保留已有 torch、torchvision、torchaudio 和 NumPy 版本；需要更换时先显示方案并获得明确同意；按 CUDA 11、12、13 家族选择 TensorRT 10.x，安装 ONNX，并在 root 用户的 Debian/Ubuntu 系统补齐缺少的 EGL/OpenGL 开发依赖。其他系统会明确提示前置依赖。机器须已有 CUDA `nvcc`。若 toolkit 不能编译该 GPU 的 SM，则选择支持的 PTX 目标；仍需实际推理验证该组合。

Weights and samples are fetched by `wapr.download_assets`; `wapr.bootstrap.fetch_example()` fetches a public GitHub usage example. The package does not change the inference backend: select `wapr.recipe.backend = "torch"` explicitly for PyTorch, or keep `"trt"` and export engines on the target GPU with `python -m wapr.export_engines`.

权重和样本由 `wapr.download_assets` 下载；`wapr.bootstrap.fetch_example()` 下载 GitHub 公开用法示例。包不改变推理后端：PyTorch 推理显式设置 `wapr.recipe.backend = "torch"`；或保留 `"trt"`，在目标显卡执行 `python -m wapr.export_engines` 导出引擎。

The wheel also contains reviewed first-party example recipes. Export them into a new writable directory with `wapr.bootstrap.export_examples(destination)`; existing user files are preserved. Examples 11 and 12 share reconstructed meshes under the resource cache, rather than writing into the installed package. Set an absolute `WAPR_CACHE_DIR` when large model downloads should use a data volume.

wheel 同时包含已审核的第一方示例配方。用 `wapr.bootstrap.export_examples(destination)` 导出到新的可写目录；已有用户文件会保留。示例 11、12 在资源缓存中共用重建网格，不向安装包目录写结果。大模型需要放在数据盘时，可设置绝对路径 `WAPR_CACHE_DIR`。

Optional SAM preparation keeps the base Torch stack and uses an independent compatible interpreter when required. `WAPR_SAM3D_ENV` can select an existing prefix. Gated SAM3D checkpoints require the user's own Hugging Face access and token, or locally supplied files; no maintainer credentials or private download service is included. Automatic download routes respect user-selected indexes and mirrors, measure available routes, and retain file checksums. Their speed depends on the target network.

可选 SAM 准备保留基础 Torch，必要时使用兼容的独立解释器。`WAPR_SAM3D_ENV` 可选用已有环境前缀。受控 SAM3D 权重需要用户自己的 Hugging Face 权限和 token，或用户已下载的本地文件；包内没有维护者凭据或私有下载服务。自动下载尊重用户指定的索引和镜像，对可用路线测速并继续核验文件；速度取决于目标网络。

## Tested existing environments / 已实测的已有环境

These Linux x86-64 environments retain their original base Torch and CUDA builds. Core pose inference, optional detection, and robot examples were run on an Ada GPU (SM 8.9). The table records tested combinations, not every version between them, and does not establish Windows/macOS or other GPU support. Successful tracking execution does not guarantee target recovery on every frame.

这些 Linux x86-64 环境保留原有基础 Torch 与 CUDA 构建。已在 Ada 显卡（SM 8.9）实际运行核心位姿推理、可选检测与机器人示例。表格记录实测组合，不代表中间所有版本或 Windows/macOS、其他显卡均支持；跟踪流程运行完成也不保证每帧恢复目标。

| Python | Existing PyTorch / 已有 PyTorch | Core pose / 核心位姿 | Detection / 检测 | Robot examples / 机器人示例 |
|---|---|---|---|---|
| 3.12.3 | 2.8.0+cu128 | Passed / 通过 | Passed / 通过 | Passed / 通过 |
| 3.12.3 | 2.12.1+cu130 | Passed / 通过 | Passed / 通过 | Passed / 通过 |
| 3.12.3 | 2.5.1+cu124 | Passed / 通过 | Passed / 通过 | Passed / 通过 |
| 3.12.3 | 2.3.0+cu121 | Passed / 通过 | Passed / 通过 | Passed / 通过 |
| 3.10.8 | 2.1.2+cu118 | Passed / 通过 | Passed / 通过 | Passed / 通过 |
| 3.10.8 | 2.1.2+cu121 | Passed / 通过 | Passed / 通过 | Passed / 通过 |

Full reconstruction and cross-scene inference also passed in all six combinations: original SAM3D generation, 100 views, 1024-pixel texture baking with 2500 optimization steps, UniPose9D, RoMa matching and WAPR scale/pose refinement. The following independent Python 3.11 / PyTorch 2.5.1 environments preserve the base installation:

六组环境也均通过完整重建及跨场景推理：原始 SAM3D 生成、100 视图、1024 像素纹理与 2500 步优化、UniPose9D、RoMa 匹配及 WAPR 尺度与位姿修正。以下独立 Python 3.11 / PyTorch 2.5.1 环境保留基础安装：

| Base Torch CUDA / 基础 Torch CUDA | Independent Torch CUDA / 独立 Torch CUDA | Full reconstruction and cross-scene pose / 完整重建与跨场景位姿 |
|---|---|---|
| cu128, cu130, cu124 | cu124 | Passed / 通过 |
| cu121 | cu121 | Passed / 通过 |
| cu118 | cu118 | Passed / 通过 |

The tested GPUs reported approximately 31.47 GiB of CUDA memory. This does not establish the full reconstruction budget on 16 GiB devices. Robot results cover simulation and planning, not physical robot hardware. Native wheels are reusable only for matching Python ABI, Torch/CUDA and GPU compilation targets. Gated checkpoints were validated using user-supplied local files; each user must obtain their own access. Tutorial sample redistribution is separate from runtime compatibility.

实测 GPU 报告约 31.47 GiB CUDA 显存，不代表 16 GiB 设备能运行同样完整重建预算。机器人验证覆盖仿真与规划，不包含实体硬件。原生 wheel 仅在 Python ABI、Torch/CUDA 与显卡编译目标匹配时复用。受控权重通过用户提供的本地文件验证；每位用户需自行取得访问权限。教程小样的再分发与运行兼容性是两项独立条件。

## Precompiled wheel / 预编译 wheel

Use Linux x86-64, Python 3.10, CUDA toolkit 12.8 with `nvcc`, CMake 3.18+, a C++17 compiler, pybind11 2.12.0, and EGL/OpenGL development libraries. Install the packaging tools, then run the existing build script from the repository root:

使用 Linux x86-64、Python 3.10、带 `nvcc` 的 CUDA toolkit 12.8、CMake 3.18+、C++17 编译器、pybind11 2.12.0，以及 EGL/OpenGL 开发库。安装打包工具后，从仓库根目录运行现有构建脚本：

```bash
python -m pip install 'setuptools>=77' 'wheel>=0.45' 'pybind11==2.12.0'
python wheel_build/build_wheel.py
```

Output is written to `wheel_build/dist/`. The script builds only a wheel, checks its runtime files and licenses, and prints SHA256. It never uploads a package. The CUDA architecture list is an explicit compiler setting, not a claim that every listed GPU has been tested.

产物写入 `wheel_build/dist/`。脚本只构建 wheel，检查运行文件与许可，并打印 SHA256；它不会上传包。CUDA 架构列表是显式编译配置，不代表已验证其中所有 GPU。

The wheel contains the Python runtime, `_gpu_render*.so`, shaders, detector resource catalog, fonts and license notices. Its `.dist-info/licenses/` directory contains `LICENSE`, `AUTHORS.md`, `THIRD_PARTY_NOTICES.txt`, `WEIGHTS_LICENSE.txt` and notices for the bundled fonts. Model weights, TensorRT engines, samples, datasets, caches, user `paths.json`, reports, benchmark results, complete third-party trees, source-checkout tools and website files are excluded.

wheel 包含 Python 运行实现、`_gpu_render*.so`、shader、检测资源清单、字体与许可说明。`.dist-info/licenses/` 中收录 `LICENSE`、`AUTHORS.md`、`THIRD_PARTY_NOTICES.txt`、`WEIGHTS_LICENSE.txt` 及随包字体的许可说明。模型权重、TensorRT 引擎、小样、数据集、缓存、用户 `paths.json`、报告、测速产物、完整第三方源码、源码安装工具和网站文件不进入 wheel。

## Runtime / 运行

The wheel contains WAPR, not a bundled Python environment. Basic installation adds NumPy, Pillow, trimesh and Hugging Face Hub without exact version pins. Prepare CUDA-enabled PyTorch for your GPU first; prepare TensorRT 10.x when using the TensorRT backend. These GPU packages are not selected by basic installation. Pose inference also needs Kornia and one OpenCV distribution. Keep a working OpenCV installation; do not install several distributions that all provide `cv2`.

wheel 只交付 WAPR，不打包完整 Python 环境。基础安装补齐 NumPy、Pillow、trimesh 和 Hugging Face Hub，不固定其精确版本。请先按 GPU 环境准备带 CUDA 的 PyTorch；使用 TensorRT 后端时再准备 TensorRT 10.x。基础安装不会选择这些 GPU 包。位姿推理还需 Kornia 和一种 OpenCV 发行包；已有可用的 OpenCV 时继续使用，不要同时安装多种提供 `cv2` 的发行包。

For a precompiled wheel, keep the existing GPU environment and prepare missing libraries through the package:

预编译 wheel 也保留已有 GPU 环境，通过包内入口准备缺失库：

```bash
python -m pip install --no-deps wheel_build/dist/wapr-0.0.3-cp310-cp310-linux_x86_64.whl
python -m wapr.bootstrap
```

### Optional features / 可选功能

The minimal runtime prepares WAPR, SAPR, WBPS and OGL. Detection, reconstruction and robot dependencies are separate. Installing the wheel does not install those stacks or fetch their models. Calling a supported optional entry for the first time invokes the package's dependency preparation; importing a module alone does not. Incompatible features report unmet requirements. A proposed package replacement shows the installed and proposed versions and requires explicit user approval before installation; declining stops preparation of that feature.

最小运行环境准备 WAPR、SAPR、WBPS 与 OGL。检测、重建和机器人依赖单独准备；安装 wheel 不安装这些完整环境，也不下载其模型。首次调用受支持的可选入口时，由包内依赖准备流程处理；仅导入模块不会触发安装。不兼容功能会明确报告缺少的条件。需要更换已有库时，先列出已安装版本与建议版本，得到用户明确同意后才安装；拒绝则停止该功能的准备。

Inspect the current environment before choosing features, or prepare one explicitly:

先检查当前环境，再选择功能；也可显式准备某项功能：

```bash
python -m wapr.bootstrap --check
python -m wapr.bootstrap --feature det2d
python -m wapr.bootstrap --feature robot
python -m wapr.bootstrap --feature compatible
```

`compatible` considers features against the existing environment and prepares those whose prerequisites can be satisfied. Its report distinguishes installed/importable packages, satisfied declared requirements, skipped features and runtime verification. Passing a version check alone does not prove model inference works. Package replacements require a user decision after the old and proposed versions are shown. Interactive sessions ask for approval; noninteractive sessions deny replacements by default. `--yes` is explicit permission for the displayed replacements, not a claim that every feature is compatible.

`compatible` 根据已有环境考虑各项功能，准备满足前置条件的部分。报告区分已安装且可导入的包、满足声明条件的功能、跳过的功能以及运行验证结果。仅通过版本检查不代表模型推理已验证。更换已有库前先显示旧版本与建议版本，由用户决定。交互终端会询问；非交互环境默认拒绝更换。`--yes` 是对所列更换的明确许可，不代表全部功能都兼容。

Standard pip extras select only lightweight helpers used for HTTP transfer, metadata/version inspection and source configuration. They contain no torch dependency or fixed CUDA-family TensorRT selection:

标准 pip extras 仅选择 HTTP 传输、元数据与版本检查、源码配置所需的轻量辅助包，不声明 torch 依赖，也不固定 TensorRT 的 CUDA 家族：

| Extra | Scope / 范围 | Helpers / 辅助包 |
| --- | --- | --- |
| `pose`, `trt` | Environment selection / 环境选择 | packaging |
| `export` | ONNX export / ONNX 导出 | onnx |
| `det2d` | 2D detection / 2D 检测 | packaging, requests, PyYAML |
| `dinov2` | DINOv2 features / DINOv2 特征 | packaging, requests |
| `sam2` | SAM 2 segmentation / SAM 2 分割 | packaging, PyYAML |
| `sam3d` | SAM 3D reconstruction / SAM 3D 重建 | packaging, requests, PyYAML |
| `roma`, `qwen` | Matching or vision-language helpers / 匹配或视觉语言辅助 | packaging, requests |
| `reconstruction` | Reconstruction prerequisites / 重建前置依赖 | packaging, requests, PyYAML |
| `robot` | Robot configuration / 机器人配置 | packaging, PyYAML |
| `compatible` | Available-feature preparation / 可用功能准备 | packaging, requests, PyYAML |

For example, in an environment whose base dependencies are already prepared:

例如，在基础依赖已准备好的环境中：

```bash
python -m pip install 'wheel_build/dist/wapr-0.0.3-py3-none-any.whl[det2d]'
python -m wapr.bootstrap --feature det2d
```

Pip accepts `wapr[det2d]` syntax, not arbitrary installation arguments that execute WAPR code. Installing an extra does **not** run bootstrap, compile a third-party component or select all compatible features. Use the explicit package command or the first-call preparation for that work. With an existing GPU stack, the `--no-deps` minimal installation above plus bootstrap is the controlled route: bootstrap protects installed torch, torchvision, torchaudio and NumPy while resolving missing dependencies, and asks before applying a displayed replacement plan. No optional feature is claimed compatible with every PyTorch/CUDA version, Windows or macOS.

pip 支持 `wapr[det2d]` 语法，不支持通过任意安装参数执行 WAPR 代码。安装 extra **不会**执行 bootstrap、编译第三方组件或自动选择全部兼容功能；这些工作由显式包内命令或首次调用准备流程执行。已有 GPU 软件栈时，推荐上方 `--no-deps` 最小安装加 bootstrap：解析缺失依赖时保护已有 torch、torchvision、torchaudio 和 NumPy；如需更换，先显示方案并征求同意。不声明所有可选功能兼容全部 PyTorch/CUDA 版本、Windows 或 macOS。

The precompiled wheel has a CPython 3.10 ABI, so it requires Python 3.10. The source wheel contains no native binary and declares Python 3.8 or later; its renderer and dependencies still need target-environment preparation. The CUDA 12.8 SDK above belongs to the controlled precompiled build; installing a precompiled wheel does not require `nvcc` or that exact toolkit installation. Its CUDA runtime libraries, GPU driver and EGL/OpenGL must still be compatible. Source setup reports the selected compiler instead of rejecting other toolkit versions, but removing a gate does not verify a new environment. The precompiled wheel does not declare other Python ABIs; Windows and general manylinux compatibility are not declared for either variant. Installed-package resources use the user cache; an absolute `WAPR_CACHE_DIR` can select another root. Weights and sample packs are downloaded separately, and TensorRT engines must be built for the target GPU and TensorRT version. Installation itself does not run inference or fetch models. Follow the Docs for the complete detector setup.

预编译 wheel 使用 CPython 3.10 ABI，因此需要 Python 3.10。源码 wheel 不包含本地二进制，声明 Python 3.8 及以上；渲染器与依赖仍需在目标环境准备。上方 CUDA 12.8 SDK 属于受控的预编译构建环境；安装预编译 wheel 不需要 `nvcc`，也不要求安装同一精确版本的 toolkit，但所需 CUDA 运行库、驱动和 EGL/OpenGL 仍须兼容。源码安装会报告所选编译器，不再拒绝其他 toolkit 版本；去掉检查不代表已验证新环境。预编译 wheel 不声明其他 Python ABI；两种构建均不声明支持 Windows 或通用 manylinux 环境。安装版资源使用用户缓存，可用绝对路径 `WAPR_CACHE_DIR` 指定其他根目录。权重与小样另行下载，TensorRT 引擎需在目标 GPU 和 TensorRT 版本上构建。安装本身不运行推理或下载模型，完整检测环境按 Docs 准备。

For source setup, `requirements.txt` contains pose/export/build dependencies and `requirements-detector.txt` contains additional 2D dependencies. The installer checks user-selected PyTorch, TensorRT and OpenCV first, and checks torchvision when 2D detection is enabled. It constrains pip to the installed torch/torchvision versions; a conflict fails instead of silently replacing them. Other packages have no exact pins. Source revisions and actual native ABI constraints still apply; a wider dependency declaration is not a claim that every release has been tested.

源码环境中，`requirements.txt` 列出位姿、导出与编译依赖，`requirements-detector.txt` 列出额外的 2D 依赖。安装脚本先检查用户选择的 PyTorch、TensorRT 和 OpenCV；启用 2D 检测时还检查 torchvision。pip 被约束为保留已安装的 torch/torchvision 版本，冲突时会报错，而不会静默替换。普通依赖不固定精确版本；第三方源码版本和实际本地模块 ABI 约束仍须满足，放宽依赖声明不代表已测试所有发行版本。

## Release / 发行

The current release version is **0.0.3**. Optional feature compatibility depends on the target environment; installation or import alone does not establish inference compatibility. The matching release tag is `v0.0.3`.

当前发行版本为 **0.0.3**。可选功能的兼容性取决于目标环境；安装或导入成功不代表实际推理已验证。对应发行标签为 `v0.0.3`。

`.github/workflows/release.yml` runs for a published, non-prerelease GitHub Release, or a maintainer retry from main that verifies the same published Release tag. The `build-wheel` job uses a controlled self-hosted Linux CUDA runner labeled `wapr-cuda-12-8`. It executes the existing builder with `WAPR_WHEEL_SOURCE_ONLY=1`, producing a `py3-none-any` wheel containing WAPR Python code and renderer sources. PyTorch, model weights, engines and complete third-party repositories are not bundled. The job checks metadata, licenses, checksum and README rendering, then installs the wheel outside the source checkout in a separate environment that reuses the controlled runner's existing GPU dependencies, and verifies CUDA, TensorRT and the target-built renderer. This checks installation into an existing GPU environment; it does not claim a clean dependency installation. Only the verified wheel and checksum reach the publishing job.

`.github/workflows/release.yml` 由已发布、非预发行的 GitHub Release 触发，或由维护者从 main 重试同一个经核验的正式 Release 标签。`build-wheel` 使用带 `wapr-cuda-12-8` 标签的受控 self-hosted Linux CUDA runner，设置 `WAPR_WHEEL_SOURCE_ONLY=1` 执行现有构建脚本，生成包含 WAPR Python 代码与渲染器源码的 `py3-none-any` wheel，不打包 PyTorch、模型权重、引擎或完整第三方仓库。该 job 检查元数据、许可、校验和与 README 渲染，并在源码检出目录之外的独立环境中安装 wheel，复用受控 runner 已有的 GPU 依赖，验证 CUDA、TensorRT 与目标环境编译的渲染器。这验证已有 GPU 环境中的安装，不声称完成从零依赖安装。仅将已验证的 wheel 与校验和交给发布 job。

The source wheel's `Requires-Python >=3.8` and `py3-none-any` tag describe package installation and the absence of a bundled native ABI. They do not establish Windows/macOS, every Python release or every GPU/CUDA combination as supported. The runtime still needs a compatible Linux/CUDA/EGL/OpenGL environment. The separate precompiled build remains available for controlled local use; its `linux_x86_64` wheel is not the artifact selected for this PyPI workflow.

源码 wheel 的 `Requires-Python >=3.8` 与 `py3-none-any` 标签描述包的安装条件及不含预编译本地 ABI，不代表已验证 Windows/macOS、所有 Python 版本或所有 GPU/CUDA 组合。运行仍需要兼容的 Linux/CUDA/EGL/OpenGL 环境。预编译构建保留用于受控本地安装；其 `linux_x86_64` wheel 不是此 PyPI workflow 选择的产物。

Before creating a formal GitHub Release:

创建正式 GitHub Release 前：

1. Build version **0.0.3** and use its matching `v0.0.3` tag.
2. Configure the controlled CUDA runner with Python 3.10, CUDA development tools, EGL/OpenGL and an NVIDIA GPU for the package checks.
3. Create **WAPR → Settings → Environments → pypi**, configure a required reviewer and protect release-tag creation.
4. Configure PyPI Pending Trusted Publisher with the following values. The publishing job uses OIDC with `id-token: write`; no token, password or sdist is required.

1. 构建 **0.0.3** 版本，使用一致的 `v0.0.3` 标签。
2. 配置受控 CUDA runner，准备 Python 3.10、CUDA 开发工具、EGL/OpenGL 和用于包校验的 NVIDIA GPU。
3. 在 **WAPR → Settings → Environments → pypi** 创建环境，设置审核人并限制正式标签创建权限。
4. 按下表配置 PyPI Pending Trusted Publisher。发布 job 使用 OIDC 与 `id-token: write`，不需要长期 token、密码或 sdist。

| Field / 字段 | Value / 值 |
| --- | --- |
| PyPI project / 项目 | `wapr` |
| Owner / 所有者 | `WangYuLin-SEU` |
| Repository / 仓库 | `WAPR` |
| Workflow / 工作流 | `release.yml` |
| Environment / 环境 | `pypi` |

Local wheel building does not publish anything. A formal Release is the publishing trigger; prepare and review the version, runner, environment and publisher configuration before creating one.

本地构建 wheel 不发布任何内容。正式 Release 是发布触发条件；创建前应先准备并审核版本、runner、Environment 与 publisher 配置。

### Model attribution / 模型署名

The four pose weights carry `wapr.metadata` alongside `model_state_dict`. WAPR ONNX exports retain this attribution in model metadata. Exported `.engine` files are WAPR containers holding the original TensorRT engine and attribution; WAPR loads these containers and native engines. For TensorRT tools that require native engine files, extract the payload:

四份位姿权重在 `model_state_dict` 同级保存 `wapr.metadata`。WAPR 导出 ONNX 时保留该署名；导出的 `.engine` 文件封装原生 TensorRT 引擎及署名信息，WAPR 可以读取封装及原生引擎。需要原生引擎的 TensorRT 工具可使用以下方法提取：

```python
from pathlib import Path
from wapr.model_metadata import unpack_engine

native, metadata = unpack_engine(Path("sapr.engine").read_bytes())
Path("sapr.native.engine").write_bytes(native)
```
