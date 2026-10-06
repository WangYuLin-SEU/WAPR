# WAPR wheel build / WAPR wheel 构建

WAPR refines unseen-object 6D poses from RGB-D observations and a metric mesh. This directory builds the installable `wapr` package and its CUDA/EGL/OpenGL renderer. Source code and usage examples are in [WAPR](https://github.com/WangYuLin-SEU/WAPR); installation and input conventions are in the [Docs](https://wangyulin-seu.github.io/WAPR/docs/).

WAPR 根据 RGB-D 观测与米制网格修正未见物体的 6D 位姿。本目录构建可安装的 `wapr` 包与 CUDA/EGL/OpenGL 渲染器。源码和示例见 [WAPR](https://github.com/WangYuLin-SEU/WAPR)，安装步骤和输入约定见 [Docs](https://wangyulin-seu.github.io/WAPR/docs/?lang=zh)。

## Build / 构建

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

After preparing PyTorch and OpenCV, install Kornia and the local wheel:

准备好 PyTorch 和 OpenCV 后，安装 Kornia 与本地 wheel：

```bash
python -m pip install kornia
python -m pip install wheel_build/dist/wapr-0.1.0.dev0-cp310-cp310-linux_x86_64.whl
```

For a new environment, the optional `pose` extra installs torch, Kornia and `opencv-python`; select a suitable CUDA-enabled torch build first. The `trt` extra selects CUDA 12 TensorRT 10.x; the `export` extra adds ONNX for engine export. These extras are opt-in, install dependencies as separate packages, and do not include third-party detector source trees. If GPU packages are already managed externally, install the base wheel and only add the missing libraries yourself. No `--upgrade` is needed.

新环境可选 `pose` 附加依赖，安装 torch、Kornia 与 `opencv-python`；应先选择适合设备的带 CUDA 的 torch。`trt` 附加依赖选择 CUDA 12 的 TensorRT 10.x；`export` 附加依赖补充导出引擎所需的 ONNX。这些附加依赖须显式选择，各自作为独立包安装，不包含第三方检测器源码。GPU 软件栈已由其他方式管理时，安装基础 wheel 并自行补齐缺失库即可，不需要使用 `--upgrade`。

The current wheel has a CPython 3.10 ABI, so it requires Python 3.10. The CUDA 12.8 SDK above belongs to the controlled release build; installing a precompiled wheel does not require `nvcc` or that exact toolkit installation. Its CUDA runtime libraries, GPU driver and EGL/OpenGL must still be compatible. Source setup reports the selected compiler instead of rejecting other toolkit versions, but removing a gate does not verify a new environment. Other Python versions, Windows and general manylinux compatibility are not declared. Installed-package resources use the user cache; an absolute `WAPR_CACHE_DIR` can select another root. Weights and sample packs are downloaded separately, and TensorRT engines must be built for the target GPU and TensorRT version. Installation itself does not run inference or fetch models. Follow the Docs for the complete detector setup.

当前 wheel 使用 CPython 3.10 ABI，因此需要 Python 3.10。上方 CUDA 12.8 SDK 属于受控的正式构建环境；安装预编译 wheel 不需要 `nvcc`，也不要求安装同一精确版本的 toolkit，但所需 CUDA 运行库、驱动和 EGL/OpenGL 仍须兼容。源码安装会报告所选编译器，不再拒绝其他 toolkit 版本；去掉检查不代表已验证新环境。不声明支持其他 Python 版本、Windows 或通用 manylinux 环境。安装版资源使用用户缓存，可用绝对路径 `WAPR_CACHE_DIR` 指定其他根目录。权重与小样另行下载，TensorRT 引擎需在目标 GPU 和 TensorRT 版本上构建。安装本身不运行推理或下载模型，完整检测环境按 Docs 准备。

For source setup, `requirements.txt` contains pose/export/build dependencies and `requirements-detector.txt` contains additional 2D dependencies. The installer checks user-selected PyTorch, TensorRT and OpenCV first, and checks torchvision when 2D detection is enabled. It constrains pip to the installed torch/torchvision versions; a conflict fails instead of silently replacing them. Other packages have no exact pins. Source revisions and actual native ABI constraints still apply; a wider dependency declaration is not a claim that every release has been tested.

源码环境中，`requirements.txt` 列出位姿、导出与编译依赖，`requirements-detector.txt` 列出额外的 2D 依赖。安装脚本先检查用户选择的 PyTorch、TensorRT 和 OpenCV；启用 2D 检测时还检查 torchvision。pip 被约束为保留已安装的 torch/torchvision 版本，冲突时会报错，而不会静默替换。普通依赖不固定精确版本；第三方源码版本和实际本地模块 ABI 约束仍须满足，放宽依赖声明不代表已测试所有发行版本。

## Release / 发行

The suggested first stable version is **0.1.0**, pending the maintainer's confirmation. `0.1.0.dev0` is a preparation version, not the final release version. Before publishing, set the confirmed stable version in `pyproject.toml` and use its matching `vX.Y.Z` release tag.

首个正式版本建议为 **0.1.0**，待作者确认。`0.1.0.dev0` 是预备版本，不是最终发行版本。发布前在 `pyproject.toml` 中填写已确认的正式版本，并使用一致的 `vX.Y.Z` Release 标签。

`.github/workflows/release.yml` runs only for a published, non-prerelease GitHub Release. Its `build-wheel` job uses a dedicated self-hosted runner labeled `wapr-cuda-12-8`; configure Linux, x64, Python 3.10, CUDA 12.8, build libraries, an NVIDIA GPU and working EGL/OpenGL on that runner. The job builds the wheel, installs it into an isolated virtual environment outside the checkout, and checks package/native imports and CUDA/EGL initialization without downloading weights or constructing the estimator. Only the verified wheel and its checksum are transferred to the publishing job.

`.github/workflows/release.yml` 仅由已发布、非预发行的 GitHub Release 触发。`build-wheel` 使用带 `wapr-cuda-12-8` 标签的专用 self-hosted runner；需配置 Linux、x64、Python 3.10、CUDA 12.8、开发库、NVIDIA GPU 及可用的 EGL/OpenGL。该 job 构建 wheel，在源码检出目录之外的独立虚拟环境安装，并检查包与本地模块导入及 CUDA/EGL 初始化；不下载权重，不构造估计器。仅将验证后的 wheel 与校验和传给发布 job。

`publish-pypi` runs on `ubuntu-latest`, checks the artifact checksum and stable version/tag, and uses the `pypi` environment with OIDC. It does not rebuild the wheel, use long-lived PyPI credentials or publish an sdist. Create `pypi` in **WAPR → Settings → Environments**, configure the required reviewer and restrict release tags to maintainers. Configure the Pending Trusted Publisher as project `wapr`, owner `WangYuLin-SEU`, repository `WAPR`, workflow `release.yml`, environment `pypi`.

`publish-pypi` 在 `ubuntu-latest` 上核对产物校验和、正式版本与标签，使用 `pypi` Environment 和 OIDC；不重新构建 wheel，不使用长期 PyPI 凭据，不发布 sdist。在 **WAPR → Settings → Environments** 创建 `pypi`，设置审核人，并限制正式标签由维护者创建。Pending Trusted Publisher 填写：项目 `wapr`，所有者 `WangYuLin-SEU`，仓库 `WAPR`，工作流 `release.yml`，环境 `pypi`。

The current native builder produces a `linux_x86_64` wheel. [PyPI platform validation](https://github.com/pypi/warehouse/blob/main/warehouse/utils/wheel.py) rejects that tag. The publishing job stops before upload for such a file. A PyPI-accepted Linux wheel requires a verified ABI and dependency policy, such as an audited manylinux build; changing its filename alone is insufficient. Keep the generated wheel for installation checks until that build is available.

当前本地编译流程生成 `linux_x86_64` wheel；[PyPI 平台检查](https://github.com/pypi/warehouse/blob/main/warehouse/utils/wheel.py) 不接受该标签，发布 job 会在上传前停止。PyPI 可接受的 Linux wheel 需要经过验证的 ABI 与依赖约束，例如通过审计的 manylinux 构建；仅修改文件名并不成立。在完成这种构建前，生成的 wheel 用于安装验证。

### Model attribution / 模型署名

The four pose weights carry `wapr.metadata` alongside `model_state_dict`. WAPR ONNX exports retain this attribution in model metadata. Exported `.engine` files are WAPR containers holding the original TensorRT engine and attribution; WAPR loads these containers and native engines. For TensorRT tools that require native engine files, extract the payload:

四份位姿权重在 `model_state_dict` 同级保存 `wapr.metadata`。WAPR 导出 ONNX 时保留该署名；导出的 `.engine` 文件封装原生 TensorRT 引擎及署名信息，WAPR 可以读取封装及原生引擎。需要原生引擎的 TensorRT 工具可使用以下方法提取：

```python
from pathlib import Path
from wapr.model_metadata import unpack_engine

native, metadata = unpack_engine(Path("sapr.engine").read_bytes())
Path("sapr.native.engine").write_bytes(native)
```
