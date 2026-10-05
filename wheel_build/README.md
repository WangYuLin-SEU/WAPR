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

The wheel contains the Python runtime, `_gpu_render*.so`, shaders, detector resource catalog, fonts and license notices. Its `.dist-info/licenses/` directory contains `LICENSE`, `AUTHORS.md`, `THIRD_PARTY_NOTICES.txt`, `WEIGHTS_LICENSE.txt` and all files from `licenses/`. Model weights, TensorRT engines, samples, datasets, caches, user `paths.json`, reports, benchmark results, complete third-party trees, source-checkout tools and website files are excluded.

wheel 包含 Python 运行实现、`_gpu_render*.so`、shader、检测资源清单、字体与许可说明。`.dist-info/licenses/` 中收录 `LICENSE`、`AUTHORS.md`、`THIRD_PARTY_NOTICES.txt`、`WEIGHTS_LICENSE.txt` 及 `licenses/` 的全部文件。模型权重、TensorRT 引擎、小样、数据集、缓存、用户 `paths.json`、报告、测速产物、完整第三方源码、源码安装工具和网站文件不进入 wheel。

## Runtime / 运行

The package requires Python 3.10. CUDA/EGL/OpenGL and NVIDIA driver compatibility must be checked on the target device. Windows, other Python versions and general manylinux compatibility are not declared. Installed-package resources use the user cache; an absolute `WAPR_CACHE_DIR` can select another root. Weights and sample packs are downloaded separately, and TensorRT engines must be built for the target GPU. Installation itself does not run inference or fetch models. Follow the Docs for the complete detector setup.

包要求 Python 3.10。CUDA/EGL/OpenGL 与 NVIDIA 驱动的兼容性须在目标设备验证；不声明支持 Windows、其他 Python 版本或通用 manylinux 环境。安装版资源使用用户缓存，可用绝对路径 `WAPR_CACHE_DIR` 指定其他根目录。权重与小样另行下载，TensorRT 引擎需在目标 GPU 上构建。安装本身不运行推理或下载模型，完整检测环境按 Docs 准备。

## Release / 发行

The suggested first stable version is **0.1.0**, pending the maintainer's confirmation. `0.1.0.dev0` is a preparation version, not the final release version. Before publishing, set the confirmed stable version in `pyproject.toml` and use its matching `vX.Y.Z` release tag.

首个正式版本建议为 **0.1.0**，待作者确认。`0.1.0.dev0` 是预备版本，不是最终发行版本。发布前在 `pyproject.toml` 中填写已确认的正式版本，并使用一致的 `vX.Y.Z` Release 标签。

`.github/workflows/release.yml` runs only for a published, non-prerelease GitHub Release. Its `build-wheel` job uses a dedicated self-hosted runner labeled `wapr-cuda-12-8`; configure Linux, x64, Python 3.10, CUDA 12.8, build libraries, an NVIDIA GPU and working EGL/OpenGL on that runner. The job builds the wheel, installs it into an isolated virtual environment outside the checkout, and checks package/native imports and CUDA/EGL initialization without downloading weights or constructing the estimator. Only the verified wheel and its checksum are transferred to the publishing job.

`.github/workflows/release.yml` 仅由已发布、非预发行的 GitHub Release 触发。`build-wheel` 使用带 `wapr-cuda-12-8` 标签的专用 self-hosted runner；需配置 Linux、x64、Python 3.10、CUDA 12.8、开发库、NVIDIA GPU 及可用的 EGL/OpenGL。该 job 构建 wheel，在源码检出目录之外的独立虚拟环境安装，并检查包与本地模块导入及 CUDA/EGL 初始化；不下载权重，不构造估计器。仅将验证后的 wheel 与校验和传给发布 job。

`publish-pypi` runs on `ubuntu-latest`, checks the artifact checksum and stable version/tag, and uses the `pypi` environment with OIDC. It does not rebuild the wheel, use long-lived PyPI credentials or publish an sdist. Create `pypi` in **WAPR → Settings → Environments**, configure the required reviewer and restrict release tags to maintainers. Configure the Pending Trusted Publisher as project `wapr`, owner `WangYuLin-SEU`, repository `WAPR`, workflow `release.yml`, environment `pypi`.

`publish-pypi` 在 `ubuntu-latest` 上核对产物校验和、正式版本与标签，使用 `pypi` Environment 和 OIDC；不重新构建 wheel，不使用长期 PyPI 凭据，不发布 sdist。在 **WAPR → Settings → Environments** 创建 `pypi`，设置审核人，并限制正式标签由维护者创建。Pending Trusted Publisher 填写：项目 `wapr`，所有者 `WangYuLin-SEU`，仓库 `WAPR`，工作流 `release.yml`，环境 `pypi`。

The current native builder produces a `linux_x86_64` wheel. [PyPI platform validation](https://github.com/pypi/warehouse/blob/main/warehouse/utils/wheel.py) rejects that tag. The publishing job stops before upload for such a file. A PyPI-accepted Linux wheel requires a verified ABI and dependency policy, such as an audited manylinux build; changing its filename alone is insufficient. Keep the generated wheel for installation checks until that build is available.

当前本地编译流程生成 `linux_x86_64` wheel；[PyPI 平台检查](https://github.com/pypi/warehouse/blob/main/warehouse/utils/wheel.py) 不接受该标签，发布 job 会在上传前停止。PyPI 可接受的 Linux wheel 需要经过验证的 ABI 与依赖约束，例如通过审计的 manylinux 构建；仅修改文件名并不成立。在完成这种构建前，生成的 wheel 用于安装验证。
