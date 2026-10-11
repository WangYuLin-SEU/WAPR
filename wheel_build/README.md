# WAPR · Wide-Angle Pose Refinement

WAPR estimates and refines unseen-object 6D poses from RGB-D observations and a metric object mesh.

WAPR 根据 RGB-D 观测与米制物体网格，估计并修正未见物体的 6D 位姿。

```bash
pip install -U wapr
python -m wapr.bootstrap
```

Use Python 3.10 or newer on Linux or Windows with an NVIDIA GPU and CUDA-enabled PyTorch. Install Git, the CUDA toolkit and a C++ compiler (Visual Studio Build Tools on Windows) before setup. The setup command prepares 2D detection, segmentation and 6D pose estimation, including TensorRT 10 and OpenGL. Existing dependency replacements require confirmation.

Reconstruction and robotics are optional Linux features. Prepare SAM3D and RoMa for reconstruction; language-guided applications additionally use Qwen and UniPose9D. SAM3D uses an independent environment and downloads weights from ModelScope under the upstream model license. Follow the documentation to prepare the features you need. macOS pose inference is not supported.

请使用 Python 3.10 或更高版本，以及配有 NVIDIA GPU 和 CUDA 版 PyTorch 的 Linux 或 Windows 环境。准备前请安装 Git、CUDA toolkit 与 C++ 编译器（Windows 使用 Visual Studio Build Tools）。准备命令覆盖 2D 检测、分割与 6D 位姿估计，包括 TensorRT 10 和 OpenGL。更换已有依赖前需确认。

重建和机器人是按需准备的 Linux 扩展。重建使用 SAM3D 与 RoMa，语言引导的应用还需要 Qwen 和 UniPose9D。SAM3D 使用独立环境，默认从魔搭下载权重，使用须遵守上游模型许可。请按文档准备所需功能。macOS 不支持位姿推理。

Resources share `cache/weights/`, `cache/environments/` and dataset-specific `cache/samples/` under `WAPR_CACHE_DIR`; each example writes separately to `outputs/<number_task>/`. / 资源在 `WAPR_CACHE_DIR` 根下共用缓存目录，各示例输出独立存放。

See the [documentation](https://wangyulin-seu.github.io/WAPR/docs/) for installation, examples and input requirements, and the [project repository](https://github.com/WangYuLin-SEU/WAPR) for source code.

安装步骤、示例和输入要求见[文档](https://wangyulin-seu.github.io/WAPR/docs/?lang=zh)，源码见[项目仓库](https://github.com/WangYuLin-SEU/WAPR)。

Code: [LGPL-2.1](https://github.com/WangYuLin-SEU/WAPR/blob/main/LICENSE). Model weights: [weight license](https://github.com/WangYuLin-SEU/WAPR/blob/main/WEIGHTS_LICENSE.txt). See [third-party notices](https://github.com/WangYuLin-SEU/WAPR/blob/main/THIRD_PARTY_NOTICES.txt) for additional terms.

代码与模型权重分别遵循上述许可；第三方组件遵循其各自许可。
