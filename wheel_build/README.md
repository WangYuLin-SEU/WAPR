# WAPR · Wide-Angle Pose Refinement

WAPR estimates and refines unseen-object 6D poses from RGB-D observations and a metric object mesh.

WAPR 根据 RGB-D 观测与米制物体网格，估计并修正未见物体的 6D 位姿。

```bash
pip install -U wapr
python -m wapr.bootstrap
```

Use a compatible Windows or Linux NVIDIA GPU environment with CUDA-enabled PyTorch. OGL requires CUDA and C++ compilers (Visual Studio Build Tools on Windows). The setup command prepares TensorRT 10 and common features; robot and SAM3D remain explicit. SAM2.1 uses installed Ultralytics through `wapr.sam2.predict_mask`, while DINOv2 uses official Torch Hub and reuses its cache. No native SAM2 or separate DINOv2 checkout is installed. Detector setup selects compatible Transformers 4 and Hugging Face Hub versions. Existing dependency replacements require confirmation. SAM3D needs your own model access and may use an independent compatible environment. macOS pose inference is not supported.

请使用兼容的 Windows 或 Linux NVIDIA GPU 环境及支持 CUDA 的 PyTorch。OGL 需要 CUDA 和 C++ 编译器（Windows 使用 Visual Studio Build Tools）。准备命令覆盖 TensorRT 10 及常用功能，机器人与 SAM3D 仍需显式准备。SAM2.1 通过已安装的 Ultralytics 调用；DINOv2 使用官方 Torch Hub 并复用已有缓存。不安装原生 SAM2 或独立 DINOv2 检出。更换已有依赖前需确认。SAM3D 需要用户自己的模型权限，并可使用兼容独立环境。macOS 不支持位姿推理。

Resources share `cache/weights/`, `cache/environments/` and dataset-specific `cache/samples/` under `WAPR_CACHE_DIR`; each example writes separately to `outputs/<number_task>/`. / 资源在 `WAPR_CACHE_DIR` 根下共用缓存目录，各示例输出独立存放。

See the [documentation](https://wangyulin-seu.github.io/WAPR/docs/) for installation, examples and input requirements, and the [project repository](https://github.com/WangYuLin-SEU/WAPR) for source code.

安装步骤、示例和输入要求见[文档](https://wangyulin-seu.github.io/WAPR/docs/?lang=zh)，源码见[项目仓库](https://github.com/WangYuLin-SEU/WAPR)。

Code: [LGPL-2.1](https://github.com/WangYuLin-SEU/WAPR/blob/main/LICENSE). Model weights: [weight license](https://github.com/WangYuLin-SEU/WAPR/blob/main/WEIGHTS_LICENSE.txt). See [third-party notices](https://github.com/WangYuLin-SEU/WAPR/blob/main/THIRD_PARTY_NOTICES.txt) for additional terms.

代码与模型权重分别遵循上述许可；第三方组件遵循其各自许可。
