# WAPR · Wide-Angle Pose Refinement

WAPR estimates and refines unseen-object 6D poses from RGB-D observations and a metric object mesh.

WAPR 根据 RGB-D 观测与米制物体网格，估计并修正未见物体的 6D 位姿。

```bash
pip install -U wapr==0.0.5
python -m wapr.bootstrap
```

Use a compatible Windows or Linux NVIDIA GPU environment with CUDA-enabled PyTorch. OGL requires a CUDA compiler and a C++ compiler (Visual Studio Build Tools on Windows). Windows without TensorRT uses PyTorch by default; an explicit TensorRT selection requires TensorRT 10. Optional detection dependencies are prepared before first use, rather than when importing examples. Ultralytics is not a core installation dependency. Optional features may require additional dependencies and model access. SAM3D and the verified robot planning stack require Linux; SAM3D weights require your own Hugging Face access or locally supplied files. macOS pose inference is not supported.

请使用兼容的 Windows 或 Linux NVIDIA GPU 环境及支持 CUDA 的 PyTorch。OGL 需要 CUDA 编译器和 C++ 编译器（Windows 使用 Visual Studio Build Tools）。Windows 未安装 TensorRT 时默认使用 PyTorch；显式选择 TensorRT 时需要 TensorRT 10。可选检测依赖在首次使用前准备，仅导入示例不会启动安装；Ultralytics 不属于基础安装依赖。可选功能可能需要额外依赖和模型访问权限。SAM3D 及已验证的机器人规划环境需要 Linux；SAM3D 权重需使用自己的 Hugging Face 访问权限或本地文件。不支持 macOS 位姿推理。

See the [documentation](https://wangyulin-seu.github.io/WAPR/docs/) for installation, examples and input requirements, and the [project repository](https://github.com/WangYuLin-SEU/WAPR) for source code.

安装步骤、示例和输入要求见[文档](https://wangyulin-seu.github.io/WAPR/docs/?lang=zh)，源码见[项目仓库](https://github.com/WangYuLin-SEU/WAPR)。

Code: [LGPL-2.1](https://github.com/WangYuLin-SEU/WAPR/blob/main/LICENSE). Model weights: [weight license](https://github.com/WangYuLin-SEU/WAPR/blob/main/WEIGHTS_LICENSE.txt). See [third-party notices](https://github.com/WangYuLin-SEU/WAPR/blob/main/THIRD_PARTY_NOTICES.txt) for additional terms.

代码与模型权重分别遵循上述许可；第三方组件遵循其各自许可。
