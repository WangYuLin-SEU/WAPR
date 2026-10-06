# Author: Yulin Wang (yulinwang@seu.edu.cn)
# SPDX-License-Identifier: LGPL-2.1-only
"""Prepare SAM3D inference without its training environment.

只准备 SAM3D 推理，不安装上游训练环境；凭据仅来自使用者本机。
"""
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path


def reconstruction_weights_status(checkpoint_directory=None):
    """Check local files before authentication or network access.

    先检查本地文件；此检查不读取 token，也不访问网络。
    """
    from wapr.resources import weights_dir
    directory = Path(checkpoint_directory or os.path.join(weights_dir(), "sam3d", "checkpoints"))
    pipeline = directory / "pipeline.yaml"
    required = ["pipeline.yaml"]
    if pipeline.is_file():
        # Published pipeline path fields name adjacent checkpoints/configs.
        # 已发布 pipeline 的 path 字段引用同目录的权重与配置。
        for line in pipeline.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^[A-Za-z0-9_]+_path:\s*([^\s#]+)", line)
            if match:
                name = match.group(1).strip("\"'")
                if Path(name).name != name:
                    raise ValueError("Checkpoint path must stay in its directory / 权重路径必须在同目录: " + name)
                required.append(name)
    missing = []
    for name in required:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
        elif path.suffix in (".ckpt", ".pt"):
            with path.open("rb") as stream:
                if stream.read(40).startswith(b"version https://git-lfs"):
                    missing.append(name)
    moge = Path(weights_dir()) / "moge-vitl" / "model.pt"
    if not moge.is_file() or moge.stat().st_size == 0:
        missing.append("MoGe/model.pt")
    return {"status": "ready" if not missing else "missing", "directory": str(directory),
            "pipeline": str(pipeline), "moge": str(moge), "missing": missing}


def ensure_reconstruction_weights(checkpoint_directory=None, check_only=False):
    """Use local checkpoints, or download with the user's own gated-model access.

    优先使用本地权重；缺失时只使用用户自己的受控模型访问权限下载。
    """
    result = reconstruction_weights_status(checkpoint_directory)
    if result["status"] == "ready":
        return result
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url, snapshot_download
    token = get_token()
    if any(name != "MoGe/model.pt" for name in result["missing"]):
        if not token:
            result.update(status="blocked", reason="Request SAM3D access and log in with your own Hugging Face token / 请申请 SAM3D 访问权限，并使用自己的 Hugging Face token 登录")
            return result
        try:
            url = hf_hub_url("facebook/sam-3d-objects", "checkpoints/pipeline.yaml", endpoint="https://huggingface.co")
            get_hf_file_metadata(url, token=token)
        except Exception as error:
            result.update(status="blocked", reason="SAM3D access check failed / SAM3D 访问检查失败: " + type(error).__name__)
            return result
    if check_only:
        result["status"] = "download_required"
        return result
    # Never send gated-model credentials to an implicit mirror.
    # 不向隐式镜像发送受控模型凭据；第三方权重也不收入 WAPR wheel。
    if any(name != "MoGe/model.pt" for name in result["missing"]):
        parent = str(Path(result["directory"]).parent)
        snapshot_download("facebook/sam-3d-objects", endpoint="https://huggingface.co", token=token,
                          allow_patterns=["checkpoints/*.yaml", "checkpoints/*.ckpt", "LICENSE"],
                          local_dir=parent, max_workers=2)
    if "MoGe/model.pt" in result["missing"]:
        hf_hub_download("Ruicheng/moge-vitl", "model.pt", endpoint="https://huggingface.co", token=False,
                        local_dir=str(Path(result["moge"]).parent))
    return reconstruction_weights_status(checkpoint_directory)


def prepare_reconstruction(allow_replacement=None, check_only=False, checkpoint_directory=None):
    """Build native inference dependencies against the existing Torch.

    按已有 Torch 构建原生推理依赖；不更换 Torch，不安装完整训练清单。
    """
    if sys.platform != "linux":
        return {"status": "blocked", "reason": "SAM3D native path requires Linux / SAM3D 原生路径需要 Linux"}
    if sys.version_info[:2] < (3, 9):
        return {"status": "blocked", "reason": "The selected MoGe source declares Python >=3.9 / 所选 MoGe 源码声明 Python >=3.9"}
    import torch
    if not torch.cuda.is_available():
        return {"status": "blocked", "reason": "Existing Torch needs CUDA / 已有 Torch 需要可用 CUDA"}
    from torch.utils import _pytree
    if not callable(getattr(_pytree, "tree_map_only", None)) or not callable(getattr(torch.nn.functional, "scaled_dot_product_attention", None)):
        return {"status": "blocked", "reason": "Existing Torch lacks SAM3D pytree/SDPA APIs; Torch is retained / 已有 Torch 缺少 SAM3D pytree/SDPA API；保留 Torch"}
    weights = ensure_reconstruction_weights(checkpoint_directory, check_only=check_only)
    if weights["status"] not in ("ready", "download_required"):
        return weights
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    result = {"status": "native_build_required", "weights": weights, "gpu_memory_gb": memory_gb,
              "note": "Official setup recommends 32GB; smaller GPUs require actual inference validation / 官方建议 32GB；较小显存需实际推理验证"}
    if check_only:
        return result
    from wapr.installation import install_requirements
    from wapr.resources import resource_root
    from wapr.source_setup import SAM3D_REVISION, _checkout, prepare_source
    # Only imports reached by the mesh/vertex-color inference path are prepared.
    # 只准备网格/顶点颜色推理路径实际使用的依赖，不安装音频、训练和 UI 工具。
    requirements = ["torchvision", "omegaconf", "hydra-core", "iopath", "xatlas", "open3d",
                    "loguru", "tqdm", "safetensors", "optree", "astor", "easydict", "scipy",
                    "pyvista", "pymeshfix", "igraph", "lightning==2.3.3", "einops", "timm"]
    # Sparse kernels do not link the Torch extension ABI. The CUDA 11/12 runtime
    # families are candidates, not a claim that all compiler/GPU pairs work.
    # 稀疏内核不链接 Torch 扩展 ABI；CUDA 11/12 发行是待实测候选，不承诺所有组合可用。
    cuda_major = int(torch.version.cuda.split(".")[0])
    if cuda_major == 11:
        requirements.append("spconv-cu118")
    elif cuda_major == 12:
        # cu120 wheels stop before CPython 3.12; cu124 publishes cp312 wheels.
        # cu120 未提供 CPython 3.12 wheel；cu124 有 cp312 发行，仍需实测内核。
        cuda_minor = int(torch.version.cuda.split(".")[1])
        requirements.append("spconv-cu124" if cuda_minor >= 4 else "spconv-cu120")
    else:
        result.update(status="blocked", reason="No verified spconv candidate for this CUDA family / 此 CUDA 家族暂无已核验 spconv 候选")
        return result
    plan = install_requirements(requirements, check_only=True)
    # OpenCV distributions share cv2 files; pip metadata does not flag this collision.
    # OpenCV 多种发行共用 cv2 文件，pip 元数据无法识别此类覆盖风险。
    from importlib.metadata import PackageNotFoundError, version
    opencv_existing = []
    for distribution in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            opencv_existing.append({"name": distribution, "version": version(distribution)})
        except PackageNotFoundError:
            pass
    opencv_new = [entry for entry in plan.get("install", [])
                  if entry["name"].startswith("opencv-") and not entry.get("installed")]
    if opencv_existing and opencv_new:
        result.update(status="blocked", dependency_plan=plan, existing_opencv=opencv_existing,
                      reason="OpenCV distributions would overwrite cv2; explicit environment review required / OpenCV 多种发行可能覆盖 cv2，需先明确审核环境变更")
        return result
    if any(entry["name"] in ("torch", "torchvision", "torchaudio") for entry in plan.get("replace", [])):
        result.update(status="blocked", dependency_plan=plan,
                      reason="Reconstruction requires changing the protected Torch stack; retain this environment / 重建需要更换受保护 Torch 软件栈；保留当前环境")
        return result
    dependencies = install_requirements(requirements, allow_replacement=allow_replacement)
    if dependencies["status"] not in ("ready", "installed"):
        return dependencies
    # The caller supplies a mask; this inference path does not import SAM2.
    # 调用者已提供 mask，此推理路径不导入 SAM2，不获取其无关源码。
    sam3d_source = _checkout("sam-3d-objects", "https://github.com/facebookresearch/sam-3d-objects.git", SAM3D_REVISION, "sam3d")
    dinov2_source = prepare_source("dinov2")
    utils_source = _checkout("utils3d", "https://github.com/EasternJournalist/utils3d.git", "3913c65d81e05e47b9f367250cf8c0f7462a0900")
    moge_source = _checkout("MoGe", "https://github.com/microsoft/MoGe.git", "a8c37341bc0325ca99b9d57981cc3bb2bd3e255b")
    pytorch3d_source = _checkout("pytorch3d", "https://github.com/facebookresearch/pytorch3d.git", "75ebeeaea0908c5527e7b1e305fbc7681382db47")
    wheel_directory = Path(resource_root()) / "native_wheels" / "pytorch3d"
    wheel_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MAX_JOBS"] = "2"
    environment["FORCE_CUDA"] = "1"
    native_healthy = False
    try:
        from pytorch3d.ops import knn_points
        points = torch.zeros((1, 2, 3), device="cuda")
        nearest = knn_points(points, points, K=1)
        native_healthy = bool(torch.isfinite(nearest.dists).all())
    except Exception:
        pass
    # Build in the existing environment; isolated builders cannot see its Torch.
    # 使用现有环境构建；隔离构建环境无法看到当前 Torch。
    native_wheel = None
    if not native_healthy:
        build = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                                "--wheel-dir", str(wheel_directory), pytorch3d_source], env=environment)
        if build.returncode != 0:
            result.update(status="failed", reason="PyTorch3D native build failed / PyTorch3D 原生构建失败", exit_code=build.returncode)
            return result
        wheels = list(wheel_directory.glob("pytorch3d-*.whl"))
        if len(wheels) != 1:
            result.update(status="blocked", reason="Expected exactly one PyTorch3D wheel / 需要唯一 PyTorch3D wheel")
            return result
        native_wheel = str(wheels[0])
        native = install_requirements([native_wheel], allow_replacement=allow_replacement)
        if native["status"] not in ("ready", "installed"):
            return native
    for path in (sam3d_source, utils_source, moge_source):
        if path not in sys.path:
            sys.path.insert(0, path)
    # An import proves setup completion, not inference success or a 16GB memory bound.
    # 导入只证明准备完成，不等于推理成功或 16GB 显存可运行。
    os.environ["LIDRA_SKIP_INIT"] = "true"
    try:
        importlib.import_module("sam3d_objects.pipeline.inference_pipeline_pointmap")
    except Exception as error:
        detail = re.sub(r"https?://\S+", "[URL omitted / 地址已隐藏]", str(error))
        result.update(status="blocked", reason="SAM3D inference import failed / SAM3D 推理导入失败: " + type(error).__name__ + " " + detail)
        return result
    result.update(status="ready", source=sam3d_source, native_wheel=native_wheel,
                  source_paths=[sam3d_source, utils_source, moge_source], dinov2_source=dinov2_source,
                  inference_verified=False)
    return result


def reconstruct(rgb, mask, checkpoint_directory=None, allow_replacement=None):
    """Run the existing uncompiled vertex-color mesh recipe on one RGB/mask.

    在单张 RGB/mask 上执行已有未编译的顶点颜色网格方案；不改变推理步数。
    """
    prepared = prepare_reconstruction(allow_replacement=allow_replacement,
                                      checkpoint_directory=checkpoint_directory)
    if prepared["status"] != "ready":
        raise RuntimeError("Reconstruction preparation stopped / 重建准备停止: " + str(prepared.get("reason", prepared["status"])))
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("Expected uint8 RGB and matching mask / 需要 uint8 RGB 与同尺寸 mask")
    if int(np.count_nonzero(mask)) < 20:
        raise ValueError("Object mask is too small / 物体 mask 太小")
    config = OmegaConf.load(prepared["weights"]["pipeline"])
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = prepared["weights"]["directory"]
    config.depth_model.model.pretrained_model_name_or_path = prepared["weights"]["moge"]
    rgba = np.concatenate([rgb, ((mask > 0)[..., None] * 255).astype(np.uint8)], axis=2)
    os.environ["LIDRA_SKIP_INIT"] = "true"
    torch.cuda.empty_cache()
    original_hub_load = torch.hub.load

    def local_dino_load(repo_or_dir, model, *args, source="github", **kwargs):
        """Keep model arguments while using the prepared DINOv2 source.

        保留模型参数与预训练选择，只将 DINOv2 源码定位到已准备的检出。
        """
        if repo_or_dir == "facebookresearch/dinov2":
            return original_hub_load(prepared["dinov2_source"], model, *args, source="local", **kwargs)
        return original_hub_load(repo_or_dir, model, *args, source=source, **kwargs)

    torch.hub.load = local_dino_load
    try:
        pipeline = instantiate(config)
    finally:
        torch.hub.load = original_hub_load
    with torch.no_grad():
        output = pipeline.run(rgba, None, seed=42, stage1_only=False,
                              with_mesh_postprocess=False, with_texture_baking=False,
                              with_layout_postprocess=False, use_vertex_color=True)
    mesh = output["glb"]
    if len(mesh.vertices) == 0 or not np.isfinite(mesh.vertices).all():
        raise RuntimeError("Reconstruction produced invalid vertices / 重建产生无效顶点")
    del output
    del pipeline
    torch.cuda.empty_cache()
    return mesh
