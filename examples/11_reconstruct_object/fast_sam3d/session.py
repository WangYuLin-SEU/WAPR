# WAPR integration and modifications: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang for WAPR integration and modifications.
# All rights reserved, except as granted under applicable licenses.
# 作者署名对应 WAPR 集成与修改；上游作者署名与条款继续适用。
# SAM 3D / Fast-SAM3D attribution and terms: THIRD_PARTY_NOTICES.txt and licenses/manifest.json.

"""One resident SAM 3D pipeline, with a Fast-SAM3D inference mode.

一套常驻的 SAM 3D。Fast-SAM3D 只改推理步，不重新加载权重。

backend sam3d leaves the local pipeline.run path unchanged.
backend fast_sam3d turns on the training-free cache, token carving, and
spectral voxel aggregation. Presets only change those inference arguments.
backend 为 sam3d 时，本地 pipeline.run 保持原样。
backend 为 fast_sam3d 时打开免训练的缓存、token carving 和频谱体素聚合。
preset 只改这些推理参数。
"""

import json
import os
import sys
import time
import types

import numpy as np


DEMO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from step01_point_mask import (  # noqa: E402
    MOGE_CHECKPOINT,
    SAM3D_CONFIG,
    SAM3D_ROOT,
    sam3d_gaussian_backend,
    sam3d_local_dino,
    sam3d_utils3d_names,
)


# Official Fast-SAM3D Pareto point. Do not raise the stride or the token ratio here.
# Fast-SAM3D 官方 Pareto 点。这里不把 stride 或 token 比例再加大。
SS_STRIDE = 3
SS_WARMUP = 2
SS_ORDER = 1
SS_MOMENTUM_BETA = 0.5
SLAT_THRESH = 1.5
SLAT_WARMUP = 3
SLAT_TOKEN_RATIO = 0.1
MESH_SPECTRAL_LOW = 0.5
MESH_SPECTRAL_HIGH = 0.7

# Geometry uses 12 SLaT steps and 25 stage-1 steps. / 几何重建采用 12 步 SLaT 与 25 步 stage 1。
GEOMETRY_STAGE2_STEPS = 12
GEOMETRY_TARGET_VERTICES = 8000
TEXTURE_SIZE = 1024
FALLBACK_IOU = 0.55


def preset_config(name):
    """
    # Return the inference-argument dict for one preset name.

        An unknown name raises KeyError.

        `name` is a string.

        The branches are `sam3d`, `fast_sam3d`, `fast_sam3d_stage1_4`, `fast_sam3d_stage1_4_stage2_12`, `fast_sam3d_geometry`, and `fast_sam3d_quality`. It is not a tensor and it is not None.

    ---

    # 返回这个名字对应的推理参数字典。

        未知名字抛出 KeyError。

        `name` 是字符串。

        分支是 `sam3d`、`fast_sam3d`、`fast_sam3d_stage1_4`、`fast_sam3d_stage1_4_stage2_12`、`fast_sam3d_geometry` 和 `fast_sam3d_quality`。

        不是张量，也不是 None。

"""
    base = {
        "backend": "sam3d",
        "preset": name,
        "enable_fast_sam3d": False,
        "enable_ss_faster": False,
        "enable_slat_token": False,
        "enable_mesh_aggregation": False,
        "ss_faster_stride": SS_STRIDE,
        "ss_warmup": SS_WARMUP,
        "ss_order": SS_ORDER,
        "ss_momentum_beta": SS_MOMENTUM_BETA,
        "slat_thresh": SLAT_THRESH,
        "slat_warmup": SLAT_WARMUP,
        "slat_token_ratio": SLAT_TOKEN_RATIO,
        "mesh_spectral_threshold_low": MESH_SPECTRAL_LOW,
        "mesh_spectral_threshold_high": MESH_SPECTRAL_HIGH,
        "stage1_distillation": False,
        "stage1_steps": None,
        "stage2_steps": None,
        "compile_model": False,
        "texture_size": TEXTURE_SIZE,
        "target_vertices": None,
        # Decoder vertex color only. The UV unwrap and the Gaussian-view bake stay off.
        # 只用解码器的顶点颜色。不展开 UV，也不做高斯多视角烘焙。
        "with_texture_baking": False,
        "with_mesh_postprocess": False,
        "save_gaussian": False,
        "auto_fallback": False,
        "fallback_iou": FALLBACK_IOU,
        "seed": 42,
    }
    if name == "sam3d":
        return dict(base)
    fast = dict(base)
    fast["backend"] = "fast_sam3d"
    fast["enable_fast_sam3d"] = True
    fast["enable_ss_faster"] = True
    fast["enable_slat_token"] = True
    fast["enable_mesh_aggregation"] = True
    if name == "fast_sam3d":
        return fast
    if name == "fast_sam3d_stage1_4":
        fast["stage1_distillation"] = True
        fast["stage1_steps"] = 4
        return fast
    if name == "fast_sam3d_stage1_4_stage2_12":
        fast["stage1_distillation"] = True
        fast["stage1_steps"] = 4
        fast["stage2_steps"] = 12
        return fast
    if name in ("fast_sam3d_geometry", "fast_sam3d_quality"):
        # 4-step stage-1 distillation opens the cracker thin face. 25-step
        # stage 1 with the Fast cache does not. SLaT stays at 12 or 25.
        # 4 步 stage-1 蒸馏会把饼干盒薄面打穿。25 步 stage 1 加 Fast 缓存不会。
        # SLaT 仍是 12 或 25。
        fast["stage1_distillation"] = False
        fast["stage1_steps"] = None
        fast["stage2_steps"] = 25 if name == "fast_sam3d_quality" else GEOMETRY_STAGE2_STEPS
        fast["target_vertices"] = GEOMETRY_TARGET_VERTICES
        fast["with_mesh_postprocess"] = True
        fast["with_texture_baking"] = False
        return fast
    raise KeyError(name)


def _sync():
    """
    # Synchronize CUDA when it is available, then return `time.perf_counter()` in seconds.

        This function takes no arguments.

    ---

    # 若 CUDA 可用则先同步，然后返回 `time.perf_counter()`，单位是秒。

        这个函数没有参数。

"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class StageClock:
    """
    # CUDA-synchronized stage times, seconds.

    ---

    # 各阶段耗时，秒。

        CUDA 先同步再读表。
    """

    def __init__(self):
        """
        # Create an empty stage-time map.

            `self` is the new `StageClock`. `seconds` starts as an empty dict of stage name to accumulated seconds.

            There is no other argument.

        ## Returns

            - Returns None.

        ---

        # 建立一个空的阶段耗时表。

            `self` 是新建的 `StageClock`。

            `seconds` 一开始是空字典，键是阶段名，值是累计秒数。

            没有其他参数。

        ## 返回

            - 返回 None。

"""
        self.seconds = {}

    def add(self, name, dt):
        """
        # Add `dt` seconds into the named stage.

            `self` is the `StageClock` whose `seconds` dict is updated.

            `name` is a string key such as `preprocess`, `condition_ss`, `sample_ss`, `pose`, `sample_slat`, `mesh_decoder`, `gaussian_decoder`, `mesh_postprocess`, `texture_baking`, or `export`. It is not a tensor and it is not None.

            `dt` is converted with `float` and added to any previous value for `name`. Callers pass a difference of `_sync()` readings, in seconds.

            It is not None.

        ## Returns

            - Returns None.

        ---

        # 把 `dt` 秒累加到指定阶段。

            `self` 是要更新 `seconds` 字典的 `StageClock`。

            `name` 是字符串键，例如 `preprocess`、`condition_ss`、`sample_ss`、`pose`、`sample_slat`、`mesh_decoder`、`gaussian_decoder`、`mesh_postprocess`、`texture_baking` 或 `export`。

            不是张量，也不是 None。

            `dt` 会用 `float` 转换，再加到 `name` 已有的值上。

            调用者传入的是两次 `_sync()` 的差，单位秒。

            不是 None。

        ## 返回

            - 返回 None。

"""
        self.seconds[name] = self.seconds.get(name, 0.0) + float(dt)

    def exclusive(self):
        """
        # Return the stage report in seconds.

            `T_total` is the sum of the reported stages, not the wall clock.

            `self` is the `StageClock`. Missing stages are read as 0.0.

            `T_stage1_ss` is `sample_ss` minus `condition_ss`, and `T_stage2_slat` is `sample_slat` minus `condition_slat`, each clamped at 0.

            The other keys are `T_preprocess`, `T_condition_encoder`, `T_pose_layout`, `T_mesh_decoder`, `T_gaussian_decoder`, `T_mesh_postprocess`, `T_texture_baking`, and `T_export`.

        ---

        # 返回各阶段报告，单位秒。

            `T_total` 是这些阶段之和，不是墙钟时间。

            `self` 是 `StageClock`。

            缺失的阶段按 0.0 读取。

            `T_stage1_ss` 是 `sample_ss` 减去 `condition_ss`，`T_stage2_slat` 是 `sample_slat` 减去 `condition_slat`，两者都不会小于 0。

            其余键是 `T_preprocess`、`T_condition_encoder`、`T_pose_layout`、`T_mesh_decoder`、`T_gaussian_decoder`、`T_mesh_postprocess`、`T_texture_baking` 和 `T_export`。

"""
        condition_ss = self.seconds.get("condition_ss", 0.0)
        condition_slat = self.seconds.get("condition_slat", 0.0)
        report = {
            "T_preprocess": self.seconds.get("preprocess", 0.0),
            "T_condition_encoder": condition_ss + condition_slat,
            "T_stage1_ss": max(0.0, self.seconds.get("sample_ss", 0.0) - condition_ss),
            "T_pose_layout": self.seconds.get("pose", 0.0),
            "T_stage2_slat": max(0.0, self.seconds.get("sample_slat", 0.0) - condition_slat),
            "T_mesh_decoder": self.seconds.get("mesh_decoder", 0.0),
            "T_gaussian_decoder": self.seconds.get("gaussian_decoder", 0.0),
            "T_mesh_postprocess": self.seconds.get("mesh_postprocess", 0.0),
            "T_texture_baking": self.seconds.get("texture_baking", 0.0),
            "T_export": self.seconds.get("export", 0.0),
        }
        report["T_total"] = float(sum(report.values()))
        return report


def _compile_warmup(num_warmup_iters=3):
    """
    # Run the pointmap, sparse-structure, and SLaT path `num_warmup_iters` times on a 512×512 RGBA image.

        `num_warmup_iters` is an int, default 3.

        The pipeline is the `pipeline` attribute previously stored on this function.

        The synthetic image is uint8 with shape (512, 512, 4).

        It is not None.

    ## Returns

        - Returns None.

    ---

    # 在一张 512×512 的 RGBA 图像上，把 pointmap、稀疏结构和 SLaT 路径跑 `num_warmup_iters` 次。

        `num_warmup_iters` 是整数，默认 3。

        pipeline 是事先存在这个函数的 `pipeline` 属性上的对象。

        合成图像是 uint8，形状 (512, 512, 4)。

        不是 None。

    ## 返回

        - 返回 None。

"""
    import torch
    from PIL import Image

    pipeline = _compile_warmup.pipeline
    test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
    test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image = pipeline.merge_image_and_mask(Image.fromarray(test_image), None)
    with torch.inference_mode(False):
        with torch.no_grad():
            for _ in range(int(num_warmup_iters)):
                pointmap = pipeline.compute_pointmap(image)["pointmap"]
                ss_input_dict = pipeline.preprocess_image(
                    image, pipeline.ss_preprocessor, pointmap=pointmap
                )
                slat_input_dict = pipeline.preprocess_image(image, pipeline.slat_preprocessor)
                ss_return_dict = pipeline.sample_sparse_structure(ss_input_dict)
                pipeline.sample_slat(slat_input_dict, ss_return_dict["coords"])


def _bind(obj, fn):
    """
    # Return `types.MethodType(fn, obj)`, so `fn` runs as a bound method of `obj`.

        `obj` is the pipeline or generator instance that becomes the first argument.

        It is not None.

        `fn` is a function whose first parameter receives `obj`. It is not None.

    ---

    # 返回 `types.MethodType(fn, obj)`，让 `fn` 作为 `obj` 的绑定方法运行。

        `obj` 是将成为第一个参数的 pipeline 或生成器实例。

        不是 None。

        `fn` 是函数，它的第一个参数接收 `obj`。

        不是 None。

"""
    return types.MethodType(fn, obj)


class Sam3dSession:
    """
    # Load SAM 3D once.

        reconstruct() can switch presets without reloading.

    ---

    # SAM 3D 只加载一次。

        reconstruct() 切换 preset 时不重新加载。
    """

    def __init__(self, device="cuda:0", compile_model=False):
        """
        # Store the device and the default `sam3d` preset.

            The pipeline stays None until `load`. Returns None.

            `self` is the new `Sam3dSession`.

            `device` is a string, default `cuda:0`. `load` uses the text after the last colon as `CUDA_VISIBLE_DEVICES` and passes the whole string to `torch.cuda.set_device`. It is not None.

            `compile_model` is converted with `bool`, default False.

            True makes `load` compile the pipeline after replacing its warmup.

            It is not None.

        ---

        # 保存设备和默认的 `sam3d` preset。

            pipeline 在 `load` 之前保持 None。

            返回 None。

            `self` 是新建的 `Sam3dSession`。

            `device` 是字符串，默认 `cuda:0`。

            `load` 把最后一个冒号后面的文字当作 `CUDA_VISIBLE_DEVICES`，并把整个字符串传给 `torch.cuda.set_device`。

            不是 None。

            `compile_model` 会用 `bool` 转换，默认 False。

            为 True 时，`load` 会先替换 warmup，再编译 pipeline。

            不是 None。

"""
        self.device = device
        self.compile_model = bool(compile_model)
        self.pipeline = None
        self.cfg = preset_config("sam3d")
        self._originals = {}
        self._ss_fast_on = False
        self._slat_fast_on = False
        self._sample_fast_on = False
        self.compile_seconds = None

    def load(self):
        """
        # Instantiate the SAM 3D pipeline once and return `self`. A second call returns the session already loaded.

            `self` is the `Sam3dSession`. The pipeline config sets `rendering_engine` to `pytorch3d`, `compile_model` to False on the config object, and the depth checkpoint to `MOGE_CHECKPOINT`. When `self.compile_model` is true, `pipeline._warmup` is replaced by `_compile_warmup` before `pipeline._compile()`.

        ---

        # 实例化 SAM 3D pipeline 一次，并返回 `self`。

            再次调用时返回已经载入的 session。

            `self` 是 `Sam3dSession`。

            pipeline 配置把 `rendering_engine` 设为 `pytorch3d`，配置对象上的 `compile_model` 设为 False，深度权重设为 `MOGE_CHECKPOINT`。

            当 `self.compile_model` 为真时，会先把 `pipeline._warmup` 换成 `_compile_warmup`，再调用 `pipeline._compile()`。

"""
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        if self.pipeline is not None:
            return self
        os.environ["LIDRA_SKIP_INIT"] = "true"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.device.split(":")[-1])
        sam3d_utils3d_names()
        if SAM3D_ROOT not in sys.path:
            sys.path.insert(0, SAM3D_ROOT)
        sam3d_gaussian_backend()
        sam3d_local_dino()
        config = OmegaConf.load(SAM3D_CONFIG)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = os.path.dirname(SAM3D_CONFIG)
        config.depth_model.model.pretrained_model_name_or_path = MOGE_CHECKPOINT
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        self.pipeline = instantiate(config)
        self._remember_originals()
        if self.compile_model:
            # Pointmap._warmup calls run_layout_model, which this class does not
            # define. Warm SS and SLaT on a pointmap instead, without editing SAM3D.
            # pointmap 的 _warmup 会调用不存在的 run_layout_model。
            # 不改 SAM3D 源码，改成带 pointmap 的 SS 和 SLaT 热身。
            self.pipeline._warmup = _compile_warmup
            _compile_warmup.pipeline = self.pipeline
            compile_started = _sync()
            self.pipeline._compile()
            self.compile_seconds = _sync() - compile_started
            print("COMPILE_SECONDS", self.compile_seconds, flush=True)
        return self

    def _remember_originals(self):
        """
        # Save the generator and pipeline callables that later presets replace.

            `self` is a loaded `Sam3dSession`. The saved names are `ss_generate_iter`, `ss_dynamics`, `slat_generate`, `slat_generate_iter`, `slat_solver`, `sample_ss`, `sample_slat`, `decode`, and `post`. There is no other argument.

        ## Returns

            - Returns None.

        ---

        # 保存之后的 preset 会替换的生成器和 pipeline 可调用对象。

            `self` 是已经载入的 `Sam3dSession`。

            保存的名字是 `ss_generate_iter`、`ss_dynamics`、`slat_generate`、`slat_generate_iter`、`slat_solver`、`sample_ss`、`sample_slat`、`decode` 和 `post`。

            没有其他参数。

        ## 返回

            - 返回 None。

"""
        pipeline = self.pipeline
        ss = pipeline.models["ss_generator"]
        slat = pipeline.models["slat_generator"]
        self._originals = {
            "ss_generate_iter": ss.generate_iter,
            "ss_dynamics": ss._generate_dynamics,
            "slat_generate": slat.generate,
            "slat_generate_iter": slat.generate_iter,
            "slat_solver": slat._solver,
            "sample_ss": pipeline.sample_sparse_structure,
            "sample_slat": pipeline.sample_slat,
            "decode": pipeline.decode_slat,
            "post": pipeline.postprocess_slat_output,
        }

    def _restore_generators(self):
        """
        # Put the saved generator and pipeline callables back and clear the fast-path flags.

            `self` is the `Sam3dSession` whose `_originals` dict was filled by `_remember_originals`. There is no other argument.

        ## Returns

            - Returns None.

        ---

        # 把保存的生成器和 pipeline 可调用对象放回去，并清掉快速路径标志。

            `self` 是 `Sam3dSession`，它的 `_originals` 字典由 `_remember_originals` 填好。

            没有其他参数。

        ## 返回

            - 返回 None。

"""
        pipeline = self.pipeline
        ss = pipeline.models["ss_generator"]
        slat = pipeline.models["slat_generator"]
        saved = self._originals
        ss.generate_iter = saved["ss_generate_iter"]
        ss._generate_dynamics = saved["ss_dynamics"]
        slat.generate = saved["slat_generate"]
        slat.generate_iter = saved["slat_generate_iter"]
        slat._solver = saved["slat_solver"]
        pipeline.sample_sparse_structure = saved["sample_ss"]
        pipeline.sample_slat = saved["sample_slat"]
        pipeline.decode_slat = saved["decode"]
        pipeline.postprocess_slat_output = saved["post"]
        self._ss_fast_on = False
        self._slat_fast_on = False
        self._sample_fast_on = False

    def apply(self, cfg):
        """
        # Point the loaded pipeline at one preset dict.

            `self` is the `Sam3dSession`. A missing pipeline is loaded first.

            Generators are restored before the new flags are applied.

            `cfg` is a dict, normally from `preset_config`. This method copies it onto `self.cfg`. If the master fast flag is off, `enable_ss_faster`, `enable_slat_token`, and `enable_mesh_aggregation` are stored as False.

            It is not None.

        ## Returns

            - Returns None.
            - Fast hooks are installed only when `enable_fast_sam3d` is true and `backend` is `fast_sam3d`.

        ---

        # 把已载入的 pipeline 指到一份 preset 字典。

            `self` 是 `Sam3dSession`。

            pipeline 还不存在时会先载入。

            应用新标志之前会先恢复生成器。

            `cfg` 是字典，通常来自 `preset_config`。

            这个方法把它复制到 `self.cfg`。

            主快速开关关闭时，`enable_ss_faster`、`enable_slat_token` 和 `enable_mesh_aggregation` 会被存成 False。

            不是 None。

        ## 返回

            - 返回 None。
            - 只有 `enable_fast_sam3d` 为真且 `backend` 是 `fast_sam3d` 时才安装快速钩子。

"""
        if self.pipeline is None:
            self.load()
        self._restore_generators()
        self.cfg = dict(cfg)
        master = bool(cfg["enable_fast_sam3d"]) and cfg["backend"] == "fast_sam3d"
        if not master:
            self.cfg["enable_ss_faster"] = False
            self.cfg["enable_slat_token"] = False
            self.cfg["enable_mesh_aggregation"] = False
            return
        if cfg["enable_ss_faster"]:
            self._install_ss_cache()
        if cfg["enable_mesh_aggregation"] or cfg["enable_slat_token"]:
            self._install_structure_sample()
        if cfg["enable_slat_token"]:
            self._install_slat_cache()

    def _install_ss_cache(self):
        """
        # Replace the sparse-structure generator's `generate_iter` and `_generate_dynamics` with the cache wrappers.

            `self` is the `Sam3dSession`. The installed `ss_params` are `ss_faster_stride`, `ss_warmup`, `ss_order`, and `ss_momentum_beta` from `self.cfg`, stored as int or float.

            There is no other argument.

        ## Returns

            - Returns None.

        ---

        # 用缓存包装替换稀疏结构生成器的 `generate_iter` 和 `_generate_dynamics`。

            `self` 是 `Sam3dSession`。

            装上的 `ss_params` 来自 `self.cfg` 的 `ss_faster_stride`、`ss_warmup`、`ss_order` 和 `ss_momentum_beta`，按 int 或 float 保存。

            没有其他参数。

        ## 返回

            - 返回 None。

"""
        from fast_sam3d.acceleration import faster_init, ss_cache_dynamics

        ss = self.pipeline.models["ss_generator"]
        cfg = self.cfg
        ss.ss_params = {
            "ss_faster_stride": int(cfg["ss_faster_stride"]),
            "ss_warmup": int(cfg["ss_warmup"]),
            "ss_order": int(cfg["ss_order"]),
            "ss_momentum_beta": float(cfg["ss_momentum_beta"]),
        }

        def generate_iter(generator, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
            """
            # Initialize the sparse-structure cache, then return the class `generate_iter` so this override is not recursive.

                `generator` is the sparse-structure generator this function is bound to.

                `faster_init` uses its `inference_steps` and `ss_params`. `prev_v` is set to None.

                It is not None.

                `x_shape` and `x_device` are forwarded to the class `generate_iter`. This wrapper does not read their contents.

                They are not None.

                `args_conditionals` and `kwargs_conditionals` are forwarded unchanged.

                This wrapper does not fix their shapes.

            ---

            # 初始化稀疏结构缓存，然后返回类上的 `generate_iter`，避免这个覆盖递归调用自己。

                `generator` 是这个函数绑定到的稀疏结构生成器。

                `faster_init` 使用它的 `inference_steps` 和 `ss_params`。

                `prev_v` 被设为 None。

                不是 None。

                `x_shape` 和 `x_device` 会转给类上的 `generate_iter`。

                这个包装不读它们的内容。

                不是 None。

                `args_conditionals` 和 `kwargs_conditionals` 原样转发。

                这个包装不规定它们的形状。

"""
            generator.faster_dic, generator.current = faster_init(
                generator.inference_steps,
                faster_interval=generator.ss_params["ss_faster_stride"],
                max_order=generator.ss_params["ss_order"],
                first_enhance=generator.ss_params["ss_warmup"],
                end_enhance=24,
            )
            generator.prev_v = None
            return generator.__class__.generate_iter(
                generator, x_shape, x_device, *args_conditionals, **kwargs_conditionals
            )

        def dynamics(generator, x_t, t, d, *args_conditionals, **kwargs_conditionals):
            """
            # Return the velocity from `ss_cache_dynamics` for one sparse-structure step.

                `generator`, `x_t`, `t`, `d`, `args_conditionals`, and `kwargs_conditionals` are forwarded to `ss_cache_dynamics`. This wrapper does not change them.

                None is not substituted here.

            ---

            # 返回 `ss_cache_dynamics` 为一步稀疏结构算出的速度。

                `generator`、`x_t`、`t`、`d`、`args_conditionals` 和 `kwargs_conditionals` 都转给 `ss_cache_dynamics`。

                这个包装不改它们。

                这里不会把它们换成 None。

"""
            return ss_cache_dynamics(
                generator, x_t, t, d, *args_conditionals, **kwargs_conditionals
            )

        # Call the class generate_iter so the instance override is not recursive.
        # 走类上的 generate_iter，避免实例方法递归。
        ss.generate_iter = _bind(ss, generate_iter)
        ss._generate_dynamics = _bind(ss, dynamics)
        self._ss_fast_on = True

    def _install_slat_cache(self):
        """
        # Install `EulerFasterSLat` and replace the SLaT generator's `generate` and `generate_iter`. Returns None.

            `self` is the `Sam3dSession`. The solver is built from `slat_thresh`, `slat_warmup`, `stage2_steps` or the generator's `inference_steps`, and `slat_token_ratio` in `self.cfg`. There is no other argument.

        ---

        # 装上 `EulerFasterSLat`，并替换 SLaT 生成器的 `generate` 和 `generate_iter`。

            `self` 是 `Sam3dSession`。

            求解器用 `self.cfg` 里的 `slat_thresh`、`slat_warmup`、`stage2_steps`（否则用生成器的 `inference_steps`）和 `slat_token_ratio` 来建。

            没有其他参数。

        ## 返回

            - 返回 None。

"""
        from fast_sam3d.acceleration import EulerFasterSLat

        slat = self.pipeline.models["slat_generator"]
        cfg = self.cfg
        solver = EulerFasterSLat(
            thresh=float(cfg["slat_thresh"]),
            ret_steps=int(cfg["slat_warmup"]),
            full_steps=int(cfg["stage2_steps"] or slat.inference_steps),
            carving_ratio=float(cfg["slat_token_ratio"]),
        )
        slat.slat_params = {
            "slat_thresh": float(cfg["slat_thresh"]),
            "slat_warmup": int(cfg["slat_warmup"]),
            "slat_token_ratio": float(cfg["slat_token_ratio"]),
        }

        def generate(generator, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
            """
            # Prepare the SLaT solver from the current generator settings, then return the class `generate`.

                `generator` is the SLaT generator this function is bound to.

                `solver.full_steps` is set from `generator.inference_steps`. Threshold, warmup, and carving ratio are copied from `generator.slat_params`. It is not None.

                `x_shape` supplies `int(x_shape[1])` as the token count for `solver.prepare`. `x_device` is the device passed to `prepare`. Scores are `pipeline._fast_coords_scores`, which may be None.

                Neither `x_shape` nor `x_device` is None.

                `args_conditionals` and `kwargs_conditionals` are forwarded to the class `generate`. This wrapper does not read them.

            ---

            # 按当前生成器设置准备 SLaT 求解器，然后返回类上的 `generate`。

                `generator` 是这个函数绑定到的 SLaT 生成器。

                `solver.full_steps` 来自 `generator.inference_steps`。

                阈值、预热步数和 carving 比例从 `generator.slat_params` 复制。

                不是 None。

                `x_shape` 用 `int(x_shape[1])` 作为 `solver.prepare` 的 token 数。

                `x_device` 是传给 `prepare` 的设备。

                分数是 `pipeline._fast_coords_scores`，它可以是 None。

                `x_shape` 和 `x_device` 本身不是 None。

                `args_conditionals` 和 `kwargs_conditionals` 转给类上的 `generate`。

                这个包装不读它们。

"""
            scores = getattr(self.pipeline, "_fast_coords_scores", None)
            solver.full_steps = int(generator.inference_steps)
            solver.thresh = float(generator.slat_params["slat_thresh"])
            solver.ret_steps = int(generator.slat_params["slat_warmup"])
            solver.carving_ratio = float(generator.slat_params["slat_token_ratio"])
            solver.prepare(int(x_shape[1]), x_device, scores)
            return generator.__class__.generate(
                generator, x_shape, x_device, *args_conditionals, **kwargs_conditionals
            )

        def generate_iter(generator, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
            """
            # Yield `(t, x_t, ())` for each state produced by `solver.solve_iter`.

                `generator` creates `x_0` with `_generate_noise(x_shape, x_device)` and the time sequence with `_prepare_t()`. The dynamics callable is `generator._generate_dynamics`. It is not None.

                `x_shape` and `x_device` are forwarded to `_generate_noise`. This wrapper does not document a further shape.

                They are not None.

                `args_conditionals` and `kwargs_conditionals` are forwarded to `solver.solve_iter`. This wrapper does not fix their shapes.

            ---

            # 对 `solver.solve_iter` 产出的每个状态，产出 `(t, x_t, ())`。

                `generator` 用 `_generate_noise(x_shape, x_device)` 建立 `x_0`，用 `_prepare_t()` 建立时间序列。

                动力学可调用对象是 `generator._generate_dynamics`。

                不是 None。

                `x_shape` 和 `x_device` 会传给 `_generate_noise`。

                这个包装不再另写形状。

                不是 None。

                `args_conditionals` 和 `kwargs_conditionals` 转给 `solver.solve_iter`。

                这个包装不规定它们的形状。

"""
            x_0 = generator._generate_noise(x_shape, x_device)
            t_seq = generator._prepare_t().to(x_device)
            for x_t, t, _velocity in solver.solve_iter(
                generator._generate_dynamics,
                x_0,
                t_seq,
                *args_conditionals,
                **kwargs_conditionals,
            ):
                yield t, x_t, ()

        slat._solver = solver
        slat.generate = _bind(slat, generate)
        slat.generate_iter = _bind(slat, generate_iter)
        self._slat_fast_on = True

    def _install_structure_sample(self):
        """
        # Replace sparse-structure sampling with the fast local sampler, and replace SLaT sampling when token carving is enabled.

            `self` is the `Sam3dSession`. `pipeline.sample_slat` is replaced only when `self.cfg["enable_slat_token"]` is true.

            There is no other argument.

        ## Returns

            - Returns None.

        ---

        # 用本地快速采样替换稀疏结构采样。

            `self` 是 `Sam3dSession`。

            只有 `self.cfg["enable_slat_token"]` 为真时才替换 `pipeline.sample_slat`。

            没有其他参数。

        ## 返回

            - token carving 打开时，也替换 SLaT 采样。
            - 返回 None。

"""
        pipeline = self.pipeline

        def sample_sparse_structure(
            self_pipeline, ss_input_dict, inference_steps=None, use_distillation=False
        ):
            """
            # Return the dict from `_sample_sparse_structure_fast`, using this session's preset as `cfg`.

                `self_pipeline` is the pipeline the method is bound to.

                It is forwarded as the pipeline argument.

                It is not None.

                `ss_input_dict` is forwarded.

                The fast sampler reads its `image` entry.

                It is not None.

                `inference_steps` defaults to None and is forwarded.

                The fast sampler changes the generator step count only when this value is true.

                `use_distillation` is a bool, default False, and is forwarded.

            ---

            # 返回 `_sample_sparse_structure_fast` 的字典，并用这个 session 的 preset 作为 `cfg`。

                `self_pipeline` 是这个方法绑定到的 pipeline。

                它会作为 pipeline 参数传下去。

                不是 None。

                `ss_input_dict` 会被转发。

                快速采样器读取其中的 `image`。

                不是 None。

                `inference_steps` 默认 None，会被转发。

                只有这个值为真时，快速采样器才改生成器的步数。

                `use_distillation` 是布尔值，默认 False，也会转发。

"""
            return _sample_sparse_structure_fast(
                self_pipeline, ss_input_dict, inference_steps, use_distillation, self.cfg
            )

        def sample_slat(
            self_pipeline,
            slat_input,
            coords,
            inference_steps=25,
            use_distillation=False,
        ):
            """
            # Attach the fast coordinate scores and map tokens, then return the saved SLaT sampler's result.

                `self_pipeline` is the pipeline.

                `coords_scores` is set from `_fast_coords_scores`, and `map_tokens` from `_fast_map_tokens`. Either attribute may be missing, in which case the generator field is None.

                `slat_input` and `coords` are forwarded to the saved `sample_slat`. This wrapper does not read their shapes.

                They are not None.

                `inference_steps` is an int, default 25, forwarded as a keyword.

                `use_distillation` is a bool, default False, also forwarded.

            ---

            # 挂上快速路径的坐标分数和 map token，然后返回原先保存的 SLaT 采样结果。

                `self_pipeline` 是 pipeline。

                `coords_scores` 来自 `_fast_coords_scores`，`map_tokens` 来自 `_fast_map_tokens`。

                这两个属性都可以不存在，那时生成器字段是 None。

                `slat_input` 和 `coords` 转给保存的 `sample_slat`。

                这个包装不读它们的形状。

                不是 None。

                `inference_steps` 是整数，默认 25，按关键字转发。

                `use_distillation` 是布尔值，默认 False，同样转发。

"""
            saved = self._originals["sample_slat"]
            slat_generator = self_pipeline.models["slat_generator"]
            slat_generator.coords_scores = getattr(self_pipeline, "_fast_coords_scores", None)
            slat_generator.map_tokens = getattr(self_pipeline, "_fast_map_tokens", None)
            return saved(
                slat_input,
                coords,
                inference_steps=inference_steps,
                use_distillation=use_distillation,
            )

        pipeline.sample_sparse_structure = _bind(pipeline, sample_sparse_structure)
        if self.cfg["enable_slat_token"]:
            pipeline.sample_slat = _bind(pipeline, sample_slat)
        self._sample_fast_on = True

    def reconstruct(self, rgb, mask, cfg=None, allow_fallback=True):
        """
        # Run one preset on an RGB image and a mask, and return a dict with `mesh`, `texture`, `metadata`, and `profile`. A geometry preset can call this method again with the quality preset when silhouette IoU is below `fallback_iou`.

            `self` is the `Sam3dSession`. The call applies `cfg`, times `pipeline.run`, exports the mesh, and may record silhouette IoU.

            `rgb` is converted with `numpy.asarray`. The first three channels are concatenated with a mask alpha channel to form the RGBA passed to `pipeline.run`. Callers pass RGB uint8 images.

            It is not None.

            `mask` is converted with `numpy.asarray`. Values `> 0` become alpha 255.

            It is also passed to `_silhouette_iou`. It is not None.

            `cfg` is a preset dict or None.

            None uses `self.cfg`. It selects steps, distillation, postprocess, and fallback.

            `allow_fallback` is a bool, default True.

            The second quality pass runs only when this is true, `cfg["auto_fallback"]` is true, the preset is `fast_sam3d_geometry`, and the IoU is above 0 and below `cfg["fallback_iou"]`. The recursive call passes False.

        ---

        # 用一个 preset 跑一张 RGB 和一张 mask，返回含 `mesh`、`texture`、`metadata` 和 `profile` 的字典。

            几何 preset 在剪影 IoU 低于 `fallback_iou` 时，会用质量 preset 再调用一次这个方法。

            `self` 是 `Sam3dSession`。

            这次调用会应用 `cfg`，给 `pipeline.run` 计时，导出网格，并可能记录剪影 IoU。

            `rgb` 会用 `numpy.asarray` 转换。

            前三个通道与 mask 的 alpha 通道拼成传给 `pipeline.run` 的 RGBA。

            调用者传入的是 RGB uint8 图像。

            不是 None。

            `mask` 会用 `numpy.asarray` 转换。

            `> 0` 的值变成 alpha 255。

            它也会传给 `_silhouette_iou`。

            不是 None。

            `cfg` 是 preset 字典或 None。

            None 时使用 `self.cfg`。

            它决定步数、蒸馏、后处理和回退。

            `allow_fallback` 是布尔值，默认 True。

            只有它为真、`cfg["auto_fallback"]` 为真、preset 是 `fast_sam3d_geometry`，并且 IoU 大于 0 且低于 `cfg["fallback_iou"]` 时，才会再跑质量 preset。

            递归调用传入 False。

"""
        import torch

        if cfg is None:
            cfg = self.cfg
        self.apply(cfg)
        pipeline = self.pipeline
        clock = StageClock()
        pipeline._clock = clock
        pipeline._cond_phase = "ss"
        pipeline._last_intrinsics = None
        pipeline._last_raw_vertices = None
        pipeline._last_raw_faces = None
        pipeline._source_rgb = np.asarray(rgb)
        pipeline._source_mask = np.asarray(mask)
        rgba = np.concatenate(
            [np.asarray(rgb)[..., :3], (np.asarray(mask)[..., None] > 0).astype(np.uint8) * 255],
            axis=2,
        )
        torch.cuda.reset_peak_memory_stats()
        started = _sync()
        pipeline._bake_texture = bool(cfg["with_texture_baking"])
        self._install_timers(clock)
        stage1_steps = cfg["stage1_steps"]
        stage2_steps = cfg["stage2_steps"]
        try:
            with torch.no_grad():
                output = pipeline.run(
                    rgba,
                    None,
                    seed=int(cfg["seed"]),
                    stage1_only=False,
                    with_mesh_postprocess=bool(cfg["with_mesh_postprocess"]) and cfg["target_vertices"] is None,
                    with_texture_baking=bool(cfg["with_texture_baking"]),
                    with_layout_postprocess=False,
                    use_vertex_color=not bool(cfg["with_texture_baking"]),
                    stage1_inference_steps=stage1_steps,
                    stage2_inference_steps=stage2_steps,
                    use_stage1_distillation=bool(cfg["stage1_distillation"]),
                    use_stage2_distillation=False,
                )
        finally:
            self._remove_timers()
        exported = _export_result(output, clock, pipeline)
        wall_s = _sync() - started
        profile = clock.exclusive()
        profile["T_total"] = wall_s
        profile["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        mesh = exported["mesh"]
        fallback = {
            "auto_fallback": bool(cfg["auto_fallback"]),
            "iou": None,
            "triggered": False,
        }
        if mesh is not None and int(exported["faces"]) <= 20000:
            fallback["iou"] = _silhouette_iou(
                mesh,
                mask,
                output.get("rotation", None),
                output.get("translation", None),
                output.get("scale", None),
                pipeline._last_intrinsics,
            )
            if (
                allow_fallback
                and cfg["auto_fallback"]
                and cfg["preset"] == "fast_sam3d_geometry"
                and fallback["iou"] is not None
                and fallback["iou"] > 0.0
                and fallback["iou"] < float(cfg["fallback_iou"])
            ):
                quality = preset_config("fast_sam3d_quality")
                quality["auto_fallback"] = False
                quality["compile_model"] = cfg["compile_model"]
                second = self.reconstruct(rgb, mask, quality, allow_fallback=False)
                second["metadata"]["fallback"] = {
                    "auto_fallback": True,
                    "iou": fallback["iou"],
                    "triggered": True,
                    "from_preset": "fast_sam3d_geometry",
                }
                return second
        metadata = {
            "backend": cfg["backend"],
            "preset": cfg["preset"],
            "enable_fast_sam3d": bool(cfg["enable_fast_sam3d"]),
            "stage1_distillation": bool(cfg["stage1_distillation"]),
            "stage1_steps": cfg["stage1_steps"],
            "stage2_steps": cfg["stage2_steps"],
            "ss_faster_stride": cfg["ss_faster_stride"] if cfg["enable_ss_faster"] else None,
            "slat_thresh": cfg["slat_thresh"] if cfg["enable_slat_token"] else None,
            "slat_token_ratio": cfg["slat_token_ratio"] if cfg["enable_slat_token"] else None,
            "mesh_spectral_low": cfg["mesh_spectral_threshold_low"] if cfg["enable_mesh_aggregation"] else None,
            "mesh_spectral_high": cfg["mesh_spectral_threshold_high"] if cfg["enable_mesh_aggregation"] else None,
            "spectral_factor": getattr(pipeline, "_fast_spectral_factor", None),
            "spectral_score": getattr(pipeline, "_fast_spectral_score", None),
            "raw_vertices": pipeline._last_raw_vertices,
            "raw_faces": pipeline._last_raw_faces,
            "vertices": exported["vertices"],
            "faces": exported["faces"],
            "texture_size": exported["texture_size"],
            "rotation": _tensor_list(output.get("rotation", None)),
            "translation": _tensor_list(output.get("translation", None)),
            "scale": _tensor_list(output.get("scale", None)),
            "gaussian_saved": False,
            "fallback": fallback,
            "profile": profile,
        }
        # Vertex-color output does not keep the Gaussian. A bake, if one was requested, has already read it.
        # 顶点色结果不保留高斯。若这次确实烘焙了贴图，高斯也已经用完。
        output.pop("gaussian", None)
        output.pop("gs", None)
        output.pop("gaussian_4", None)
        output.pop("gs_4", None)
        return {
            "mesh": exported["mesh"],
            "texture": exported["texture"],
            "metadata": metadata,
            "profile": profile,
        }

    def _install_timers(self, clock):
        """
        # Wrap the pipeline stages so their CUDA-synchronized durations accumulate on `clock`. Returns None.

            `self` is the `Sam3dSession`. The previous callables are saved on `self._timed` before the wrappers replace them.

            `clock` is a `StageClock`. The wrappers close over it and call `add`. It is not None.

        ---

        # 包装 pipeline 的各个阶段，把 CUDA 同步后的耗时累加到 `clock`。

            `self` 是 `Sam3dSession`。

            包装替换之前，原先的可调用对象保存在 `self._timed`。

            `clock` 是 `StageClock`。

            包装函数闭包住它并调用 `add`。

            不是 None。

        ## 返回

            - 返回 None。

"""
        pipeline = self.pipeline
        self._timed = {
            "compute_pointmap": pipeline.compute_pointmap,
            "preprocess_image": pipeline.preprocess_image,
            "get_condition_input": pipeline.get_condition_input,
            "sample_ss": pipeline.sample_sparse_structure,
            "pose_decoder": pipeline.pose_decoder,
            "sample_slat": pipeline.sample_slat,
            "decode": pipeline.decode_slat,
            "post": pipeline.postprocess_slat_output,
        }

        def compute_pointmap(self_pipeline, image, pointmap=None):
            """
            # Time the saved point-map call under the stage name `preprocess`, store its intrinsics, and return that call's output.

                `self_pipeline` receives `_last_intrinsics` from `out.get("intrinsics", None)`, so a missing key stores None.

                `image` and `pointmap` are forwarded to the saved `compute_pointmap`. `pointmap` defaults to None and is forwarded as given.

                This wrapper does not state their shapes.

            ---

            # 给保存的 point-map 调用计时，阶段名是 `preprocess`，保存它的内参，并返回那次调用的输出。

                `self_pipeline` 会从 `out.get("intrinsics", None)` 得到 `_last_intrinsics`，因此缺少这个键时存的是 None。

                `image` 和 `pointmap` 转给保存的 `compute_pointmap`。

                `pointmap` 默认 None，并按传入值转发。

                这个包装不写它们的形状。

"""
            t0 = _sync()
            out = self._timed["compute_pointmap"](image, pointmap)
            clock.add("preprocess", _sync() - t0)
            self_pipeline._last_intrinsics = out.get("intrinsics", None)
            return out

        def preprocess_image(self_pipeline, image, preprocessor, pointmap=None):
            """
            # Time the saved image preprocessor under the stage name `preprocess` and return its output.

                `self_pipeline` is the bound pipeline and is not read beyond the method binding.

                `image`, `preprocessor`, and `pointmap` are forwarded.

                `pointmap` defaults to None and is passed as the keyword `pointmap`. This wrapper does not state their shapes.

            ---

            # 给保存的图像预处理器计时，阶段名是 `preprocess`，并返回它的输出。

                `self_pipeline` 是绑定的 pipeline，除了方法绑定之外这里不读它。

                `image`、`preprocessor` 和 `pointmap` 会被转发。

                `pointmap` 默认 None，并以关键字 `pointmap` 传入。

                这个包装不写它们的形状。

"""
            t0 = _sync()
            out = self._timed["preprocess_image"](image, preprocessor, pointmap=pointmap)
            clock.add("preprocess", _sync() - t0)
            return out

        def get_condition_input(self_pipeline, condition_embedder, input_dict, input_mapping):
            """
            # Time the saved condition embedder.

                The stage is `condition_ss` while `_cond_phase` is `ss`, otherwise `condition_slat`. Returns the embedder output.

                `self_pipeline` supplies `_cond_phase`. It is not None.

                `condition_embedder`, `input_dict`, and `input_mapping` are forwarded to the saved function.

                This wrapper does not state their shapes.

                They are not None.

            ---

            # 给保存的条件嵌入计时。

                `_cond_phase` 为 `ss` 时阶段名是 `condition_ss`，否则是 `condition_slat`。

                返回特征编码器的输出。

                `self_pipeline` 提供 `_cond_phase`。

                不是 None。

                `condition_embedder`、`input_dict` 和 `input_mapping` 转给保存的函数。

                这个包装不写它们的形状。

                不是 None。

"""
            t0 = _sync()
            out = self._timed["get_condition_input"](
                condition_embedder, input_dict, input_mapping
            )
            name = "condition_ss" if self_pipeline._cond_phase == "ss" else "condition_slat"
            clock.add(name, _sync() - t0)
            return out

        def sample_ss(self_pipeline, ss_input_dict, inference_steps=None, use_distillation=False):
            """
            # Set the condition phase to `ss`, time the saved sparse-structure sampler under `sample_ss`, and return its output.

                `self_pipeline` receives `_cond_phase = "ss"`. It is not None.

                `ss_input_dict` is forwarded.

                `inference_steps` defaults to None and is forwarded as a keyword.

                `use_distillation` is a bool, default False, also forwarded.

                This wrapper does not state the dict layout.

            ---

            # 把条件阶段设为 `ss`，用阶段名 `sample_ss` 给保存的稀疏结构采样器计时，并返回它的输出。

                `self_pipeline` 会被写成 `_cond_phase = "ss"`。

                不是 None。

                `ss_input_dict` 会被转发。

                `inference_steps` 默认 None，按关键字转发。

                `use_distillation` 是布尔值，默认 False，同样转发。

                这个包装不写字典的布局。

"""
            self_pipeline._cond_phase = "ss"
            t0 = _sync()
            out = self._timed["sample_ss"](
                ss_input_dict,
                inference_steps=inference_steps,
                use_distillation=use_distillation,
            )
            clock.add("sample_ss", _sync() - t0)
            return out

        def pose_decoder(self_pipeline, *args, **kwargs):
            """
            # Time the saved pose decoder under the stage name `pose` and return its output.

                `self_pipeline` is the bound pipeline and is not read beyond the method binding.

                `args` and `kwargs` are forwarded to the saved `pose_decoder`. This wrapper does not state their shapes.

            ---

            # 用阶段名 `pose` 给保存的位姿解码器计时，并返回它的输出。

                `self_pipeline` 是绑定的 pipeline，除了方法绑定之外这里不读它。

                `args` 和 `kwargs` 转给保存的 `pose_decoder`。

                这个包装不写它们的形状。

"""
            t0 = _sync()
            out = self._timed["pose_decoder"](*args, **kwargs)
            clock.add("pose", _sync() - t0)
            return out

        def sample_slat(self_pipeline, slat_input, coords, inference_steps=25, use_distillation=False):
            """
            # Set the condition phase to `slat`, time the saved SLaT sampler under `sample_slat`, and return its output.

                `self_pipeline` receives `_cond_phase = "slat"`. It is not None.

                `slat_input` and `coords` are forwarded.

                `inference_steps` is an int, default 25.

                `use_distillation` is a bool, default False.

                Both are forwarded as keywords.

                This wrapper does not state the tensor shapes.

            ---

            # 把条件阶段设为 `slat`，用阶段名 `sample_slat` 给保存的 SLaT 采样器计时，并返回它的输出。

                `self_pipeline` 会被写成 `_cond_phase = "slat"`。

                不是 None。

                `slat_input` 和 `coords` 会被转发。

                `inference_steps` 是整数，默认 25。

                `use_distillation` 是布尔值，默认 False。

                二者都按关键字转发。

                这个包装不写张量形状。

"""
            self_pipeline._cond_phase = "slat"
            t0 = _sync()
            out = self._timed["sample_slat"](
                slat_input,
                coords,
                inference_steps=inference_steps,
                use_distillation=use_distillation,
            )
            clock.add("sample_slat", _sync() - t0)
            return out

        def decode_slat(self_pipeline, slat, formats=None):
            """
            # Return the timed decode dict from `_decode_timed`.

                `self_pipeline` is forwarded as the pipeline.

                It is not None.

                `slat` is forwarded to the mesh and Gaussian decoders inside `_decode_timed`. This wrapper does not state its shape.

                It is not None.

                `formats` defaults to None and is forwarded.

                `_decode_timed` replaces None with `pipeline.decode_formats`.

            ---

            # 返回 `_decode_timed` 的计时解码字典。

                `self_pipeline` 作为 pipeline 转发。

                不是 None。

                `slat` 会转到 `_decode_timed` 里的网格和高斯解码器。

                这个包装不写它的形状。

                不是 None。

                `formats` 默认 None 并被转发。

                `_decode_timed` 会把 None 换成 `pipeline.decode_formats`。

"""
            return _decode_timed(self_pipeline, slat, formats, clock)

        def postprocess(self_pipeline, outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color):
            """
            # Return the timed postprocess dict from `_postprocess_timed`, including this session's preset and clock.

                `self_pipeline`, `outputs`, `with_mesh_postprocess`, `with_texture_baking`, and `use_vertex_color` are forwarded in that order.

                This wrapper does not change them.

                They are not None.

            ---

            # 返回 `_postprocess_timed` 的计时后处理字典，并带上这个 session 的 preset 和时钟。

                `self_pipeline`、`outputs`、`with_mesh_postprocess`、`with_texture_baking` 和 `use_vertex_color` 按这个顺序转发。

                这个包装不改它们。

                不是 None。

"""
            return _postprocess_timed(
                self_pipeline,
                outputs,
                with_mesh_postprocess,
                with_texture_baking,
                use_vertex_color,
                self.cfg,
                clock,
            )

        pipeline.compute_pointmap = _bind(pipeline, compute_pointmap)
        pipeline.preprocess_image = _bind(pipeline, preprocess_image)
        pipeline.get_condition_input = _bind(pipeline, get_condition_input)
        pipeline.sample_sparse_structure = _bind(pipeline, sample_ss)
        pipeline.pose_decoder = _bind(pipeline, pose_decoder)
        pipeline.sample_slat = _bind(pipeline, sample_slat)
        pipeline.decode_slat = _bind(pipeline, decode_slat)
        pipeline.postprocess_slat_output = _bind(pipeline, postprocess)

    def _remove_timers(self):
        """
        # Restore the pipeline callables saved in `self._timed`. Returns None.

            `self` is the `Sam3dSession`. After a restore, `_timed` is set to None.

            There is no other argument.

        ## Returns

            - A missing or empty `_timed` returns immediately.

        ---

        # 恢复 `self._timed` 里保存的 pipeline 可调用对象。

            `self` 是 `Sam3dSession`。

            恢复之后 `_timed` 被设为 None。

            没有其他参数。

        ## 返回

            - 返回 None。
            - `_timed` 缺失或为空时立刻返回。

"""
        pipeline = self.pipeline
        timed = getattr(self, "_timed", None)
        if not timed:
            return
        pipeline.compute_pointmap = timed["compute_pointmap"]
        pipeline.preprocess_image = timed["preprocess_image"]
        pipeline.get_condition_input = timed["get_condition_input"]
        pipeline.sample_sparse_structure = timed["sample_ss"]
        pipeline.pose_decoder = timed["pose_decoder"]
        pipeline.sample_slat = timed["sample_slat"]
        pipeline.decode_slat = timed["decode"]
        pipeline.postprocess_slat_output = timed["post"]
        self._timed = None


def _sample_sparse_structure_fast(pipeline, ss_input_dict, inference_steps, use_distillation, cfg):
    """
    # Sample the local sparse structure, optionally merge voxels with the spectral factor, and return the decoder dict with `coords` and `downsample_factor`. Also stores scores and the spectral values on the pipeline.

        `pipeline` is the SAM 3D pipeline.

        It supplies the generators, the condition embedder, `shape_model_dtype`, and the source image attributes.

        `_fast_coords_scores`, `_fast_map_tokens`, `_fast_spectral_factor`, and `_fast_spectral_score` are written here.

        It is not None.

        `ss_input_dict` is a dict whose `image` tensor supplies the batch and the device.

        It is not None.

        `inference_steps` is forwarded to `ss_generator.inference_steps` only when it is true.

        None or another false value leaves the previous step count until the function restores it.

        `use_distillation` is a bool.

        True clears the generator shortcut flags and the reverse-function strengths.

        False restores the pipeline guidance strengths.

        It is not None.

        `cfg` is the preset dict.

        `enable_mesh_aggregation` and the two spectral thresholds decide whether `downsample_with_feature_fusion` runs.

        It is not None.

    ---

    # 采样本地稀疏结构，可选地按频谱倍数合并体素，并返回带 `coords` 和 `downsample_factor` 的解码字典。

        同时把分数和频谱数值存到 pipeline 上。

        `pipeline` 是 SAM 3D pipeline。

        它提供生成器、条件特征编码器、`shape_model_dtype` 和源图像属性。

        这里会写 `_fast_coords_scores`、`_fast_map_tokens`、`_fast_spectral_factor` 和 `_fast_spectral_score`。

        不是 None。

        `ss_input_dict` 是字典，其中的 `image` 张量提供 batch 和设备。

        不是 None。

        `inference_steps` 只有为真时才写入 `ss_generator.inference_steps`。

        None 或其他假值会保留原先的步数，直到函数结束时恢复。

        `use_distillation` 是布尔值。

        True 时清掉生成器的 shortcut 标志和反向函数强度。

        False 时恢复 pipeline 的引导强度。

        不是 None。

        `cfg` 是 preset 字典。

        `enable_mesh_aggregation` 和两个频谱阈值决定是否运行 `downsample_with_feature_fusion`。

        不是 None。

"""
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure,
        prune_sparse_structure,
    )
    import torch

    from fast_sam3d.acceleration import (
        downsample_with_feature_fusion,
        hfer_from_rgba,
        spectral_downsample_factor,
        voxel_frequency,
    )

    ss_generator = pipeline.models["ss_generator"]
    ss_decoder = pipeline.models["ss_decoder"]
    if use_distillation:
        ss_generator.no_shortcut = False
        ss_generator.reverse_fn.strength = 0
        ss_generator.reverse_fn.strength_pm = 0
    else:
        ss_generator.no_shortcut = True
        ss_generator.reverse_fn.strength = pipeline.ss_cfg_strength
        ss_generator.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm
    previous_steps = ss_generator.inference_steps
    if inference_steps:
        ss_generator.inference_steps = inference_steps
    image = ss_input_dict["image"]
    batch = image.shape[0]
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            if pipeline.is_mm_dit():
                latent_shape = {
                    key: (batch,) + (value.pos_emb.shape[0], value.input_layer.in_features)
                    for key, value in ss_generator.reverse_fn.backbone.latent_mapping.items()
                }
            else:
                latent_shape = (batch,) + (4096, 8)
            condition_args, condition_kwargs = pipeline.get_condition_input(
                pipeline.condition_embedders["ss_condition_embedder"],
                ss_input_dict,
                pipeline.ss_condition_input_mapping,
            )
            return_dict = ss_generator(
                latent_shape,
                image.device,
                *condition_args,
                **condition_kwargs,
            )
            if not pipeline.is_mm_dit():
                return_dict = {"shape": return_dict}
            shape_latent = return_dict["shape"]
            occupancy = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]].int()
            return_dict["coords_original"] = coords
            scores, hfer_3d = voxel_frequency(occupancy)
            factor = 1.0
            score = None
            if cfg["enable_mesh_aggregation"] and scores is not None and scores.shape[0] == coords.shape[0]:
                source_rgb = getattr(pipeline, "_source_rgb", None)
                source_mask = getattr(pipeline, "_source_mask", None)
                if source_rgb is None or source_mask is None:
                    source_rgb = (
                        image[0, :3].detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
                    )
                    source_mask = image[0, 3].detach().float().cpu().numpy()
                hfer_2d = hfer_from_rgba(source_rgb, source_mask)
                factor, score = spectral_downsample_factor(
                    hfer_2d,
                    hfer_3d,
                    cfg["mesh_spectral_threshold_low"],
                    cfg["mesh_spectral_threshold_high"],
                )
                coords, scores, downsample_factor = downsample_with_feature_fusion(
                    coords,
                    scores.to(coords.device),
                    downsample_factor=factor,
                )
            else:
                original_count = coords.shape[0]
                if pipeline.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=pipeline.downsample_ss_dist,
                    )
                coords, downsample_factor = downsample_sparse_structure(coords)
                if scores is None or scores.shape[0] != original_count or coords.shape[0] != original_count:
                    scores = None
            return_dict["coords"] = coords
            return_dict["downsample_factor"] = downsample_factor
    ss_generator.inference_steps = previous_steps
    pipeline._fast_coords_scores = None if scores is None else scores.detach()
    pipeline._fast_map_tokens = getattr(ss_generator, "k_map_shap", None)
    pipeline._fast_spectral_factor = float(factor)
    pipeline._fast_spectral_score = None if score is None else float(score)
    return return_dict


def _decode_timed(pipeline, slat, formats, clock):
    """
    # Decode the requested SLaT formats, time the mesh and Gaussian decoders, and return the dict of those outputs.

        `pipeline` is the SAM 3D pipeline.

        When `_bake_texture` is false, formats are forced to `["mesh"]`. A mesh result records `_last_raw_vertices` and `_last_raw_faces`. It is not None.

        `slat` is passed to `slat_decoder_mesh` and, when requested, to the Gaussian decoders.

        This function does not state its shape.

        It is not None.

        `formats` is a list of format names or None.

        None uses `pipeline.decode_formats` before the bake rule above.

        `clock` is a `StageClock`. Mesh decode time is added as `mesh_decoder`, and the first Gaussian decode as `gaussian_decoder`. It is not None.

    ---

    # 解码要求的 SLaT 格式，给网格和高斯解码器计时，并返回这些输出组成的字典。

        `pipeline` 是 SAM 3D pipeline。

        `_bake_texture` 为假时，格式被强制成 `["mesh"]`。

        网格结果会记下 `_last_raw_vertices` 和 `_last_raw_faces`。

        不是 None。

        `slat` 会传给 `slat_decoder_mesh`，需要时也传给高斯解码器。

        这个函数不写它的形状。

        不是 None。

        `formats` 是格式名列表或 None。

        None 会先用 `pipeline.decode_formats`，然后再套用上面的烘焙规则。

        `clock` 是 `StageClock`。

        网格解码时间记在 `mesh_decoder`，第一次高斯解码记在 `gaussian_decoder`。

        不是 None。

"""
    if formats is None:
        formats = pipeline.decode_formats
    # The UV bake is the only reader of the Gaussian. Vertex color comes from the mesh decoder.
    # 只有 UV 烘焙会读高斯。顶点颜色来自网格解码器。
    if not getattr(pipeline, "_bake_texture", False):
        formats = ["mesh"]
    ret = {}
    import torch

    with torch.no_grad():
        if "mesh" in formats:
            pipeline.models["slat_decoder_mesh"].map = getattr(pipeline, "_fast_map_tokens", None)
            t0 = _sync()
            ret["mesh"] = pipeline.models["slat_decoder_mesh"](slat)
            clock.add("mesh_decoder", _sync() - t0)
            mesh = ret["mesh"][0]
            pipeline._last_raw_vertices = int(mesh.vertices.shape[0])
            pipeline._last_raw_faces = int(mesh.faces.shape[0])
        if "gaussian" in formats:
            t0 = _sync()
            ret["gaussian"] = pipeline.models["slat_decoder_gs"](slat)
            clock.add("gaussian_decoder", _sync() - t0)
        if "gaussian_4" in formats:
            ret["gaussian_4"] = pipeline.models["slat_decoder_gs_4"](slat)
    return ret


def _postprocess_timed(
    pipeline, outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color, cfg, clock
):
    """
    # Time mesh simplification and texture baking, or build a vertex-color trimesh, and return `outputs` with `glb` set.

        `pipeline` supplies `rendering_engine` to `to_glb` when a Gaussian bake is requested.

        It is not None.

        `outputs` is a dict.

        A `mesh` entry may be replaced by quadric decimation when `cfg["target_vertices"]` is not None.

        A bake also requires `gaussian`. The returned dict may gain `glb`, `gs`, and `gs_4`. It is not None.

        `with_mesh_postprocess`, `with_texture_baking`, and `use_vertex_color` are the bools forwarded from the pipeline call.

        Quadric decimation forces the later mesh postprocess and hole fill off.

        `cfg` is the preset dict.

        `target_vertices` and `texture_size` are read from it.

        It is not None.

        `clock` is a `StageClock` for `mesh_postprocess` and `texture_baking`. It is not None.

    ---

    # 给网格简化和贴图烘焙计时，或者建立带顶点色的 trimesh，并返回写好 `glb` 的 `outputs`。

        `pipeline` 在需要高斯烘焙时，把 `rendering_engine` 提供给 `to_glb`。

        不是 None。

        `outputs` 是字典。

        当 `cfg["target_vertices"]` 不是 None 时，`mesh` 可能被二次误差简化替换。

        烘焙还需要 `gaussian`。

        返回的字典可能多出 `glb`、`gs` 和 `gs_4`。

        不是 None。

        `with_mesh_postprocess`、`with_texture_baking` 和 `use_vertex_color` 是 pipeline 调用转发来的布尔值。

        走了二次误差简化后，后面的网格后处理和补洞都会关掉。

        `cfg` 是 preset 字典。

        这里读取 `target_vertices` 和 `texture_size`。

        不是 None。

        `clock` 是记录 `mesh_postprocess` 和 `texture_baking` 的 `StageClock`。

        不是 None。

"""
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils

    simplify = 0.95
    fill_holes = True
    do_post = bool(with_mesh_postprocess)
    if cfg["target_vertices"] is not None and "mesh" in outputs:
        # Decimate before export. pyvista's ratio is the fraction removed;
        # the vertex target is applied with quadric edge collapse instead.
        # 先简化再导出。pyvista 的比例是删掉的面，顶点目标改走二次误差边折叠。
        t0 = _sync()
        outputs["mesh"] = [_quadric_vertex_budget(outputs["mesh"][0], int(cfg["target_vertices"]))]
        clock.add("mesh_postprocess", _sync() - t0)
        do_post = False
        fill_holes = False
    saved = {
        "postprocess_mesh": postprocessing_utils.postprocess_mesh,
        "parametrize_mesh": postprocessing_utils.parametrize_mesh,
        "render_multiview": postprocessing_utils.render_multiview,
        "bake_texture": postprocessing_utils.bake_texture,
    }

    def timed(name, fn):
        """
        # Return `inner`, a wrapper that times one saved postprocess callable under `name`.

            `name` is the string passed to `clock.add`, either `mesh_postprocess` or `texture_baking`. It is not None.

            `fn` is the saved callable.

            `inner` calls it with the arguments it receives.

            It is not None.

        ---

        # 返回 `inner`，它用 `name` 给一个保存的后处理可调用对象计时。

            `name` 是传给 `clock.add` 的字符串，取 `mesh_postprocess` 或 `texture_baking`。

            不是 None。

            `fn` 是保存的可调用对象。

            `inner` 会用自己收到的参数调用它。

            不是 None。

"""
        def inner(*args, **kwargs):
            """
            # Call `fn`, add the elapsed seconds to `clock` under `name`, and return `fn`'s result.

                `args` and `kwargs` are forwarded to the `fn` closed over by `timed`. This wrapper does not state their shapes.

            ---

            # 调用 `fn`，把经过的秒数按 `name` 加到 `clock`，并返回 `fn` 的结果。

                `args` 和 `kwargs` 转给 `timed` 闭包里的 `fn`。

                这个包装不写它们的形状。

"""
            t0 = _sync()
            out = fn(*args, **kwargs)
            clock.add(name, _sync() - t0)
            return out

        return inner

    postprocessing_utils.postprocess_mesh = timed("mesh_postprocess", saved["postprocess_mesh"])
    postprocessing_utils.parametrize_mesh = timed("texture_baking", saved["parametrize_mesh"])
    postprocessing_utils.render_multiview = timed("texture_baking", saved["render_multiview"])
    postprocessing_utils.bake_texture = timed("texture_baking", saved["bake_texture"])
    try:
        glb = None
        if "mesh" in outputs and with_texture_baking and "gaussian" in outputs:
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=simplify,
                fill_holes=fill_holes,
                texture_size=int(cfg["texture_size"]),
                verbose=False,
                with_mesh_postprocess=do_post,
                with_texture_baking=True,
                use_vertex_color=False,
                rendering_engine=pipeline.rendering_engine,
            )
        elif "mesh" in outputs:
            glb = _vertex_color_trimesh(outputs["mesh"][0], keep_color=bool(use_vertex_color))
        outputs["glb"] = glb
        if "gaussian" in outputs:
            outputs["gs"] = outputs["gaussian"][0]
        if "gaussian_4" in outputs:
            outputs["gs_4"] = outputs["gaussian_4"][0]
        return outputs
    finally:
        postprocessing_utils.postprocess_mesh = saved["postprocess_mesh"]
        postprocessing_utils.parametrize_mesh = saved["parametrize_mesh"]
        postprocessing_utils.render_multiview = saved["render_multiview"]
        postprocessing_utils.bake_texture = saved["bake_texture"]


def _boundary_edge_count(triangles):
    """
    # Return the number of triangle edges that occur once.

        An empty triangle array returns 0.

        `triangles` is an array indexed as rows of three vertex indices.

        Edges are the pairs `(0,1)`, `(1,2)`, and `(2,0)`, sorted per pair, then counted.

        It is not None.

        This function does not name a coordinate frame.

    ---

    # 返回只出现一次的三角形边的数量。

        三角形数组为空时返回 0。

        `triangles` 是数组，每一行是三个顶点下标。

        边是每一行的 `(0,1)`、`(1,2)` 和 `(2,0)`，每对先排序再计数。

        不是 None。

        这个函数不写坐标系。

"""
    if triangles.size == 0:
        return 0
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    span = int(edges[:, 1].max()) + 1
    keys = edges[:, 0].astype(np.int64) * span + edges[:, 1]
    _unique, counts = np.unique(keys, return_counts=True)
    return int(np.sum(counts == 1))


def _quadric_vertex_budget(mesh_result, target_vertices):
    """
    # Quadric-decimate a mesh above 15000 vertices and return a `MeshExtractResult`. A mesh that is already at or below 15000 vertices is returned unchanged.

        If every candidate fails the size check, the original mesh is returned.

        `mesh_result` has tensor `vertices`, `faces`, `vertex_attrs`, and `res`. Vertices and faces are moved to NumPy for Open3D.

        The device and dtype of the returned tensors follow `vertices`. It is not None.

        `target_vertices` is converted with `int`. The first candidate triangle count is `clip(target_vertices * 2, 10000, 30000)`, then 22000 and 30000 are tried while the boundary stays within 64 edges of the source.

        It is not None.

    ---

    # 对超过 15000 个顶点的网格做二次误差简化，并返回 `MeshExtractResult`。

        已经不超过 15000 个顶点的网格原样返回。

        若每个候选都过不了尺寸检查，也返回原网格。

        `mesh_result` 有张量 `vertices`、`faces`、`vertex_attrs` 和 `res`。

        顶点和面先转到 NumPy 再交给 Open3D。

        返回张量的设备和 dtype 跟随 `vertices`。

        不是 None。

        `target_vertices` 会用 `int` 转换。

        第一个候选三角形数是 `clip(target_vertices * 2, 10000, 30000)`，然后在边界边数不超过源边界加 64 的前提下再试 22000 和 30000。

        不是 None。

"""
    import open3d as o3d
    import torch
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import (
        MeshExtractResult,
    )

    vertices = mesh_result.vertices.detach().float().cpu().numpy()
    faces = mesh_result.faces.detach().cpu().numpy().astype(np.int32)
    count = int(vertices.shape[0])
    # Already inside the complex-object cap. Leave the triangulation alone.
    # 已经落在复杂物体的上限里，就不再改三角化。
    if count <= 15000:
        return mesh_result
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(vertices)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(faces)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    source_faces = np.asarray(mesh.triangles, dtype=np.int64)
    source_boundary = _boundary_edge_count(source_faces)
    # Do not delete non-manifold faces first. That step itself opens the surface.
    # 不先删非流形面，那一步自己就会把表面撕开。
    preferred = int(np.clip(int(target_vertices) * 2, 10000, 30000))
    candidates = []
    for triangles in (preferred, 22000, 30000):
        if triangles not in candidates and triangles < len(mesh.triangles):
            candidates.append(triangles)
    chosen = None
    chosen_boundary = None
    for triangles in candidates:
        simplified = mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(triangles),
            boundary_weight=1000.0,
        )
        new_vertices = np.asarray(simplified.vertices, dtype=np.float32)
        new_faces = np.asarray(simplified.triangles, dtype=np.int64)
        if new_vertices.shape[0] < 1000 or new_faces.shape[0] < 1000:
            continue
        opened = _boundary_edge_count(new_faces)
        # A closed box may gain a few edges from a non-manifold collapse.
        # 闭合盒子允许少量新边界；出现明显开孔时改用更密的目标。
        if opened <= source_boundary + 64:
            chosen = (new_vertices, new_faces)
            chosen_boundary = opened
            break
        if chosen is None or opened < chosen_boundary:
            chosen = (new_vertices, new_faces)
            chosen_boundary = opened
    if chosen is None:
        return mesh_result
    new_vertices, new_faces = chosen
    device = mesh_result.vertices.device
    verts_t = torch.tensor(new_vertices, device=device, dtype=mesh_result.vertices.dtype)
    faces_t = torch.tensor(new_faces, device=device, dtype=torch.long)
    # Quadric collapse does not carry colors. Each new vertex keeps the nearest old color.
    # 二次误差折叠不带走颜色。新顶点用最近的旧顶点颜色。
    attrs = _nearest_vertex_colors(vertices, mesh_result.vertex_attrs, new_vertices)
    attrs_t = torch.tensor(attrs, device=device, dtype=verts_t.dtype)
    return MeshExtractResult(verts_t, faces_t, vertex_attrs=attrs_t, res=mesh_result.res)


def _nearest_vertex_colors(old_vertices, old_attrs, new_vertices):
    """
    # Return float32 colors of shape `(M, 3)` for the collapsed vertices.

        A missing or mismatched color array returns ones of that shape.

        `old_vertices` is the NumPy vertex array from before the collapse, used as the KD-tree.

        Its row count is compared with the color rows.

        It is not None.

        `old_attrs` is a tensor of shape `(N, C)`, a 1D tensor, or None.

        None returns ones.

        A 1D tensor is reshaped to a column.

        Only the first three columns are kept.

        A row count other than `len(old_vertices)` returns ones.

        `new_vertices` is the NumPy array of collapsed vertices.

        `M` is `new_vertices.shape[0]`. Each row takes the color of the nearest old vertex.

        It is not None.

    ---

    # 返回折叠后顶点的 float32 颜色，形状 `(M, 3)`。

        颜色缺失或行数不一致时，返回这个形状的全 1。

        `old_vertices` 是折叠前的 NumPy 顶点数组，用来建 KD 树。

        它的行数会和颜色行数比较。

        不是 None。

        `old_attrs` 是形状 `(N, C)` 的张量、一维张量，或 None。

        None 返回全 1。

        一维张量会拉成一列。

        只保留前三列。

        行数不是 `len(old_vertices)` 时也返回全 1。

        `new_vertices` 是折叠后顶点的 NumPy 数组。

        `M` 是 `new_vertices.shape[0]`。

        每一行取最近旧顶点的颜色。

        不是 None。

"""
    count = int(new_vertices.shape[0])
    if old_attrs is None:
        return np.ones((count, 3), dtype=np.float32)
    colors = old_attrs.detach().float().cpu().numpy()
    if colors.ndim == 1:
        colors = colors.reshape(-1, 1)
    colors = np.asarray(colors[:, :3], dtype=np.float32)
    if colors.shape[0] != old_vertices.shape[0]:
        return np.ones((count, 3), dtype=np.float32)
    from scipy.spatial import cKDTree

    _distance, index = cKDTree(np.asarray(old_vertices, dtype=np.float64)).query(
        np.asarray(new_vertices, dtype=np.float64), k=1,
    )
    return colors[np.asarray(index, dtype=np.int64)]


def _vertex_color_trimesh(mesh_result, keep_color=True):
    """
    # Return a trimesh whose vertex colors come from the decoder attributes, after the z-up to y-up axis swap used by `to_glb`.

        `mesh_result` provides tensor `vertices`, `faces`, and `vertex_attrs`. Vertices and faces are detached to NumPy.

        The axis swap is the matrix `[[1,0,0],[0,0,-1],[0,1,0]]`. It is not None.

        `keep_color` is a bool, default True.

        False ignores `vertex_attrs` and uses uint8 white.

        Otherwise attributes are clipped to uint8 RGB, multiplying by 255 when the maximum is at most 1.5.

        A missing attribute array also uses white.

        Alpha is 255.

    ---

    # 返回带解码器顶点颜色的 trimesh，并做了与 `to_glb` 相同的 z-up 到 y-up 轴交换。

        `mesh_result` 提供张量 `vertices`、`faces` 和 `vertex_attrs`。

        顶点和面会 detach 成 NumPy。

        轴交换矩阵是 `[[1,0,0],[0,0,-1],[0,1,0]]`。

        不是 None。

        `keep_color` 是布尔值，默认 True。

        False 时忽略 `vertex_attrs`，使用 uint8 白色。

        否则属性会收成 uint8 RGB，最大值不超过 1.5 时先乘 255。

        没有属性数组时也用白色。

        Alpha 是 255。

"""
    import trimesh

    vertices = mesh_result.vertices.detach().float().cpu().numpy()
    faces = mesh_result.faces.detach().cpu().numpy()
    attrs = mesh_result.vertex_attrs if keep_color else None
    if attrs is None:
        rgb = np.full((vertices.shape[0], 3), 255, dtype=np.uint8)
    else:
        rgb = attrs.detach().float().cpu().numpy()
        if rgb.ndim == 1:
            rgb = np.repeat(rgb.reshape(-1, 1), 3, axis=1)
        rgb = rgb[:, :3]
        if float(np.nanmax(rgb)) <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    # Same axis swap as postprocessing_utils.to_glb.
    # 和 postprocessing_utils.to_glb 相同的坐标轴交换。
    vertices = vertices @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    mesh.visual.vertex_colors = np.concatenate([rgb, alpha], axis=1)
    return mesh


def _export_result(output, clock, pipeline):
    """
    # Read the exported trimesh and return a dict with `mesh`, `texture`, `vertices`, `faces`, and `texture_size`. The export stage time is added to `clock`.

        `output` is the pipeline result dict.

        `glb` may be None, in which case the counts stay 0 and `texture` stays None.

        A material image becomes `texture`, and a PIL image with `size` sets `texture_size` to `[width, height]`. It is not None.

        `clock` is a `StageClock`. The measured section is stored as `export`. It is not None.

        `pipeline` is accepted and not read by this function.

    ---

    # 读取导出的 trimesh，返回含 `mesh`、`texture`、`vertices`、`faces` 和 `texture_size` 的字典。

        导出阶段的时间会加到 `clock`。

        `output` 是 pipeline 的结果字典。

        `glb` 可以是 None，那时计数保持 0，`texture` 保持 None。

        材质图像成为 `texture`。

        带 `size` 的 PIL 图像会把 `texture_size` 设为 `[宽, 高]`。

        不是 None。

        `clock` 是 `StageClock`。

        测到的这一段记在 `export`。

        不是 None。

        `pipeline` 会被接收，但这个函数不读它。

"""
    mesh = output.get("glb", None)
    texture = None
    vertices = 0
    faces = 0
    texture_size = None
    t0 = _sync()
    if mesh is not None:
        vertices = int(len(mesh.vertices))
        faces = int(len(mesh.faces))
        material = getattr(getattr(mesh, "visual", None), "material", None)
        image = getattr(material, "baseColorTexture", None) if material is not None else None
        if image is None and material is not None:
            image = getattr(material, "image", None)
        texture = image
        if image is not None and hasattr(image, "size"):
            texture_size = [int(image.size[0]), int(image.size[1])]
    clock.add("export", _sync() - t0)
    return {
        "mesh": mesh,
        "texture": texture,
        "vertices": vertices,
        "faces": faces,
        "texture_size": texture_size,
    }


def _tensor_list(value):
    """
    # Return a flat Python list of floats, or None when `value` is None.

        `value` may be None, a torch tensor, or an array-like object.

        A tensor is detached, cast to float, moved to CPU, and reshaped to one dimension.

        Any other value goes through `numpy.asarray`, float64, and `reshape(-1)`.

    ---

    # 返回拉平的 Python 浮点列表。

        `value` 是 None 时返回 None。

        `value` 可以是 None、torch 张量，或其他可转成数组的对象。

        张量会 detach、转成 float、放到 CPU，再拉成一维。

        其他值走 `numpy.asarray`、float64 和 `reshape(-1)`。

"""
    if value is None:
        return None
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().reshape(-1).tolist()
    return np.asarray(value).astype(np.float64).reshape(-1).tolist()


def _silhouette_iou(mesh, mask, rotation, translation, scale, intrinsics):
    """
    # Rasterize the posed mesh and return its intersection-over-union with the mask as a float.

        Missing pose data, a mesh with more than 20000 faces, or a projection that leaves the frame returns None.

        A failure inside the try block also returns None.

        `mesh` is a trimesh or None.

        None returns None.

        Vertices are read as float32 and multiplied by the inverse of the z-up to y-up swap before the pose is applied.

        `mask` is an array.

        A 3D mask uses channel 0.

        Its height and width set the image size, and pixels `> 0` are the target after the same resize as the raster.

        It is not None.

        `rotation`, `translation`, and `scale` come from the pipeline output.

        Any None among them returns None.

        They are flattened by `_tensor_list`. A rotation whose first element is smaller in magnitude than its last is reordered from the last component to the front before `quaternion_to_matrix`. A one-element scale is repeated to three components.

        `intrinsics` is a 3×3 camera matrix, a tensor that can be detached to that shape, or None.

        None returns None.

        It projects the camera-frame vertices to pixel `u` and `v`.

    ---

    # 把摆好位姿的网格栅格化，返回它和 mask 的交并比，一个浮点数。

        位姿数据缺失、面数超过 20000，或投影离开画面时返回 None。

        try 块里失败时也返回 None。

        `mesh` 是 trimesh 或 None。

        None 直接返回 None。

        顶点按 float32 读取，并在施加位姿之前乘上 z-up 到 y-up 交换的逆变换。

        `mask` 是数组。

        三维 mask 用第 0 通道。

        它的高和宽决定图像尺寸，`> 0` 的像素在与光栅相同的缩放后作为目标。

        不是 None。

        `rotation`、`translation` 和 `scale` 来自 pipeline 输出。

        其中任何一个是 None 就返回 None。

        它们由 `_tensor_list` 拉平。

        若旋转的第一个分量绝对值小于最后一个，会把最后一个分量挪到前面，再交给 `quaternion_to_matrix`。

        只有一个元素的 scale 会重复成三个分量。

        `intrinsics` 是 3×3 相机矩阵、可以 detach 成这个形状的张量，或 None。

        None 返回 None。

        它把相机坐标系里的顶点投到像素 `u` 和 `v`。

"""
    if mesh is None or rotation is None or translation is None or scale is None or intrinsics is None:
        return None
    if len(mesh.faces) > 20000:
        return None
    try:
        import torch
        from pytorch3d.transforms import quaternion_to_matrix

        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        height, width = mask_np.shape[:2]
        verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
        faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64)
        # to_glb rotates z-up to y-up. The pose was predicted in the z-up frame.
        # to_glb 把 z-up 转到 y-up。位姿是在 z-up 坐标系里预测的。
        undo = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)
        verts = verts @ undo
        quat = torch.tensor(_tensor_list(rotation), dtype=torch.float32).reshape(-1)
        if quat.numel() == 4 and abs(float(quat[0])) < abs(float(quat[3])):
            quat = torch.stack([quat[3], quat[0], quat[1], quat[2]])
        rotation_m = quaternion_to_matrix(quat.reshape(1, 4))[0]
        scale_v = torch.tensor(_tensor_list(scale), dtype=torch.float32).reshape(-1)
        translation_v = torch.tensor(_tensor_list(translation), dtype=torch.float32).reshape(-1)
        if scale_v.numel() == 1:
            scale_v = scale_v.repeat(3)
        verts = verts * scale_v.reshape(1, 3)
        verts = verts @ rotation_m.transpose(0, 1) + translation_v.reshape(1, 3)
        k = intrinsics.detach().float().cpu() if hasattr(intrinsics, "detach") else torch.tensor(intrinsics)
        k = k.reshape(3, 3)
        z = verts[:, 2]
        u = k[0, 0] * verts[:, 0] / z.clamp(min=1e-4) + k[0, 2]
        v = k[1, 1] * verts[:, 1] / z.clamp(min=1e-4) + k[1, 2]
        finite = torch.isfinite(u) & torch.isfinite(v) & (z > 1e-4)
        in_frame = finite & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if float(in_frame.float().mean()) < 0.05:
            # The exported mesh frame and the predicted camera do not line up yet.
            # 导出网格的坐标系和预测相机仍不一致，这时不把 IoU 当成失败。
            return None
        rendered = _raster_silhouette(u, v, faces, width, height)
        scale = 160.0 / float(max(width, height))
        yy = np.clip((np.arange(rendered.shape[0]) / scale).astype(int), 0, height - 1)
        xx = np.clip((np.arange(rendered.shape[1]) / scale).astype(int), 0, width - 1)
        target = (mask_np > 0)[yy][:, xx]
        inter = np.logical_and(rendered, target).sum()
        union = np.logical_or(rendered, target).sum()
        if union <= 0:
            return 0.0
        return float(inter) / float(union)
    except Exception as exc:
        print("SILHOUETTE_IOU_FAIL", type(exc).__name__, exc, flush=True)
        return None


def _raster_silhouette(u, v, faces, width, height, max_side=160):
    """
    # Return a bool canvas of the projected triangles, scaled so the longer image side is `max_side` pixels.

        `u` and `v` are tensors of projected pixel coordinates.

        They are detached, moved to CPU, and multiplied by the scale.

        They are not None.

        `faces` is a tensor of triangle indices, detached to a NumPy integer array.

        It is not None.

        `width` and `height` are the full-image sizes in pixels.

        They set the canvas size after scaling, with a minimum of 8 on each side.

        They are not None.

        `max_side` is an int, default 160.

        The scale is `max_side / max(width, height)`. It is not None.

    ---

    # 返回投影三角形的布尔画布。

        缩放后，图像较长的一边是 `max_side` 像素。

        `u` 和 `v` 是投影后的像素坐标张量。

        它们会 detach、放到 CPU，再乘上缩放。

        不是 None。

        `faces` 是三角形下标张量，detach 成 NumPy 整数数组。

        不是 None。

        `width` 和 `height` 是整幅图像的像素尺寸。

        它们决定缩放后的画布大小，每一边至少 8。

        不是 None。

        `max_side` 是整数，默认 160。

        缩放是 `max_side / max(width, height)`。

        不是 None。

"""
    scale = max_side / float(max(width, height))
    small_w = max(8, int(round(width * scale)))
    small_h = max(8, int(round(height * scale)))
    uu = u.detach().cpu().numpy() * scale
    vv = v.detach().cpu().numpy() * scale
    tri = faces.detach().cpu().numpy()
    canvas = np.zeros((small_h, small_w), dtype=np.bool_)
    for f0, f1, f2 in tri:
        pts = np.stack([uu[[f0, f1, f2]], vv[[f0, f1, f2]]], axis=1)
        if not np.isfinite(pts).all():
            continue
        min_u = int(np.floor(pts[:, 0].min()))
        max_u = int(np.ceil(pts[:, 0].max()))
        min_v = int(np.floor(pts[:, 1].min()))
        max_v = int(np.ceil(pts[:, 1].max()))
        if max_u < 0 or max_v < 0 or min_u >= small_w or min_v >= small_h:
            continue
        min_u = max(min_u, 0)
        min_v = max(min_v, 0)
        max_u = min(max_u, small_w - 1)
        max_v = min(max_v, small_h - 1)
        xs = np.arange(min_u, max_u + 1)
        ys = np.arange(min_v, max_v + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        inside = _points_in_triangle(grid_x, grid_y, pts)
        canvas[min_v:max_v + 1, min_u:max_u + 1] |= inside
    return canvas


def _points_in_triangle(xs, ys, pts):
    """
    # Return a bool array of the same shape as `xs`, true where the grid point lies in the triangle.

        A near-zero denominator returns an all-false array.

        `xs` and `ys` are NumPy grids of pixel coordinates in the small canvas.

        They are not None.

        `pts` is a (3, 2) array of the triangle corners in the same pixel coordinates.

        The test uses barycentric coordinates.

        It is not None.

    ---

    # 返回与 `xs` 同形状的布尔数组，格点落在三角形内时为真。

        分母接近 0 时返回全假数组。

        `xs` 和 `ys` 是小画布上像素坐标的 NumPy 网格。

        不是 None。

        `pts` 是三角形三个角点的 (3, 2) 数组，坐标与像素网格相同。

        判断使用重心坐标。

        不是 None。

"""
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) < 1e-8:
        return np.zeros(xs.shape, dtype=np.bool_)
    a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
    b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
    c = 1.0 - a - b
    return (a >= 0) & (b >= 0) & (c >= 0)


def save_reconstruction(result, out_dir):
    """
    # Write `mesh.glb`, `mesh.obj` when export succeeds, `texture.png` when a texture image exists, and `metadata.json`. Returns a dict of those paths.

        A missing texture path is None, and a failed OBJ export stores None for `obj`.

        `result` is the dict from `reconstruct`. It must contain `mesh` and `metadata`. `texture` may be None.

        A mesh is exported to GLB and OBJ.

        A texture image is saved with its own `save`. It is not None.

        `out_dir` is a directory path created if needed.

        The four files are children of it.

        It is not None.

    ---

    # 写出 `mesh.glb`。

        OBJ 导出成功时写出 `mesh.obj`。

        有贴图图像时写出 `texture.png`。

        并写出 `metadata.json`。

        返回这些路径组成的字典。

        没有贴图时对应路径是 None。

        OBJ 导出失败时 `obj` 是 None。

        `result` 是 `reconstruct` 返回的字典。

        它必须含有 `mesh` 和 `metadata`。

        `texture` 可以是 None。

        网格会导出为 GLB 和 OBJ。

        贴图图像用它自己的 `save` 保存。

        不是 None。

        `out_dir` 是目录路径，不存在时会创建。

        那四个文件都放在它下面。

        不是 None。

"""
    os.makedirs(out_dir, exist_ok=True)
    mesh = result["mesh"]
    glb_path = os.path.join(out_dir, "mesh.glb")
    obj_path = os.path.join(out_dir, "mesh.obj")
    tex_path = os.path.join(out_dir, "texture.png")
    meta_path = os.path.join(out_dir, "metadata.json")
    if mesh is not None:
        mesh.export(glb_path)
        try:
            mesh.export(obj_path)
        except Exception:
            obj_path = None
    texture = result.get("texture", None)
    if texture is not None:
        texture.save(tex_path)
    elif os.path.isfile(tex_path):
        os.remove(tex_path)
        tex_path = None
    else:
        tex_path = None
    with open(meta_path, "w", encoding="utf-8") as stream:
        json.dump(result["metadata"], stream, indent=2)
    return {"glb": glb_path, "obj": obj_path, "texture": tex_path, "metadata": meta_path}
