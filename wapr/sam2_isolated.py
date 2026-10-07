# Author: Yulin Wang (yulinwang@seu.edu.cn)
# Copyright (c) 2026 Yulin Wang. SPDX-License-Identifier: LGPL-2.1-only
"""Run SAM2 without replacing the pose process's Torch.

独立进程运行 SAM2，不替换位姿进程的 Torch。
"""

import json
import os
import subprocess
import sys
import tempfile


def prepare_environment(allow_replacement=None, check_only=False):
    """Prepare the shared independent SAM prefix and its SAM2 source.

    准备共享的独立 SAM 前缀以及 SAM2 源码；基础环境保持不变。
    """
    from wapr import sam3d_isolated
    prepared = sam3d_isolated.prepare_environment(allow_replacement, check_only)
    if prepared["status"] not in ("ready", "installed"):
        return prepared
    with tempfile.TemporaryDirectory(prefix="wapr-sam2-prepare-") as directory:
        path = os.path.join(directory, "request.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump({"allow_replacement": allow_replacement, "check_only": check_only}, stream)
        environment = os.environ.copy()
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        result = sam3d_isolated._run_preparation_worker("prepare", path, environment,
                                                       worker_file=os.path.abspath(__file__))
        result["bootstrap"] = prepared
        return result


def predict_mask(rgb, point_uv=None, box_xyxy=None, device="cuda", checkpoint=None,
                 config="configs/sam2.1/sam2.1_hiera_l.yaml"):
    """Return the highest scoring large SAM2 mask, using one point or one box.

    返回 large SAM2 得分最高的 mask；只传一个像素点或一个像素框，保持原分割配方。
    """
    import numpy as np
    from wapr.sam3d_isolated import SAM3D_PYTHON
    if (point_uv is None) == (box_xyxy is None):
        raise ValueError("Provide exactly one point or box / 必须只提供一个点或框")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB HWC / 需要 uint8 RGB HWC")
    prepared = prepare_environment()
    if prepared["status"] not in ("ready", "installed"):
        raise RuntimeError("SAM2 independent preparation / SAM2 独立准备: " + json.dumps(prepared, ensure_ascii=False))
    with tempfile.TemporaryDirectory(prefix="wapr-sam2-infer-") as directory:
        input_path = os.path.join(directory, "input.npz")
        output_path = os.path.join(directory, "mask.npy")
        request_path = os.path.join(directory, "request.json")
        np.savez(input_path, rgb=rgb)
        request = {"input_path": input_path, "output_path": output_path, "device": device,
                   "checkpoint": checkpoint, "config": config,
                   "point_uv": None if point_uv is None else np.asarray(point_uv).reshape(2).tolist(),
                   "box_xyxy": None if box_xyxy is None else np.asarray(box_xyxy).reshape(4).tolist()}
        with open(request_path, "w", encoding="utf-8") as stream:
            json.dump(request, stream)
        environment = os.environ.copy()
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        from wapr.sam3d_isolated import _worker_cache_environment
        environment = _worker_cache_environment(environment)
        completed = subprocess.run([SAM3D_PYTHON, os.path.abspath(__file__), "infer", request_path],
                                   env=environment, check=False)
        if completed.returncode != 0:
            raise RuntimeError("SAM2 independent inference failed / SAM2 独立推理失败: " + str(completed.returncode))
        mask = np.load(output_path, allow_pickle=False)
        if mask.shape != rgb.shape[:2] or mask.dtype != np.uint8:
            raise RuntimeError("SAM2 returned an invalid mask / SAM2 返回无效 mask")
        return mask


def _worker(mode, path):
    """Prepare or infer inside the independent interpreter only.

    仅在独立解释器内准备或推理；数据交换不用 pickle。
    """
    with open(path, encoding="utf-8") as stream:
        request = json.load(stream)
    if mode == "prepare":
        from wapr.sam3d_isolated import _prepare_cuda_toolchain
        toolchain = _prepare_cuda_toolchain(check_only=request["check_only"])
        if toolchain["status"] not in ("ready", "needs_toolchain"):
            print(json.dumps(toolchain, ensure_ascii=False), flush=True)
            return 1
        from wapr.bootstrap import prepare_feature
        result = prepare_feature("sam2", allow_replacement=request["allow_replacement"],
                                 check_only=request["check_only"])
        result["toolchain"] = toolchain
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["status"] in ("ready", "installed", "native_build_required") else 1
    import numpy as np
    from wapr.bootstrap import ensure_optional
    ensure_optional("sam2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    checkpoint = request["checkpoint"]
    if checkpoint is None or not os.path.isfile(checkpoint):
        from wapr.source_setup import prepare_sam2_weights
        # An explicitly missing custom file is an error, never silently substituted.
        # 明确指定但缺失的自定义文件报错，不静默替换。
        if checkpoint is not None:
            raise FileNotFoundError(checkpoint)
        checkpoint = prepare_sam2_weights("large")["checkpoint"]
    rgb = np.load(request["input_path"], allow_pickle=False)["rgb"]
    model = build_sam2(request["config"], checkpoint, device=request["device"])
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(rgb)
    if request["point_uv"] is not None:
        masks, scores, _ = predictor.predict(point_coords=np.asarray(request["point_uv"], dtype=np.float32).reshape(1, 2),
                                             point_labels=np.ones(1, dtype=np.int32), multimask_output=True)
    else:
        masks, scores, _ = predictor.predict(box=np.asarray(request["box_xyxy"], dtype=np.float32), multimask_output=True)
    best = int(np.argmax(scores))
    mask = (masks[best] > 0).astype(np.uint8)
    print("SAM", int(mask.sum()), "score", round(float(scores[best]), 3), flush=True)
    np.save(request["output_path"], mask, allow_pickle=False)
    return 0


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[1], sys.argv[2]))
