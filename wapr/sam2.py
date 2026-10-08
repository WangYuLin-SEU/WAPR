# Author: Yulin Wang (yulinwang@seu.edu.cn)
# Copyright (c) 2026 Yulin Wang. SPDX-License-Identifier: LGPL-2.1-only
"""Run SAM2.1 through Ultralytics in the current WAPR interpreter.

通过 Ultralytics 在当前 WAPR 解释器运行 SAM2.1。
"""

import os

import numpy as np


def predict_mask(rgb, point_uv=None, box_xyxy=None, device="cuda", checkpoint=None):
    """Return the highest scoring large SAM2 mask, using one point or one box.

    返回 large SAM2 得分最高的 mask；只传一个像素点或一个像素框，保持原分割配方。
    """
    from wapr.bootstrap import ensure_optional
    ensure_optional("sam2")
    from ultralytics import SAM
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.models.sam.predict import SAM2Predictor
    from wapr.source_setup import prepare_sam2_weights
    if (point_uv is None) == (box_xyxy is None):
        raise ValueError("Provide exactly one point or box / 必须只提供一个点或框")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB HWC / 需要 uint8 RGB HWC")
    checkpoint = checkpoint or prepare_sam2_weights("large")["checkpoint"]
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    class CandidatePredictor(SAM2Predictor):
        # Keep the original three-candidate selection for point and box prompts.
        # 点与框提示继续使用原先的三个候选，再按质量分选择。
        def inference(self, im, bboxes=None, points=None, labels=None,
                      masks=None, multimask_output=False, *args, **kwargs):
            return super().inference(im, bboxes=bboxes, points=points,
                                     labels=labels, masks=masks,
                                     multimask_output=True, *args, **kwargs)

    model = SAM(checkpoint)
    # Keep FP32 across old and new Ultralytics precision APIs.
    # 兼容新旧 Ultralytics 精度接口；均保持原有 FP32 推理。
    precision = {"quantize": 32} if "quantize" in DEFAULT_CFG_DICT else {"half": False}
    prompts = ({"points": np.asarray(point_uv).reshape(1, 2).tolist(), "labels": [1]}
               if point_uv is not None else {"bboxes": np.asarray(box_xyxy).reshape(1, 4).tolist()})
    # Ultralytics ndarray input is BGR; masks retain the original pixel grid.
    # Ultralytics 的 ndarray 输入是 BGR；掩码保留原始像素网格。
    result = model.predict(source=np.ascontiguousarray(rgb[..., ::-1]), device=device,
                           predictor=CandidatePredictor,
                           retina_masks=True, verbose=False, **precision, **prompts)[0]
    if result.masks is None or len(result.masks.data) == 0:
        raise RuntimeError("SAM2 returned no mask / SAM2 未返回掩码")
    scores = result.boxes.conf if result.boxes is not None else None
    pick = int(scores.argmax().item()) if scores is not None else 0
    mask = result.masks.data[pick].detach().cpu().numpy()
    if mask.shape != rgb.shape[:2]:
        raise RuntimeError("SAM2 mask shape mismatch / SAM2 掩码尺寸不匹配")
    return (mask > 0).astype(np.uint8)
