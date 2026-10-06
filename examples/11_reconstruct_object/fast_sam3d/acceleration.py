# WAPR integration and modifications: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang for WAPR integration and modifications.
# All rights reserved, except as granted under applicable licenses.
# 作者署名对应 WAPR 集成与修改；上游作者署名与条款继续适用。
# SAM 3D / Fast-SAM3D attribution and terms: THIRD_PARTY_NOTICES.txt and the accompanying LICENSE.

"""Fast-SAM3D inference hooks on the local SAM 3D generators.

Fast-SAM3D 的推理加速，挂在本地 SAM 3D 的生成器上。

The math follows the training-free Fast-SAM3D modules (step cache, token
carving, spectral voxel aggregation). No weights are trained or rewritten.
Upstream Fast-SAM3D forks an older SAM 3D tree. These hooks call the local
generator and solver instead of replacing that tree.
公式按 Fast-SAM3D 的免训练模块：步缓存、token carving、频谱体素聚合。
不训练，也不改权重。官方仓库分叉的是更老的 SAM 3D。这里调用本地生成器，
不覆盖那套源码。
"""

import math

import numpy as np
import torch

from sam3d_objects.model.backbone.generator.flow_matching.solver import (
    ODESolver,
    linear_approximation_step,
)


# Sparse-structure step cache. Order 1 is a linear extrapolation of the velocity.
# 稀疏结构的步缓存。order 1 是速度的线性外推。
def faster_init(num_steps, faster_interval=3, max_order=1, first_enhance=2, end_enhance=24):
    """
    # Build the sparse-structure step-cache state and the step counter.

        `num_steps` is an int stored as `current["num_steps"]`. It is the planned number of solver steps.

        It is not a tensor and is not None.

        `faster_interval` is an int, default 3.

        A full network step is forced when the counter reaches `interval - 1`. It is not a tensor and is not None.

        `max_order` is an int, default 1.

        It is the highest finite-difference order stored later by `derivative_approximation`. It is not a tensor and is not None.

        `first_enhance` is an int, default 2.

        Steps with `current["step"]` below this value are full steps.

        It is not a tensor and is not None.

        `end_enhance` is an int, default 24.

        Steps at or after this index are full steps.

        It is not a tensor and is not None.

    ## Returns

        - Returns the two dicts.

    ---

    # 建立稀疏结构步缓存的状态和步计数。

        `num_steps` 是整数，存进 `current["num_steps"]`。

        它是计划中的求解步数。

        不是张量，也不是 None。

        `faster_interval` 是整数，默认 3。

        计数到 `interval - 1` 时强制走完整网络。

        不是张量，也不是 None。

        `max_order` 是整数，默认 1。

        它是 `derivative_approximation` 之后保存的最高差分阶。

        不是张量，也不是 None。

        `first_enhance` 是整数，默认 2。

        `current["step"]` 小于它的步是完整步。

        不是张量，也不是 None。

        `end_enhance` 是整数，默认 24。

        下标大于等于它的步是完整步。

        不是张量，也不是 None。

    ## 返回

        - 返回这两个字典。

"""
    faster_dic = {
        "faster_counter": 0,
        "faster_interval": int(faster_interval),
        "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance),
        "faster_enabled": True,
        "cache": {-1: {"final": {"final": {"final": {}}}}},
    }
    current = {
        "activated_steps": [],
        "step": 0,
        "num_steps": int(num_steps),
    }
    return faster_dic, current


def faster_cal_type(faster_dic, current):
    """
    # Choose a full network step or a cached step and write `current["type"]`. Returns None.

        `faster_dic` is the dict from `faster_init`. This function reads `first_enhance`, `end_enhance`, `faster_interval`, `faster_enabled`, and `faster_counter`, and sets `faster_counter` back to 0 on a full step.

        It is not a tensor and is not None.

        `current` is the step dict from `faster_init`. This function reads `step` and appends it to `activated_steps` on a full step.

        It sets `type` to the string `full` or `faster`. It is not a tensor and is not None.

        An enabled cache that is neither a warmup, an interval, nor an end step becomes `faster`. Any other state raises ValueError.

    ---

    # 选择完整网络步或缓存步，并写入 `current["type"]`。

        `faster_dic` 是 `faster_init` 返回的字典。

        这里读取 `first_enhance`、`end_enhance`、`faster_interval`、`faster_enabled` 和 `faster_counter`，完整步会把 `faster_counter` 置回 0。

        不是张量，也不是 None。

        `current` 是 `faster_init` 返回的步字典。

        这里读取 `step`，完整步时把它追加到 `activated_steps`。

        `type` 被写成字符串 `full` 或 `faster`。

        不是张量，也不是 None。

        缓存开着、又不属于预热、间隔或收尾时，类型是 `faster`。

        其余状态抛出 ValueError。

    ## 返回

        - 返回 None。

"""
    first_steps = current["step"] < faster_dic["first_enhance"]
    end_steps = current["step"] >= faster_dic["end_enhance"]
    interval = faster_dic["faster_interval"]
    if first_steps or faster_dic["faster_counter"] == interval - 1 or end_steps:
        current["type"] = "full"
        faster_dic["faster_counter"] = 0
        current["activated_steps"].append(current["step"])
    elif faster_dic["faster_enabled"]:
        faster_dic["faster_counter"] += 1
        current["type"] = "faster"
    else:
        raise ValueError("unsupported sparse-structure cache state")


def derivative_approximation(faster_dic, current, feature):
    """
    # Store finite-difference orders of the latest full-step feature in the step cache.

        `faster_dic` is the cache dict.

        `max_order` limits the orders, `first_enhance` limits which steps may use a previous order, and `cache[-1]` holds the per-layer modules.

        It is not a tensor and is not None.

        `current` is the step dict.

        It must already contain `activated_steps`, `step`, `num_steps`, `layer`, and `module`. With fewer than two activated steps the difference distance is 1.0.

        It is not a tensor and is not None.

        `feature` is a tensor, or a dict of tensors.

        A tensor is stored under the key `default`. Each value becomes a dict of order to tensor.

        None is not accepted.

    ## Returns

        - Returns None.

    ---

    # 把最近一次完整步特征的各阶差分写进步缓存。

        `faster_dic` 是缓存字典。

        `max_order` 限制阶数，`first_enhance` 限制哪些步能用上一阶，`cache[-1]` 按层保存模块。

        不是张量，也不是 None。

        `current` 是步字典。

        它必须已经有 `activated_steps`、`step`、`num_steps`、`layer` 和 `module`。

        已激活步少于两步时，差分距离用 1.0。

        不是张量，也不是 None。

        `feature` 是一个张量，或张量字典。

        单个张量会放在键 `default` 下。

        每个值变成阶到张量的字典。

        不接受 None。

    ## 返回

        - 返回 None。

"""
    if len(current["activated_steps"]) < 2:
        difference_distance = 1.0
    else:
        difference_distance = (
            current["activated_steps"][-1] - current["activated_steps"][-2]
        )
    if isinstance(feature, torch.Tensor):
        feature = {"default": feature}
    cache_root = faster_dic["cache"][-1]
    prev_module_cache = cache_root[current["layer"]][current["module"]]
    updated = {}
    for key, tensor_val in feature.items():
        factors = {0: tensor_val}
        prev_key_cache = prev_module_cache.get(key, None) if prev_module_cache else None
        for order in range(faster_dic["max_order"]):
            has_prev = prev_key_cache is not None and order in prev_key_cache
            within = current["step"] < (
                current["num_steps"] - faster_dic["first_enhance"] + 1
            )
            if has_prev and within:
                factors[order + 1] = (factors[order] - prev_key_cache[order]) / difference_distance
            else:
                break
        updated[key] = factors
    layer_cache = cache_root.setdefault(current["layer"], {})
    layer_cache[current["module"]] = updated


def faster_formula(faster_dic, current, prev_v, beta=0.5):
    """
    # Extrapolate the cached sparse-structure feature with a Taylor sum in the step distance.

        `faster_dic` is the cache dict.

        The entry at `cache[-1][layer][module]` supplies the factors.

        It is not a tensor and is not None.

        `current` is the step dict.

        The distance is `step` minus the last `activated_steps` entry.

        `layer` and `module` select the cache entry.

        It is not a tensor and is not None.

        `prev_v` is accepted and then deleted.

        This function does not read it.

        The published linear result does not use the previous velocity argument.

        `beta` is a float, default 0.5.

        It is deleted and does not enter the sum.

        The preset still passes `ss_momentum_beta` here.

        None is not used.

    ## Returns

        - Returns that tensor, or a dict of tensors when the cache entry is a dict.

    ---

    # 用步距上的泰勒和，外推缓存里的稀疏结构特征。

        `faster_dic` 是缓存字典。

        `cache[-1][layer][module]` 提供各阶系数。

        不是张量，也不是 None。

        `current` 是步字典。

        距离是 `step` 减去 `activated_steps` 的最后一项。

        `layer` 和 `module` 用来选缓存项。

        不是张量，也不是 None。

        `prev_v` 被收下后随即删除。

        这个函数不读它。

        实际执行的线性结果不用这个上一速度参数。

        `beta` 是浮点数，默认 0.5。

        它被删除，不进入求和。

        preset 仍会把 `ss_momentum_beta` 传到这里。

        不使用 None。

    ## 返回

        - 缓存项是字典时返回张量字典，否则返回张量。

"""
    del beta
    distance = current["step"] - current["activated_steps"][-1]
    module_cache = faster_dic["cache"][-1][current["layer"]][current["module"]]

    def expand(factors, x_dist):
        """
        # Sum `factors[order] * distance ** order / order!` over the stored orders.

            `factors` is a dict whose keys are orders 0, 1, ... and whose values are tensors added into the sum.

            It is not None.

            `x_dist` is the step distance passed by `faster_formula`, `current["step"]` minus the last activated step.

            It is a Python number, not a batch tensor, and it is not None.

        ## Returns

            - Returns that total.

        ---

        # 对已保存的阶计算 `factors[order] * distance ** order / order!` 并求和。

            `factors` 是字典，键是 0、1、… 阶，值是加进求和的张量。

            不是 None。

            `x_dist` 是 `faster_formula` 传入的步距，即 `current["step"]` 减去最后一次激活步。

            它是 Python 数值，不是批量张量，也不是 None。

        ## 返回

            - 返回这个和。

"""
        total = 0
        for order in range(len(factors)):
            total = total + (1.0 / math.factorial(order)) * factors[order] * (x_dist ** order)
        return total

    first = next(iter(module_cache.values()))
    if isinstance(first, dict):
        return {key: expand(factors, distance) for key, factors in module_cache.items()}
    return expand(module_cache, distance)


def ss_cache_dynamics(generator, x_t, t, d, *args_conditionals, **kwargs_conditionals):
    """
    # Take one sparse-structure step.

        A full step runs `generator.reverse_fn` and updates the cache.

        A faster step returns the extrapolated velocity.

        Returns that velocity.

        `generator` is the sparse-structure generator.

        It must have `time_scale`, `faster_dic`, `current`, `reverse_fn`, `prev_v`, and `ss_params["ss_momentum_beta"]`. This function increments `current["step"]` and stores the velocity on `prev_v`. It is not None.

        `x_t` is the current sample tensor forwarded to `reverse_fn` on a full step.

        Its shape is whatever the generator produced.

        It is not None.

        `t` and `d` are scalars.

        Each is multiplied by `generator.time_scale` and stored as a float32 tensor of shape (1,) on the device of `x_t`. They are not None.

        `args_conditionals` and `kwargs_conditionals` are forwarded to `reverse_fn` on a full step.

        This function does not fix their shapes.

        On a cached step they are not read.

    ---

    # 走一步稀疏结构。

        完整步调用 `generator.reverse_fn` 并更新缓存。

        加速步返回外推速度。

        返回这个速度。

        `generator` 是稀疏结构生成器。

        它必须有 `time_scale`、`faster_dic`、`current`、`reverse_fn`、`prev_v` 和 `ss_params["ss_momentum_beta"]`。

        这里会把 `current["step"]` 加一，并把速度写到 `prev_v`。

        不是 None。

        `x_t` 是当前样本张量，完整步会把它传给 `reverse_fn`。

        形状由生成器给出。

        不是 None。

        `t` 和 `d` 是标量。

        各自乘上 `generator.time_scale`，变成 `x_t` 所在设备上形状为 (1,) 的 float32 张量。

        不是 None。

        `args_conditionals` 和 `kwargs_conditionals` 在完整步转给 `reverse_fn`。

        这个函数不规定它们的形状。

        缓存步不读它们。

"""
    from sam3d_objects.model.backbone.generator.flow_matching.model import _get_device

    device = _get_device(x_t)
    t_tensor = torch.tensor([t * generator.time_scale], device=device, dtype=torch.float32)
    d_tensor = torch.tensor([d * generator.time_scale], device=device, dtype=torch.float32)
    faster_cal_type(generator.faster_dic, generator.current)
    generator.current["stream"] = "final"
    generator.current["layer"] = "final"
    generator.current["module"] = "final"
    if generator.current["type"] == "full":
        velocity = generator.reverse_fn(
            x_t, t_tensor, *args_conditionals, d=d_tensor, **kwargs_conditionals
        )
        generator.prev_v = velocity
        derivative_approximation(generator.faster_dic, generator.current, velocity)
    else:
        velocity = faster_formula(
            generator.faster_dic,
            generator.current,
            generator.prev_v,
            beta=generator.ss_params["ss_momentum_beta"],
        )
        generator.prev_v = velocity
    generator.current["step"] += 1
    return velocity


# SLaT cache. A full step runs when the accumulated input/output error crosses thresh.
# SLaT 缓存。累计的输入/输出误差超过阈值时才再跑完整一步。
def slat_cache_init(num_steps):
    """
    # Build the SLaT error-cache dict and the step dict used by `EulerFasterSLat`. Returns the two dicts.

        `num_steps` is an int stored as `current["num_steps"]`. The cache starts with `thresh` 1.0, `dir_weight` 0.5, `max_order` 2, and `first_enhance` 1.

        Tensor slots such as `prev_x` and `prev_v` start as None.

        It is not None.

    ---

    # 建立 `EulerFasterSLat` 用的 SLaT 误差缓存字典和步字典。

        `num_steps` 是整数，存进 `current["num_steps"]`。

        缓存初始 `thresh` 为 1.0，`dir_weight` 为 0.5，`max_order` 为 2，`first_enhance` 为 1。

        `prev_x`、`prev_v` 等张量槽一开始是 None。

        这个参数不是 None。

    ## 返回

        - 返回这两个字典。

"""
    return {
        "thresh": 1.0,
        "dir_weight": 0.5,
        "faster_counter": 0,
        "faster_enabled": True,
        "max_order": 2,
        "first_enhance": 1,
        "cache": {
            "k": None,
            "prev_x": None,
            "prev_v": None,
            "prev_prev_x": None,
            "error": None,
            "feature": None,
            "easy": None,
        },
    }, {
        "type": None,
        "activated_steps": [],
        "step": 0,
        "num_steps": int(num_steps),
        "use_token": True,
        "is_token_active": False,
        "num_to_skip": 0,
        "cache_indices": None,
        "fast_update_indices": None,
    }


def slat_should_run_full(faster_dic, current, x_t):
    """
    # Decide whether this SLaT step runs the network.

        `faster_dic` is the SLaT cache dict.

        Warmup steps below `first_enhance` are full.

        Later, `cache["error"]` accumulates `k` times the mean absolute input change over the mean absolute previous velocity.

        A full step is required when that error reaches `thresh`, and the stored error is then set to 0.

        It is not None.

        `current` is the step dict.

        A full decision appends `step` to `activated_steps` and clears `faster_counter`. A cached decision increments `faster_counter`. It is not None.

        `x_t` is the current SLaT tensor.

        It is subtracted from `cache["prev_x"]` when a previous state exists.

        Its shape is the generator state.

        It is not None.

        Before history exists, the step is full.

    ## Returns

        - Returns that bool, and writes `current["type"]` as `full` or `faster`.

    ---

    # 判断这一步 SLaT 要不要跑网络。

        `faster_dic` 是 SLaT 缓存字典。

        低于 `first_enhance` 的预热步是完整步。

        之后 `cache["error"]` 累加 `k` 乘以输入绝对变化的均值，再除以上一速度绝对值的均值。

        误差达到 `thresh` 时必须完整计算，并把保存的误差置 0。

        不是 None。

        `current` 是步字典。

        判定为完整步时，把 `step` 追加到 `activated_steps`，并把 `faster_counter` 清零。

        判定为缓存步时，`faster_counter` 加一。

        不是 None。

        `x_t` 是当前 SLaT 张量。

        已有上一状态时，用它减去 `cache["prev_x"]`。

        形状是生成器状态。

        不是 None。

        还没有历史时，这一步是完整步。

    ## 返回

        - 返回这个布尔值，并把 `current["type"]` 写成 `full` 或 `faster`。

"""
    state = faster_dic["cache"]
    if current["step"] < faster_dic["first_enhance"]:
        should_calc = True
    else:
        has_history = (
            state["prev_x"] is not None
            and state["prev_v"] is not None
            and state["prev_prev_x"] is not None
        )
        should_calc = True
        if has_history and state["k"] is not None:
            delta_x = x_t - state["prev_x"]
            input_change = delta_x.abs().mean()
            output_norm = state["prev_v"].abs().mean() + 1e-6
            state["error"] = state["error"] + state["k"] * (input_change / output_norm)
            should_calc = bool(state["error"] >= faster_dic["thresh"])
    if should_calc:
        state["error"] = 0
        current["type"] = "full"
        faster_dic["faster_counter"] = 0
        current["activated_steps"].append(current["step"])
    else:
        current["type"] = "faster"
        faster_dic["faster_counter"] += 1
    return should_calc


class TokenLeader:
    """
    # Gates token carving until the warmup steps have finished.

    ---

    # token carving 在预热步数走完之前不打开。
    """

    def __init__(self):
        """
        # Store the warmup gate for SLaT token carving.

            `self` is the new `TokenLeader`. `full_sampling_steps` is 2, `current_step` is 0, and `total_tokens` is 0.

            None is not used.

        ## Returns

            - Returns None.

        ---

        # 保存 SLaT token carving 的预热门闩。

            `self` 是新建的 `TokenLeader`。

            `full_sampling_steps` 为 2，`current_step` 为 0，`total_tokens` 为 0。

            不使用 None。

        ## 返回

            - 返回 None。

"""
        self.full_sampling_steps = 2
        self.current_step = 0
        self.total_tokens = 0

    def reset(self):
        """
        # Set `current_step` back to 0.

            `self` is the `TokenLeader`. `total_tokens` is left unchanged.

            There is no other argument and no None case.

        ## Returns

            - Returns None.

        ---

        # 把 `current_step` 设回 0。

            `self` 是 `TokenLeader`。

            `total_tokens` 保持不变。

            没有其他参数，也没有 None 情形。

        ## 返回

            - 返回 None。

"""
        self.current_step = 0

    def increase_step(self):
        """
        # Add one to `current_step`. Returns None.

            `self` is the `TokenLeader`. Carving in `solve_iter` waits until `current_step` reaches `full_sampling_steps`. There is no other argument and no None case.

        ---

        # 把 `current_step` 加一。

            `self` 是 `TokenLeader`。

            `solve_iter` 里的 carving 会等到 `current_step` 达到 `full_sampling_steps`。

            没有其他参数，也没有 None 情形。

        ## 返回

            - 返回 None。

"""
        self.current_step += 1


class StabilityTracker:
    """
    # Pick the least-changing SLaT tokens to reuse.

    ---

    # 选出变化最小的 SLaT token，这一步不再送进网络。
    """

    def __init__(self):
        """
        # Prepare empty SLaT token-stability state.

            `self` is the new `StabilityTracker`. `num_tokens` starts at 0, `device` is CPU, and `cached_streak_counter`, `prev_pred_v`, and `coords_scores` start as None.

            `fixed_threshold` is 5 and `acceleration_weight` is 0.7.

        ## Returns

            - Returns None.

        ---

        # 准备空的 SLaT token 稳定度状态。

            `self` 是新建的 `StabilityTracker`。

            `num_tokens` 从 0 开始，`device` 是 CPU，`cached_streak_counter`、`prev_pred_v` 和 `coords_scores` 一开始是 None。

            `fixed_threshold` 是 5，`acceleration_weight` 是 0.7。

        ## 返回

            - 返回 None。

"""
        self.num_tokens = 0
        self.device = torch.device("cpu")
        self.cached_streak_counter = None
        self.prev_pred_v = None
        self.coords_scores = None
        self.fixed_threshold = 5
        self.acceleration_weight = 0.7

    def reset(self, num_tokens, device):
        """
        # Allocate a zero streak counter for this many tokens.

            `self` is the `StabilityTracker`. `prev_pred_v` is set to None.

            The streak counter is a long tensor of shape `(num_tokens,)` on `device`.

            `num_tokens` is an int.

            It is not a tensor and is not None.

            `device` is the torch device stored and used for the counter.

            It is not None.

        ## Returns

            - Returns None.

        ---

        # 为这么多个 token 分配全零的连续计数。

            `self` 是 `StabilityTracker`。

            `prev_pred_v` 被设为 None。

            连续计数是 `device` 上形状为 `(num_tokens,)` 的 long 张量。

            `num_tokens` 是整数。

            不是张量，也不是 None。

            `device` 是保存下来、用来放计数器的 torch 设备。

            不是 None。

        ## 返回

            - 返回 None。

"""
        self.num_tokens = int(num_tokens)
        self.device = device
        self.cached_streak_counter = torch.zeros(self.num_tokens, device=device, dtype=torch.long)
        self.prev_pred_v = None

    @torch.no_grad()
    def select(self, pred_v, num_to_skip, spatial_weight=0.3):
        """
        # Pick the least-changing token indices to reuse, and the indices that still need an update.

            `self` is the `StabilityTracker`. It reads `coords_scores`, `prev_pred_v`, `num_tokens`, `device`, `acceleration_weight`, `cached_streak_counter`, and `fixed_threshold`. If `coords_scores` is None or its first dimension is not `num_tokens`, the reuse index is empty and every token is updated.

            `pred_v` is cast to float.

            Its L2 norm and its change from `prev_pred_v` score motion.

            The first call, or a shape mismatch, treats acceleration as zero and stores this tensor as `prev_pred_v`. It is not None.

            `num_to_skip` is an int.

            Zero or less clears the streak counter and returns no reused indices.

            Otherwise the lowest combined scores are candidates, up to `num_tokens`. It is not None.

            `spatial_weight` is a float, default 0.3.

            The score is this weight times the normalized first column of `coords_scores`, plus the rest times motion.

            It is not None.

        ## Returns

            - Returns those two long tensors.

        ---

        # 选出变化最小、可以复用的 token 下标，以及仍要更新的下标。

            `self` 是 `StabilityTracker`。

            它读取 `coords_scores`、`prev_pred_v`、`num_tokens`、`device`、`acceleration_weight`、`cached_streak_counter` 和 `fixed_threshold`。

            若 `coords_scores` 是 None，或第一维不是 `num_tokens`，复用下标为空，所有 token 都更新。

            `pred_v` 会被转成 float。

            它的 L2 范数，以及相对 `prev_pred_v` 的变化，用来给运动打分。

            第一次调用或形状不一致时，加速度当作零，并把这个张量存成 `prev_pred_v`。

            不是 None。

            `num_to_skip` 是整数。

            小于等于零时清空连续计数，并且不复用任何下标。

            否则按综合分从低到高取候选，最多 `num_tokens` 个。

            不是 None。

            `spatial_weight` 是浮点数，默认 0.3。

            分数是它乘以 `coords_scores` 第一列的归一化值，其余权重乘运动项。

            不是 None。

        ## 返回

            - 返回这两个 long 张量。

"""
        pred_v = pred_v.float()
        scores = self.coords_scores
        if scores is None or scores.shape[0] != self.num_tokens:
            return (
                torch.zeros(0, device=self.device, dtype=torch.long),
                torch.arange(self.num_tokens, device=self.device),
            )
        scores = scores.to(device=self.device, dtype=torch.float32)
        if self.prev_pred_v is None or self.prev_pred_v.shape != pred_v.shape:
            self.prev_pred_v = torch.zeros_like(pred_v)
            first = True
        else:
            first = False
        if num_to_skip <= 0:
            self.cached_streak_counter.zero_()
            self.prev_pred_v.copy_(pred_v)
            return (
                torch.zeros(0, device=self.device, dtype=torch.long),
                torch.arange(self.num_tokens, device=self.device),
            )
        current_coords = scores[:, 1:]
        l2_scores = torch.norm(pred_v, p=2, dim=-1).reshape(-1)
        if first:
            acceleration = torch.zeros_like(l2_scores)
        else:
            acceleration = torch.norm(pred_v - self.prev_pred_v, p=2, dim=-1).reshape(-1)
        l2_norm = (l2_scores - l2_scores.min()) / (l2_scores.max() - l2_scores.min() + 1e-6)
        accel_norm = (acceleration - acceleration.min()) / (
            acceleration.max() - acceleration.min() + 1e-6
        )
        weight = self.acceleration_weight
        motion = weight * accel_norm + (1.0 - weight) * l2_norm
        spatial_raw = scores[:, 0].reshape(-1)
        spatial = (spatial_raw - spatial_raw.min()) / (
            spatial_raw.max() - spatial_raw.min() + 1e-6
        )
        final_scores = spatial_weight * spatial + (1.0 - spatial_weight) * motion
        num_to_pick = min(int(num_to_skip), self.num_tokens)
        _, preliminary = torch.topk(final_scores, k=num_to_pick, largest=False)
        streaks = self.cached_streak_counter[preliminary]
        keep = preliminary[streaks < self.fixed_threshold - 1]
        update_mask = torch.ones(self.num_tokens, dtype=torch.bool, device=self.device)
        if keep.numel() > 0:
            update_mask[keep] = False
        fast_update = torch.where(update_mask)[0]
        self.cached_streak_counter[fast_update] = 0
        if keep.numel() > 0:
            self.cached_streak_counter[keep] += 1
        self.prev_pred_v.copy_(pred_v)
        return keep, fast_update


class EulerFasterSLat(ODESolver):
    """
    # SLaT Euler step with error cache and token carving.

    ---

    # 带误差缓存和 token carving 的 SLaT Euler 步。
    """

    def __init__(self, thresh=1.5, ret_steps=3, full_steps=12, carving_ratio=0.1):
        """
        # Store the SLaT cache thresholds and create the token leader and stability tracker.

            `self` is the new solver.

            `faster_dic`, `current`, and `full_coords_backup` start as None until `prepare` or `solve_iter`.

            `thresh` is a float, default 1.5.

            It becomes the cache error threshold in `prepare`. It is not None.

            `ret_steps` is an int, default 3.

            `prepare` copies it to `first_enhance`, so that many leading steps stay full.

            It is not None.

            `full_steps` is an int, default 12.

            `prepare` passes it to `slat_cache_init` as the step count.

            It is not None.

            `carving_ratio` is a float, default 0.1.

            `solve_iter` multiplies it by the token count to choose how many tokens may be skipped.

            It is not None.

        ## Returns

            - Returns None.

        ---

        # 保存 SLaT 缓存阈值，并创建 token 门闩和稳定度跟踪器。

            `self` 是新建的求解器。

            `faster_dic`、`current` 和 `full_coords_backup` 在 `prepare` 或 `solve_iter` 之前是 None。

            `thresh` 是浮点数，默认 1.5。

            `prepare` 时它成为缓存误差阈值。

            不是 None。

            `ret_steps` 是整数，默认 3。

            `prepare` 把它抄到 `first_enhance`，因此开头这么多步保持完整计算。

            不是 None。

            `full_steps` 是整数，默认 12。

            `prepare` 把它作为步数传给 `slat_cache_init`。

            不是 None。

            `carving_ratio` 是浮点数，默认 0.1。

            `solve_iter` 用它乘 token 数，决定最多跳过多少个 token。

            不是 None。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.thresh = float(thresh)
        self.ret_steps = int(ret_steps)
        self.full_steps = int(full_steps)
        self.carving_ratio = float(carving_ratio)
        self.faster_dic = None
        self.current = None
        self.leader = TokenLeader()
        self.tracker = StabilityTracker()
        self.full_coords_backup = None

    def prepare(self, num_tokens, device, coords_scores):
        """
        # Reset the SLaT cache, the leader, and the tracker for one generate call.

            `self` is the `EulerFasterSLat`. `thresh` and `ret_steps` are copied onto the new cache dict.

            `num_tokens` is an int.

            It is stored on the leader and used to size the tracker.

            It is not None.

            `device` is the torch device passed to `StabilityTracker.reset`. It is not None.

            `coords_scores` is a tensor or None.

            A tensor is detached and stored on the tracker.

            None clears `tracker.coords_scores`.

        ## Returns

            - Returns None.

        ---

        # 为一次生成重置 SLaT 缓存、门闩和跟踪器。

            `self` 是 `EulerFasterSLat`。

            `thresh` 和 `ret_steps` 会抄到新的缓存字典上。

            `num_tokens` 是整数。

            它写到门闩上，并用来确定跟踪器的长度。

            不是 None。

            `device` 是传给 `StabilityTracker.reset` 的 torch 设备。

            不是 None。

            `coords_scores` 是张量或 None。

            张量会 detach 后存到跟踪器上。

            None 会把 `tracker.coords_scores` 清掉。

        ## 返回

            - 返回 None。

"""
        self.faster_dic, self.current = slat_cache_init(self.full_steps)
        self.faster_dic["thresh"] = self.thresh
        self.faster_dic["first_enhance"] = self.ret_steps
        self.leader.reset()
        self.leader.total_tokens = int(num_tokens)
        self.tracker.reset(num_tokens, device)
        if coords_scores is not None:
            self.tracker.coords_scores = coords_scores.detach()
        else:
            self.tracker.coords_scores = None

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        """
        # Advance one SLaT Euler step, using the cached velocity or a full or carved network call.

            `self` is the `EulerFasterSLat`. It reads `faster_dic`, `current`, `tracker`, and `full_coords_backup`.

            `dynamics_fn` is the callable that returns a velocity tensor.

            On a carved step it may receive only the tokens in `fast_update`. It is not None.

            `x_t` is the current state tensor.

            A cached step sets velocity to `x_t + cache["easy"]`. It is not None.

            `t` is the current time value forwarded to `dynamics_fn` on a full step.

            `dt` is the time increment passed to `linear_approximation_step`. Neither is None.

            This function does not name a unit for them.

            `args` and `kwargs` are forwarded to `dynamics_fn`. If token carving is active and `args` has a second entry, that entry is replaced by the rows of `full_coords_backup` selected by the update indices, as a contiguous int32 array.

        ## Returns

            - Returns the next state and the velocity used.

        ---

        # 推进一步 SLaT Euler。

            `self` 是 `EulerFasterSLat`。

            它读取 `faster_dic`、`current`、`tracker` 和 `full_coords_backup`。

            `dynamics_fn` 是返回速度张量的可调用对象。

            裁 token 时，它可能只收到 `fast_update` 里的 token。

            不是 None。

            `x_t` 是当前状态张量。

            缓存步把速度设为 `x_t + cache["easy"]`。

            不是 None。

            `t` 是当前时间值，完整步会传给 `dynamics_fn`。

            `dt` 是传给 `linear_approximation_step` 的时间增量。

            二者都不是 None。

            这个函数没有给它们写单位。

            `args` 和 `kwargs` 会转给 `dynamics_fn`。

            若 token carving 开着，并且 `args` 有第二项，该项会换成 `full_coords_backup` 里由更新下标选出的行，类型是连续的 int32 数组。

        ## 返回

            - 速度来自缓存，或来自完整网络，或来自被裁过的网络。
            - 返回下一状态和所用速度。

"""
        should_calc = slat_should_run_full(self.faster_dic, self.current, x_t)
        cache = self.faster_dic["cache"]
        if should_calc:
            step_args = args
            token_on = self.current["is_token_active"] and self.current["use_token"]
            if token_on:
                _, fast_update = self.tracker.select(
                    cache["prev_v"], self.current["num_to_skip"], spatial_weight=0.3
                )
                self.current["fast_update_indices"] = fast_update
                if fast_update.numel() == 0 or fast_update.numel() == x_t.shape[1]:
                    velocity = dynamics_fn(x_t, t, *args, **kwargs)
                else:
                    x_input = x_t[:, fast_update, :]
                    step_args_list = list(args)
                    if len(step_args_list) > 1 and self.full_coords_backup is not None:
                        picked = fast_update.detach().cpu().numpy()
                        step_args_list[1] = np.ascontiguousarray(
                            self.full_coords_backup[picked]
                        ).astype(np.int32)
                    velocity = dynamics_fn(x_input, t, *tuple(step_args_list), **kwargs)
                    full_velocity = cache["prev_v"].clone()
                    full_velocity[:, fast_update, :] = velocity.to(full_velocity.dtype)
                    velocity = full_velocity
            else:
                velocity = dynamics_fn(x_t, t, *step_args, **kwargs)
            prev_x = cache["prev_x"]
            prev_prev_x = cache["prev_prev_x"]
            prev_v = cache["prev_v"]
            if prev_x is not None and prev_prev_x is not None and prev_v is not None:
                output_change = (velocity - prev_v).abs().mean()
                prev_input_change = (prev_x - prev_prev_x).abs().mean() + 1e-8
                current_k = output_change / prev_input_change
                if cache["k"] is None:
                    cache["k"] = current_k
                else:
                    cache["k"] = 0.7 * cache["k"] + 0.3 * current_k
            if prev_x is not None:
                cache["prev_prev_x"] = prev_x
            cache["prev_x"] = x_t.detach().clone()
            cache["prev_v"] = velocity.detach().clone()
            cache["easy"] = velocity - x_t
        else:
            velocity = x_t + cache["easy"]
            cache["prev_x"] = x_t.detach().clone()
            cache["prev_v"] = velocity.detach().clone()
        x_next = linear_approximation_step(x_t, dt, velocity)
        self.current["step"] += 1
        return x_next, velocity

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        """
        # Walk the time sequence and yield each Euler step as `(state, time, velocity)`.

            `self` is the `EulerFasterSLat`. If `args` has a second entry, it is saved as `full_coords_backup`. Token carving turns on only after the leader warmup, when `carving_ratio * num_tokens` is strictly between 0 and the token count.

            `num_tokens` is `x_init.shape[1]`.

            `dynamics_fn` is forwarded to `step`. It is not None.

            `x_init` is the initial state tensor.

            The token axis is dimension 1.

            It is not None.

            `times` is a sequence of time values.

            Each consecutive pair supplies `t0` and `dt = t1 - t0`. It is not None.

            `args` and `kwargs` are forwarded to every `step` call.

            Their shapes are fixed by the generator, not by this method.

        ---

        # 沿时间序列走 Euler 步，每步产出 `(状态, 时间, 速度)`。

            `self` 是 `EulerFasterSLat`。

            若 `args` 有第二项，它会存成 `full_coords_backup`。

            只有门闩预热结束，并且 `carving_ratio * num_tokens` 严格介于 0 和 token 数之间时，才打开 token carving。

            `num_tokens` 取 `x_init.shape[1]`。

            `dynamics_fn` 会传给 `step`。

            不是 None。

            `x_init` 是初始状态张量。

            token 轴是第 1 维。

            不是 None。

            `times` 是一串时间值。

            相邻一对给出 `t0`，以及 `dt = t1 - t0`。

            不是 None。

            `args` 和 `kwargs` 会传给每一次 `step`。

            它们的形状由生成器决定，不由这个方法决定。

"""
        if len(args) > 1:
            self.full_coords_backup = args[1]
        num_tokens = int(x_init.shape[1])
        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            self.current["is_token_active"] = False
            cache = self.faster_dic["cache"]
            if (
                self.current["use_token"]
                and cache["prev_v"] is not None
                and self.leader.current_step >= self.leader.full_sampling_steps
            ):
                num_to_skip = int(self.carving_ratio * num_tokens)
                if 0 < num_to_skip < num_tokens:
                    self.current["is_token_active"] = True
                    self.current["num_to_skip"] = num_to_skip
            dt = t1 - t0
            x_t, velocity = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            self.leader.increase_step()
            yield x_t, t0, velocity


def hfer_from_rgba(rgb_uint8, mask_uint8, radius_ratio=0.15, target_size=256):
    """
    # Return the high-frequency energy ratio of the masked crop as one float.

        An empty mask returns 0.0.

        `rgb_uint8` is an array.

        A 2D array is used as gray.

        A wider array uses the first three channels as RGB and converts them to gray.

        This function does not check a dtype beyond what `cv2.cvtColor` receives.

        It is not None.

        `mask_uint8` is an array.

        A 3D mask uses channel 0.

        Pixels `> 0` become the alpha.

        The crop is the bounding box of that alpha, padded by 2 pixels and clamped to the image.

        It is not None.

        `radius_ratio` is a float, default 0.15.

        After the resize, the low-frequency disk radius is `int(min(rows, cols) * radius_ratio)` pixels in the Fourier magnitude.

        It is not None.

        `target_size` is an int, default 256.

        The crop is resized to a square of that side before the FFT.

        It is not None.

    ---

    # 返回扣图裁剪的高频能量比，一个浮点数。

        空 mask 返回 0.0。

        `rgb_uint8` 是数组。

        二维数组当作灰度。

        更宽的数组用前三个通道当 RGB，再转成灰度。

        除了 `cv2.cvtColor` 实际收到的类型，这个函数不再检查 dtype。

        不是 None。

        `mask_uint8` 是数组。

        三维 mask 用第 0 通道。

        `> 0` 的像素成为 alpha。

        裁剪框是这块 alpha 的包围盒，四周加 2 像素，并限制在图像内。

        不是 None。

        `radius_ratio` 是浮点数，默认 0.15。

        缩放之后，低频圆盘半径是傅里叶幅度上的 `int(min(rows, cols) * radius_ratio)` 像素。

        不是 None。

        `target_size` 是整数，默认 256。

        FFT 之前，裁剪被缩放到这个边长的正方形。

        不是 None。

"""
    import cv2

    rgb = np.asarray(rgb_uint8)
    mask = np.asarray(mask_uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    alpha = (mask > 0).astype(np.uint8) * 255
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(coords)
    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)
    crop_gray = gray[y:y + h, x:x + w]
    crop_alpha = alpha[y:y + h, x:x + w]
    resized_gray = cv2.resize(crop_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    resized_alpha = cv2.resize(crop_alpha, (target_size, target_size), interpolation=cv2.INTER_AREA)
    object_pixels = resized_gray[resized_alpha > 0]
    mean_val = float(np.mean(object_pixels)) if object_pixels.size else 128.0
    filled = np.ones_like(resized_gray, dtype=np.float32) * mean_val
    np.copyto(filled, resized_gray.astype(np.float32), where=(resized_alpha > 0))
    window = np.outer(np.hanning(target_size), np.hanning(target_size))
    image = filled * window
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.abs(spectrum)
    magnitude[crow, ccol] = 0
    total = float(np.sum(magnitude)) + 1e-8
    radius = int(min(rows, cols) * radius_ratio)
    yy, xx = np.ogrid[:rows, :cols]
    low = (xx - ccol) ** 2 + (yy - crow) ** 2 <= radius * radius
    high = float(np.sum(magnitude[~low]))
    return high / total


def voxel_frequency(occupancy):
    """
    # Return per-occupied-voxel scores and one 3D high-frequency energy ratio.

        No occupied voxel returns `(None, 0.0)`.

        `occupancy` is a tensor.

        Occupied locations are `argwhere(occupancy > 0)`, and the grid size is `occupancy.shape[-1]`. The score rows follow those occupied voxels that fall inside the grid, as float32 `(N, 4)`: normalized spatial high-frequency response, then z, y, x index.

        The scalar is the share of FFT magnitude outside a radius of 8 voxels.

        The tensor is detached.

        It is not None.

    ---

    # 返回每个占据体素的分数，以及一个三维高频能量比。

        没有占据体素时返回 `(None, 0.0)`。

        `occupancy` 是张量。

        占据位置是 `argwhere(occupancy > 0)`，网格边长是 `occupancy.shape[-1]`。

        分数行对应落在网格内的占据体素，float32，形状 `(N, 4)`：归一化后的空间高频响应，然后是 z、y、x 下标。

        标量是半径 8 体素以外的 FFT 幅度占比。

        张量会先 detach。

        不是 None。

"""
    occupancy = occupancy.detach()
    indices = torch.argwhere(occupancy > 0)
    if indices.numel() == 0:
        return None, 0.0
    values = occupancy[occupancy > 0].detach().float().cpu().numpy()
    data = indices.detach().cpu().numpy()
    grid_size = int(occupancy.shape[-1])
    zs = data[:, 2].astype(np.int64)
    ys = data[:, 3].astype(np.int64)
    xs = data[:, 4].astype(np.int64)
    valid = (zs < grid_size) & (ys < grid_size) & (xs < grid_size) & (zs >= 0) & (ys >= 0) & (xs >= 0)
    zs, ys, xs, values = zs[valid], ys[valid], xs[valid], values[valid]
    dense = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    dense[zs, ys, xs] = values
    spectrum = np.fft.fftshift(np.fft.fftn(dense))
    magnitude = np.abs(spectrum)
    center = grid_size // 2
    zz, yy, xx = np.ogrid[:grid_size, :grid_size, :grid_size]
    dist_sq = (zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2
    freq_mask = np.ones((grid_size, grid_size, grid_size), dtype=np.float32)
    freq_mask[dist_sq < 8 ** 2] = 0
    total = float(np.sum(magnitude))
    hfer = float(np.sum(magnitude * freq_mask) / total) if total > 1e-6 else 0.0
    spatial = np.abs(np.fft.ifftn(np.fft.ifftshift(spectrum * freq_mask)))
    token_score = spatial[zs, ys, xs]
    if token_score.max() > token_score.min():
        token_score = (token_score - token_score.min()) / (token_score.max() - token_score.min())
    scores = np.stack([token_score, zs, ys, xs], axis=1).astype(np.float32)
    return torch.from_numpy(scores), hfer


def spectral_downsample_factor(hfer_2d, hfer_3d, low_thresh, high_thresh):
    """
    # Map the combined frequency score to a downsample factor.

        `hfer_2d` and `hfer_3d` are scalars converted with `float`. The combined score is `0.9 * hfer_2d + 0.1 * hfer_3d`. They are not tensors and are not None.

        `low_thresh` and `high_thresh` are scalars converted with `float`. A combined score at or above `high_thresh` returns factor 1.25.

        A score above `low_thresh` returns 1.50.

        Any other score returns 2.00.

        They are not None.

    ## Returns

        - Returns that factor and the combined score.

    ---

    # 把合成的频率分数映射成下采样倍数。

        `hfer_2d` 和 `hfer_3d` 是用 `float` 转换的标量。

        合成分数是 `0.9 * hfer_2d + 0.1 * hfer_3d`。

        不是张量，也不是 None。

        `low_thresh` 和 `high_thresh` 是用 `float` 转换的标量。

        合成分数大于等于 `high_thresh` 时倍数是 1.25。

        高于 `low_thresh` 时是 1.50。

        其余是 2.00。

        不是 None。

    ## 返回

        - 返回这个倍数和合成分数。

"""
    combined = 0.9 * float(hfer_2d) + 0.1 * float(hfer_3d)
    if combined >= float(high_thresh):
        factor = 1.25
    elif combined > float(low_thresh):
        factor = 1.50
    else:
        factor = 2.00
    return factor, combined


def downsample_with_feature_fusion(
    coord_batch,
    coords_scores,
    max_coords=42000,
    downsample_factor=2,
):
    """
    # Merge voxels onto a coarser grid and keep the maximum score in each cell.

        `coord_batch` is a tensor of shape `(N, 4)`: batch index, then z, y, x.

        It lives on the device used for the unique keys.

        A count of at least `max_coords` forces the factor to 2.0.

        A count below 2048 forces the factor to 1.0.

        It is not None.

        `coords_scores` is a tensor whose column 0 is the score fused with `amax`. The returned scores are `(M, 4)`: fused score, then the merged z, y, x as float.

        It is not None.

        `max_coords` is an int, default 42000.

        If the merged set is still larger, a random subset of this size is kept.

        It is not None.

        `downsample_factor` is a float, default 2.

        It is the starting cell size before the count rules above replace it.

        It is not None.

    ## Returns

        - Returns the new coordinates, the new scores, and the factor actually used.

    ---

    # 把体素并到更粗的网格上，每个格子保留最大分数。

        `coord_batch` 是形状 `(N, 4)` 的张量：batch 下标，然后是 z、y、x。

        唯一键使用它所在的设备。

        数量达到 `max_coords` 时倍数强制为 2.0。

        数量低于 2048 时倍数强制为 1.0。

        不是 None。

        `coords_scores` 是张量，第 0 列是用 `amax` 融合的分数。

        返回的分数是 `(M, 4)`：融合分数，然后是合并后的 z、y、x 浮点坐标。

        不是 None。

        `max_coords` 是整数，默认 42000。

        合并后仍然更大时，随机留下这个数量。

        不是 None。

        `downsample_factor` 是浮点数，默认 2。

        它是上面的数量规则改写之前的初始格子倍数。

        不是 None。

    ## 返回

        - 返回新坐标、新分数，以及实际使用的倍数。

"""
    min_coords = 2048
    current_factor = float(downsample_factor)
    if coord_batch.shape[0] >= max_coords:
        current_factor = 2.0
    if coord_batch.shape[0] < min_coords:
        current_factor = 1.0
    device = coord_batch.device
    coords = coord_batch[:, 1:].float()
    batch_indices = coord_batch[:, 0:1]
    coords_min = coords.min(dim=0)[0]
    coords_max = coords.max(dim=0)[0]
    original_size = coords_max - coords_min + 1
    target_size = original_size / current_factor
    offset = (original_size - target_size) / 2
    target_min = coords_min + offset
    target_max = target_min + target_size - 1
    coords_normalized = (coords - coords_min) / (coords_max - coords_min).clamp(min=1)
    coords_rescaled = torch.round(coords_normalized * (target_size - 1) + target_min).int()
    coords_rescaled = torch.stack(
        [
            coords_rescaled[:, i].clamp(int(target_min[i].item()), int(target_max[i].item()))
            for i in range(3)
        ],
        dim=1,
    )
    combined_keys = torch.cat([batch_indices, coords_rescaled], dim=1)
    unique_keys, inverse = torch.unique(combined_keys, dim=0, return_inverse=True)
    raw_scores = coords_scores[:, 0].float().to(device)
    fused = torch.full((unique_keys.shape[0],), -1e9, device=device, dtype=raw_scores.dtype)
    fused.scatter_reduce_(0, inverse, raw_scores, reduce="amax", include_self=False)
    new_coord_batch = unique_keys.int()
    new_scores = torch.cat([fused.unsqueeze(1), new_coord_batch[:, 1:].float()], dim=1)
    if new_coord_batch.shape[0] > max_coords:
        perm = torch.randperm(new_coord_batch.shape[0], device=device)[:max_coords]
        new_coord_batch = new_coord_batch[perm]
        new_scores = new_scores[perm]
    return new_coord_batch, new_scores, current_factor
