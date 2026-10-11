# Author: Yulin Wang (yulinwang@seu.edu.cn)
# SPDX-License-Identifier: LGPL-2.1-only
"""Keep RoMa and its OpenCV dependencies outside the pose interpreter.

RoMa 及其 OpenCV 依赖留在独立解释器；匹配像素返回基础位姿流程。
"""
import atexit
import json
import socket
import threading
import os
import subprocess
import sys
from wapr.sam3d_isolated import _temporary_directory


# One worker owns the GPU model for this caller's lifetime.
# 每个调用进程保留一个工作进程，模型驻留到调用进程退出。
_match_worker = None
_match_channel = None
_match_owner = None
_match_lock = threading.Lock()


def _close_match_worker():
    """Release only this process's matching worker. / 仅释放当前进程拥有的匹配工作进程。"""
    global _match_worker, _match_channel, _match_owner
    if _match_owner != os.getpid():
        _match_worker = _match_channel = None
        return
    if _match_channel is not None:
        _match_channel.close()
    if _match_worker is not None and _match_worker.poll() is None:
        try:
            _match_worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _match_worker.terminate()
            try:
                _match_worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _match_worker.kill()
                _match_worker.wait()
    _match_worker = _match_channel = None


atexit.register(_close_match_worker)

def _matching_weights(device):
    """Fetch the same upstream checkpoints through the resumable package downloader.

    用包内续传入口获取相同上游权重；指纹来自已审查缓存，不称为官方公布摘要。
    """
    import hashlib
    import torch
    from wapr.det2d import _download_file
    from wapr.resources import weights_dir
    directory = os.path.join(torch.hub.get_dir(), "checkpoints")
    assets = (
        ("roma_outdoor.pth", "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
         445647516, "c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba"),
        ("dinov2_vitl14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
         1217586395, "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"),
    )
    loaded = []
    for name, url, size, expected in assets:
        path = os.path.join(directory, name)
        if name == "dinov2_vitl14_pretrain.pth" and not os.path.isfile(path):
            # The detector uses this exact backbone; verify before reusing its file.
            # 检测器使用同一骨干；复用已有文件并在下方核验，避免重复下载 1.2 GB。
            shared_path = os.path.join(weights_dir(), "det2d", name)
            if os.path.isfile(shared_path):
                path = shared_path
        if not os.path.isfile(path):
            _download_file(url, path)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if os.path.getsize(path) != size or digest.hexdigest() != expected:
            raise RuntimeError("RoMa checkpoint fingerprint differs; retain and review the file / RoMa 权重指纹不符，保留文件并检查: " + path)
        # Load tensors only, preserving the model values and selected device.
        # 只加载张量；模型数值与所选设备不变。
        loaded.append(torch.load(path, map_location=device, weights_only=True))
    return loaded[0], loaded[1]


def prepare_environment(allow_replacement=None, check_only=False):
    """Prepare matching in the shared compatible prefix without SAM weights.

在共享兼容前缀准备匹配；不请求 SAM 权重，不替换基础 Torch 或 OpenCV。
"""
    from wapr import sam3d_isolated
    prepared = sam3d_isolated.prepare_environment(allow_replacement, check_only)
    if prepared["status"] not in ("ready", "installed"):
        return prepared
    with _temporary_directory(prefix="wapr-roma-prepare-") as directory:
        path = os.path.join(directory, "request.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump({"allow_replacement": allow_replacement, "check_only": check_only}, stream)
        environment = os.environ.copy()
        result = sam3d_isolated._run_preparation_worker("prepare", path, environment,
                                                       worker_file=os.path.abspath(__file__))
    result["bootstrap"] = prepared
    return result


def match_pixels(src_rgb, dst_rgb, device, coarse_res, upsample_res, num):
    """Return RoMa's sampled source/destination pixels and confidence unchanged.

原样返回 RoMa 采样的源／目标像素与置信度；几何过滤仍在原配方进行。
"""
    import numpy as np
    import torch
    from wapr.sam3d_isolated import SAM3D_PYTHON, _worker_cache_environment
    src_rgb, dst_rgb = np.asarray(src_rgb), np.asarray(dst_rgb)
    for value in (src_rgb, dst_rgb):
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("RoMa expects uint8 RGB HWC / RoMa 需要 uint8 RGB HWC")
    with _temporary_directory(prefix="wapr-roma-match-") as directory:
        input_path = os.path.join(directory, "input.npz")
        output_path = os.path.join(directory, "pixels.npz")
        request_path = os.path.join(directory, "request.json")
        # Carry the caller's sampling state; do not introduce another seed.
        # 传递调用者的采样随机状态，不引入新的种子。
        np.savez(input_path, src_rgb=src_rgb, dst_rgb=dst_rgb,
                 cpu_rng=torch.get_rng_state().numpy(), cuda_rng=torch.cuda.get_rng_state(device).cpu().numpy())
        request = {"input_path": input_path, "output_path": output_path, "device": device,
                   "coarse_res": coarse_res, "upsample_res": upsample_res, "num": num}
        with open(request_path, "w", encoding="utf-8") as stream:
            json.dump(request, stream)
        global _match_worker, _match_channel, _match_owner
        with _match_lock:
            if _match_owner != os.getpid() or _match_worker is None or _match_worker.poll() is not None:
                _close_match_worker()
                prepared = prepare_environment()
                if prepared["status"] not in ("ready", "installed"):
                    raise RuntimeError("RoMa preparation / RoMa 准备: " + json.dumps(prepared, ensure_ascii=False))
                parent_socket, child_socket = socket.socketpair()
                parent_socket.settimeout(600)
                environment = _worker_cache_environment(os.environ.copy())
                try:
                    _match_worker = subprocess.Popen(
                        [SAM3D_PYTHON, os.path.abspath(__file__), "serve", str(child_socket.fileno())],
                        env=environment, pass_fds=(child_socket.fileno(),))
                    _match_channel = parent_socket.makefile("rwb", buffering=0)
                    _match_owner = os.getpid()
                finally:
                    parent_socket.close()
                    child_socket.close()
            try:
                _match_channel.write((json.dumps(request) + "\n").encode("utf-8"))
                response = json.loads(_match_channel.readline())
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "RoMa worker failed / RoMa 工作进程失败"))
            except Exception:
                _close_match_worker()
                raise
        with np.load(output_path, allow_pickle=False) as result:
            src_xy, dst_xy, weight = result["src_xy"].copy(), result["dst_xy"].copy(), result["weight"].copy()
            cpu_rng, cuda_rng = result["cpu_rng"].copy(), result["cuda_rng"].copy()
        if src_xy.ndim != 2 or src_xy.shape[1] != 2 or dst_xy.shape != src_xy.shape or weight.shape != (len(src_xy),):
            raise RuntimeError("Invalid RoMa pixel output / RoMa 像素返回形状无效")
        if not all(np.isfinite(value).all() for value in (src_xy, dst_xy, weight)):
            raise RuntimeError("Nonfinite RoMa pixels / RoMa 像素含非有限值")
        if any(value.dtype != np.uint8 or value.ndim != 1 for value in (cpu_rng, cuda_rng)):
            raise RuntimeError("Invalid RoMa sampling state / RoMa 采样状态无效")
        torch.set_rng_state(torch.from_numpy(cpu_rng))
        torch.cuda.set_rng_state(torch.from_numpy(cuda_rng), device)
        return src_xy, dst_xy, weight


def _worker(mode, path):
    """Use the fixed package worker file, never generated experiment source.

执行固定的包内工作文件，不合成或执行实验源码。
"""
    with open(path, encoding="utf-8") as stream:
        request = json.load(stream)
    if mode == "prepare":
        from wapr.bootstrap import prepare_feature
        prepared = prepare_feature("roma", allow_replacement=request.get("allow_replacement"),
                                   check_only=request.get("check_only", False))
        print(json.dumps(prepared, ensure_ascii=False), flush=True)
        return 0 if prepared["status"] in ("ready", "installed", "approval_required", "installable", "needs_dependencies") else 1
    return _match_request(request, {})


def _match_request(request, models):
    """Reuse a model for matching settings, preserving caller sampling state.

    相同匹配设置复用模型，保留调用者采样状态。
    """
    import numpy as np
    import torch
    from PIL import Image
    from romatch import roma_outdoor
    key = (str(request["device"]), request["coarse_res"], request["upsample_res"])
    if models.get("key") != key:
        models.clear()
        weights, dinov2_weights = _matching_weights(request["device"])
        models["model"] = roma_outdoor(device=request["device"], coarse_res=request["coarse_res"],
                                      upsample_res=request["upsample_res"], symmetric=False, use_custom_corr=False,
                                      weights=weights, dinov2_weights=dinov2_weights)
        models["key"] = key
        print("WAPR_ROMA_MODEL_LOADED", os.getpid(), key, flush=True)
    model = models["model"]
    with np.load(request["input_path"], allow_pickle=False) as values:
        src_rgb, dst_rgb = values["src_rgb"].copy(), values["dst_rgb"].copy()
        torch.set_rng_state(torch.from_numpy(values["cpu_rng"].copy()))
        torch.cuda.set_rng_state(torch.from_numpy(values["cuda_rng"].copy()), request["device"])
    warp, certainty = model.match(Image.fromarray(src_rgb), Image.fromarray(dst_rgb), device=request["device"])
    matches, cert = model.sample(warp, certainty, num=request["num"])
    height_src, width_src = src_rgb.shape[:2]
    height_dst, width_dst = dst_rgb.shape[:2]
    src_xy, dst_xy = model.to_pixel_coordinates(matches, height_src, width_src, height_dst, width_dst)
    np.savez(request["output_path"], src_xy=src_xy.detach().float().cpu().numpy(),
             dst_xy=dst_xy.detach().float().cpu().numpy(), weight=cert.detach().float().cpu().numpy(),
             cpu_rng=torch.get_rng_state().numpy(), cuda_rng=torch.cuda.get_rng_state(request["device"]).cpu().numpy())
    print("WAPR_ROMA_ISOLATED_MATCH", len(src_xy), flush=True)
    return 0


def _serve(descriptor):
    """Serve local requests over an inherited socket; no network listener or installer.

    通过继承的本地 socket 接收请求；不监听网络，不重复启动安装器。
    """
    from wapr.resources import cache_dir
    source = os.path.join(cache_dir(), "sources", "RoMa")
    if source not in sys.path:
        sys.path.insert(0, source)
    models = {}
    with socket.socket(fileno=descriptor) as channel, channel.makefile("rwb", buffering=0) as stream:
        for line in stream:
            try:
                _match_request(json.loads(line), models)
                response = {"ok": True, "pid": os.getpid()}
            except Exception as error:
                response = {"ok": False, "error": type(error).__name__ + ": " + str(error)}
            stream.write((json.dumps(response) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("prepare", "match", "serve"):
        raise SystemExit("Expected worker mode and request path / 需要工作模式与请求文件")
    if sys.argv[1] == "serve":
        raise SystemExit(_serve(int(sys.argv[2])))
    raise SystemExit(_worker(sys.argv[1], sys.argv[2]))
