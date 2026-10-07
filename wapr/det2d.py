# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""RGB CAD-prompted instance detection: GroundingDINO Swin-B, SAM 2.1-L, and DINOv2-L or TensorRT. It does not depend on the pose research repository.

RGB CAD 提示的实例检测：GroundingDINO Swin-B、SAM 2.1-L，以及 DINOv2-L 或 TensorRT。它不依赖姿态研究仓库。

The usual unseen-object path proposes boxes, cuts a mask, then matches the crop to a fixed CAD template bank.

常见的未见物体流程是先给出包围盒，再切 mask，再把裁剪和固定 CAD 模板库匹配。

Boxes come from GroundingDINO Swin-B. Masks come from SAM 2.1-L.

包围盒来自 GroundingDINO Swin-B。mask 来自 SAM 2.1-L。

The absolute/relative similarity combination follows the MUSE principle. DINOv2-L CLS and GeM each take a Tanimoto score, the two are averaged on the same view, absolute similarity is mixed with a class-axis softmax, and that product is multiplied by GroundingDINO objectness to the power 0.1.

The CLS + GeM feature combination and pooling exponent 1.5 follow MUSE Section 3.2:
https://arxiv.org/html/2510.17866v1#S3.SS2
GeM pooling itself comes from Radenović, Tolias and Chum (2017, revised 2018):
https://arxiv.org/abs/1711.02512

绝对与相对相似度的组合参考 MUSE。DINOv2-L 的 CLS 和 GeM 各自做 Tanimoto，两者在同一视角上平均，绝对相似度再和类别轴 softmax 混合，然后乘以 GroundingDINO objectness 的 0.1 次方。

CLS 与 GeM 的特征组合、池化指数 1.5 参考 MUSE 第 3.2 节；GeM 池化本身源自
Radenović、Tolias 和 Chum（2017，2018 修订），论文链接见上方。

The CAD-view shortlist follows CNOS: the five strongest global scores, plus the best CLS view. The local patch term is added on that same view.

CAD 视角候选缩减参考 CNOS：全局分最高的 5 个，再加 CLS 最好的一个。局部 patch 项加在同一个视角上。

Models and templates load once. FP16 TensorRT accelerates DINO. GroundingDINO and SAM remain FP32.

模型和模板只加载一次。FP16 TensorRT 加速 DINO。GroundingDINO 和 SAM 保持 FP32。

detect_many_categories_many_instances returns (instances, timing). Masks are visible COCO RLE. bbox is the detection xywh in pixels.

detect_many_categories_many_instances 返回 (instances, timing)。mask 是可见 COCO RLE。bbox 是检测 xywh，单位像素。

Construct the detector with WAPRDet2D(template, ...). Call detect_many_categories_many_instances(rgb, ...) for 2D instances. onboard_meshes(...) builds the CAD bank.

用 WAPRDet2D(template, ...) 构造检测器。调用 detect_many_categories_many_instances(rgb, ...) 获取 2D 实例。onboard_meshes(...) 建立 CAD 库。
"""
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from wapr.resources import weights_dir, resource_root, source_checkout


release_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_weights_dir = os.path.join(weights_dir(), 'det2d')
# The checksum catalog lives beside this module in both source and wheel layouts.
# 校验清单在源码目录和 wheel 中都与此模块同处 wapr/。
det2d_manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'det2d_assets.json')
source_parent = os.path.join(release_dir, 'third_party') if source_checkout else os.path.join(resource_root(), 'sources')
default_dino_repo = os.path.join(source_parent, 'dinov2')
# A fresh GroundingDINO clone still calls its CUDA extension on GPU.
# The checkout used here calls multi_scale_deformable_attn_pytorch instead.
# The text encoder is a local bert-base-uncased directory, not a hub download.
# 新克隆的 GroundingDINO 在 GPU 上仍会调用它的 CUDA 扩展。
# 这里的检出改为调用 multi_scale_deformable_attn_pytorch。
# 文本编码器是本地的 bert-base-uncased 目录，不从 hub 下载。
grounding_repo = os.path.join(source_parent, 'GroundingDINO')
if os.path.isdir(grounding_repo) and grounding_repo not in sys.path:
    sys.path.insert(0, grounding_repo)
sam_repo = os.path.join(source_parent, 'ultralytics')
if os.path.isdir(sam_repo) and sam_repo not in sys.path:
    sys.path.insert(0, sam_repo)

# Scientific choices stay explicit; changing them requires a new accuracy evaluation.
# 科学选择显式保留；修改后必须重新评价精度，不能继承固定的实验配置的AP。
caption = 'items .'
box_threshold = .1
box_nms_threshold = .7
crop_px = 224
crop_margin = 1.1
# Within one frame, DINO crops and local matching are split only to fit memory.
# 同一帧里，DINO 裁剪和局部匹配只按显存分块。
# DINOv2 uses LayerNorm, so a longer chunk does not change the per-crop features.
# DINOv2 用 LayerNorm，块更长不改变每个裁剪自己的特征。
# The DINO engine profile max is this crop chunk. A shorter tail is its own chunk.
# DINO 引擎的 batch 上限就是这个裁剪块长。不足一块的尾部单独成块。
crop_batch = 128
patch_coverage = .5
global_views = 5
# 128 local rows on a 30-object library materialize about 25 GiB before the detector weights.
# 30 个物体、128 行局部匹配在载入检测器权重之前就要约 25 GiB。
matching_batch = 64
global_weight = .5
absolute_weight = .8
class_temperature = .02
objectness_power = .1
# 0.1 is the CAD-match gate. A box stays when its best-class score is at least this value.
# The published AP table was measured at 0.35. A later check at 0.50, on those saved
# outputs, kept 89/88/99/99/89 percent of the same-class visible-bbox matches at IoU 0.5
# on LMO, ICBIN, TUDL, YCBV, and TLESS. Mean boxes per frame fell from 55/45/36/16/20
# to 11/15/6/7/7. That check is not this gate.
# 0.1 是 CAD 匹配阈值。最佳类别分数不低于这个值，包围盒才留下。
# 已发表的 AP 表是在 0.35 下测的。后来在 0.50 上核对过同一批输出：
# LMO、ICBIN、TUDL、YCBV、TLESS 上同类可见包围盒 IoU 0.5 的匹配还保留
# 89/88/99/99/89%，每帧平均包围盒从 55/45/36/16/20 降到 11/15/6/7/7。
# 那次核对不是现在这个阈值。
score_threshold = .1
mask_nms_threshold = .5
# Foreground RGB below this has no appearance. Shape is drawn instead.
# 前景 RGB 低于这个值时，没有可用来匹配的外观，改画形状。
black_foreground_max = .01
template_views = 42
inplane_turns = [0, 1, 2, 3]

# Official checkpoints the detector can load. vitl14 and swinb are the published pair.
# 检测器能加载的官方权重。已发表的一对是 vitl14 和 swinb。
# Other names download their own file and, for DINOv2, their own TensorRT engine.
# 其他名字下载各自的文件。DINOv2 再为这份权重建自己的 TensorRT 引擎。
# ViT-g and the register models are not in this table. Patch size stays 14.
# 表里没有 ViT-g，也没有带 register 的模型。图块大小仍是 14。
DINO_CHOICES = {
    'vits14': {
        'hub': 'dinov2_vits14',
        'file': 'dinov2_vits14_pretrain.pth',
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'embed': 384,
    },
    'vitb14': {
        'hub': 'dinov2_vitb14',
        'file': 'dinov2_vitb14_pretrain.pth',
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        'embed': 768,
    },
    'vitl14': {
        'hub': 'dinov2_vitl14',
        'file': 'dinov2_vitl14_pretrain.pth',
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        'embed': 1024,
        'sha256': 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428',
    },
}
GROUNDING_CHOICES = {
    'swinb': {
        'file': 'groundingdino_swinb_cogcoor.pth',
        'config': 'groundingdino_swinb.json',
        'config_src': 'groundingdino_swinb.json',
        'url': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth',
        'sha256': '46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7',
    },
    'swint': {
        'file': 'groundingdino_swint_ogc.pth',
        'config': 'groundingdino_swint.json',
        'config_src': 'groundingdino_swint.json',
        'url': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
    },
}
# SAM and BERT stay one pair. They are fetched when the file is missing.
# SAM 和 BERT 仍是这一对。仅当文件缺失时下载。
SHARED_DOWNLOADS = (
    ('sam2.1_l.pt', 'https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt',
     'ab7e1ac9cb9f6eb3bcf197ece044f06a707ec49129361a2b47e93e1db6989efd'),
    ('bert-base-uncased/config.json', 'https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json',
     '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20'),
    ('bert-base-uncased/pytorch_model.bin', 'https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin',
     '097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a'),
    ('bert-base-uncased/tokenizer.json', 'https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json',
     'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'),
    ('bert-base-uncased/tokenizer_config.json', 'https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json',
     'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'),
    ('bert-base-uncased/vocab.txt', 'https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt',
     '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
)


def sha256(path):
    """
    # Hash one file and return its SHA-256 hex digest.

    ## Args

        - path: the filesystem path. The file is read in 1 MiB chunks. A missing file raises.

    ## Returns

        - The return is a hex string.
        - Loading and onboarding call this.
        - predict does not hash each frame.

    ---

    # 对一个文件做 SHA-256，返回十六进制摘要。

    ## 参数

        - path: 文件路径。文件按 1 MiB 一块读取。文件不存在时抛出异常。

    ## 返回

        - 返回值是十六进制字符串。
        - 加载和建库会调用它。
        - predict 不会对每一帧做哈希。

"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def det2d_choice(table, name, kind):
    """
    # Return one catalog row.

        An unknown name raises ValueError before any download.

    ## Args

        - table: the catalog dict. Callers pass DINO_CHOICES or GROUNDING_CHOICES.
        - name: the choice key. It is converted with str before the lookup.
        - kind: the word placed in the ValueError. Callers pass 'dino' or 'grounding'.

    ## Returns

        - The return is table[str(name)], the matching row dict.

    ---

    # 返回目录里的一行。

        名字不在表里时，在下载之前抛出 ValueError。

    ## 参数

        - table: 目录字典。调用方传入 DINO_CHOICES 或 GROUNDING_CHOICES。
        - name: 选项键。查找前用 str 转成字符串。
        - kind: 写进 ValueError 的词。调用方传入 'dino' 或 'grounding'。

    ## 返回

        - 返回值是 table[str(name)]，也就是匹配的那一行字典。

"""
    key = str(name)
    if key not in table:
        known = ', '.join(sorted(table))
        raise ValueError('%s must be one of %s' % (kind, known))
    return table[key]


def dino_artifact_names(dino):
    """
    # Return the TensorRT engine, manifest, and ONNX filenames for one DINOv2 choice.

    ## Args

        - dino: a key in DINO_CHOICES: vits14, vitb14, or vitl14. An unknown name raises ValueError.

    ## Returns

        - The return is three strings, engine then manifest then ONNX.
        - vitl14 uses dino_patches_fp16.engine, dino_engine_manifest.json, and dino_patches.onnx.
        - Any other choice puts its name into those three filenames.

    ---

    # 返回一种 DINOv2 选择对应的 TensorRT 引擎、清单和 ONNX 文件名。

    ## 参数

        - dino: DINO_CHOICES 的键：vits14、vitb14 或 vitl14。未知名字抛出 ValueError。

    ## 返回

        - 返回三个字符串，依次是引擎、清单、ONNX。
        - vitl14 使用 dino_patches_fp16.engine、dino_engine_manifest.json 和 dino_patches.onnx。
        - 其他选择把名字写进这三个文件名。

"""
    spec = det2d_choice(DINO_CHOICES, dino, 'dino')
    if spec['hub'] == 'dinov2_vitl14':
        return 'dino_patches_fp16.engine', 'dino_engine_manifest.json', 'dino_patches.onnx'
    tag = str(dino)
    return (
        'dino_patches_fp16_%s.engine' % tag,
        'dino_engine_manifest_%s.json' % tag,
        'dino_patches_%s.onnx' % tag,
    )


def _download_file(url, dest):
    """Serialize writers of one resource across processes and retain Range retries.

    跨进程串行写入同一资源，避免首次调用及预下载共同破坏续传文件。
    """
    from filelock import FileLock
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    existed_before_lock = os.path.isfile(dest)
    with FileLock(dest + '.lock', timeout=1800):
        if not existed_before_lock and os.path.isfile(dest):
            # Another caller completed the atomic placement; the caller still validates its hash.
            # 其他调用者已完成原子放置；上层仍须校验预期摘要。
            print('DET2D_DOWNLOAD_SHARED_COMPLETE', {'path': dest}, flush=True)
            return
        _download_file_body(url, dest)


def _download_parallel_ranges(url, partial, total, etag, direct):
    """Download one large HTTP entity with four workers and bounded ranges.

    使用四个线程及有界分片下载同一大文件，核验范围、字节数与强 ETag。
    """
    import concurrent.futures
    import threading
    worker_count = 4
    minimum_bytes = 256 * 1024 * 1024
    segment_bytes = 16 * 1024 * 1024
    if total is None or total < minimum_bytes:
        return False
    if not etag or etag.startswith('W/') or not (etag.startswith('"') and etag.endswith('"')):
        print('DET2D_RANGE_SERIAL', {'reason': 'no_strong_entity_validator', 'total': total}, flush=True)
        return False
    metadata_path = partial + '.entity.json'
    metadata = None
    try:
        with open(metadata_path, encoding='utf-8') as stream:
            metadata = json.load(stream)
    except (OSError, ValueError):
        pass
    offset = os.path.getsize(partial) if os.path.isfile(partial) else 0
    reuse_prefix = offset > 0 and metadata == {'etag': etag, 'total': total}
    start = offset if reuse_prefix else 0
    if start >= total:
        return False
    # A legacy prefix without entity provenance is retained until complete new ranges validate.
    # 旧前缀没有实体凭据时保留原件，重新并行核验完整实体，不把未知字节混入新文件。
    # Bound straggler tails and retain completed validated pieces across failed attempts.
    # 限制慢连接尾段长度；失败重试时保留已完整校验的分片。
    ranges = [(index, max(start, index * segment_bytes), min(total - 1, (index + 1) * segment_bytes - 1))
              for index in range(start // segment_bytes, (total + segment_bytes - 1) // segment_bytes)]
    pieces = [partial + '.range%d' % index for index, _, _ in ranges]
    assembled = partial + '.assembled'
    stop = threading.Event()
    progress_lock = threading.Lock()
    progress = {'bytes': start, 'time': time.monotonic()}
    completed = set()
    committed = False
    url_fingerprint = hashlib.sha256(url.encode('utf-8')).hexdigest()

    def download_range(item):
        index, first, last = item
        piece = partial + '.range%d' % index
        piece_metadata = piece + '.json'
        identity = {'etag': etag, 'total': total, 'first': first, 'last': last, 'url': url_fingerprint}
        cached = None
        try:
            with open(piece_metadata, encoding='utf-8') as stream:
                cached = json.load(stream)
        except (OSError, ValueError):
            pass
        if isinstance(cached, dict) and cached.get('identity') == identity and os.path.isfile(piece):
            if os.path.getsize(piece) == last - first + 1 and sha256(piece) == cached.get('sha256'):
                with progress_lock:
                    completed.add(piece)
                    progress['bytes'] += last - first + 1
                print('DET2D_RANGE_REUSE', {'first': first, 'last': last}, flush=True)
                return
        if stop.is_set():
            raise RuntimeError('Range batch interrupted / 分片批次中断')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
        request = urllib.request.Request(url, headers={'User-Agent': 'WAPR-det2d', 'Accept-Encoding': 'identity',
                                                     'Range': 'bytes=%d-%d' % (first, last), 'If-Match': etag})
        received = 0
        digest = hashlib.sha256()
        with opener.open(request, timeout=120) as response:
            content_range = response.headers.get('Content-Range') or ''
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range.strip())
            if getattr(response, 'status', 200) != 206 or match is None or tuple(map(int, match.groups())) != (first, last, total):
                raise RuntimeError('Server ignored or changed Range / 服务端忽略或改变下载范围')
            if response.headers.get('ETag') != etag:
                raise RuntimeError('HTTP entity changed / HTTP 下载实体改变')
            declared = response.headers.get('Content-Length')
            if declared is not None and int(declared) != last - first + 1:
                raise RuntimeError('Range byte count differs / 分片声明字节数不符')
            with open(piece, 'wb') as stream:
                while not stop.is_set():
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if received + len(block) > last - first + 1:
                        raise RuntimeError('Range body exceeds bounds / 分片主体越界')
                    stream.write(block)
                    digest.update(block)
                    received += len(block)
                    with progress_lock:
                        progress['bytes'] += len(block)
                        if time.monotonic() - progress['time'] >= 10:
                            print('DET2D_RANGE_PROGRESS', {'bytes': progress['bytes'], 'total': total}, flush=True)
                            progress['time'] = time.monotonic()
        if received != last - first + 1:
            raise RuntimeError('Truncated HTTP range / HTTP 分片截断')
        with open(piece_metadata + '.pending', 'w', encoding='utf-8') as stream:
            json.dump({'identity': identity, 'sha256': digest.hexdigest()}, stream)
        os.replace(piece_metadata + '.pending', piece_metadata)
        with progress_lock:
            completed.add(piece)

    try:
        print('DET2D_RANGE_PARALLEL', {'workers': min(worker_count, len(ranges)), 'pieces': len(ranges), 'segment_bytes': segment_bytes, 'total': total,
                                     'verified_prefix_bytes': start, 'legacy_prefix_retained': offset if not reuse_prefix else 0}, flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(download_range, item) for item in ranges]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except Exception:
                stop.set()
                raise
        with open(assembled, 'wb') as output:
            if reuse_prefix:
                with open(partial, 'rb') as prefix:
                    shutil.copyfileobj(prefix, output)
            for piece in pieces:
                with open(piece, 'rb') as stream:
                    shutil.copyfileobj(stream, output)
        if os.path.getsize(assembled) != total:
            raise RuntimeError('Assembled download length differs / 合并下载长度不符')
        # Commit is the last fallible operation; no fallback may append after success.
        # 原子提交是最后一个可能失败的操作，成功后不得回退再追加旧响应。
        os.replace(assembled, partial)
        committed = True
        return True
    except Exception as error:
        detail = str(error)[:160] if isinstance(error, RuntimeError) else ''
        print('DET2D_RANGE_FALLBACK', {'reason': type(error).__name__, 'detail': detail,
                                     'retained_complete_pieces': len(completed)}, flush=True)
        return False
    finally:
        removable = [assembled, metadata_path + '.pending']
        for piece in pieces:
            removable.append(piece + '.json.pending')
            if committed or piece not in completed:
                removable.extend((piece, piece + '.json'))
        for path in removable:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                # Cleanup must not turn a committed entity into a retry that appends old bytes.
                # 清理失败不得把已提交的实体变成追加旧字节的重试。
                print('DET2D_RANGE_CLEANUP_RETAINED', {'temporary_file': os.path.basename(path)}, flush=True)


def _download_file_body(url, dest):
    """
    # Download url into dest.

        The final path is replaced only after the body finishes.

    ## Args

        - url: the HTTP address. The request sends User-Agent WAPR-det2d.
        - dest: the final file path. Verified response bytes remain in dest + '.part' for bounded Range retries. Only a complete file is moved to dest.

    ---

    # 把 url 下载到 dest。

        文件主体下载完成后再替换最终路径。

    ## 参数

        - url: HTTP 地址。请求带的 User-Agent 是 WAPR-det2d。
        - dest: 最终文件路径。响应字节暂存于 dest + '.part' 并有限续传，仅完整文件才移到 dest。

"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    original_partial = dest + '.part'
    partial = original_partial
    from wapr.download_route import public_url
    print('DET2D_DOWNLOAD', {'url': public_url(url), 'path': dest}, flush=True)
    expected_total = None
    for attempt in range(6):
        direct = attempt > 0
        # Retry the same official URL without a failing regional proxy.
        # 地区代理失败时直连同一官方地址，不替换下载资源。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
        replacement_partial = original_partial + '.new-entity'
        partial = replacement_partial if os.path.isfile(replacement_partial) else original_partial
        offset = os.path.getsize(partial) if os.path.isfile(partial) else 0
        resume_entity = None
        if offset:
            try:
                with open(partial + '.entity.json', encoding='utf-8') as metadata:
                    resume_entity = json.load(metadata)
                validator = resume_entity['etag']
                if not isinstance(validator, str) or validator.startswith('W/') or not (validator.startswith('"') and validator.endswith('"')):
                    resume_entity = None
            except (OSError, ValueError, KeyError, TypeError):
                resume_entity = None
        if offset and resume_entity is None:
            # Unknown legacy bytes stay untouched; request an entire new entity in a separate file.
            # 未知来源旧前缀保持原状；在独立文件中请求完整新实体，绝不能串行追加混用。
            print('DET2D_DOWNLOAD_UNKNOWN_PREFIX_RETAINED', {'bytes': offset, 'restart_full_entity': True}, flush=True)
            partial = replacement_partial
            offset = 0
        headers = {'User-Agent': 'WAPR-det2d', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = 'bytes=%d-' % offset
            headers['If-Range'] = resume_entity['etag']
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(request, timeout=120) as response:
                if 'html' in (response.headers.get('Content-Type') or '').lower():
                    raise RuntimeError('Download returned an HTML page / 下载返回网页而非资源')
                content_length = response.headers.get('Content-Length')
                expected_bytes = int(content_length) if content_length is not None else None
                content_range = response.headers.get('Content-Range')
                append = False
                if content_range is not None:
                    # A proxy may return only a cached byte range despite a full request.
                    # 代理可能把完整请求错误地返回为缓存片段，不能将该片段发布为权重。
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range.strip())
                    if match is None:
                        raise RuntimeError('Incomplete HTTP range / HTTP 只返回部分文件: ' + str(content_range))
                    start, end, total = map(int, match.groups())
                    if resume_entity is not None and (response.headers.get('ETag') != resume_entity['etag'] or total != resume_entity['total']):
                        raise RuntimeError('Resume HTTP entity changed / 续传 HTTP 实体改变')
                    if start != offset or end < start or end >= total or (expected_total is not None and total != expected_total):
                        raise RuntimeError('Invalid HTTP range / HTTP 续传范围不匹配: ' + str(content_range))
                    if expected_bytes is not None and expected_bytes != end - start + 1:
                        raise RuntimeError('Invalid range length / 续传响应长度不匹配')
                    expected_bytes = end - start + 1
                    expected_total = total
                    append = offset > 0
                else:
                    # Servers ignoring Range must replace, never append, their full response.
                    # 服务端忽略 Range 时重新完整写入，不能把完整文件追加到片段。
                    if getattr(response, 'status', 200) == 206:
                        raise RuntimeError('Missing Content-Range / 续传响应缺少范围头')
                    offset = 0
                    expected_total = expected_bytes
                downloaded_bytes = 0
                progress_time = time.monotonic()
                etag = response.headers.get('ETag')
                parallel_complete = _download_parallel_ranges(url, partial, expected_total, etag, direct)
                if parallel_complete:
                    downloaded_bytes = expected_bytes
                else:
                    with open(partial, 'ab' if append else 'wb') as stream:
                        if offset == 0 and etag and not etag.startswith('W/') and expected_total is not None:
                            # Record provenance only when this response owns every prefix byte.
                            # 仅在当前响应负责全部前缀字节时记录实体凭据，不伪造旧缓存的来源。
                            with open(partial + '.entity.json', 'w', encoding='utf-8') as metadata:
                                json.dump({'etag': etag, 'total': expected_total}, metadata)
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            if expected_bytes is not None and downloaded_bytes + len(chunk) > expected_bytes:
                                raise RuntimeError('HTTP body exceeds declared length / HTTP 主体超过声明长度')
                            stream.write(chunk)
                            downloaded_bytes += len(chunk)
                            if time.monotonic() - progress_time >= 10:
                                print('DET2D_DOWNLOAD_PROGRESS', {'bytes': offset + downloaded_bytes,
                                                                 'total': expected_total}, flush=True)
                                progress_time = time.monotonic()
                # Keep the final path untouched when a connection ends prematurely.
                # 连接提前结束时保留最终路径原状，并保留 .part 供续传。
                if expected_bytes is not None and downloaded_bytes != expected_bytes:
                    raise RuntimeError('Incomplete download / 下载不完整: expected %d bytes, received %d' %
                                       (expected_bytes, downloaded_bytes))
                if expected_total is not None and os.path.getsize(partial) != expected_total:
                    raise RuntimeError('Incomplete total download / 文件尚未下载完整')
            zip_checkpoints = {spec['file'] for spec in DINO_CHOICES.values()}
            zip_checkpoints.update(('sam2.1_hiera_large.pt', 'sam2.1_hiera_tiny.pt'))
            if os.path.basename(dest) in zip_checkpoints:
                # Validate before publishing: a proxy may mislabel a truncated body as complete.
                # 发布前核验：代理可能把截断主体错误标成完整 HTTP 响应。
                try:
                    with zipfile.ZipFile(partial) as archive:
                        broken_record = archive.testzip()
                        if broken_record is not None:
                            raise RuntimeError('Invalid checkpoint record / 权重数据记录损坏: ' + broken_record)
                except Exception:
                    os.remove(partial)
                    expected_total = None
                    raise
            os.replace(partial, dest)
            if partial != original_partial:
                # Full replacement passed byte/format checks before releasing the preserved prefix.
                # 完整替换通过字节与格式检查后，才释放保留的未知旧前缀。
                for legacy_path in (original_partial, original_partial + '.entity.json'):
                    try:
                        if os.path.isfile(legacy_path):
                            os.remove(legacy_path)
                    except OSError:
                        print('DET2D_RANGE_CLEANUP_RETAINED', {'temporary_file': os.path.basename(legacy_path)}, flush=True)
            # Remove only this downloader's temporary artifacts after successful publication.
            # 发布成功后只清理当前下载器的临时文件；清理失败不触发重新下载。
            temporary_prefix = os.path.basename(partial) + '.range'
            for name in os.listdir(os.path.dirname(partial)):
                if name == os.path.basename(partial) + '.entity.json' or re.fullmatch(re.escape(temporary_prefix) + r'\d+(?:\.json(?:\.pending)?)?', name):
                    try:
                        os.remove(os.path.join(os.path.dirname(partial), name))
                    except OSError:
                        print('DET2D_RANGE_CLEANUP_RETAINED', {'temporary_file': name}, flush=True)
            return
        except Exception:
            # Retain verified response bytes for bounded retries and the next invocation.
            # 保留已写入的响应字节，供有限重试及下次调用续传；错误响应不写入。
            retained_bytes = os.path.getsize(partial) if os.path.isfile(partial) else 0
            print('DET2D_DOWNLOAD_RETRY', {'attempt': attempt + 1, 'bytes': retained_bytes, 'total': expected_total}, flush=True)
            if attempt == 5:
                raise
            # The public Hugging Face host often stalls. The same path on the
            # public mirror is the same file; no token is sent.
            # 官方 Hugging Face 常卡住。公开镜像上的同一路径是同一文件，不发送令牌。
            if '://huggingface.co/' in url and '://hf-mirror.com/' not in url:
                url = url.replace('://huggingface.co/', '://hf-mirror.com/', 1)
                expected_total = None
                print('DET2D_DOWNLOAD_MIRROR', {'url': public_url(url)}, flush=True)
            # GitHub archive redirects can fail while its official codeload works.
            # GitHub 归档重定向可能失败；切换同一官方版本的 codeload 地址。
            archive_match = re.fullmatch(r'https://github.com/([^/]+)/([^/]+)/archive/(.+)\.zip', url)
            if archive_match:
                owner, repository, revision = archive_match.groups()
                url = 'https://codeload.github.com/%s/%s/zip/%s' % (owner, repository, revision)
                if os.path.isfile(partial):
                    os.remove(partial)
                expected_total = None
            print('DET2D_DOWNLOAD_RETRY_DIRECT', flush=True)


def _place_file(weights_dir, relative, url, expected_sha):
    """
    # Return the local path of one weight file.

        Download it only when that file is missing.

    ## Args

        - weights_dir: the directory joined with relative.
        - relative: the path under weights_dir.
        - url: passed to _download_file when the destination is missing.
        - expected_sha: a hex digest, or None. None skips the hash check. A mismatch raises RuntimeError and does not delete the file.

    ## Returns

        - The return is the joined destination path.
        - A file that is already there is not downloaded again.

    ---

    # 返回一个权重文件的本地路径。

        仅当目标文件缺失时下载。

    ## 参数

        - weights_dir: 目录，和 relative 拼成路径。
        - relative: weights_dir 下面的相对路径。
        - 目标文件缺失时，url 传给 _download_file。
        - expected_sha: 十六进制摘要，或 None。None 跳过哈希检查。不一致时抛出 RuntimeError，并且不删除该文件。

    ## 返回

        - 返回值为拼接得到的目标路径。
        - 已经在磁盘上的文件不会再次下载。

"""
    dest = os.path.join(weights_dir, relative)
    if relative == 'dinov2_vitl14_pretrain.pth' and not os.path.isfile(dest):
        # Reuse only the audited identical upstream checkpoint; custom files remain untouched.
        # 仅复用已核验的同一上游权重；保留用户已有的自定义文件。
        from filelock import FileLock
        fingerprint = 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'
        torch_home = os.environ.get('TORCH_HOME', os.path.join(resource_root(), 'torchhub'))
        candidates = [os.path.join(torch_home, 'hub', 'checkpoints', relative)]
        # An explicit cache root must not silently read another user's/default cache.
        # 显式缓存根不能静默读取其他用户或默认缓存；未指定时仍可复用默认 Hub。
        if not os.environ.get('TORCH_HOME'):
            candidates.append(os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', relative))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with FileLock(dest + '.lock', timeout=1800):
            for candidate in candidates:
                if os.path.isfile(dest):
                    break
                if not os.path.isfile(candidate) or os.path.getsize(candidate) != 1217586395:
                    continue
                if sha256(candidate) != fingerprint:
                    continue
                temporary = dest + '.reuse.tmp'
                try:
                    if os.path.lexists(temporary):
                        os.remove(temporary)
                    try:
                        os.link(candidate, temporary)
                    except OSError:
                        shutil.copyfile(candidate, temporary)
                    os.replace(temporary, dest)
                    print('DET2D_VERIFIED_HUB_REUSE', {'path': dest}, flush=True)
                finally:
                    if os.path.lexists(temporary):
                        os.remove(temporary)
                break
    if not os.path.isfile(dest):
        grounding_sha = '46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7'
        grounding_url = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth'
        if relative == 'groundingdino_swinb_cogcoor.pth' and expected_sha == grounding_sha and url == grounding_url:
            # The author's Hub copy has the identical Git LFS SHA256 and byte count.
            # 作者 Hub 副本的 Git LFS SHA256 与字节数完全一致；下载后仍核验完整摘要。
            from wapr.download_route import probe_prefix, public_url
            candidates = [url,
                          'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/' + relative,
                          'https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/' + relative]
            measured = []
            for candidate in candidates:
                sample = probe_prefix(candidate, expected_total=938057991)
                if sample is not None and sample['bytes'] > 0:
                    measured.append((sample['seconds'] / sample['bytes'], candidate))
            if measured:
                url = min(measured)[1]
                print('DET2D_IDENTICAL_CHECKPOINT_ROUTE', {'url': public_url(url), 'sha256': grounding_sha}, flush=True)
        _download_file(url, dest)
    if expected_sha is not None and sha256(dest) != expected_sha:
        raise RuntimeError('Detection weight hash mismatch: ' + relative)
    return dest


def _place_config(weights_dir, spec):
    """
    # Return the GroundingDINO config path under weights_dir.

        Copy the shipped file when that copy is missing.

    ## Args

        - weights_dir: the destination directory.
        - spec: one GROUNDING_CHOICES row. spec['config'] is the filename written under weights_dir. spec['config_src'] is the filename read from third_party/GroundingDINO/groundingdino/config/.

    ## Returns

        - The return is that destination path.
        - An existing file is returned unchanged.
        - A missing source raises FileNotFoundError.

    ---

    # 返回 weights_dir 下的 GroundingDINO 配置路径。

        该副本不存在时，从 third_party/GroundingDINO/groundingdino/config/ 复制到目标目录。

    ## 参数

        - weights_dir: 目标目录。
        - spec: GROUNDING_CHOICES 的一行。spec['config'] 是写到 weights_dir 下的文件名。spec['config_src'] 是从 third_party/GroundingDINO/groundingdino/config/ 读取的文件名。

    ## 返回

        - 返回值就是该目标路径。
        - 文件已在则原样返回。
        - 源文件不存在时抛出 FileNotFoundError。

"""
    dest = os.path.join(weights_dir, spec['config'])
    if os.path.isfile(dest):
        return dest
    source = os.path.join(grounding_repo, 'groundingdino', 'config', spec['config_src'])
    if not os.path.isfile(source):
        raise FileNotFoundError('Missing GroundingDINO config: ' + source)
    os.makedirs(weights_dir, exist_ok=True)
    shutil.copyfile(source, dest)
    print('DET2D_CONFIG', {'path': dest}, flush=True)
    return dest


def _load_tensor_file(path, map_location='cpu'):
    """Load verified tensor resources across PyTorch versions.

    跨 PyTorch 版本读取已校验的张量资源；旧版没有 weights_only 参数。
    """
    load_options = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_options['weights_only'] = True
    return torch.load(path, **load_options)


def prepare_det2d_weights(weights_dir, dino='vitl14', grounding='swinb'):
    """
    # Place the selected DINOv2, GroundingDINO, SAM, and BERT files under weights_dir.

        A catalog sha256 on the chosen row is checked when that row has one.

        SAM and BERT always have hashes.

        The published pair, vitl14 with swinb, returns the SHA-256 of det2d_assets.json from verify_assets.

        Any other pair checks the shared SAM and BERT names against that JSON and returns None.

    ## Args

        - weights_dir: the directory. It is made absolute and created if needed.
        - dino: a DINO_CHOICES key. The default is vitl14. An unknown name raises ValueError before a download.
        - grounding: a GROUNDING_CHOICES key. The default is swinb. An unknown name raises ValueError before a download.

    ## Returns

        - Return the published asset id, or None.

    ---

    # 把所选 DINOv2、GroundingDINO、SAM 和 BERT 文件放到 weights_dir 下。

        所选行里如果写了 sha256，就会核对该哈希。

        SAM 和 BERT 总有哈希。

        已发布的一对是 vitl14 加 swinb，这时返回 verify_assets 给出的 det2d_assets.json 的 SHA-256。

        其他组合则按该 JSON 检查共用的 SAM 和 BERT 名字，并返回 None。

    ## 参数

        - weights_dir: 目录。会先变成绝对路径，不存在就创建。
        - dino: DINO_CHOICES 的键。默认是 vitl14。未知名字在下载前抛出 ValueError。
        - grounding: GROUNDING_CHOICES 的键。默认是 swinb。未知名字在下载前抛出 ValueError。

    ## 返回

        - 返回已发布资源的标识，或 None。

"""
    from wapr.bootstrap import ensure_optional
    ensure_optional('det2d')
    weights_dir = os.path.abspath(os.fspath(weights_dir))
    os.makedirs(weights_dir, exist_ok=True)
    dino_spec = det2d_choice(DINO_CHOICES, dino, 'dino')
    grounding_spec = det2d_choice(GROUNDING_CHOICES, grounding, 'grounding')
    _place_file(weights_dir, dino_spec['file'], dino_spec['url'], dino_spec.get('sha256'))
    _place_file(weights_dir, grounding_spec['file'], grounding_spec['url'], grounding_spec.get('sha256'))
    _place_config(weights_dir, grounding_spec)
    for relative, url, expected in SHARED_DOWNLOADS:
        _place_file(weights_dir, relative, url, expected)
    if dino_spec['hub'] == 'dinov2_vitl14' and grounding_spec['file'].startswith('groundingdino_swinb'):
        return verify_assets(weights_dir)
    with open(det2d_manifest_path) as stream:
        artifacts = json.load(stream)['artifacts']
    shared_names = [relative for relative, _url, _expected in SHARED_DOWNLOADS]
    for name in shared_names:
        if sha256(os.path.join(weights_dir, name)) != artifacts[name]['sha256']:
            raise RuntimeError('Detection weight differs from det2d_assets.json: ' + name)
    return None


def prepare_dino_weight(weights_dir, dino='vitl14'):
    """
    # Make the selected DINOv2 weight available.

        Only the selected DINOv2 checkpoint is fetched; detector assets are independent.

        Return the catalog id for vitl14, or None. This is not full detector verification.

    ## Args

        - weights_dir: the directory the file is placed in.
        - dino: a DINO_CHOICES key. The default is vitl14. Only its checkpoint is placed.

    ---

    # 保证所选 DINOv2 权重可用。

        只获取选中的 DINOv2 权重，检测器资源独立准备。

        vitl14 返回目录标识，其他返回 None；不表示已核对整个检测器资源包。

    ## 参数

        - weights_dir: 放置该文件的目录。
        - dino: DINO_CHOICES 的键。默认是 vitl14。仅放置该权重。

"""
    spec = det2d_choice(DINO_CHOICES, dino, 'dino')
    weights_dir = os.path.abspath(os.fspath(weights_dir))
    os.makedirs(weights_dir, exist_ok=True)
    _place_file(weights_dir, spec['file'], spec['url'], spec.get('sha256'))
    if spec.get('sha256') is None:
        # The small-model upstream catalog has no checksum; validate its ZIP records.
        # 小型号上游目录没有校验值，至少核验 PyTorch ZIP 索引及数据完整性。
        checkpoint = os.path.join(weights_dir, spec['file'])
        try:
            with zipfile.ZipFile(checkpoint) as archive:
                broken_record = archive.testzip()
                if broken_record is not None:
                    raise RuntimeError('Invalid checkpoint record / 权重数据记录损坏: ' + broken_record)
        except (zipfile.BadZipFile, EOFError, OSError) as error:
            raise RuntimeError('Incomplete DINOv2 checkpoint / DINOv2 权重不完整: ' + checkpoint) from error
    if str(dino) == 'vitl14':
        return sha256(det2d_manifest_path)
    return None


def detection_gate(score_min, confidence):
    """
    # Return the shared 2D score gate.

        confidence and score_min must be the same number when both are set.

        Two different numbers raise ValueError.

        This function does not replace None with score_threshold.

    ## Args

        - score_min: the gate, or None.
        - confidence: the same gate, or None. A set confidence is returned. Otherwise score_min is returned, including None.

    ---

    # 返回共用的 2D 分数阈值。

        两者都给出时，confidence 和 score_min 必须是同一个数。

        两个不同的数会抛出 ValueError。

        这个函数不会把 None 换成 score_threshold。

    ## 参数

        - score_min: 阈值，或 None。
        - confidence: 同一个阈值，或 None。confidence 已给出时返回它。否则返回 score_min，包括 None。

"""
    if confidence is not None and score_min is not None and float(confidence) != float(score_min):
        raise ValueError('confidence is %s and score_min is %s' % (confidence, score_min))
    if confidence is not None:
        return confidence
    return score_min


def _engine_is_current(engine_path, manifest_path, weight_path, device):
    """
    # Return True when this TensorRT engine matches this weight, this TensorRT, and this GPU.

        True requires all four of these: manifest tensorrt equals the installed TensorRT version, compute_capability equals that device pair, engine_sha256 equals the engine file, and weight_sha256 equals the weight file.

    ## Args

        - engine_path: the serialized engine file. A missing file returns False.
        - manifest_path: the JSON beside the engine. A missing file returns False.
        - weight_path: the DINOv2 weight file. A missing file returns False.
        - device: passed to torch.cuda.get_device_capability.

    ---

    # 这份 TensorRT 引擎对得上这份权重、这份 TensorRT 和这块 GPU 时，返回 True。

        只有下面四项全部成立才是 True：清单里的 tensorrt 等于已安装的 TensorRT 版本，compute_capability 等于该设备的 CUDA 计算能力版本号（主版本号、次版本号），engine_sha256 等于引擎文件，weight_sha256 等于权重文件。

    ## 参数

        - engine_path: 序列化后的引擎文件。文件不存在时返回 False。
        - manifest_path: 引擎旁边的 JSON。文件不存在时返回 False。
        - weight_path: DINOv2 权重文件。文件不存在时返回 False。
        - device 会传给 torch.cuda.get_device_capability。

"""
    if not os.path.isfile(engine_path) or not os.path.isfile(manifest_path) or not os.path.isfile(weight_path):
        return False
    import tensorrt as trt

    with open(manifest_path) as stream:
        manifest = json.load(stream)
    if manifest.get('tensorrt') != trt.__version__:
        return False
    if manifest.get('compute_capability') != list(torch.cuda.get_device_capability(device)):
        return False
    if sha256(engine_path) != manifest.get('engine_sha256'):
        return False
    if sha256(weight_path) != manifest.get('weight_sha256'):
        return False
    return True


def ensure_dino_engine(weights_dir=default_weights_dir, device='cuda:0', dino='vitl14', dino_repo=default_dino_repo):
    """
    # Return the FP16 DINOv2 engine path.

        Build one when the selected weight has no matching engine.

    ## Args

        - weights_dir: the weight directory. The default is default_weights_dir. It is made absolute.
        - device: the CUDA device used for the capability check and for the build. The default is cuda:0.
        - dino: a DINO_CHOICES key. The default is vitl14. It selects the weight and the engine filenames.
        - dino_repo: the local DINOv2 checkout passed into the build. The default is default_dino_repo. A matching engine does not use it and is not built again.

    ---

    # 返回 FP16 DINOv2 引擎路径。

        所选权重还没有匹配的引擎时，构建一份。

    ## 参数

        - weights_dir: 权重目录。默认是 default_weights_dir。会变成绝对路径。
        - device: 用于 GPU 架构兼容性检查和引擎构建的 CUDA 设备。默认是 cuda:0。
        - dino: DINO_CHOICES 的键。默认是 vitl14。它决定权重和引擎文件名。
        - dino_repo: 传给构建的本地 DINOv2 源码目录。默认是 default_dino_repo。引擎已经匹配时不会用到它，也不会再次构建。

"""
    weights_dir = os.path.abspath(os.fspath(weights_dir))
    spec = det2d_choice(DINO_CHOICES, dino, 'dino')
    engine_name, manifest_name, _onnx_name = dino_artifact_names(dino)
    engine_path = os.path.join(weights_dir, engine_name)
    manifest_path = os.path.join(weights_dir, manifest_name)
    weight_path = os.path.join(weights_dir, spec['file'])
    if _engine_is_current(engine_path, manifest_path, weight_path, device):
        print('DET2D_ENGINE_READY', {'dino': str(dino), 'path': engine_path}, flush=True)
        return engine_path
    print('DET2D_ENGINE_BUILD', {'dino': str(dino), 'weight': weight_path}, flush=True)
    return build_dino_engine(weights_dir, device, dino_repo, dino=dino)


def verify_assets(weights_dir):
    """
    # Check downloaded weight and BERT bytes against det2d_assets.json.

        Source checkouts under third_party/ are not hashed here.

        The DINO engine is checked by its own TensorRT manifest, not by this function.

    ## Args

        - weights_dir: the directory of those files. Names that end with .engine are skipped. A hash mismatch raises RuntimeError.

    ## Returns

        - Return the SHA-256 of that JSON file.

    ---

    # 按 det2d_assets.json 核验已下载的权重和 BERT。

        third_party/ 下的源码不在这里哈希。

        DINO 引擎由它自己的 TensorRT 清单核验，不由这个函数核验。

    ## 参数

        - weights_dir: 这些文件所在的目录。以 .engine 结尾的名字会跳过。哈希不一致时抛出 RuntimeError。

    ## 返回

        - 返回该 JSON 文件的 SHA-256。

"""
    with open(det2d_manifest_path) as stream:
        manifest = json.load(stream)
    for name, record in manifest['artifacts'].items():
        if name.endswith('.engine'):
            continue
        if sha256(os.path.join(weights_dir, name)) != record['sha256']:
            raise RuntimeError('Detection weight differs from det2d_assets.json: ' + name)
    return sha256(det2d_manifest_path)


def _save_json(path, value):
    """
    # Write value as JSON at path.

        The destination is replaced only after the temporary file is finished.

    ## Args

        - path: the final JSON path. The text is written to path + '.tmp', then os.replace moves it to path.
        - value: a JSON-serializable object. It is written with indent 2.

    ---

    # 把 value 写成 path 上的 JSON。

        临时文件写完之后才替换目标。

    ## 参数

        - path: 最终 JSON 路径。文本先写到 path + '.tmp'，再用 os.replace 移到 path。
        - value: 可序列化为 JSON 的对象。写出时缩进为 2。

"""
    with open(path + '.tmp', 'w') as stream:
        json.dump(value, stream, indent=2)
    os.replace(path + '.tmp', path)


def _view_foreground_max(rgb, mask):
    """
    # Return the maximum RGB inside each mask, one number per view.

    ## Args

        - rgb: float (V, H, W, 3), or float (V, 3, H, W) when mask is also 4D. Background pixels are treated as 0 before the max.
        - mask: (V, H, W), or (V, 1, H, W) for the channel-first rgb. Foreground is mask greater than 0.

    ## Returns

        - The return is a 1D tensor of length V.
        - A view with no foreground returns 0.

    ---

    # 返回每个 mask 内的 RGB 最大值，每个视角一个数。

    ## 参数

        - rgb: float，形状 (V, H, W, 3)；当 mask 也是 4 维时，也可以是 float (V, 3, H, W)。取最大值前，背景像素按 0 处理。
        - mask: (V, H, W)；rgb 为通道在前时，mask 是 (V, 1, H, W)。前景是 mask 大于 0 的位置。

    ## 返回

        - 返回值是长度为 V 的一维张量。
        - 没有前景的视角返回 0。

"""
    if rgb.shape[1] == 3 and rgb.ndim == 4 and mask.ndim == 4:
        spatial = mask[:, 0] > 0
        hidden = rgb.permute(0, 2, 3, 1)
    else:
        spatial = mask > 0
        hidden = rgb
    filled = hidden.masked_fill(~spatial[..., None], 0)
    return filled.flatten(1).amax(1)


def _lambert_gray_from_depth(depth, mask):
    """
    # Return gray Lambert shading from depth normals.

        Background stays 0.

        Shading uses light_dir, w_ambient, and w_diffuse from wapr/recipe.py.

        The albedo is DEFAULT_UNTEXTURED_VERTEX_COLOR.

        With those recipe weights, a face aimed at the light is that gray.

    ## Args

        - depth: (V, H, W). It is the z used by the pinhole unprojection. The camera is the template camera: focal_x is the width in pixels, focal_y is the height in pixels, and the principal point is half of each.
        - mask: (V, H, W). Pixels outside the mask are 0. A masked pixel whose normal length is at most 1e-6 keeps ambient gray only.

    ## Returns

        - The return is float (V, H, W, 3).

    ---

    # 用深度法向做灰色朗伯明暗，返回该颜色。

        背景保持 0。

        明暗使用 wapr/recipe.py 的 light_dir、w_ambient 和 w_diffuse。

        反照率是 DEFAULT_UNTEXTURED_VERTEX_COLOR。

        在这组配置的权重下，朝向光源的面就是这个灰。

    ## 参数

        - depth: (V, H, W)。它是针孔反投影用的 z。相机是模板相机：focal_x 是宽度像素，focal_y 是高度像素，主点是宽和高各自的一半。
        - mask: (V, H, W)。mask 外的像素是 0。mask 内法向长度不超过 1e-6 的像素只保留环境光灰色。

    ## 返回

        - 返回值是 float (V, H, W, 3)。

"""
    from wapr.ogl import DEFAULT_UNTEXTURED_VERTEX_COLOR
    from wapr import recipe as pose_recipe

    height, width = depth.shape[-2:]
    base = float(DEFAULT_UNTEXTURED_VERTEX_COLOR)
    z = depth
    us = torch.arange(width, device=depth.device, dtype=depth.dtype)
    vs = torch.arange(height, device=depth.device, dtype=depth.dtype)
    focal_x = float(width)
    focal_y = float(height)
    cx = focal_x / 2
    cy = focal_y / 2
    points = torch.stack([
        (us.view(1, 1, width) - cx) / focal_x * z,
        (vs.view(1, height, 1) - cy) / focal_y * z,
        z,
    ], -1)
    across = torch.zeros_like(points)
    down = torch.zeros_like(points)
    valid_across = mask[:, :, 2:] & mask[:, :, :-2]
    valid_down = mask[:, 2:, :] & mask[:, :-2, :]
    across[:, :, 1:-1] = torch.where(
        valid_across[..., None], points[:, :, 2:] - points[:, :, :-2], across[:, :, 1:-1],
    )
    down[:, 1:-1, :] = torch.where(
        valid_down[..., None], points[:, 2:, :] - points[:, :-2, :], down[:, 1:-1, :],
    )
    normal = torch.cross(across, down, dim=-1)
    length = normal.norm(dim=-1, keepdim=True)
    toward_camera = F.normalize(-points, dim=-1, eps=1e-6)
    facing = (normal * toward_camera).sum(-1, keepdim=True)
    normal = torch.where(facing < 0, -normal, normal)
    normal = normal / length.clamp_min(1e-6)
    light = normal.new_tensor(pose_recipe.light_dir)
    direction = F.normalize(-light, dim=0)
    diffuse = (normal * direction).sum(-1, keepdim=True).clamp(0, 1)
    lit = base * float(pose_recipe.w_ambient) + base * diffuse * float(pose_recipe.w_diffuse)
    usable = mask & (length[..., 0] > 1e-6)
    gray = lit.new_full(lit.shape, base * float(pose_recipe.w_ambient))
    shaded = torch.where(usable[..., None], lit, gray)
    return torch.where(mask[..., None], shaded, torch.zeros_like(shaded))


def shade_black_templates(rgb, depth):
    """
    # Replace a black CAD view with gray Lambert shading from its depth.

        A view is black when it has foreground and the maximum RGB there is below black_foreground_max, which is 0.01.

        A view with no foreground is left unchanged.

        If every view has no foreground, ValueError is raised.

        If a replaced view is still below black_foreground_max, ValueError is raised.

    ## Args

        - rgb: float (V, H, W, 3) on the 0-1 scale.
        - depth: (V, H, W). Foreground is depth greater than 0.

    ## Returns

        - Return the RGB views.
        - The return is float (V, H, W, 3).
        - Views that are not black are the input pixels.

    ---

    # 把黑色 CAD 视角换成由深度得到的灰色朗伯明暗。

        若某个视角含有前景，且前景 RGB 最大值低于 black_foreground_max（0.01），则将其视为黑色模板。

        没有前景的视角保持原样。

        如果每个视角都没有前景，抛出 ValueError。

        替换后仍低于 black_foreground_max 的视角也会抛出 ValueError。

    ## 参数

        - rgb: float (V, H, W, 3)，范围是 0 到 1。
        - depth: (V, H, W)。前景是 depth 大于 0 的位置。

    ## 返回

        - 返回这些 RGB 视角。
        - 返回值是 float (V, H, W, 3)。
        - 不是黑色的视角仍是输入像素。

"""
    mask = depth > 0
    present = mask.flatten(1).any(1)
    dark = present & (_view_foreground_max(rgb, mask) < black_foreground_max)
    if not bool(dark.any()):
        if not bool(present.any()):
            raise ValueError('Nearly black CAD render')
        return rgb
    shaded = _lambert_gray_from_depth(depth, mask)
    out = rgb.clone()
    out[dark] = shaded[dark]
    still = dark & (_view_foreground_max(out, mask) < black_foreground_max)
    if bool(still.any()):
        raise ValueError('Nearly black CAD render')
    return out


def lift_black_queries(crops, crop_masks):
    """
    # Paint a black query crop with flat gray inside its mask.

        A crop is black when it has foreground and the maximum RGB there is below black_foreground_max, which is 0.01.

        A brighter crop, and a crop with no foreground, stays unchanged.

        The paint color is DEFAULT_UNTEXTURED_VERTEX_COLOR, the same albedo a black template uses before Lambert shading.

    ## Args

        - crops: float (N, 3, H, W) on the 0-1 scale.
        - crop_masks: (N, 1, H, W). Foreground is crop_masks greater than 0.

    ## Returns

        - Return the crops.
        - The return is float (N, 3, H, W).

    ---

    # 把黑色查询裁剪在 mask 内涂成均匀的灰。

        若某个裁剪含有前景，且前景 RGB 最大值低于 black_foreground_max（0.01），则将其视为黑色裁剪。

        更亮的裁剪，以及没有前景的裁剪，保持原样。

        涂上的颜色是 DEFAULT_UNTEXTURED_VERTEX_COLOR，也就是黑色模板在朗伯明暗之前使用的反照率。

    ## 参数

        - crops: float (N, 3, H, W)，范围是 0 到 1。
        - crop_masks: (N, 1, H, W)。前景是 crop_masks 大于 0 的位置。

    ## 返回

        - 返回这些裁剪。
        - 返回值是 float (N, 3, H, W)。

"""
    from wapr.ogl import DEFAULT_UNTEXTURED_VERTEX_COLOR

    foreground = crop_masks > 0
    present = foreground.flatten(1).any(1)
    dark = present & (_view_foreground_max(crops, foreground) < black_foreground_max)
    if not bool(dark.any()):
        return crops
    gray = float(DEFAULT_UNTEXTURED_VERTEX_COLOR)
    painted = torch.where(foreground, crops.new_full((), gray), crops)
    out = crops.clone()
    out[dark] = painted[dark]
    return out


def square_crops(image_chw, masks, boxes_xyxy, coordinates=None):
    """
    # Return square bilinear crops of RGB and of the mask.

        The mask multiplies both before the sample.

    ## Args

        - image_chw: float RGB, (3, H, W) for one frame and many masks, or (N, 3, H, W) for one mask on each image.
        - masks: (N, H, W), one mask per box, in the same pixel frame as the image.
        - boxes_xyxy: (N, 4) pixel xyxy. The square side is the longer box edge times crop_margin, which is 1.1, and at least 1 pixel. The square is centered on the box.
        - coordinates: a 1D tensor of length crop_px on the image device, or None. None builds (index + 0.5) / crop_px - 0.5. crop_px is 224.

    ## Returns

        - The return is crops (N, 3, 224, 224) and a mask sample (N, 1, 224, 224).
        - The mask multiplies RGB and an extra ones channel before that one bilinear grid_sample.

    ---

    # 返回 RGB 和 mask 的正方形双线性裁剪。

        采样之前，mask 会同时乘到这两者上。

    ## 参数

        - image_chw: float RGB。(3, H, W) 表示一帧对多个 mask。(N, 3, H, W) 表示每张图一个 mask。
        - masks: (N, H, W)，每个包围盒一个 mask，像素坐标系与图像相同。
        - boxes_xyxy: (N, 4) 的像素 xyxy。正方形边长是包围盒较长边乘以 crop_margin（1.1），并且至少 1 像素。正方形以包围盒中心为中心。
        - coordinates: 图像设备上长度为 crop_px 的一维张量，或 None。None 会生成 (index + 0.5) / crop_px - 0.5。crop_px 是 224。

    ## 返回

        - 返回 crops (N, 3, 224, 224) 和 mask 采样 (N, 1, 224, 224)。
        - 在这一次双线性 grid_sample 之前，mask 乘到 RGB 和额外的全 1 通道上。

"""
    height, width = image_chw.shape[-2:]
    center = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2
    edge = ((boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]).amax(-1) * crop_margin).clamp_min(1)
    if coordinates is None:
        coordinates = (torch.arange(crop_px, device=image_chw.device) + .5) / crop_px - .5
    x = center[:, 0, None, None] + edge[:, None, None] * coordinates[None, None, :]
    y = center[:, 1, None, None] + edge[:, None, None] * coordinates[None, :, None]
    grid = torch.stack([x.expand(-1, crop_px, -1) * 2 / width - 1,
                        y.expand(-1, -1, crop_px) * 2 / height - 1], -1)
    if image_chw.ndim == 3:
        channels = torch.cat([image_chw, torch.ones_like(image_chw[:1])], 0)[None] * masks[:, None]
    else:
        channels = torch.cat([image_chw, torch.ones_like(image_chw[:, :1])], 1) * masks[:, None]
    sampled = F.grid_sample(channels, grid, mode='bilinear', align_corners=False)
    return sampled[:, :3], sampled[:, 3:4]


class NativeDino:
    """
    # Local DINOv2 encoder for the torch backend and for CAD onboarding.

        encode returns CLS, GeM, patches, and a valid mask.

        Construct it with NativeDino(weights_dir, device, dino_repo, dino).

        The .pth file must already be on disk.

        This class does not download it.

        Callers use encode(crops, crop_masks).

        Constructed by build_dino_engine, by WAPRDet2D.__init__ when backend is torch, by _bank_on_device, and by encode_render_templates.

        A direct call looks like NativeDino(...).

    ---

    # 本地 DINOv2 编码器，用于 torch 后端和 CAD 建库。

        encode 返回 CLS、GeM、patches 和 valid。

        用 NativeDino(weights_dir, device, dino_repo, dino) 构造。

        .pth 文件必须已经在磁盘上。

        这个类不下载它。

        调用方使用 encode(crops, crop_masks)。

        build_dino_engine、backend 为 torch 时的 WAPRDet2D.__init__、_bank_on_device 和 encode_render_templates 构造它。

        直接调用写成 NativeDino(...)。
    """
    def __init__(self, weights_dir=default_weights_dir, device='cuda:0', dino_repo=default_dino_repo, dino='vitl14'):
        """
        # Load one local DINOv2 weight onto device.

            ImageNet mean and standard deviation are stored for encode.

            No value is returned.

            Constructed by build_dino_engine, by WAPRDet2D.__init__ when backend is torch, by _bank_on_device, and by encode_render_templates.

            A direct call looks like NativeDino(...).

        ## Args

            - weights_dir: the .pth named by the catalog. The default is default_weights_dir. A missing file raises FileNotFoundError.
            - device: the CUDA device. The default is cuda:0.
            - dino_repo: the local DINOv2 checkout. The default is default_dino_repo. It must contain hubconf.py. The hub load uses pretrained False, then the state dict is loaded with strict True.
            - dino: vits14, vitb14, or vitl14. The default is vitl14. The patch size must be 14. The embed width must match the catalog row.

        ---

        # 把一份本地 DINOv2 权重载到 device 上。

            encode 会用到这里保存的 ImageNet 均值和标准差。

            没有返回值。

            build_dino_engine、backend 为 torch 时的 WAPRDet2D.__init__、_bank_on_device 和 encode_render_templates 构造它。

            直接调用写成 NativeDino(...)。

        ## 参数

            - weights_dir 放目录所指定的 .pth。
            - 默认是 default_weights_dir。
            - 文件不存在时抛出 FileNotFoundError。
            - device: CUDA 设备。默认是 cuda:0。
            - dino_repo: 本地 DINOv2 源码目录。默认是 default_dino_repo。它必须包含 hubconf.py。hub 加载使用 pretrained False，然后以 strict True 载入 state dict。
            - dino: vits14、vitb14 或 vitl14。默认是 vitl14。patch 大小必须是 14。embed 宽度必须和目录行一致。
        """
        from wapr.bootstrap import ensure_optional
        from wapr.installation import prepare_optional
        if os.path.abspath(dino_repo) == os.path.abspath(default_dino_repo):
            ensure_optional('dinov2')
        else:
            dependency_result = prepare_optional('dinov2')
            if dependency_result.get('status') not in ('ready', 'installed'):
                raise RuntimeError('DINOv2 dependencies not ready / DINOv2 依赖未就绪: ' + str(dependency_result))
        self.device = torch.device(device)
        self.spec = det2d_choice(DINO_CHOICES, dino, 'dino')
        self.dino = str(dino)
        self.weight_path = os.path.join(weights_dir, self.spec['file'])
        if not os.path.isfile(self.weight_path):
            raise FileNotFoundError('Missing DINOv2 weight %s. prepare_det2d_weights downloads it.' % self.weight_path)
        if not os.path.isfile(os.path.join(dino_repo, 'hubconf.py')):
            raise FileNotFoundError('Missing local DINOv2 source: ' + dino_repo)
        with torch.cuda.device(self.device):
            # Load the catalog backbone directly, preserving an already imported UniPose namespace.
            # 直接载入目录中指定的骨干，保留 UniPose 已导入的命名空间。
            if dino_repo not in sys.path:
                sys.path.insert(0, dino_repo)
            from dinov2.hub import backbones
            print('DINO_BACKBONE_SOURCE', backbones.__file__, flush=True)
            self.model = getattr(backbones, self.spec['hub'])(pretrained=False)
            self.model.load_state_dict(_load_tensor_file(self.weight_path, map_location='cpu'), strict=True)
            self.model = self.model.eval().to(self.device)
        if int(self.model.patch_size) != 14:
            raise ValueError('DINOv2 patch size must be 14')
        if int(self.model.embed_dim) != int(self.spec['embed']):
            raise ValueError('DINOv2 embed dim is %s, %s expects %s' % (self.model.embed_dim, self.dino, self.spec['embed']))
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def encode(self, crops, crop_masks):
        """
        # Encode crops into DINOv2 features.

            Each crop is one detection region. This module writes that crop's features.

        ## Args

            - crops: float (N, 3, H, W) on the 0-1 scale. This method applies ImageNet mean and standard deviation. The forward uses CUDA autocast in float16, then CLS and GeM are stored as float32.
            - crop_masks: (N, 1, H, W). A 14 by 14 patch is valid when its pooled mask is greater than patch_coverage, which is 0.5.

        ## Returns

            - gem: (N, embed) float32, GeM with p 1.5.
            - patches: (N, tokens, embed) float16. Invalid patch vectors are zero. A 224 crop has 16 by 16 tokens.
            - valid: (N, tokens) bool.
            - Rows: split by crop_batch. That split does not change the per-crop features.

        ---

        # 把裁剪编码成 DINOv2 特征。

            每一块裁剪是一个检测区域。这个模块写出这块裁剪的特征。

        ## 参数

            - crops: float (N, 3, H, W)，范围是 0 到 1。这个方法会做 ImageNet 均值和标准差归一化。前向使用 CUDA autocast 的 float16，然后 CLS 和 GeM 存成 float32。
            - crop_masks: (N, 1, H, W)。14 乘 14 的 patch 在池化后的 mask 大于 patch_coverage（0.5）时有效。

        ## 返回

            - gem: (N, embed) float32，GeM 的 p 为 1.5。
            - patches: (N, tokens, embed) float16。无效 patch 向量为 0。224 的裁剪有 16 乘 16 个 token。
            - 行按 crop_batch 切开。
            - 这样切开不改变每个裁剪自己的特征。

"""
        chunks = {'cls': [], 'gem': [], 'patches': [], 'valid': []}
        for start in range(0, len(crops), crop_batch):
            images = (crops[start:start + crop_batch] - self.mean) / self.std
            with torch.autocast('cuda', dtype=torch.float16):
                tokens = self.model.forward_features(images)
            patches = tokens['x_norm_patchtokens'].float()
            valid = F.avg_pool2d(crop_masks[start:start + crop_batch], 14, 14).flatten(1) > patch_coverage
            chunks['cls'].append(tokens['x_norm_clstoken'].float())
            chunks['gem'].append(patches.clamp_min(1e-6).pow(1.5).mean(1).pow(1 / 1.5))
            chunks['patches'].append(F.normalize(patches, dim=-1).half() * valid[..., None])
            chunks['valid'].append(valid)
        return {key: torch.cat(values) for key, values in chunks.items()}


class DinoExport(torch.nn.Module):
    """
    # ONNX wrapper around one NativeDino at a fixed 224 crop.

        forward returns float32 CLS, GeM, and patches.

        Construct it with DinoExport(encoder).

        encoder is a NativeDino.

        The 224 positional encoding is interpolated once and stored.

        Callers use forward(crops).

        The exported graph includes ImageNet normalization.

        GeM and the three outputs stay float32.

        Constructed by build_dino_engine.

        A direct call looks like DinoExport(encoder).

    ---

    # 包住一个 NativeDino 的 ONNX 包装，裁剪固定为 224。

        forward 返回 float32 的 CLS、GeM 和 patches。

        用 DinoExport(encoder) 构造。

        encoder 是 NativeDino。

        224 的位置编码只插值一次并保存。

        调用方使用 forward(crops)。

        导出的图包含 ImageNet 归一化。

        GeM 和三个输出保持 float32。

        build_dino_engine 构造它。

        直接调用写成 DinoExport(encoder)。
    """
    def __init__(self, encoder):
        """
        # Store the native module, its ImageNet norm, and the fixed 224 positional encoding.

            No other arguments.

            No value is returned.

        ## Args

            - encoder: a NativeDino. Its model, mean, and standard deviation are copied. position_224 is interpolate_pos_encoding at crop_px, which is 224.

        ## Returns

            - Return None.

        ---

        # 保存PyTorch 模型、它的 ImageNet 归一化，以及固定的 224 位置编码。

            没有其他参数。

            没有返回值。

        ## 参数

            - encoder: NativeDino。它的 model、均值和标准差被复制。position_224 是在 crop_px（224）上的 interpolate_pos_encoding。

        ## 返回

            - 返回 None。

"""
        super().__init__()
        self.model = encoder.model
        self.register_buffer('mean', encoder.mean)
        self.register_buffer('std', encoder.std)
        tokens = torch.empty((1, (crop_px // self.model.patch_size) ** 2 + 1, self.model.embed_dim), device=encoder.device)
        with torch.no_grad():
            position = self.model.interpolate_pos_encoding(tokens, crop_px, crop_px)
        self.register_buffer('position_224', position)

    def forward(self, crops):
        """
        # Encode one crop batch into DINOv2 features.

            The outputs are float32 CLS, GeM, and patch tokens.

            This path has no valid-patch mask.

        ## Args

            - crops: (N, 3, crop_px, crop_px) with crop_px 224. ImageNet normalization runs inside this method. A model with register tokens raises ValueError, because this table is plain DINOv2.

        ## Returns

            - The return is cls (N, embed), gem (N, embed) with GeM p 1.5, and patches (N, tokens, embed), all float32.

        ---

        # 把一批裁剪编码成 DINOv2 特征。

            输出是 float32 的 CLS、GeM 和 patch token。

            这条路径没有有效 patch mask。

        ## 参数

            - crops: (N, 3, crop_px, crop_px)，crop_px 为 224。ImageNet 归一化在这个方法内部进行。带 register token 的模型会抛出 ValueError，因为这张表是不带 register 的 DINOv2。

        ## 返回

            - 返回 cls (N, embed)、gem (N, embed)（GeM 的 p 为 1.5）和 patches (N, tokens, embed)，都是 float32。

"""
        tokens = self.model.patch_embed((crops - self.mean) / self.std)
        tokens = torch.cat([self.model.cls_token.expand(tokens.shape[0], -1, -1), tokens], 1) + self.position_224
        if self.model.register_tokens is not None:
            raise ValueError('This table is plain DINOv2, without register tokens')
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        cls, patches = tokens[:, 0].float(), tokens[:, 1:].float()
        gem = patches.clamp_min(1e-6).pow(1.5).mean(1).pow(1 / 1.5)
        return cls, gem, patches


def build_dino_engine(weights_dir=default_weights_dir, device='cuda:0', dino_repo=default_dino_repo, dino='vitl14'):
    """
    # Build the FP16 TensorRT engine for one DINOv2 choice and return its path.

        TensorRT 10.x is required.

        Any TensorRT version string other than 10.x raises RuntimeError.

        The export wrapper is checked against native CLS and patch tokens on a zero crop before ONNX export.

        A failed engine build raises RuntimeError.

    ## Args

        - weights_dir: the output directory. The default is default_weights_dir. The selected .pth is placed there if it is missing.
        - device: the CUDA device for the native check and the build. The default is cuda:0.
        - dino_repo: the local DINOv2 checkout passed to NativeDino. The default is default_dino_repo.
        - dino: vits14, vitb14, or vitl14. The default is vitl14. vitl14 writes the published engine filename.

    ## Returns

        - Tactics: FP16. Normalization, GeM, and the CLS, GeM, and patch outputs stay FP32.

    ---

    # 为一种 DINOv2 选择构建 FP16 TensorRT 引擎，并返回它的路径。

        必须是 TensorRT 10.x。

        不是 TensorRT 10.x 的版本字符串会抛出 RuntimeError。

        导出 ONNX 之前，会在一张零裁剪上把导出包装和PyTorch 模型的 CLS、patch token 核对。

        引擎构建失败抛出 RuntimeError。

    ## 参数

        - weights_dir: 输出目录。默认是 default_weights_dir。所选 .pth 不在时会放到这里。
        - device: 与 PyTorch 模型核对和构建使用的 CUDA 设备。默认是 cuda:0。
        - dino_repo: 传给 NativeDino 的本地 DINOv2 源码目录。默认是 default_dino_repo。
        - dino: vits14、vitb14 或 vitl14。默认是 vitl14。vitl14 写出已发布的引擎文件名。

    ## 返回

        - 策略是 FP16。
        - 归一化、GeM，以及 CLS、GeM 和 patch 输出保持 FP32。

"""
    import tensorrt as trt

    if not trt.__version__.startswith('10.'):
        raise RuntimeError('This verified exporter requires TensorRT10.x; do not reuse it silently on11.x')
    started = time.perf_counter()
    os.makedirs(weights_dir, exist_ok=True)
    spec = det2d_choice(DINO_CHOICES, dino, 'dino')
    _place_file(weights_dir, spec['file'], spec['url'], spec.get('sha256'))
    encoder = NativeDino(weights_dir, device, dino_repo, dino=dino)
    wrapper = DinoExport(encoder).eval()
    example = torch.zeros((1, 3, crop_px, crop_px), device=encoder.device)
    with torch.inference_mode():
        native = encoder.model.forward_features((example - encoder.mean) / encoder.std)
        cls, _, patches = wrapper(example)
        assert torch.equal(cls, native['x_norm_clstoken']) and torch.equal(patches, native['x_norm_patchtokens'])
    engine_name, manifest_name, onnx_name = dino_artifact_names(dino)
    onnx_path = os.path.join(weights_dir, onnx_name)
    engine_path = os.path.join(weights_dir, engine_name)
    # PyTorch 2.5+ defaults to the dynamo exporter. This engine path keeps the legacy exporter.
    # Older Torch has no dynamo argument, so the keyword is omitted there.
    # PyTorch 2.5 起默认使用 dynamo 导出器。这条引擎路径保持旧导出器。
    # 更早的 Torch 没有 dynamo 参数，因此不传这个关键字。
    export_arguments = {
        'input_names': ['crops'],
        'output_names': ['cls', 'gem', 'patches'],
        'opset_version': 17,
        'dynamic_axes': {name: {0: 'batch'} for name in ['crops', 'cls', 'gem', 'patches']},
    }
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        export_arguments['dynamo'] = False
    torch.onnx.export(wrapper, example, onnx_path, **export_arguments)
    logger = trt.Logger(trt.Logger.WARNING)
    with torch.cuda.device(encoder.device):
        builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as stream:
        if not parser.parse(stream.read()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'crops', (1, 3, crop_px, crop_px), (crop_batch, 3, crop_px, crop_px), (crop_batch, 3, crop_px, crop_px),
    )
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)
    config.builder_optimization_level = 3
    constrained = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.type == trt.LayerType.NORMALIZATION or (layer.type in {trt.LayerType.ELEMENTWISE, trt.LayerType.REDUCE}
                                                        and not layer.name.startswith('/model/')):
            layer.precision = trt.float32
            for output_index in range(layer.num_outputs):
                layer.set_output_type(output_index, trt.float32)
            constrained.append(layer.name)
    with torch.cuda.device(encoder.device):
        serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('FP16 DINO engine build failed')
    with open(engine_path + '.tmp', 'wb') as stream:
        stream.write(bytes(serialized))
    os.replace(engine_path + '.tmp', engine_path)
    manifest = {'dino': encoder.dino, 'weight': encoder.spec['file'],
                'weight_sha256': sha256(encoder.weight_path), 'engine_sha256': sha256(engine_path),
                'onnx_sha256': sha256(onnx_path), 'export_sha256': sha256(__file__), 'tensorrt': trt.__version__,
                'torch': torch.__version__, 'compute_capability': list(torch.cuda.get_device_capability(encoder.device)),
                'profile': [1, crop_batch, crop_batch], 'fp32_layers': constrained, 'native_export_cls_exact': True,
                'native_export_patches_exact': True, 'build_s': time.perf_counter() - started,
                'precision': 'FP16 tactics; FP32 normalization/GeM and outputs',
                'embed': int(encoder.model.embed_dim)}
    _save_json(os.path.join(weights_dir, manifest_name), manifest)
    return engine_path


class DinoTRT:
    """
    # One TensorRT execution of the exported DINOv2.

        encode returns CLS, GeM, patches, and a valid mask.

        Construct it with DinoTRT(weights_dir, device, dino).

        The engine and manifest must already match this weight, this TensorRT, and this GPU.

        This class does not build the engine.

        Callers use encode(crops, crop_masks).

        Outputs are float32.

        One context and one set of output buffers are reused, so two calls must not run at the same time.

        GroundingDINO and SAM are not this engine.

        Constructed by WAPRDet2D.__init__ when backend is trt.

        A direct call looks like DinoTRT(...).

    ---

    # 导出的 DINOv2 的一份 TensorRT 执行。

        encode 返回 CLS、GeM、patches 和 valid。

        用 DinoTRT(weights_dir, device, dino) 构造。

        引擎和清单必须已经对上这份权重、这份 TensorRT 和这块 GPU。

        这个类不构建引擎。

        调用方使用 encode(crops, crop_masks)。

        输出是 float32。

        一个 context 和一套输出缓冲被复用，所以两次调用不能同时进行。

        GroundingDINO 和 SAM 不是这个引擎。

        backend 为 trt 时由 WAPRDet2D.__init__ 构造。

        直接调用写成 DinoTRT(...)。
    """
    def __init__(self, weights_dir=default_weights_dir, device='cuda:0', dino='vitl14'):
        """
        # Load one serialized DINOv2 engine.

            A mismatch raises RuntimeError.

            weights_dir contains the engine, the manifest, and the .pth.

            The default is default_weights_dir.

            The engine inputs and outputs are crops, cls, gem, and patches, all float32.

            The profile spatial size is crop_px, which is 224, and the max batch must be at least crop_batch.

            No value is returned.

            Constructed by WAPRDet2D.__init__ when backend is trt.

            A direct call looks like DinoTRT(...).

        ## Args

            - device: the CUDA device. The default is cuda:0. The manifest compute capability must equal this device, and the manifest TensorRT version must equal the installed one.
            - dino: vits14, vitb14, or vitl14. The default is vitl14. It selects the engine and weight filenames. The engine SHA-256 and the weight SHA-256 must match the manifest.

        ---

        # 载入一份序列化的 DINOv2 引擎。

            不一致时抛出 RuntimeError。

            weights_dir 里有引擎、清单和 .pth。

            默认是 default_weights_dir。

            引擎的输入和输出是 crops、cls、gem 和 patches，都是 float32。

            profile 的空间尺寸是 crop_px，也就是 224，最大 batch 必须至少是 crop_batch。

            没有返回值。

            backend 为 trt 时由 WAPRDet2D.__init__ 构造。

            直接调用写成 DinoTRT(...)。

        ## 参数

            - device: CUDA 设备。默认是 cuda:0。清单里的 CUDA 计算能力版本号必须与这块 GPU 一致，清单里的 TensorRT 版本必须等于已安装的版本。
            - dino: vits14、vitb14 或 vitl14。默认是 vitl14。它决定引擎和权重文件名。引擎的 SHA-256 和权重的 SHA-256 必须和清单一致。
        """
        import tensorrt as trt

        self.device = torch.device(device)
        self.spec = det2d_choice(DINO_CHOICES, dino, 'dino')
        self.dino = str(dino)
        engine_name, manifest_name, _onnx_name = dino_artifact_names(dino)
        engine_path = os.path.join(weights_dir, engine_name)
        self.weight_path = os.path.join(weights_dir, self.spec['file'])
        with open(os.path.join(weights_dir, manifest_name)) as stream:
            manifest = json.load(stream)
        if manifest['tensorrt'] != trt.__version__ or manifest['compute_capability'] != list(torch.cuda.get_device_capability(self.device)):
            raise RuntimeError('TRT version/GPU architecture mismatch: explicitly rebuild the DINO engine')
        if sha256(engine_path) != manifest['engine_sha256'] or sha256(self.weight_path) != manifest['weight_sha256']:
            raise RuntimeError('DINO engine/weight identity mismatch')
        self.logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            with open(engine_path, 'rb') as stream:
                self.engine = self.runtime.deserialize_cuda_engine(stream.read())
            if self.engine is None:
                raise RuntimeError('Cannot deserialize DINO engine')
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream(device=self.device)
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != {'crops', 'cls', 'gem', 'patches'} or any(self.engine.get_tensor_dtype(name) != trt.float32 for name in names):
            raise RuntimeError('Unexpected DINO engine I/O')
        maximum = self.engine.get_tensor_profile_shape('crops', 0)[2]
        if tuple(maximum[1:]) != (3, crop_px, crop_px) or maximum[0] < crop_batch:
            raise RuntimeError('Incompatible within-frame crop profile')
        self.context.set_input_shape('crops', (crop_batch, 3, crop_px, crop_px))
        self.buffers = {name: torch.empty(tuple(self.context.get_tensor_shape(name)), device=self.device, dtype=torch.float32)
                        for name in ['cls', 'gem', 'patches']}

    @torch.inference_mode()
    def encode(self, crops, crop_masks):
        """
        # Encode crops into DINOv2 features.

            Each crop is one detection region. The engine writes that crop's features.

        ## Args

            - crops: (N, 3, H, W) on the 0-1 scale, cast to float32 here. H and W are crop_px, 224. The engine applies the exported ImageNet norm. A shape outside the profile raises RuntimeError.
            - crop_masks: (N, 1, H, W). A 14 by 14 patch is valid when its pooled mask is greater than patch_coverage, which is 0.5.

        ## Returns

            - Normalization: inside the FP16 engine, and the copied outputs are float32.
            - The return dict has cls and gem as float32 clones, patches as float16 unit vectors with invalid rows zeroed, and valid as bool.
            - CLS and GeM: cloned before the next chunk reuses the buffers. A failed execute raises RuntimeError.

        ---

        # 把裁剪编码成 DINOv2 特征。

            每一块裁剪是一个检测区域。引擎写出这块裁剪的特征。

        ## 参数

            - crops: (N, 3, H, W)，范围是 0 到 1，在这里转成 float32。H 和 W 是 crop_px，即 224。引擎应用导出的 ImageNet 归一化。超出 profile 的形状抛出 RuntimeError。
            - crop_masks: (N, 1, H, W)。14 乘 14 的 patch 在池化后的 mask 大于 patch_coverage（0.5）时有效。

        ## 返回

            - 归一化在 FP16 引擎内部，复制出来的输出是 float32。
            - 返回字典里，cls 和 gem 是 float32 克隆，patches 是 float16 单位向量且无效行为 0，valid 是 bool。
            - 下一块复用缓冲之前，CLS 和 GeM 已经克隆出来。
            - 执行失败抛出 RuntimeError。

"""
        chunks = {'cls': [], 'gem': [], 'patches': [], 'valid': []}
        for start in range(0, len(crops), crop_batch):
            inputs = crops[start:start + crop_batch].float().contiguous()
            if not self.context.set_input_shape('crops', tuple(inputs.shape)):
                raise RuntimeError('DINO profile rejected crop shape')
            outputs = {name: tensor[:len(inputs)] for name, tensor in self.buffers.items()}
            self.context.set_tensor_address('crops', inputs.data_ptr())
            for name, tensor in outputs.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            current = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(current)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('DINO TRT execution failed')
            current.wait_stream(self.stream)
            inputs.record_stream(self.stream)
            for tensor in outputs.values():
                tensor.record_stream(self.stream)
            valid = F.avg_pool2d(crop_masks[start:start + crop_batch], 14, 14).flatten(1) > patch_coverage
            chunks['cls'].append(outputs['cls'].clone())
            chunks['gem'].append(outputs['gem'].clone())
            chunks['patches'].append(F.normalize(outputs['patches'], dim=-1).half() * valid[..., None])
            chunks['valid'].append(valid)
        return {key: torch.cat(values) for key, values in chunks.items()}


class GroundingSAM:
    """
    # Frozen FP32 GroundingDINO proposals and one SAM 2.1-L box prompt per image.

        predict returns boxes, visible masks, and objectness.

        Construct it with GroundingSAM(weights_dir, device, grounding).

        Weights must already be on disk.

        SAM is sam2.1_l.pt.

        The text encoder is the local bert-base-uncased directory.

        Callers use predict(rgb).

        Candidates are not truncated, and the SAM prompts are not split across calls.

        This is not a TensorRT engine.

        Constructed by WAPRDet2D.__init__. A direct call looks like GroundingSAM(...).

    ---

    # 冻结的 FP32 GroundingDINO 候选，以及每张图一次 SAM 2.1-L 包围盒提示。

        predict 返回包围盒、可见 mask 和 objectness。

        用 GroundingSAM(weights_dir, device, grounding) 构造。

        权重必须已经在磁盘上。

        SAM 是 sam2.1_l.pt。

        文本编码器是本地的 bert-base-uncased 目录。

        调用方使用 predict(rgb)。

        候选不会被截断，SAM 提示也不会拆成多次调用。

        这不是 TensorRT 引擎。

        WAPRDet2D.__init__ 构造它。

        直接调用写成 GroundingSAM(...)。
    """
    def __init__(self, weights_dir=default_weights_dir, device='cuda:0', grounding='swinb'):
        """
        # Load GroundingDINO and SAM 2.1-L in FP32.

            weights_dir contains the GroundingDINO config, the checkpoint, bert-base-uncased, and sam2.1_l.pt.

            The default is default_weights_dir.

            No value is returned.

        ## Args

            - device: the CUDA device. The default is cuda:0.
            - grounding: swinb or swint. The default is swinb. A checkpoint key mismatch, outside the allowed position-id and label-encoder keys, raises RuntimeError.

        ---

        # 以 FP32 载入 GroundingDINO 和 SAM 2.1-L。

            weights_dir 里有 GroundingDINO 配置、权重、bert-base-uncased 和 sam2.1_l.pt。

            默认是 default_weights_dir。

            没有返回值。

        ## 参数

            - device: CUDA 设备。默认是 cuda:0。
            - grounding: swinb 或 swint。默认是 swinb。权重的键若超出允许的 position-id 和 label-encoder 键，就抛出 RuntimeError。

"""
        from wapr.bootstrap import ensure_optional
        ensure_optional('det2d')
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        from groundingdino.datasets import transforms
        # Package changes must pass WAPR's reviewed installation plan.
        # 包变更必须经过 WAPR 安装计划确认，禁用上游隐式自动安装。
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        from ultralytics import SAM

        self.device = torch.device(device)
        self.spec = det2d_choice(GROUNDING_CHOICES, grounding, 'grounding')
        self.grounding = str(grounding)
        with open(os.path.join(weights_dir, self.spec['config'])) as stream:
            values = json.load(stream)
        values.update(text_encoder_type=os.path.join(weights_dir, 'bert-base-uncased'), device=str(self.device))
        self.model = build_model(SLConfig(values))
        checkpoint = _load_tensor_file(os.path.join(weights_dir, self.spec['file']), map_location='cpu')
        incompatible = self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        if set(incompatible.missing_keys) - {'bert.embeddings.position_ids'} or set(incompatible.unexpected_keys) - {'label_enc.weight', 'bert.embeddings.position_ids'}:
            raise RuntimeError('GroundingDINO checkpoint mismatch: ' + str(incompatible))
        self.model = self.model.eval().to(self.device)
        self.transform = transforms.Compose([transforms.RandomResize([800], max_size=1333), transforms.ToTensor(),
                                             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.segmentor = SAM(os.path.join(weights_dir, 'sam2.1_l.pt'))

    @torch.inference_mode()
    def predict(self, rgb):
        """
        # Propose 2D boxes before a class name.

            No category yet. Boxes, visible masks, and objectness. Class identity is assigned later.

            Class identity does not come from the text.

            Boxes with objectness below box_threshold, which is 0.1, are dropped.

            Remaining boxes are pixel xyxy.

            Surviving boxes are suppressed with box NMS at box_nms_threshold, which is 0.7, then sent to SAM in one call.

            Masks are clipped to their boxes, and empty masks are dropped.

        ## Args

            - rgb: an H by W by 3 RGB array. The caption is the module caption, 'items .'. CUDA autocast is off, and SAM runs with half False, so this call stays FP32.

        ## Returns

            - boxes: (N, 4) float xyxy pixels on the detector device. N can be 0.
            - masks: bool (N, H, W) on that device. A SAM shape other than (N, H, W) raises RuntimeError.
            - objectness: (N,) on that device.

        ---

        # 在有类别名之前提出二维框。

            这时候还没有类别。返回包围盒、可见 mask 和 objectness。类别要到后面才定。

            类别身份不来自文本。

            objectness 低于 box_threshold（0.1）的包围盒被丢掉。

            留下的包围盒是像素 xyxy。

            留下的包围盒用 box_nms_threshold（0.7）做包围盒 NMS，然后一次送给 SAM。

            mask 被裁到各自的包围盒内，空 mask 被丢掉。

        ## 参数

            - rgb: 高乘宽乘 3 的 RGB 数组。caption 是模块里的 caption，即 'items .'。CUDA autocast 关闭，SAM 以 half False 运行，所以这次调用保持 FP32。

        ## 返回

            - boxes: 检测器设备上的 (N, 4) float 像素 xyxy。N 可以是 0。
            - masks: 该设备上的 bool (N, H, W)。SAM 的形状不是 (N, H, W) 时抛出 RuntimeError。
            - objectness: 该设备上的 (N,)。

"""
        from PIL import Image
        from torchvision.ops import box_convert, nms

        height, width = rgb.shape[:2]
        image, _ = self.transform(Image.fromarray(rgb), None)
        with torch.autocast('cuda', enabled=False):
            output = self.model(image.to(self.device)[None], captions=[caption])
        objectness = output['pred_logits'][0].sigmoid().max(-1).values
        selected = objectness >= box_threshold
        boxes = box_convert(output['pred_boxes'][0][selected], 'cxcywh', 'xyxy')
        boxes *= torch.tensor([width, height, width, height], device=self.device)
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        objectness = objectness[selected]
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, objectness = boxes[valid], objectness[valid]
        keep = nms(boxes, objectness, box_nms_threshold)
        boxes, objectness = boxes[keep], objectness[keep]
        if not len(boxes):
            return boxes, torch.empty((0, height, width), device=self.device, dtype=torch.bool), objectness
        with torch.autocast('cuda', enabled=False):
            result = self.segmentor.predict(np.ascontiguousarray(rgb[:, :, ::-1]), bboxes=boxes.cpu().numpy().tolist(),
                                            device=self.device, half=False, verbose=False, save=False)[0]
        masks = result.masks.data.bool()
        if masks.shape != (len(boxes), height, width):
            raise RuntimeError('SAM returned unexpected mask dimensions')
        xs = torch.arange(width, device=self.device)[None, None, :]
        ys = torch.arange(height, device=self.device)[None, :, None]
        masks &= (xs >= boxes[:, 0, None, None]) & (xs < boxes[:, 2, None, None]) & (ys >= boxes[:, 1, None, None]) & (ys < boxes[:, 3, None, None])
        valid = masks.flatten(1).any(1)
        return boxes[valid], masks[valid], objectness[valid]


class TemplateMatcher:
    """
    # Resident CAD feature bank and the cached global terms used to score proposals.

        score returns one FP32 score per class.

        Construct it with TemplateMatcher(bank).

        bank is the template dict already on device.

        The view count must be template_views times the four in-plane turns, which is 168.

        Callers use score(query, objectness).

        The library is not reduced inside this class.

        Constructed by WAPRDet2D.__init__. A direct call looks like TemplateMatcher(bank).

    ---

    # 常驻的 CAD 特征库，以及给候选打分时用的缓存全局项。

        score 为每个类别返回一个 FP32 分数。

        用 TemplateMatcher(bank) 构造。

        bank 是已经在设备上的模板字典。

        视角数必须是 template_views 乘以四个平面内旋转，也就是 168。

        调用方使用 score(query, objectness)。

        这个类内部不会缩减库。

        WAPRDet2D.__init__ 构造它。

        直接调用写成 TemplateMatcher(bank)。
    """
    def __init__(self, bank):
        """
        # Cache the bank's global CLS and GeM terms.

            A failed shape check raises AssertionError.

            No value is returned.

        ## Args

            - bank: a dict with cls, gem, and patches on one device. cls is (objects, views, channels). views must equal template_views times len(inplane_turns). gem has the same shape. patches is (objects, views, 256, channels).

        ---

        # 缓存库里的全局 CLS 和 GeM 项。

            形状检查失败抛出 AssertionError。

            没有返回值。

        ## 参数

            - bank: 同一设备上带 cls、gem 和 patches 的字典。cls 是 (objects, views, channels)。views 必须等于 template_views 乘以 len(inplane_turns)。gem 形状相同。patches 是 (objects, views, 256, channels)。

"""
        self.bank = bank
        objects, views, channels = bank['cls'].shape
        assert views == template_views * len(inplane_turns) and channels > 0
        assert bank['gem'].shape == bank['cls'].shape and bank['patches'].shape == (objects, views, 256, channels)
        self.references = {name: bank[name].reshape(-1, channels) for name in ['cls', 'gem']}
        self.squared = {name: values.square().sum(-1)[None] for name, values in self.references.items()}
        self.semantic = F.normalize(self.references['cls'], dim=-1).T.contiguous()
        self.object_indices = torch.arange(objects, device=bank['cls'].device)[None, :, None]

    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def score(self, query, objectness):
        """
        # Fuse global and local evidence on one shared view shortlist.

            The shortlist is the global_views strongest mean CLS/GeM Tanimoto views, global_views being 5, plus the best CLS view.

            Each class keeps the best of those views.

            That view mixes the global term and the local patch term.

            Absolute similarity is then mixed with a class-axis softmax, and the powered objectness multiplies the result.

        ## Args

            - query: the encode dict: cls, gem, patches, and valid, one row per proposal. CUDA autocast is off for this method.
            - objectness: the GroundingDINO score, shape (N,). It is clamped at 1e-6 and raised to objectness_power, which is 0.1.

        ## Returns

            - Return one FP32 score per class.
            - The return is (N, objects) float32.

        ---

        # 在同一组入选视角上融合全局和局部证据。

            入选视角是 CLS/GeM 平均 Tanimoto 最强的 global_views 个，global_views 是 5，再加 CLS 最好的视角。

            每个类别保留这些视角里最好的一个。

            该视角先混合全局项和局部 patch 项。

            绝对相似度再和类别轴 softmax 混合，objectness 的幂乘在结果上。

        ## 参数

            - query: encode 的字典：cls、gem、patches 和 valid，每个候选一行。这个方法关闭 CUDA autocast。
            - objectness: GroundingDINO 分数，形状 (N,)。先限制到不低于 1e-6，再取 objectness_power 次方，这个指数是 0.1。

        ## 返回

            - 每个类别返回一个 FP32 分数。
            - 返回值是 (N, objects) float32。

"""
        objects, views, _ = self.bank['cls'].shape
        semantic = (F.normalize(query['cls'], dim=-1) @ self.semantic).reshape(-1, objects, views)
        query_pair = torch.stack([query['cls'], query['gem']], 0)
        reference_pair = torch.stack([self.references['cls'], self.references['gem']], 0)
        squared_pair = torch.stack([self.squared['cls'], self.squared['gem']], 0)
        dot = query_pair @ reference_pair.transpose(-1, -2)
        denominator = query_pair.square().sum(-1).unsqueeze(-1) + squared_pair - dot
        absolute = (dot / denominator.clamp_min(1e-6)).reshape(2, -1, objects, views)
        per_view = absolute.mean(0)
        selected = torch.argsort(per_view, dim=-1, descending=True, stable=True)[..., :global_views]
        selected = torch.cat([selected, semantic.argmax(-1)[..., None]], -1)
        local = []
        for start in range(0, len(objectness), matching_batch):
            reference = self.bank['patches'][self.object_indices, selected[start:start + matching_batch]]
            similarity = query['patches'][start:start + matching_batch, None, None] @ reference.transpose(-1, -2)
            count = query['valid'][start:start + matching_batch].sum(-1)
            local.append((similarity.amax(-1).float().sum(-1) / count[:, None, None].clamp_min(1)).clamp(0, 1))
        combined = (global_weight * torch.gather(per_view, -1, selected) + (1 - global_weight) * torch.cat(local)).amax(-1)
        return objectness[:, None].clamp_min(1e-6).pow(objectness_power) * (
            absolute_weight * combined + (1 - absolute_weight) * F.softmax(combined / class_temperature, -1))


def mask_nms_same_category(scores, obj_ids, masks, score_min=None):
    """
    # Mask-IoU NMS inside one category.

        Different categories are not compared, so an overlap between two categories is kept.

        The first return is numpy indices of kept rows, highest score first, stable.

        The second return is the NumPy best-class score for every input row, including dropped rows. It is not a calibrated probability.

        The third return is the numpy obj_id of every input row, including rows that were dropped.

    ## Args

        - scores: a tensor (N, C) of class scores.
        - obj_ids: a 1D tensor of length C, aligned with the class axis. The identity of a row is the obj_id of its best class.
        - masks: bool (N, H, W). Overlap is counted on the GPU with a float32 dot product, which is exact up to 2**24 pixels. A larger image raises ValueError.
        - score_min: the detection-score gate, or None. None uses score_threshold, which is 0.1. A row stays when its best-class score is greater than or equal to that gate.

    ## Returns

        - Return kept indices, per-row best-class scores, and per-row class ids.
        - Suppression: per obj_id. IoU greater than mask_nms_threshold, which is 0.5, drops the lower score. IoU equal to 0.5 is kept.

    ---

    # 同一类别内的掩膜 IoU NMS。

        不同类别互不比较，两个类别的 mask 重叠时都留下。

        第一个返回值是保留行的 numpy 下标，按分数从高到低，稳定排序。

        第二个返回值是每个输入行的 NumPy 最优类别分数，包括被丢掉的行；它不是校准概率。

        第三个返回值是每个输入行的 numpy obj_id，包括被丢掉的行。

    ## 参数

        - scores: 类别分数张量 (N, C)。
        - obj_ids: 长度为 C 的一维张量，与类别轴对齐。一行的身份是它最佳类别的 obj_id。
        - masks: bool (N, H, W)。交叠在 GPU 上用 float32 点积计数，像素数不超过 2**24 时是精确的。更大的图像抛出 ValueError。
        - score_min: 检测分数阈值，或 None。None 使用 score_threshold，也就是 0.1。一行的最佳类别分数不低于这个阈值才留下。

    ## 返回

        - 返回保留下标、每行最优类别分数和每行类别 id。
        - 抑制按 obj_id 进行。
        - IoU 大于 mask_nms_threshold（0.5）时丢掉较低分。
        - IoU 等于 0.5 会保留。

"""
    gate = float(score_threshold if score_min is None else score_min)
    confidence, identity = scores.max(1)
    identities = obj_ids[identity]
    selected = torch.nonzero(confidence >= gate).flatten()
    from wapr.suppression import mask_iou_nms_indices_gpu
    
    # Only the final public NumPy output returns to the host.
    # 保留原有的空掩码处理与稳定分数顺序；抑制在 CUDA，最后仅为既有 NumPy
    # 返回协议传回下标、分数与类别。
    kept = mask_iou_nms_indices_gpu(masks[selected], confidence[selected], identities[selected],
        mask_nms_threshold, drop_empty=False)
    kept = selected[kept]
    order = torch.argsort(confidence[kept], descending=True, stable=True)
    return kept[order].cpu().numpy(), confidence.cpu().numpy(), identities.cpu().numpy()


class WAPRDet2D:
    """
    # Unseen-object RGB 2D detection.

        One detector for objects that were not in a fixed training set. The image is RGB. It is bound to one CAD bank.

        detect_many_categories_many_instances returns instances and timing.

        Construct it with WAPRDet2D(template, weights_dir, device, backend, dino, grounding).

        template is a GPU bank dict or a .pt path.

        device must be CUDA.

        backend trt is the default and does not fall back to torch.

        Callers use detect_many_categories_many_instances(rgb, ...).

        GroundingDINO and SAM stay FP32.

        backend trt uses the FP16 DINOv2 TensorRT engine and builds it when the selected weight has no matching engine.


        A direct call looks like WAPRDet2D(template, ...).

    ---

    # 未见物体的 RGB 二维检测。

        给不在固定训练集里的物体做检测。图像是 RGB。绑在一个 CAD 库上。

        detect_many_categories_many_instances 返回实例和计时。

        用 WAPRDet2D(template, weights_dir, device, backend, dino, grounding) 构造。

        template 是显存里的库或一个 .pt 路径。

        device 必须是 CUDA。

        backend 默认 trt，不会退回 torch。

        调用方使用 detect_many_categories_many_instances(rgb, ...)。

        GroundingDINO 和 SAM 保持 FP32。

        backend 为 trt 时使用 FP16 DINOv2 TensorRT 引擎，所选权重还没有匹配引擎时会构建一份。


        直接调用写成 WAPRDet2D(template, ...)。
    """
    def __init__(self, template, weights_dir=default_weights_dir, device='cuda:0', backend='trt', dino='vitl14', grounding='swinb'):
        """
        # Unseen-object RGB 2D detection.

            Load the weights, the CAD bank, the DINOv2 encoder, GroundingDINO, and SAM.

            obj_ids in the bank must be 1D, unique, and the same length as the class axis.


            A direct call looks like WAPRDet2D(template, ...).

        ## Args

            - template: a bank dict, or a .pt path. A dict has its tensors moved to device. A path also reads the sibling .json, and both the bank hash and the DINO weight hash must match that manifest.
            - weights_dir: the detection-weight directory. The default is default_weights_dir. Missing files are downloaded there.
            - device: a CUDA device. The default is cuda:0. A non-CUDA device raises ValueError.
            - backend: trt or torch. The default is trt. Any other name raises ValueError. trt builds the DINOv2 engine when needed and uses DinoTRT. torch uses NativeDino. There is no fallback from trt to torch.
            - dino: vits14, vitb14, or vitl14. The default is vitl14. The bank's last dimension must equal that embed width.
            - grounding: swinb or swint. The default is swinb. GroundingDINO and SAM stay FP32 and are not TensorRT engines.

        ## Returns

            - Return None.
            - loading_s: the load time in seconds. last_evidence starts as None. No value is returned.

        ---

        # 未见物体的 RGB 二维检测。

            载入权重、CAD 库、DINOv2 编码器、GroundingDINO 和 SAM。

            库里的 obj_ids 必须是一维、互异，并且和类别轴等长。


            直接调用写成 WAPRDet2D(template, ...)。

        ## 参数

            - template: 库字典，或一个 .pt 路径。字典里的张量会移到 device。路径还会读取旁边的 .json，库哈希和 DINO 权重哈希都必须和该清单一致。
            - weights_dir: 检测权重目录。默认是 default_weights_dir。缺的文件会下载到这里。
            - device: CUDA 设备。默认是 cuda:0。不是 CUDA 的设备会抛出 ValueError。
            - backend: trt 或 torch。默认是 trt。其他名字抛出 ValueError。trt 在需要时构建 DINOv2 引擎并使用 DinoTRT。torch 使用 NativeDino。不会从 trt 退回 torch。
            - dino: vits14、vitb14 或 vitl14。默认是 vitl14。库的最后一维必须等于该 embed 宽度。
            - grounding: swinb 或 swint。默认是 swinb。GroundingDINO 和 SAM 保持 FP32，不是 TensorRT 引擎。

        ## 返回

            - 返回 None。
            - loading_s: 加载耗时，单位秒。last_evidence 初始为 None。没有返回值。
        """
        if backend not in ['trt', 'torch']:
            raise ValueError('backend must be trt or torch')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('The released accuracy recipe requires a CUDA GPU')
        self.backend = backend
        self.dino = str(dino)
        self.grounding = str(grounding)
        self.weights_dir = os.path.abspath(os.fspath(weights_dir))
        started = time.perf_counter()
        self.assets_identity = prepare_det2d_weights(self.weights_dir, dino=self.dino, grounding=self.grounding)
        if backend == 'trt':
            ensure_dino_engine(self.weights_dir, self.device, dino=self.dino)
        dino_path = os.path.join(self.weights_dir, DINO_CHOICES[self.dino]['file'])
        if isinstance(template, dict):
            bank = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in template.items()}
            template_bytes = sum(value.nbytes for value in bank.values() if torch.is_tensor(value))
            template_store = 'gpu'
        else:
            template_path = os.fspath(template)
            with open(template_path + '.json') as stream:
                manifest = json.load(stream)
            if sha256(template_path) != manifest['bank_sha256']:
                raise RuntimeError('Template bank hash mismatch')
            if sha256(dino_path) != manifest['weight_sha256']:
                raise RuntimeError('Template bank and DINO weight differ')
            bank = _load_tensor_file(template_path, map_location=self.device)
            template_bytes = os.path.getsize(template_path)
            template_store = 'disk'
        obj_ids = bank['obj_ids']
        if obj_ids.ndim != 1 or len(obj_ids) != len(bank['cls']) or len(set(obj_ids.tolist())) != len(obj_ids):
            raise ValueError('Template obj_ids must be unique and ordered')
        embed = int(DINO_CHOICES[self.dino]['embed'])
        if int(bank['cls'].shape[-1]) != embed:
            raise ValueError('Template bank width is %s, dino %s uses %s. Rebuild the bank with this dino.' % (
                int(bank['cls'].shape[-1]), self.dino, embed))
        self.matcher = TemplateMatcher(bank)
        with torch.cuda.device(self.device):
            self.encoder = DinoTRT(self.weights_dir, self.device, dino=self.dino) if backend == 'trt' else NativeDino(self.weights_dir, self.device, dino=self.dino)
            self.proposals = GroundingSAM(self.weights_dir, self.device, grounding=self.grounding)
        self.coordinates = (torch.arange(crop_px, device=self.device) + .5) / crop_px - .5
        torch.cuda.synchronize(self.device)
        self.loading_s = time.perf_counter() - started
        self.last_evidence = None
        print('WAPR_DET2D', {'backend': backend, 'dino': self.dino, 'grounding': self.grounding,
                            'GD_SAM': 'FP32', 'obj_ids': obj_ids.tolist(),
                            'views': bank['cls'].shape[1], 'crop_px': crop_px, 'crop_batch': crop_batch,
                            'global_views': global_views, 'matching_batch': matching_batch,
                            'gate': score_threshold, 'mask_NMS': mask_nms_threshold,
                            'template_store': template_store, 'template_bytes': template_bytes,
                            'loading_s': self.loading_s}, flush=True)

    @torch.inference_mode()
    def detect_many_categories_many_instances(self, rgb, profile=False, obj_ids=None, score_min=None, confidence=None):
        """
        # 2D detection of many categories and many instances.

            2D boxes and masks only. No 6D pose. obj_ids None scores every category in the template bank. Returns (instances, timing).

            The gate passed onward is confidence when it is set, otherwise score_min.

            None there means mask_nms_same_category uses score_threshold, which is 0.1.

            A row stays when its best allowed class is greater than or equal to that gate.

            A crop whose foreground maximum is below black_foreground_max is painted flat gray before DINO.

            backend trt encodes with the FP16 TensorRT engine and copies float32 outputs.

            backend torch encodes with NativeDino, whose encode uses float16 autocast.

            timing has total_ms, candidates, outputs, and stage_times_synchronized.

            profile True also adds candidate_segmentation_ms, crop_feature_ms, matching_ms, and postprocess_output_ms.

            self.last_evidence stores the proposal boxes, masks, objectness, and the class-score matrix.

        ## Args

            - rgb: uint8 HWC RGB, shape (H, W, 3). Any other array raises ValueError.
            - profile: False by default. False does not add stage synchronizes. True synchronizes around proposal, encode, matching, and postprocess, and adds those times to timing.
            - obj_ids: a sequence of class ids, or None. None scores every class in the bank. An id that is not in the bank raises ValueError. Other classes are removed before the winning class is chosen.
            - score_min: the detection-score gate, or None.
            - confidence: the same score gate, or None. The name does not imply a calibrated probability. Both set to different numbers raises ValueError.

        ## Returns

            - Autocast: off around proposals, SAM, and scoring. GroundingDINO and SAM stay FP32.
            - instances: a list of dicts. It may be empty, and one class may appear more than once.
            - obj_id: the winning class id.
            - score_2d: that class's float ranking score, not a calibrated probability.
            - bbox: xywh in pixels: left, top, width, and height of the detector box.
            - mask: the visible COCO RLE.
            - counts: ascii.
            - mask_bbox: xywh in pixels from that RLE.

        ---

        # 多类别、多实例的二维检测。

            只要 2D 框和 mask。没有 6D 位姿。obj_ids 为 None 时给模板库里的每个类别打分。返回 (instances, timing)。

            向后传递的阈值是：confidence 已给出时用它，否则用 score_min。

            若传入 None，mask_nms_same_category 会使用默认 score_threshold（0.1）。

            允许类别里的最高分不低于这个阈值，该行才留下。

            前景最大值低于 black_foreground_max 的裁剪，在进入 DINO 之前被涂成均匀的灰。

            backend 为 trt 时用 FP16 TensorRT 引擎编码，并复制出 float32 输出。

            backend 为 torch 时用 NativeDino 编码，它的 encode 使用 float16 autocast。

            timing 含 total_ms、candidates、outputs 和 stage_times_synchronized。

            profile 为 True 时还会加入 candidate_segmentation_ms、crop_feature_ms、matching_ms 和 postprocess_output_ms。

            self.last_evidence 保存候选 boxes、masks、objectness 和类别分数矩阵。

        ## 参数

            - rgb: uint8 HWC RGB，形状 (H, W, 3)。其他数组抛出 ValueError。
            - profile 默认是 False。
            - False 不会为各阶段额外同步。
            - True 会在候选、编码、匹配和后处理四周同步，并把这些时间写进 timing。
            - obj_ids: 类别 id 序列，或 None。None 对库里的每个类别打分。不在库里的 id 抛出 ValueError。其他类别会在选出获胜类别之前被去掉。
            - score_min: 检测分数阈值，或 None。
            - confidence: 同一个分数阈值，或 None。该名称不表示校准概率；两者都给出且数值不同会抛出 ValueError。

        ## 返回

            - 候选、SAM 和打分周围关闭 autocast。
            - GroundingDINO 和 SAM 保持 FP32。
            - instances: 字典列表。它可以是空的，同一个类别也可以出现多次。
            - obj_id: 获胜类别的 id。
            - score_2d: 该类别的 float 排序分数，不是校准概率。
            - bbox: 像素 xywh：检测包围盒的左、上、宽、高。
            - mask: 可见 COCO RLE。
            - counts: ascii。
            - mask_bbox: 由该 RLE 得到的像素 xywh。

"""
        from pycocotools import mask as mask_utils

        gate = detection_gate(score_min, confidence)

        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('Expected uint8 HWC RGB')
        if profile:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        # A pose caller's outer AMP context must not change detector/SAM/scoring precision.
        # 姿态调用方的外层AMP不能改变检测器、SAM和评分精度；编码器自行选择AMP/TRT。
        with torch.cuda.device(self.device), torch.autocast('cuda', enabled=False):
            boxes, masks, objectness = self.proposals.predict(rgb)
            if profile:
                torch.cuda.synchronize(self.device)
            proposed = time.perf_counter()
            query = None
            if len(boxes):
                image_array = np.ascontiguousarray(rgb) if rgb.flags.writeable else rgb.copy()
                image = torch.from_numpy(image_array).to(self.device).permute(2, 0, 1).float() / 255
                chunks = []
                for start in range(0, len(boxes), crop_batch):
                    crops, crop_masks = square_crops(image, masks[start:start + crop_batch], boxes[start:start + crop_batch], self.coordinates)
                    crops = lift_black_queries(crops, crop_masks)
                    chunks.append(self.encoder.encode(crops, crop_masks))
                query = {key: torch.cat([row[key] for row in chunks]) for key in chunks[0]}
            if profile:
                torch.cuda.synchronize(self.device)
            encoded = time.perf_counter()
            scores = self.matcher.score(query, objectness) if query is not None else torch.empty((0, len(self.matcher.bank['obj_ids'])), device=self.device)
            # Classes outside obj_ids are not candidates. The winning class is the best one that remains.
            # obj_ids 以外的类别不是候选。留下的类别里，取得分最高的那个。
            if obj_ids is not None:
                bank_ids = [int(item) for item in self.matcher.bank['obj_ids'].detach().cpu().tolist()]
                allowed = {int(item) for item in obj_ids}
                missing = sorted(allowed.difference(bank_ids))
                if missing:
                    raise ValueError('obj_ids not in the template bank: %s' % missing)
                keep_col = torch.tensor([item in allowed for item in bank_ids], device=self.device, dtype=torch.bool)
                scores = scores.masked_fill(~keep_col.view(1, -1), float('-inf'))
            if profile:
                torch.cuda.synchronize(self.device)
            matched = time.perf_counter()
            instances = []
            if len(boxes):
                keep, confidence_values, identities = mask_nms_same_category(
                    scores, self.matcher.bank['obj_ids'], masks, score_min=gate,
                )
                indices = torch.as_tensor(keep, device=self.device)
                kept_masks = masks[indices].byte().cpu().numpy()
                kept_boxes = boxes[indices].cpu().numpy()
                if len(keep):
                    rles = mask_utils.encode(np.asfortranarray(kept_masks.transpose(1, 2, 0)))
                    for position, index in enumerate(keep):
                        box, rle = kept_boxes[position], rles[position]
                        instances.append({'obj_id': int(identities[index]), 'score_2d': float(confidence_values[index]),
                                          'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                                          'mask': {'size': rle['size'], 'counts': rle['counts'].decode('ascii')},
                                          'mask_bbox': mask_utils.toBbox(rle).tolist()})
            if profile:
                torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        self.last_evidence = {'boxes': boxes, 'masks': masks, 'objectness': objectness, 'scores': scores}
        timing = {'total_ms': (finished - started) * 1000, 'candidates': len(boxes), 'outputs': len(instances),
                  'stage_times_synchronized': bool(profile)}
        if profile:
            timing.update(candidate_segmentation_ms=(proposed - started) * 1000, crop_feature_ms=(encoded - proposed) * 1000,
                          matching_ms=(matched - encoded) * 1000, postprocess_output_ms=(finished - matched) * 1000)
        return instances, timing


def _views_to_device(rgb, masks, device):
    """
    # Move RGB and masks onto device and return that pair.

    ## Args

        - rgb: a numpy array or a tensor. A numpy value is made contiguous and converted with from_numpy. Callers pass uint8 RGB.
        - masks: a numpy array or a tensor. It is converted with from_numpy only when rgb is numpy. Callers pass a bool mask.
        - device: the torch device. Both values are moved with to(device).

    ## Returns

        - Nothing: written to disk.
        - The return is (rgb, masks) on that device.

    ---

    # 把 RGB 和 mask 放到 device 上，返回这一对。

    ## 参数

        - rgb: numpy 数组或张量。numpy 会先变成连续内存再用 from_numpy。调用方传入 uint8 RGB。
        - masks: numpy 数组或张量。只有 rgb 是 numpy 时才用 from_numpy 转换它。调用方传入 bool mask。
        - device: torch 设备。两者都用 to(device) 移动。

    ## 返回

        - 不写磁盘。
        - 返回值是该设备上的 (rgb, masks)。

"""
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(np.ascontiguousarray(rgb))
        masks = torch.from_numpy(np.ascontiguousarray(masks))
    return rgb.to(device), masks.to(device)


def _encode_views(encoder, rgb_uint8, masks, obj_id):
    """
    # Encode one object's 42 rendered views, including four in-plane turns.

        Each view is cropped by its tight pixel xyxy box, then rotated by inplane_turns, which are 0, 1, 2, and 3 quarter turns.

    ## Args

        - encoder: a NativeDino in both callers. The features are whatever encoder.encode returns.
        - rgb_uint8: torch.uint8 (42, H, W, 3). Any other dtype or count raises ValueError.
        - masks: torch.bool (42, H, W), the same spatial size as rgb_uint8. An empty mask raises ValueError.
        - obj_id: used only in that empty-mask error.

    ## Returns

        - Return the encoder dict on device.
        - The return is the encode dict for 168 crops.
        - Crops and features stay on the encoder device.

    ---

    # 编码一个物体的 42 张渲染视角，含四个平面内旋转。

        每个视角按其紧像素 xyxy 包围盒裁剪，再按 inplane_turns 旋转，即 0、1、2、3 个四分之一转。

    ## 参数

        - encoder 在两处调用方都是 NativeDino。
        - 特征就是 encoder.encode 的返回值。
        - rgb_uint8: torch.uint8 (42, H, W, 3)。其他 dtype 或数量抛出 ValueError。
        - masks: torch.bool (42, H, W)，空间尺寸与 rgb_uint8 相同。空 mask 抛出 ValueError。
        - obj_id 只出现在该空 mask 的报错里。

    ## 返回

        - 返回设备上的编码器字典。
        - 返回值是 168 个裁剪的 encode 字典。
        - 裁剪和特征都留在编码器设备上。

"""
    if rgb_uint8.dtype != torch.uint8 or masks.dtype != torch.bool:
        raise ValueError('Template views must be uint8 RGB and bool masks')
    if tuple(rgb_uint8.shape[:3]) != tuple(masks.shape) or rgb_uint8.shape[0] != template_views:
        raise ValueError('Each object needs %d views' % template_views)
    rgb_tensor = rgb_uint8.permute(0, 3, 1, 2).float() / 255
    mask_tensor = masks
    if not bool(mask_tensor.flatten(1).any(1).all()):
        raise ValueError('Empty template mask for object ' + str(obj_id))
    view_h, view_w = mask_tensor.shape[-2:]
    xs = torch.arange(view_w, device=mask_tensor.device)
    ys = torch.arange(view_h, device=mask_tensor.device)
    far = float(max(view_h, view_w) + 1)
    x_plane = xs.view(1, 1, view_w).expand_as(mask_tensor)
    y_plane = ys.view(1, view_h, 1).expand_as(mask_tensor)
    boxes = torch.stack([
        torch.where(mask_tensor, x_plane, far).amin((1, 2)),
        torch.where(mask_tensor, y_plane, far).amin((1, 2)),
        torch.where(mask_tensor, x_plane, -far).amax((1, 2)) + 1,
        torch.where(mask_tensor, y_plane, -far).amax((1, 2)) + 1,
    ], 1).float()
    crops, crop_masks = square_crops(rgb_tensor, mask_tensor, boxes)
    turned_crops = torch.cat([torch.rot90(crops, turns, (-2, -1)) for turns in inplane_turns])
    turned_masks = torch.cat([torch.rot90(crop_masks, turns, (-2, -1)) for turns in inplane_turns])
    return encoder.encode(turned_crops, turned_masks)


def _bank_on_device(renders, weights_dir, device, dino='vitl14'):
    """
    # Encode every object and return the template bank on the GPU.

        No file is written.

    ## Args

        - renders: obj_id to (rgb, masks). The keys are sorted. An empty mapping raises ValueError. rgb and masks may be numpy or tensors, as accepted by _views_to_device.
        - weights_dir: passed to prepare_dino_weight and NativeDino.
        - device: the CUDA device passed to NativeDino.
        - dino: a DINO_CHOICES key. The default is vitl14.

    ## Returns

        - The return dict stacks cls, gem, patches, and valid on a new object axis, and stores obj_ids as int64 on the encoder device.

    ---

    # 编码每个物体，返回 GPU 上的模板库。

        不写文件。

    ## 参数

        - renders 把 obj_id 映射到 (rgb, masks)。
        - 键会排序。
        - 空映射抛出 ValueError。
        - rgb 和 masks 可以是 numpy 或张量，只要 _views_to_device 接受。
        - weights_dir 传给 prepare_dino_weight 和 NativeDino。
        - device: 传给 NativeDino 的 CUDA 设备。
        - dino: DINO_CHOICES 的键。默认是 vitl14。

    ## 返回

        - 返回字典把 cls、gem、patches 和 valid 在新的物体轴上堆叠，并把 obj_ids 存成编码器设备上的 int64。

"""
    obj_ids = sorted(renders)
    if not obj_ids:
        raise ValueError('The CAD library must contain at least one object')
    prepare_dino_weight(weights_dir, dino=dino)
    encoder = NativeDino(weights_dir, device, dino=dino)
    features = []
    for obj_id in obj_ids:
        rgb, masks = renders[obj_id]
        rgb_tensor, mask_tensor = _views_to_device(rgb, masks, encoder.device)
        features.append(_encode_views(encoder, rgb_tensor, mask_tensor, obj_id))
        print('DET2D_ONBOARD', obj_id, 'gpu', flush=True)
    bank = {key: torch.stack([row[key] for row in features]) for key in features[0]}
    bank['obj_ids'] = torch.tensor(obj_ids, dtype=torch.int64, device=encoder.device)
    nbytes = sum(value.nbytes for value in bank.values() if torch.is_tensor(value))
    print('DET2D_TEMPLATE_GPU', {'objects': len(obj_ids), 'views': int(bank['cls'].shape[1]), 'bytes': nbytes}, flush=True)
    return bank


def encode_render_templates(renders, cache_path="", weights_dir=default_weights_dir, device='cuda:0', dino='vitl14'):
    """
    # Build a template bank from rendered views.

        An empty cache path returns the GPU bank.

        A path returns that cache path.

        cache_path "" writes nothing and returns the GPU dict from _bank_on_device.

        A non-empty path is optional disk cache.

        A matching cache returns that path.

        An existing cache with different inputs or code raises ValueError and is not overwritten.

    ## Args

        - renders: obj_id to (rgb, masks). rgb is uint8 (42, H, W, 3). masks is bool (42, H, W).
        - weights_dir: the DINOv2 weight directory. The default is default_weights_dir.
        - device: the CUDA device. The default is cuda:0. Encoding uses NativeDino.
        - dino: vits14, vitb14, or vitl14. The default is vitl14. Cache identity checks only the selected encoder weight.

    ## Returns

        - Numpy: required when cache_path is set. An empty cache_path also accepts tensors. An empty library raises ValueError.

    ---

    # 用渲染视角建立模板库。

        缓存路径为空时返回 GPU 上的库。

        给出路径时返回该缓存路径。

        cache_path 为 "" 时不写文件，并返回 _bank_on_device 的 GPU 字典。

        非空路径是可选的磁盘缓存。

        匹配的缓存返回该路径。

        已有缓存的输入或代码不同则抛出 ValueError，并且不会覆盖。

    ## 参数

        - renders 把 obj_id 映射到 (rgb, masks)。
        - rgb 是 uint8 (42, H, W, 3)。
        - masks 是 bool (42, H, W)。
        - weights_dir: DINOv2 权重目录。默认是 default_weights_dir。
        - device: CUDA 设备。默认是 cuda:0。编码使用 NativeDino。
        - dino: vits14、vitb14 或 vitl14。默认是 vitl14。缓存标识只检查所选编码器权重。

    ## 返回

        - 设置了 cache_path 时必须是 numpy。
        - 空的 cache_path 也接受张量。
        - 空库抛出 ValueError。

"""
    if not cache_path:
        return _bank_on_device(renders, weights_dir, device, dino=dino)
    cache_path = os.path.abspath(os.fspath(cache_path))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    obj_ids = sorted(renders)
    started = time.perf_counter()
    if not obj_ids:
        raise ValueError('The CAD library must contain at least one object')
    render_hashes = {}
    for obj_id in obj_ids:
        rgb, masks = renders[obj_id]
        assert rgb.dtype == np.uint8 and masks.dtype == np.bool_ and rgb.shape[:3] == masks.shape and rgb.shape[0] == template_views
        render_hashes[str(obj_id)] = hashlib.sha256(np.ascontiguousarray(rgb).tobytes() + np.ascontiguousarray(masks).tobytes()).hexdigest()
    prepare_dino_weight(weights_dir, dino=dino)
    weight_hash = sha256(os.path.join(weights_dir, DINO_CHOICES[str(dino)]['file']))
    # Template encoding uses DINO only; detector weights are prepared by the detector entry.
    # 模板编码只使用 DINO；检测器入口另行准备检测器权重。
    assets_identity = weight_hash
    recipe = 'native AMP; uint8;224 black square margin1.1;rot4;patch valid>.5'
    if os.path.isfile(cache_path) and os.path.isfile(cache_path + '.json'):
        with open(cache_path + '.json') as stream:
            previous = json.load(stream)
        if (previous.get('render_hashes') == render_hashes and previous.get('weight_sha256') == weight_hash
                and previous.get('recipe') == recipe and previous.get('source_sha256') == sha256(__file__)
                and previous.get('assets_identity') == assets_identity
                and previous.get('bank_sha256') == sha256(cache_path)):
            print('DET2D_TEMPLATE_CACHE_HIT', cache_path, flush=True)
            return cache_path
        raise ValueError('Existing template cache has different inputs/code; choose a new cache path')
    encoder = NativeDino(weights_dir, device, dino=dino)
    features = []
    for obj_id in obj_ids:
        rgb, masks = renders[obj_id]
        rgb_tensor, mask_tensor = _views_to_device(rgb, masks, encoder.device)
        encoded = _encode_views(encoder, rgb_tensor, mask_tensor, obj_id)
        # Disk cache is the caller's choice. The copy to CPU happens only for that file.
        # 写磁盘是调用者的选择。只有这份文件才把特征拷到 CPU。
        features.append({key: value.cpu() for key, value in encoded.items()})
        print('DET2D_ONBOARD', obj_id, flush=True)
    bank = {key: torch.stack([row[key] for row in features]) for key in features[0]}
    bank['obj_ids'] = torch.tensor(obj_ids, dtype=torch.int64)
    torch.save(bank, cache_path + '.tmp')
    os.replace(cache_path + '.tmp', cache_path)
    manifest = {'bank_sha256': sha256(cache_path), 'weight_sha256': sha256(encoder.weight_path),
                'assets_identity': assets_identity,
                'render_hashes': render_hashes, 'obj_ids': obj_ids, 'views': template_views * len(inplane_turns),
                'source_sha256': sha256(__file__), 'onboarding_s': time.perf_counter() - started,
                'recipe': recipe}
    _save_json(cache_path + '.json', manifest)
    return cache_path


def onboard_meshes(meshes_m, cache_path="", weights_dir=default_weights_dir, device='cuda:0', dino='vitl14'):
    """
    # Render a CAD library and return its template bank.

        An empty cache path keeps the bank on the GPU.

        cache_path "" returns the GPU bank dict and writes no file.

        A non-empty path is an optional disk cache and the return is that path string.

        A matching cache returns the path without rendering again.

        A different existing cache raises ValueError and is not overwritten.

        Rendering uses template_views, which is 42, focal length 224 pixels, principal point 112, and a camera distance of 1.5 times the bounds diagonal.

        A view with foreground whose maximum RGB is below black_foreground_max is replaced by gray Lambert shading from depth.

        The features then come from encode_render_templates.

    ## Args

        - meshes_m: obj_id to a trimesh mesh whose vertices are meters. An empty mapping raises ValueError. Each mesh is centered on its bounds. A zero extent raises ValueError.
        - weights_dir: the DINOv2 weight directory. The default is default_weights_dir.
        - device: the CUDA device. The default is cuda:0.
        - dino: vits14, vitb14, or vitl14. The default is vitl14.

    ---

    # 渲染一个 CAD 库并返回它的模板库。

        缓存路径为空时，库留在 GPU 上。

        cache_path 为 "" 时返回 GPU 库字典，不写文件。

        非空路径是可选磁盘缓存，返回值是该路径字符串。

        匹配的缓存直接返回路径，不再渲染。

        已有缓存不同则抛出 ValueError，并且不会覆盖。

        渲染使用 template_views（42）、焦距 224 像素、主点 112，相机距离是包围盒对角线的 1.5 倍。

        有前景且 RGB 最大值低于 black_foreground_max 的视角，会换成由深度得到的灰色朗伯明暗。

        特征随后来自 encode_render_templates。

    ## 参数

        - meshes_m 把 obj_id 映射到 trimesh 网格，顶点单位是米。
        - 空映射抛出 ValueError。
        - 每个网格按其包围盒居中。
        - 范围为零抛出 ValueError。
        - weights_dir: DINOv2 权重目录。默认是 default_weights_dir。
        - device: CUDA 设备。默认是 cuda:0。
        - dino: vits14、vitb14 或 vitl14。默认是 vitl14。

"""
    started = time.perf_counter()
    from wapr.ogl import GpuRenderRuntime, ensure_ogl, _native_dir, parse_cuda_device
    from wapr import recipe as pose_recipe

    if not meshes_m:
        raise ValueError('The CAD library must contain at least one object')
    prepare_dino_weight(weights_dir, dino=dino)
    write_cache = bool(cache_path)
    identity = None
    mesh_manifest = ""
    if write_cache:
        cache_path = os.path.abspath(os.fspath(cache_path))
        encoder_weight_hash = sha256(os.path.join(weights_dir, DINO_CHOICES[str(dino)]['file']))
        # GLB includes mesh geometry, colors, UVs and textures; cache reuse is explicit.
        # GLB包含几何、颜色、UV和纹理；只有要求写缓存时才做这份检查。
        identity = {'meshes_glb_sha256': {str(obj_id): hashlib.sha256(mesh.export(file_type='glb')).hexdigest()
                                         for obj_id, mesh in sorted(meshes_m.items())},
                    'source_sha256': sha256(__file__), 'ogl_source_sha256': sha256(os.path.join(release_dir, 'wapr', 'ogl.py')),
                    'shader_sha256': {name: sha256(os.path.join(_native_dir(), 'shaders', name))
                                      for name in sorted(os.listdir(_native_dir() / 'shaders'))
                                      if os.path.isfile(_native_dir() / 'shaders' / name)},
                    'weight_sha256': encoder_weight_hash,
                    'assets_identity': encoder_weight_hash,
                    'lighting': [float(pose_recipe.w_ambient), float(pose_recipe.w_diffuse), list(pose_recipe.light_dir)],
                    'torch': torch.__version__, 'mesh_units': 'm', 'views': template_views,
                    'recipe': 'centered;diameter1.5;focal224;uint8;native AMP;rot4'}
        mesh_manifest = cache_path + '.meshes.json'
        if os.path.isfile(cache_path):
            if os.path.isfile(mesh_manifest) and os.path.isfile(cache_path + '.json'):
                with open(mesh_manifest) as stream:
                    previous = json.load(stream)
                with open(cache_path + '.json') as stream:
                    bank_manifest = json.load(stream)
                if previous == identity and bank_manifest['bank_sha256'] == sha256(cache_path):
                    print('DET2D_CAD_CACHE_HIT', cache_path, flush=True)
                    return cache_path
            raise ValueError('Existing CAD cache has different inputs/code; choose a new cache path')
    ordinal = parse_cuda_device(device)
    runtime = GpuRenderRuntime(ensure_ogl(), ordinal, str(_native_dir() / 'shaders'))
    intrinsics = np.array([[224., 0, 112.], [0, 224., 112.], [0, 0, 1.]])
    renders = {}
    for obj_id, original in sorted(meshes_m.items()):
        mesh = original.copy()
        mesh.vertices -= (mesh.bounds[0] + mesh.bounds[1]) / 2
        diameter_m = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        if diameter_m <= 0:
            raise ValueError('Zero mesh extent')
        mesh_id = runtime.load_mesh_trimesh(mesh, name='det2d_' + str(obj_id))
        index = np.arange(template_views, dtype=np.float64)
        z = 1 - 2 * (index + .5) / template_views
        angle = index * np.pi * (3 - np.sqrt(5))
        radial = np.sqrt(1 - z * z)
        forward = -np.stack([radial * np.cos(angle), radial * np.sin(angle), z], axis=1)
        right = np.cross(np.array([0., 1., 0.]), forward)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        down = np.cross(forward, right)
        poses = np.tile(np.eye(4, dtype=np.float32), (template_views, 1, 1))
        poses[:, 0, :3] = right
        poses[:, 1, :3] = down
        poses[:, 2, :3] = forward
        poses[:, 2, 3] = np.float32(diameter_m * 1.5)
        windows = np.tile([0, 0, crop_px, crop_px], (template_views, 1))
        rgb, depth = runtime.render_tiles(mesh_id, poses, windows, intrinsics, crop_px, crop_px, crop_px)
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1.001:
            raise ValueError('Renderer must return finite unit RGB')
        masks = depth > 0
        # A black texture has no appearance. Depth normals draw the shape in gray.
        # 黑色纹理没有外观。深度法向把形状画成灰。
        rgb = shade_black_templates(rgb, depth)
        # uint8 stays on the GPU. A disk cache is the only path that copies it out.
        # uint8 留在显存。只有写磁盘缓存时才拷出去。
        renders[obj_id] = ((rgb * 255).to(torch.uint8), masks)
    if not write_cache:
        return encode_render_templates(renders, "", weights_dir, device, dino=dino)
    numpy_renders = {obj_id: (rgb.detach().cpu().numpy(), mask.detach().cpu().numpy())
                     for obj_id, (rgb, mask) in renders.items()}
    encode_render_templates(numpy_renders, cache_path, weights_dir, device, dino=dino)
    # The duration is evidence, not a cache key. / 耗时是证据，不是缓存key。
    duration = time.perf_counter() - started
    _save_json(mesh_manifest, identity)
    with open(cache_path + '.json') as stream:
        manifest = json.load(stream)
    manifest['cad_onboarding_total_s'] = duration
    manifest['cad_renderer'] = 'release-local OGL; new renders require accuracy validation'
    _save_json(cache_path + '.json', manifest)
    return cache_path
