# Author: Yulin Wang (yulinwang@seu.edu.cn)
# SPDX-License-Identifier: LGPL-2.1-only
"""Carry weight attribution through WAPR exports. / 在 WAPR 导出中保留权重署名。"""
import hashlib
import inspect
import json
from pathlib import Path
import struct

ENGINE_MAGIC = b"WAPR-ENGINE-METADATA-v1\n"
MODEL_NAMES = ("wapr_w_mask", "wapr_wo_mask", "sapr", "wbps")
METADATA_KEY = "wapr.metadata"


def weight_metadata(name):
    """Build public attribution using bundled terms. / 使用随包条款生成公开署名。"""
    if name not in MODEL_NAMES:
        raise ValueError("Unknown WAPR model: " + name)
    license_dir = Path(__file__).resolve().parent / "weight_license"
    return {
        "schema_version": 1,
        "model_name": name,
        "model_version": "unversioned",
        "author": "Yulin Wang",
        "copyright": "Copyright (c) 2026 Yulin Wang",
        "license": "CC-BY-ND-4.0",
        "license_text": (license_dir / "CC-BY-ND-4.0.txt").read_text(encoding="utf-8"),
        "license_notice": (license_dir / "WEIGHTS_LICENSE.txt").read_text(encoding="utf-8"),
        "source": "https://github.com/WangYuLin-SEU/WAPR",
    }


def validate_metadata(metadata):
    """Validate attribution without changing tensors. / 验证署名，不改动张量。"""
    required = ("model_name", "model_version", "author", "copyright", "license", "license_text", "license_notice", "source")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported WAPR attribution metadata")
    if any(not isinstance(metadata.get(key), str) or not metadata[key] for key in required):
        raise ValueError("Incomplete WAPR attribution metadata")
    if metadata["model_name"] not in MODEL_NAMES:
        raise ValueError("Unknown WAPR model metadata")
    return metadata


def checkpoint_metadata(path, name):
    """Read attribution or supply it for legacy weights. / 读取署名，兼容旧权重。"""
    import torch
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    checkpoint = torch.load(path, **load_options)
    metadata = checkpoint.get(METADATA_KEY) if isinstance(checkpoint, dict) else None
    metadata = weight_metadata(name) if metadata is None else validate_metadata(metadata)
    if metadata["model_name"] != name:
        raise ValueError("Checkpoint model name differs from export model")
    return metadata


def write_onnx_metadata(path, metadata):
    """Preserve other ONNX metadata too. / 同时保留其他 ONNX 元数据。"""
    import onnx
    validate_metadata(metadata)
    model = onnx.load(str(path))
    props = {item.key: item.value for item in model.metadata_props}
    props[METADATA_KEY] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    onnx.helper.set_model_props(model, props)
    onnx.save(model, str(path))


def read_onnx_metadata(path):
    """Read embedded ONNX attribution. / 读取 ONNX 内嵌署名。"""
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    props = {item.key: item.value for item in model.metadata_props}
    if METADATA_KEY not in props:
        raise ValueError("ONNX model lacks WAPR attribution")
    return validate_metadata(json.loads(props[METADATA_KEY]))


def pack_engine(blob, metadata):
    """Wrap the original engine unchanged. / 原样封装原生引擎字节。"""
    validate_metadata(metadata)
    record = {"metadata": metadata, "engine_sha256": hashlib.sha256(blob).hexdigest()}
    header = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return ENGINE_MAGIC + struct.pack("<Q", len(header)) + header + blob


def unpack_engine(blob):
    """Return native bytes and attribution; accept legacy engines.

    返回原生字节与署名，同时兼容旧引擎。
    """
    if not blob.startswith(ENGINE_MAGIC):
        return blob, None
    offset = len(ENGINE_MAGIC)
    if len(blob) < offset + 8:
        raise ValueError("Truncated WAPR engine header")
    size = struct.unpack_from("<Q", blob, offset)[0]
    offset += 8
    if size > len(blob) - offset:
        raise ValueError("Truncated WAPR engine metadata")
    record = json.loads(blob[offset:offset + size].decode("utf-8"))
    metadata = validate_metadata(record["metadata"])
    native = blob[offset + size:]
    if hashlib.sha256(native).hexdigest() != record["engine_sha256"]:
        raise ValueError("WAPR engine integrity check failed")
    return native, metadata
