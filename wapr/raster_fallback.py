# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

# Tile renderer used when EGL/GL is unavailable. No GL context.
# EGL/GL 不可用时的分块渲染。不创建 GL 上下文。
# Outputs match the GL path: rgb (N, tile, tile, 3) in 0-1, depth (N, tile, tile) in meters.
# 输出与 GL 路径一致：rgb (N, tile, tile, 3)，范围 0-1；depth (N, tile, tile)，米。
from collections import OrderedDict
import hashlib
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from wapr import recipe


def _projection_y_down(K, width, height, znear, zfar):
    """
    # OpenCV K, y down, into a clip matrix.

    ## Args

        - K: the intrinsic tensor.
        - width and height: the full frame in pixels.
        - znear and zfar: meters.

    ## Returns

        - Returns a 4×4 tensor.
        - proj[..., 3, 2] is -1.

    ---

    # OpenCV 的 K，y 向下，变成裁剪矩阵。

    ## 参数

        - K: 内参张量。
        - width 和 height: 整幅图，像素。
        - znear 和 zfar: 米。

    ## 返回

        - 返回 4×4 张量。
        - proj[..., 3, 2] 是 -1。

"""
    w = K.new_tensor(float(width))
    h = K.new_tensor(float(height))
    depth = zfar - znear
    q = -(zfar + znear) / depth
    qn = -2.0 * (zfar * znear) / depth
    proj = K.new_zeros(K.shape[:-2] + (4, 4))
    proj[..., 0, 0] = 2.0 * K[..., 0, 0] / w
    proj[..., 0, 1] = -2.0 * K[..., 0, 1] / w
    proj[..., 0, 2] = (-2.0 * K[..., 0, 2] + w) / w
    proj[..., 1, 1] = 2.0 * K[..., 1, 1] / h
    proj[..., 1, 2] = (2.0 * K[..., 1, 2] - h) / h
    proj[..., 2, 2] = q
    proj[..., 2, 3] = qn
    proj[..., 3, 2] = -1.0
    return proj


def _clip_planes(poses, pos):
    """
    # Take near and far from posed vertex z.

    ## Args

        - poses: (N, 4, 4), object-to-camera.
        - pos: (V, 3) object-frame vertices. Valid z uses 0.8 times the minimum and 1.2 times the maximum. No valid z uses 0.001 m and 100 m.

    ## Returns

        - Returns znear and zfar, meters.

    ---

    # 由摆好姿态后的顶点 z 取近平面和远平面。

    ## 参数

        - poses: (N, 4, 4)，物体到相机。
        - pos: (V, 3)，物体坐标系顶点。有效 z 用最小值的 0.8 倍和最大值的 1.2 倍。没有有效 z 时用 0.001 米和 100 米。

    ## 返回

        - 返回 znear 和 zfar，米。

"""
    pts = torch.matmul(poses[:, None, :3, :3], pos[None, :, :, None]).squeeze(-1)
    pts = pts + poses[:, None, :3, 3]
    z = pts[..., 2]
    valid = z > 1e-3
    zmin = torch.where(valid, z, z.new_full((), 1.0e30)).amin()
    zmax = torch.where(valid, z, z.new_zeros(())).amax()
    any_valid = valid.any()
    znear = torch.where(any_valid, torch.clamp(zmin * 0.8, min=1.0e-3), z.new_tensor(0.001))
    zfar = torch.where(any_valid, torch.maximum(znear * 10.0, zmax * 1.2), z.new_tensor(100.0))
    return znear, zfar


def _clip_tf(bboxes, frame_h, frame_w):
    """
    # The bbox mapping that matches clip_affine4_from_bbox2d.

    ## Args

        - bboxes: (N, 4): left, top, right, bottom, full-frame pixels, y down.
        - frame_h and frame_w: the full frame in pixels.

    ## Returns

        - Returns (N, 4, 4).

    ---

    # 与 clip_affine4_from_bbox2d 相同的包围盒变换。

    ## 参数

        - bboxes: (N, 4)：left、top、right、bottom，整幅图像素，y 向下。
        - frame_h 和 frame_w: 整幅图，像素。

    ## 返回

        - 返回 (N, 4, 4)。

"""
    left = bboxes[:, 0]
    vmin = bboxes[:, 1]
    right = bboxes[:, 2]
    vmax = bboxes[:, 3]
    top = float(frame_h) - vmin
    bottom = float(frame_h) - vmax
    rw = (right - left).clamp(min=1.0e-6)
    rh = (top - bottom).clamp(min=1.0e-6)
    sx = float(frame_w) / rw
    sy = float(frame_h) / rh
    tx = (float(frame_w) - right - left) / rw
    ty = (float(frame_h) - top - bottom) / rh
    tf = torch.eye(4, device=bboxes.device, dtype=bboxes.dtype).expand(bboxes.shape[0], 4, 4).contiguous().clone()
    tf[:, 0, 0] = sx
    tf[:, 1, 1] = sy
    tf[:, 3, 0] = tx
    tf[:, 3, 1] = ty
    return tf


def _rasterize_pair(dr, glctx, pos_clip, faces, resolution):
    """
    # Rasterize and return the nvdiffrast result tuple.

        A returned RuntimeError object is raised.

    ## Args

        - dr: the nvdiffrast.torch module.
        - glctx: the RasterizeCudaContext.
        - pos_clip: the clip-space vertices.
        - faces: (F, 3) int.
        - resolution: (tile, tile).

    ---

    # 光栅化并返回 nvdiffrast 的结果元组。

        如果返回的是 RuntimeError 对象，就把它抛出。

    ## 参数

        - dr: nvdiffrast.torch。
        - glctx: RasterizeCudaContext。
        - pos_clip: 裁剪空间的顶点。
        - faces: (F, 3) 整数。
        - resolution: (tile, tile)。

"""
    out = dr.rasterize(glctx, pos_clip, faces, resolution)
    if isinstance(out, RuntimeError):
        raise out
    return cast(tuple, out)


def _interpolate_pair(dr, attr, rast, faces):
    """
    # Interpolate an attribute and return the nvdiffrast result tuple.

        A non-tuple result raises RuntimeError.

    ## Args

        - dr: nvdiffrast.torch.
        - attr: the attribute tensor.
        - rast: the raster output.
        - faces: the index tensor, vertex faces or wedge UV faces.

    ---

    # 插值属性并返回 nvdiffrast 的结果元组。

        结果不是 tuple 时抛出 RuntimeError。

    ## 参数

        - dr: nvdiffrast.torch。
        - attr: 属性张量。
        - rast: 光栅输出。
        - faces: 索引，顶点面或者 wedge UV 的面。

"""
    out = dr.interpolate(attr, rast, faces)
    if not isinstance(out, tuple):
        raise RuntimeError("interpolate")
    return out


def _shade(base, normal_map, xyz_map):
    """
    # Lambert from recipe.w_ambient, recipe.w_diffuse, and recipe.light_dir, then a highlight cap.

    ## Args

        - base: the albedo, 0–1.
        - normal_map and xyz_map: the interpolated camera-frame fields. A normal that faces away from the camera is flipped.

    ## Returns

        - Returns RGB in 0–1.

    ---

    # 用 recipe 的环境项、漫反射和光照方向做 Lambert，再限制高光。

    ## 参数

        - base: 反照率，0–1。
        - normal_map 和 xyz_map: 插值后的相机系量。背对相机的法线会被翻过来。

    ## 返回

        - 返回 0–1 的 RGB。

"""
    n = F.normalize(normal_map, dim=-1, eps=1.0e-6)
    toward_cam = F.normalize(-xyz_map, dim=-1, eps=1.0e-6)
    n = torch.where((n * toward_cam).sum(dim=-1, keepdim=True) < 0, -n, n)
    light = base.new_tensor(recipe.light_dir)
    direction = F.normalize(-light, dim=0)
    diffuse = (n * direction).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    lit = base * float(recipe.w_ambient) + base * diffuse * float(recipe.w_diffuse)
    floor = base.clamp(min=0.04)
    lit = torch.minimum(lit, floor * 1.15 + 0.10)
    knee = 0.90
    head = 0.08
    y = torch.minimum(lit, lit.new_full((), knee))
    over = (lit - knee).clamp(min=0.0)
    lit = y + head * over / (1.0 + over)
    return lit.clamp(0.0, 1.0)


class NvRuntime:
    """
    # CUDA rasterizer used when describing the no-GL tile path.

        One mesh, separate tiles, same units as the GL runtime.

        No function in this release constructs it.

        runtime_for returns GpuRenderRuntime.

        The methods below are what a caller would use: load_mesh_trimesh, then render_tiles.

    ---

    # 不走 GL 时的 CUDA 分块光栅。

        一个网格、分块渲染，单位与 GL 运行时相同。

        这个发布包里没有函数构造它。

        runtime_for 返回的是 GpuRenderRuntime。

        调用方会用的方法是 load_mesh_trimesh，然后 render_tiles。
    """
    def __init__(self, device):
        """
        # RasterizeCudaContext on this CUDA ordinal.

            The mesh cache starts at 8 uploads.

            Returns None.

        ## Args

            - device: the CUDA ordinal, an int. The caller is whoever constructs NvRuntime. This release does not construct it.

        ---

        # 在这个 CUDA 序号上建立 RasterizeCudaContext。

            网格缓存一开始放 8 份。

            返回 None。

        ## 参数

            - device: CUDA 序号，整数。
        """
        import nvdiffrast.torch as dr

        self._dr = dr
        self._device = int(device)
        self._dev = torch.device("cuda", self._device)
        torch.cuda.set_device(self._dev)
        self._glctx = dr.RasterizeCudaContext(self._dev)
        self._cache = OrderedDict()
        self._meshes = {}
        self._next_id = 1
        self._cache_limit = 8
        self._packed_tiles = None

    def ensure_mesh_slots(self, count):
        """
        # Keep enough cached meshes for one multi-object batch.

        ## Args

            - count: the number of rows. The cache limit becomes at least that count. No function in this release calls it.

        ## Returns

            - Returns None.

        ---

        # 为一批多物体留够网格缓存。

        ## 参数

            - count: 行数。缓存上限至少变成这个数量。这个发布包里没有函数调用它。

        ## 返回

            - 返回 None。
        """
        self._cache_limit = max(int(self._cache_limit), int(count))

    def load_mesh_trimesh(self, mesh, name="mesh"):
        """
        # Pack one mesh and keep it on this device.

            render_tiles uses the id this returns.

            No function outside this class calls it in this release.

        ## Args

            - mesh: a trimesh.
            - name: to mesh and is not used as the cache key. The key is the uploaded arrays. The cache drops the oldest id past _cache_limit.

        ## Returns

            - Returns the cached mesh id, an int.

        ---

        # 打包一个网格并留在这个设备上。

            render_tiles 使用它返回的编号。

            在这个发布包里，类外面没有函数调用它。

        ## 参数

            - mesh: trimesh。name 默认是 mesh，不拿来做缓存键。键是上传的数组。超过 _cache_limit 时丢掉最旧的编号。

        ## 返回

            - 返回缓存里的网格编号，整数。
        """
        from wapr.ogl import pack_trimesh_upload

        upload = pack_trimesh_upload(
            mesh,
            max_texture_size=1024,
            override_max_texture_size=1024,
            force_flat_normal=False,
        )
        key = self._cache_key(upload)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return int(cached)
        record = self._record_from_upload(upload)
        mesh_id = int(self._next_id)
        self._next_id += 1
        self._meshes[mesh_id] = record
        self._cache[key] = mesh_id
        while len(self._cache) > int(self._cache_limit):
            old = self._cache.popitem(last=False)
            self._meshes.pop(int(old[1]), None)
        return mesh_id

    def _cache_key(self, upload):
        """
        # Identity of the uploaded arrays, so the same mesh is not stored twice.

        ## Args

            - upload: the MeshUploadArrays from pack_trimesh_upload. None arrays are recorded as empty.

        ## Returns

            - Returns a hex string.

        ---

        # 上传数组的标识，同一个网格不重复保存。

        ## 参数

            - upload: pack_trimesh_upload 给出的 MeshUploadArrays。空数组记成空。

        ## 返回

            - 返回十六进制字符串。

"""
        digest = hashlib.blake2b(digest_size=12)
        for value in (upload.vertices, upload.faces, upload.normals, upload.vertex_colors, upload.uvs, upload.texture_rgb):
            if value is None:
                digest.update(b"\x00")
                continue
            arr = np.ascontiguousarray(value)
            digest.update(str(arr.dtype).encode("ascii"))
            digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            digest.update(arr.tobytes())
        digest.update(np.uint8(bool(upload.uv_per_corner)).tobytes())
        return digest.hexdigest()

    def _record_from_upload(self, upload):
        """
        # Copy one mesh onto the GPU.

        ## Args

            - upload: MeshUploadArrays. Texture RGB is stored in 0–1. A missing vertex color becomes the untextured gray. Wedge UV uses a corner index. Vertex UV reuses the face index.

        ## Returns

            - Returns a dict of tensors on this device.

        ---

        # 把一份网格复制到 GPU。

        ## 参数

            - upload: MeshUploadArrays。纹理 RGB 存成 0–1。没有顶点色时用无纹理的灰。wedge UV 用角点索引。顶点 UV 复用面索引。

        ## 返回

            - 返回这个设备上的张量字典。

"""
        dev = self._dev
        pos = torch.as_tensor(np.asarray(upload.vertices, dtype=np.float32).reshape(-1, 3), device=dev)
        faces = torch.as_tensor(np.asarray(upload.faces, dtype=np.int32).reshape(-1, 3), device=dev)
        normals = torch.as_tensor(np.asarray(upload.normals, dtype=np.float32).reshape(-1, 3), device=dev)
        if upload.vertex_colors is None:
            # Textured meshes leave vertex color empty. Shading then uses the texture.
            # 带纹理的网格不带顶点色。这时着色用纹理。
            from wapr.ogl import DEFAULT_UNTEXTURED_VERTEX_COLOR
            colors = torch.full(
                (pos.shape[0], 3), float(DEFAULT_UNTEXTURED_VERTEX_COLOR), device=dev, dtype=torch.float32
            )
        else:
            colors = torch.as_tensor(
                np.asarray(upload.vertex_colors, dtype=np.float32).reshape(-1, 3), device=dev
            )
        uvs = None
        uv_idx = None
        if upload.uvs is not None:
            uvs = torch.as_tensor(np.asarray(upload.uvs, dtype=np.float32).reshape(-1, 2), device=dev)
            if upload.uv_per_corner:
                uv_idx = torch.arange(uvs.shape[0], device=dev, dtype=torch.int32).reshape(-1, 3)
            else:
                uv_idx = faces
        tex = None
        if upload.texture_rgb is not None and int(upload.tex_w) > 0 and int(upload.tex_h) > 0:
            img = np.asarray(upload.texture_rgb, dtype=np.uint8).reshape(int(upload.tex_h), int(upload.tex_w), 3)
            tex = torch.as_tensor(img, device=dev, dtype=torch.float32).div_(255.0)[None]
        return {
            "pos": pos,
            "faces": faces,
            "normals": normals,
            "colors": colors,
            "uvs": uvs,
            "uv_idx": uv_idx,
            "tex": tex,
        }

    def render_tiles(self, mesh_id, poses, bboxes, K, frame_h, frame_w, tile, mesh_ids=None):
        """
        # Render tiles.

            No function outside this class calls it in this release.

            A caller would use it after load_mesh_trimesh.

        ## Args

            - mesh_id: the int from load_mesh_trimesh.
            - poses: (N, 4, 4) object-to-camera, meters, OpenCV.
            - bboxes: (N, 4) left, top, right, bottom in full-frame pixels, y down.
            - K: 3×3 or (N, 3, 3), pixels.
            - frame_h and frame_w: the full frame.
            - tile: the square side in pixels.
            - mesh_ids: one mesh id per tile, or None to use mesh_id for every tile. Different meshes are drawn in runs of the same id, then concatenated. An unknown mesh id raises RuntimeError.

        ## Returns

            - Returns rgb (N, tile, tile, 3) in 0–1 and depth (N, tile, tile) in meters.

        ---

        # 画分块。

            在这个发布包里，类外面没有函数调用它。

            调用方会在 load_mesh_trimesh 之后使用它。

        ## 参数

            - mesh_id: load_mesh_trimesh 返回的整数。
            - poses: (N, 4, 4)，物体到相机，米，OpenCV。
            - bboxes: (N, 4)，left、top、right、bottom，整幅图像素，y 向下。
            - K: 3×3 或 (N, 3, 3)，像素。
            - frame_h 和 frame_w: 整幅图。
            - tile: 正方形边长，像素。
            - mesh_ids: 每块一个网格编号，或者 None，表示每块都用 mesh_id。不同网格按相同编号的连续段绘制，再拼回去。未知的网格编号抛出 RuntimeError。

        ## 返回

            - 返回 rgb (N, tile, tile, 3)，范围 0–1，以及 depth (N, tile, tile)，米。
        """
        if mesh_ids is not None:
            ids = [int(value) for value in mesh_ids]
            if len(set(ids)) > 1:
                return self._render_mixed_tiles(ids, poses, bboxes, K, frame_h, frame_w, tile)
            mesh_id = ids[0]
        record = self._meshes.get(int(mesh_id))
        if record is None:
            raise RuntimeError("mesh")
        dev = self._dev
        pose_t = torch.as_tensor(poses, device=dev, dtype=torch.float32)
        if pose_t.ndim == 2:
            pose_t = pose_t[None]
        box_t = torch.as_tensor(bboxes, device=dev, dtype=torch.float32).reshape(-1, 4)
        n = int(pose_t.shape[0])
        K_t = torch.as_tensor(K, device=dev, dtype=torch.float32).reshape(-1, 3, 3)
        if K_t.shape[0] == 1 and n > 1:
            K_t = K_t.expand(n, -1, -1)
        pos = record["pos"]
        ones = torch.ones((pos.shape[0], 1), device=dev, dtype=torch.float32)
        pos_h = torch.cat((pos, ones), dim=1)
        znear, zfar = _clip_planes(pose_t, pos)
        proj = _projection_y_down(K_t, int(frame_w), int(frame_h), znear, zfar)
        # OpenCV camera to the rasterizer camera: flip Y and Z.
        # OpenCV 相机变到光栅相机：翻转 Y 和 Z。
        glcam = pose_t.new_tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        mtx = torch.matmul(proj, glcam) @ pose_t
        pos_clip = torch.matmul(mtx[:, None], pos_h[None, :, :, None]).squeeze(-1)
        pos_clip = torch.matmul(pos_clip[:, :, None, :], _clip_tf(box_t, frame_h, frame_w)[:, None]).squeeze(-2)
        rast, _ = _rasterize_pair(
            self._dr,
            self._glctx,
            pos_clip.contiguous(),
            record["faces"],
            (int(tile), int(tile)),
        )
        xyz_map, _ = _interpolate_pair(self._dr, pos[None], rast, record["faces"])
        xyz_map = torch.matmul(pose_t[:, None, None, :3, :3], xyz_map[..., None]).squeeze(-1)
        xyz_map = xyz_map + pose_t[:, None, None, :3, 3]
        if record["tex"] is not None:
            # A texture is stored with its UV. Both have to be present.
            # 纹理和它的 UV 一起保存。两者都必须存在。
            uvs = record["uvs"]
            uv_idx = record["uv_idx"]
            if uvs is None or uv_idx is None:
                raise RuntimeError("uv")
            texc, _ = _interpolate_pair(self._dr, uvs[None], rast, uv_idx)
            base = self._dr.texture(record["tex"], texc, filter_mode="linear")
        else:
            base, _ = _interpolate_pair(self._dr, record["colors"][None], rast, record["faces"])
        normals = torch.matmul(pose_t[:, None, :3, :3], record["normals"][None, :, :, None]).squeeze(-1)
        normal_map, _ = _interpolate_pair(self._dr, normals, rast, record["faces"])
        rgb = _shade(base, normal_map, xyz_map)
        covered = (rast[..., 3:] > 0).to(rgb.dtype)
        rgb = rgb * covered
        depth = xyz_map[..., 2] * covered[..., 0]
        # The rasterizer row order is opposite the GL tile, so flip both to match.
        # 光栅器的行序与 GL 块相反，两路都翻转后才一致。
        rgb = torch.flip(rgb, dims=(1,))
        depth = torch.flip(depth, dims=(1,))
        return rgb.contiguous(), depth.contiguous()

    def _render_mixed_tiles(self, ids, poses, bboxes, K, frame_h, frame_w, tile):
        """Rasterize all meshes with one nvdiffrast range-mode call.

        使用 nvdiffrast range 模式一次光栅化全部网格。
        Only geometry packing loops over records; rendering, interpolation,
        texture sampling and shading are batched. Cache the current layout.
        仅几何打包遍历记录；渲染、插值、纹理采样和着色均批量计算。缓存当前布局。
        """
        if any(mesh_id not in self._meshes for mesh_id in ids):
            raise RuntimeError("Unknown mesh in mixed tile batch")
        key = tuple(ids)
        dev = self._dev
        cached = self._packed_tiles
        if cached is None or cached["key"] != key:
            records = [self._meshes[mesh_id] for mesh_id in ids]
            unique = list(dict.fromkeys(ids))
            textured = [mesh_id for mesh_id in unique if self._meshes[mesh_id]["tex"] is not None]
            atlas_h = max((int(self._meshes[mid]["tex"].shape[1]) + 2 for mid in textured), default=1)
            atlas_w = sum(int(self._meshes[mid]["tex"].shape[2]) + 2 for mid in textured) or 1
            atlas = torch.ones((1, atlas_h, atlas_w, 3), device=dev)
            tex_regions = {}
            offset = 0
            for mid in textured:
                texture = self._meshes[mid]["tex"]
                height, width = texture.shape[1:3]
                # Circular padding preserves the original wrap sampling at UV borders.
                # 循环填充保持原纹理 UV 边界的 wrap 采样。
                padded = F.pad(texture.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="circular")
                atlas[:, :height + 2, offset:offset + width + 2] = padded.permute(0, 2, 3, 1)
                tex_regions[mid] = (width / atlas_w, height / atlas_h,
                                    (offset + 1) / atlas_w, 1 / atlas_h)
                offset += width + 2
            positions, normals, colors, uv_rows, vertex_rows, ranges = [], [], [], [], [], []
            triangle_start = 0
            for row, record in enumerate(records):
                corners = record["faces"].reshape(-1).long()
                positions.append(record["pos"][corners])
                normals.append(record["normals"][corners])
                colors.append(record["colors"][corners])
                if record["tex"] is None:
                    uv_rows.append(torch.zeros((len(corners), 2), device=dev))
                else:
                    uv_rows.append(record["uvs"][record["uv_idx"].reshape(-1).long()])
                vertex_rows.append(torch.full((len(corners),), row, device=dev, dtype=torch.long))
                triangles = int(record["faces"].shape[0])
                ranges.append((triangle_start, triangles))
                triangle_start += triangles
            pos = torch.cat(positions)
            cached = {"key": key, "pos": pos, "normal": torch.cat(normals),
                      "color": torch.cat(colors), "uv": torch.cat(uv_rows),
                      "row": torch.cat(vertex_rows), "atlas": atlas,
                      "faces": torch.arange(len(pos), device=dev, dtype=torch.int32).reshape(-1, 3),
                      "ranges": torch.tensor(ranges, dtype=torch.int32),
                      "tex_region": torch.tensor([tex_regions.get(mid, (0, 0, 0, 0)) for mid in ids], device=dev),
                      "textured": torch.tensor([mid in tex_regions for mid in ids], device=dev)}
            self._packed_tiles = cached
        pose_t = torch.as_tensor(poses, device=dev, dtype=torch.float32).reshape(-1, 4, 4)
        box_t = torch.as_tensor(bboxes, device=dev, dtype=torch.float32).reshape(-1, 4)
        if len(pose_t) != len(ids) or len(box_t) != len(ids):
            raise ValueError("Mesh, pose and box counts differ")
        K_t = torch.as_tensor(K, device=dev, dtype=torch.float32).reshape(-1, 3, 3)
        if len(K_t) == 1:
            K_t = K_t.expand(len(ids), -1, -1)
        row = cached["row"]
        row_pose = pose_t[row]
        xyz = (row_pose[:, :3, :3] @ cached["pos"][..., None]).squeeze(-1) + row_pose[:, :3, 3]
        z = xyz[:, 2]
        valid = z > 1e-3
        near = torch.where(valid.any(), torch.where(valid, z, 1e30).amin() * 0.8, 0.001).clamp(min=0.001)
        far = torch.where(valid.any(), torch.maximum(near * 10, torch.where(valid, z, 0).amax() * 1.2), 100.0)
        proj = _projection_y_down(K_t, int(frame_w), int(frame_h), near, far)
        camera_xyz = torch.cat((xyz * xyz.new_tensor([1, -1, -1]), torch.ones_like(xyz[:, :1])), dim=1)
        clip = (proj[row] @ camera_xyz[..., None]).squeeze(-1)
        clip = (clip[:, None] @ _clip_tf(box_t, frame_h, frame_w)[row]).squeeze(1)
        raster = self._dr.rasterize(self._glctx, clip.contiguous(), cached["faces"],
                                    tuple(tile) if isinstance(tile, tuple) else (int(tile), int(tile)),
                                    ranges=cached["ranges"])
        if isinstance(raster, RuntimeError):
            raise raster
        rast = raster[0]
        world_normals = (row_pose[:, :3, :3] @ cached["normal"][..., None]).squeeze(-1)
        attributes = torch.cat((xyz, world_normals, cached["color"], cached["uv"]), dim=1)
        maps, _ = _interpolate_pair(self._dr, attributes[None], rast, cached["faces"])
        uv = maps[..., 9:11].remainder(1.0)
        region = cached["tex_region"][:, None, None]
        texc = uv * region[..., :2] + region[..., 2:]
        texture = self._dr.texture(cached["atlas"], texc.contiguous(), filter_mode="linear")
        base = torch.where(cached["textured"][:, None, None, None], texture, maps[..., 6:9])
        covered = (rast[..., 3:] > 0).to(base.dtype)
        rgb = _shade(base, maps[..., 3:6], maps[..., :3]) * covered
        depth = maps[..., 2] * covered[..., 0]
        return torch.flip(rgb, (1,)).contiguous(), torch.flip(depth, (1,)).contiguous()
