# Author: Yulin Wang (yulinwang@seu.edu.cn)
# SPDX-License-Identifier: LGPL-2.1-only
"""Plan optional installations in the current environment. / 在当前环境规划可选安装。"""
import importlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import glob


def _installed_requirements_healthy(requirements):
    """Check installed metadata and real imports without network access.

    不联网检查已有依赖声明及真实导入；缺包、版本冲突或导入异常均不跳过解析。
    """
    try:
        from packaging.requirements import Requirement
        pending = [Requirement(value) for value in requirements]
        checked = set()
        while pending:
            requirement = pending.pop()
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            # Direct sources must be resolved again; versions alone do not prove provenance.
            # 直接源码地址必须重新解析；仅版本一致不能证明来源相同。
            if requirement.url:
                return False
            distribution = metadata.distribution(requirement.name)
            if not requirement.specifier.contains(distribution.version, prereleases=True):
                return False
            identity = (distribution.metadata.get("Name", requirement.name), tuple(sorted(requirement.extras)))
            if identity in checked:
                continue
            checked.add(identity)
            for dependency in distribution.requires or []:
                parsed = Requirement(dependency)
                marker_matches = parsed.marker is None or any(
                    parsed.marker.evaluate({"extra": extra}) for extra in (set(requirement.extras) | {""}))
                if marker_matches:
                    parsed.marker = None
                    pending.append(parsed)
        for value in requirements:
            requirement = Requirement(value)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            name = re.sub(r"[-_.]+", "-", requirement.name).lower()
            if name == "pillow":
                module_name = "PIL.Image"
            elif name == "hydra-core":
                module_name = "hydra"
            elif name == "pyyaml":
                module_name = "yaml"
            elif name == "py-cpuinfo":
                module_name = "cpuinfo"
            elif name == "ultralytics-thop":
                module_name = "thop"
            elif name == "mani-skill":
                module_name = "mani_skill"
            else:
                module_name = name.replace("-", "_")
            importlib.import_module(module_name)
        return True
    except Exception:
        return False


def install_requirements(requirements, allow_replacement=None, check_only=False):
    """Resolve before installation; ask before replacing existing packages.

    安装前先解析；更换已有包必须明确同意。check_only 不安装也不询问。
    Returns a status and the resolved installation/replacement plan.
    返回状态及解析得到的安装、更换计划。
    """
    requirements = list(requirements)
    result = {"status": "ready", "requirements": requirements, "install": [], "replace": []}
    if not requirements:
        return result
    if _installed_requirements_healthy(requirements):
        result["verified"] = "installed dependency metadata and imports / 已有依赖声明与导入"
        return result
    try:
        pip_version = metadata.version("pip")
    except metadata.PackageNotFoundError:
        result.update(status="blocked", reason="pip is missing / 缺少 pip")
        return result
    version_numbers = re.match(r"(\d+)\.(\d+)", pip_version)
    if version_numbers is None or tuple(map(int, version_numbers.groups())) < (22, 2):
        # This release supports Python 3.8 and provides the required dry-run report.
        # 此发行支持 Python 3.8，并提供安装前所需的 dry-run 报告。
        planning_pip = "25.0.1"
        pip_change = {"name": "pip", "installed": pip_version, "target": planning_pip, "action": "replace"}
        result.update(status="approval_required", install=[pip_change], replace=[pip_change],
                      reason="Update pip to resolve the feature's full dependency plan / 先更新 pip，才能解析该功能的完整依赖计划")
        print("WAPR_REPLACEMENT_PLAN / 已有包更换计划", json.dumps([pip_change], ensure_ascii=False), flush=True)
        if check_only:
            return result
        approved = allow_replacement is True
        if allow_replacement is None and sys.stdin.isatty():
            answer = input("Allow this pip update? [y/N] / 同意此次 pip 更新吗？[y/N] ").strip().lower()
            approved = answer in ("y", "yes", "同意")
        if not approved:
            result["status"] = "declined"
            return result
        completed = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "pip==" + planning_pip],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            result.update(status="failed", exit_code=completed.returncode,
                          reason=re.sub(r"(?:https?|git\+https?)://\S+", "[download URL omitted / 下载地址已隐藏]", completed.stdout))
            return result
        if metadata.version("pip") != planning_pip:
            raise RuntimeError("pip update did not install the approved version / pip 更新结果不是已同意的版本")
        result = {"status": "ready", "requirements": requirements, "install": [], "replace": []}
    installed = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[re.sub(r"[-_.]+", "-", name).lower()] = distribution.version
    # First preserve every installed version, then try the protected GPU stack.
    # 首先保留全部已有版本，再尝试保留 GPU 软件栈；任何替换仍须确认。
    protected = [name + "==" + installed[name]
                 for name in ("torch", "torchvision", "torchaudio", "numpy") if name in installed]
    with tempfile.TemporaryDirectory(prefix="wapr-install-plan-") as temporary_dir:
        constraint_path = os.path.join(temporary_dir, "existing.txt")
        report_path = os.path.join(temporary_dir, "report.json")
        with open(constraint_path, "w", encoding="utf-8") as stream:
            stream.write("\n".join(name + "==" + version for name, version in sorted(installed.items())) + "\n")
        from wapr.download_route import choose_pypi_route
        pypi_route = choose_pypi_route()
        # One index for this resolution. A measured mirror is not mixed with another index.
        # 这一次解析只用一个索引。测得的镜像不与另一个索引混用。
        index_args = []
        if pypi_route.get("reason") == "measured" and pypi_route.get("name") != "official":
            index_args = ["--index-url", pypi_route["index"]]
        command = [sys.executable, "-m", "pip", "install", "--dry-run", "--report", report_path,
                   "--timeout", "120", "--retries", "5"] + index_args
        constrained = subprocess.run(command + ["-c", constraint_path] + requirements,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        use_constraints = constrained.returncode == 0
        resolution = constrained
        if not use_constraints:
            with open(constraint_path, "w", encoding="utf-8") as stream:
                stream.write("\n".join(protected) + "\n")
            resolution = subprocess.run(command + ["-c", constraint_path] + requirements,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            use_constraints = resolution.returncode == 0
        if not use_constraints:
            resolution = subprocess.run(command + requirements, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if resolution.returncode != 0 and index_args:
            print("WAPR_DOWNLOAD_ROUTE", {"kind": "pypi", "reason": "index_failed_use_official"}, flush=True)
            with open(constraint_path, "w", encoding="utf-8") as stream:
                stream.write("\n".join(name + "==" + version for name, version in sorted(installed.items())) + "\n")
            command = [sys.executable, "-m", "pip", "install", "--dry-run", "--report", report_path,
                       "--timeout", "120", "--retries", "5"]
            constrained = subprocess.run(command + ["-c", constraint_path] + requirements,
                                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            use_constraints = constrained.returncode == 0
            resolution = constrained
            if not use_constraints:
                with open(constraint_path, "w", encoding="utf-8") as stream:
                    stream.write("\n".join(protected) + "\n")
                resolution = subprocess.run(command + ["-c", constraint_path] + requirements,
                                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                use_constraints = resolution.returncode == 0
            if not use_constraints:
                resolution = subprocess.run(command + requirements, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                use_constraints = resolution.returncode == 0
        if resolution.returncode != 0:
            reason = re.sub(r"(?:https?|git\+https?)://\S+", "[download URL omitted / 下载地址已隐藏]", resolution.stdout)
            result.update(status="blocked", reason=reason)
            return result
        with open(report_path, encoding="utf-8") as stream:
            report = json.load(stream)
        targets = []
        opencv_switches = []
        opencv_names = {"opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"}
        for item in report.get("install", []):
            package = item["metadata"]
            name = re.sub(r"[-_.]+", "-", package["name"]).lower()
            target = str(package["version"])
            entry = {"name": name, "installed": installed.get(name), "target": target}
            result["install"].append(entry)
            # A report entry for an installed name means pip intends to overwrite it,
            # even when a direct URL or checkout has the same version number.
            # report 出现已有包表示 pip 将覆盖它；直接地址同版本重装也须明确同意。
            if name in installed:
                entry["action"] = "replace" if installed[name] != target else "reinstall"
                result["replace"].append(entry)
            # OpenCV distributions share cv2 files despite having different names.
            # OpenCV 发行名称不同却共用 cv2 文件，不能当作无影响的新依赖叠装。
            other_providers = {provider: installed[provider] for provider in sorted(opencv_names)
                               if provider != name and provider in installed} if name in opencv_names else {}
            if other_providers:
                entry.update(action="distribution_switch", remove=other_providers)
                if entry not in result["replace"]:
                    result["replace"].append(entry)
                opencv_switches.append((len(targets), entry))
            download_info = item.get("download_info", {})
            url = download_info.get("url")
            if not url:
                result.update(status="blocked", reason="Resolved package has no download URL / 解析包缺少下载地址: " + name)
                return result
            vcs_info = download_info.get("vcs_info")
            if vcs_info:
                commit = vcs_info.get("commit_id")
                if not commit:
                    result.update(status="blocked", reason="Source revision is unresolved / 源码版本未解析: " + name)
                    return result
                url = vcs_info["vcs"] + "+" + url + "@" + commit
            else:
                digest = download_info.get("archive_info", {}).get("hashes", {}).get("sha256")
                if digest:
                    url += ("&" if "#" in url else "#") + "sha256=" + digest
                from wapr.download_route import same_pypi_file_url
                # Keep the resolved hash. Only the host may change after the size check.
                # 保留解析得到的校验值。大小核对通过后只允许更换主机。
                url = same_pypi_file_url(url)
            targets.append(name + " @ " + url)
        planned_opencv = {entry["name"] for entry in result["install"] if entry["name"] in opencv_names}
        if len(planned_opencv) > 1:
            result.update(status="blocked", reason="Dependencies request multiple cv2 providers; choose one compatible OpenCV distribution / 依赖同时要求多个 cv2 提供者，需要选定兼容的单一 OpenCV 发行")
            return result
        if result["replace"]:
            result["status"] = "approval_required"
            print("WAPR_REPLACEMENT_PLAN / 已有包更换计划", json.dumps(result["replace"], ensure_ascii=False), flush=True)
        if check_only:
            return result
        if result["replace"]:
            approved = allow_replacement is True
            if allow_replacement is None and sys.stdin.isatty():
                answer = input("Allow these package replacements? [y/N] / 同意上述包更换吗？[y/N] ").strip().lower()
                approved = answer in ("y", "yes", "同意")
            if not approved:
                result["status"] = "declined"
                return result
        if not result["install"]:
            result["status"] = "ready"
            return result
        restore_wheels = []
        switched_names = []
        if opencv_switches:
            staged_directory = os.path.join(temporary_dir, "opencv-wheels")
            os.makedirs(staged_directory)
            requested_wheels = [targets[index] for index, _ in opencv_switches]
            for _, entry in opencv_switches:
                requested_wheels.extend(name + "==" + version for name, version in entry["remove"].items())
            # Obtain rollback files before removing the working cv2 provider.
            # 卸载可用 cv2 之前先准备恢复文件；下载失败不会改变环境。
            staged = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                                     "--dest", staged_directory] + requested_wheels,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if staged.returncode != 0:
                result.update(status="failed", exit_code=staged.returncode,
                              reason="OpenCV replacement/rollback download failed; existing provider retained / OpenCV 替换或恢复包下载失败，保留已有发行")
                return result
            remove_names = []
            for index, entry in opencv_switches:
                providers = dict(entry["remove"])
                providers[entry["name"]] = entry["target"]
                for name, version in providers.items():
                    matches = glob.glob(os.path.join(staged_directory, name.replace("-", "_") + "-" + version + "-*.whl"))
                    if len(matches) != 1:
                        result.update(status="blocked", reason="Ambiguous OpenCV rollback files / OpenCV 恢复文件不唯一")
                        return result
                    if name == entry["name"]:
                        targets[index] = matches[0]
                        switched_names.append(name)
                    else:
                        restore_wheels.append(matches[0])
                        remove_names.append(name)
            removed = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y"] + remove_names,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if removed.returncode != 0:
                restored = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall"] + restore_wheels,
                                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                result.update(status="failed", exit_code=removed.returncode, restore_exit_code=restored.returncode,
                              reason="OpenCV uninstall failed; attempted restoration / OpenCV 卸载失败，已尝试恢复")
                return result
        # Use reviewed URLs and hashes; preserve CUDA wheels and direct-source provenance.
        # 使用审阅地址与校验值；保留 CUDA wheel 来源及直接源码来源。
        install_command = [sys.executable, "-m", "pip", "install", "--no-deps", "--timeout", "120", "--retries", "5"]
        if use_constraints:
            install_command += ["-c", constraint_path]
        completed = subprocess.run(install_command + targets, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result["status"] = "installed" if completed.returncode == 0 else "failed"
        result["exit_code"] = completed.returncode
        # Replacing loaded binary modules does not replace their in-memory libraries.
        # 替换已导入的二进制包不会刷新进程内动态库，必须重启后再使用。
        restart_packages = []
        if completed.returncode == 0:
            for entry in result["replace"]:
                module_name = entry["name"].replace("-", "_")
                if entry["name"] in ("torch", "torchvision", "torchaudio", "numpy", "scipy", "opencv-python", "opencv-python-headless", "tensorrt"):
                    if entry["name"] in ("opencv-python", "opencv-python-headless"):
                        module_name = "cv2"
                    if module_name in sys.modules:
                        restart_packages.append(entry["name"])
            if restart_packages:
                result.update(status="restart_required", restart_packages=restart_packages,
                              reason="Installation completed; restart Python before calling this feature / 安装已完成，请重启 Python 后再调用此功能")
        if completed.returncode != 0:
            if restore_wheels:
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y"] + switched_names,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                restored = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall"] + restore_wheels,
                                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                result["restore_exit_code"] = restored.returncode
            result["reason"] = re.sub(r"(?:https?|git\+https?)://\S+", "[download URL omitted / 下载地址已隐藏]", completed.stdout)
        return result


def prepare_optional(feature, allow_replacement=None, check_only=False, source_requirement=None):
    """Prepare only the requested feature's Python dependencies.

    只准备显式请求功能的 Python 依赖；第三方源码与权重由调用方另行获取。
    Importing this module never installs dependencies. / 导入本模块不触发安装。
    source_requirement supplies the selected SAM2/RoMa checkout for one resolution.
    source_requirement 提供所选 SAM2/RoMa 检出，以一次解析完成源码及其依赖计划。
    """
    notes = []
    if feature == "dinov2":
        requirements = ["torchvision", "Pillow"]
        notes.append("DINOv2 source checkout is required; xFormers is optional / 需要 DINOv2 源码，xFormers 非必需")
    elif feature == "det2d":
        # The adapted BERT wrapper uses Transformers 4 attention/head-mask methods.
        # 适配的 BERT 包装调用 Transformers 4 的 attention/head-mask 方法。
        transformers_requirement = "transformers<5"
        try:
            from packaging.version import Version
            if Version(metadata.version("torch")) < Version("2.6"):
                # Transformers v4.52.0 import_utils disables Torch older than 2.1.
                # Its BERT .bin loader additionally requires Torch >=2.6.
                # Transformers v4.52.0 禁用低于 2.1 的 Torch；BERT .bin 加载还需 >=2.6。
                transformers_requirement = "transformers<4.52"
                notes.append("The current BERT pack uses pytorch_model.bin; Transformers >=4.52 requires torch >=2.6 to load it. Older releases must only read trusted, verified checkpoints / 当前 BERT 包使用 pytorch_model.bin；Transformers >=4.52 加载它需要 torch >=2.6，较早发行只应读取可信且已校验权重")
        except metadata.PackageNotFoundError:
            pass
        requirements = ["torchvision", transformers_requirement, "timm", "addict", "yapf", "pycocotools",
                        "scipy", "matplotlib", "pandas", "seaborn", "psutil", "py-cpuinfo",
                        "PyYAML", "requests", "tqdm", "safetensors", "ultralytics-thop"]
        notes.append("Use adapted GroundingDINO and Ultralytics v8.3.70 source, not a substitute pip detector / 使用适配 GroundingDINO 与 Ultralytics v8.3.70 源码，不替换为其他 pip 检测器")
    elif feature == "sam2":
        requirements = ["torch>=2.5.1", "torchvision", "hydra-core>=1.3.2", "iopath", "Pillow"]
        notes.append("Facebook SAM2 recommends Linux, Python >=3.10 and torch >=2.5.1; older versions are unverified here / Facebook SAM2 推荐 Linux、Python >=3.10、torch >=2.5.1；此处未验证更低版本")
    elif feature == "sam3d":
        from wapr.reconstruction_setup import prepare_reconstruction
        return prepare_reconstruction(allow_replacement=allow_replacement, check_only=check_only)
    elif feature == "roma":
        requirements = ["romatch"]
        notes.append("Python/Torch requirements depend on the romatch release; use its resolved metadata / Python 与 Torch 要求取决于 romatch 发行版本，以实际解析声明为准")
    elif feature == "unipose9d":
        requirements = ["torchvision", "PyYAML", "scikit-learn"]
        notes.append("Use pinned UniPose9D inference source with supplied RGB-D and mask; no segmentation or depth model is installed here / 使用固定 UniPose9D 推理源码及已有 RGB-D、掩码；此处不安装分割或深度模型")
        try:
            module = importlib.import_module("unipose9d_inference")
            healthy = all(callable(getattr(module, name, None)) for name in ("estimate_pose", "load_pose_model", "set_seed"))
        except Exception:
            healthy = False
        if healthy:
            return {"status": "ready", "feature": feature, "requirements": requirements,
                    "install": [], "replace": [], "notes": notes, "existing_source_api": True}
    elif feature == "qwen":
        # Keep the Transformers 4 API used by this Qwen example and GroundingDINO.
        # 保持本 Qwen 示例及 GroundingDINO 使用的 Transformers 4 API。
        requirements = ["transformers>=4.49,<5", "accelerate", "qwen-vl-utils", "Pillow"]
        notes.append("Qwen2.5-VL support and model access must be checked after installation / 安装后仍需检查 Qwen2.5-VL 支持与模型访问")
    elif feature == "robot":
        requirements = ["mani-skill", "sapien", "mplib;platform_system=='Linux'", "gymnasium", "transforms3d", "imageio", "imageio-ffmpeg"]
        notes.append("ManiSkill v3.0.1 declares Python >=3.9 and Linux mplib==0.1.1; resolved releases supply their own dependency metadata / ManiSkill v3.0.1 声明 Python >=3.9、Linux mplib==0.1.1；以实际发行依赖声明解析")
        notes.append("Simulation additionally needs a working Vulkan driver; physical robots have separate SDKs / 仿真另需可用 Vulkan 驱动；真实机器人 SDK 独立准备")
    else:
        raise ValueError("Unknown optional feature / 未知可选功能: " + str(feature))
    # SAM3D declares no Requires-Python in upstream pyproject; do not invent a floor.
    # SAM3D 上游 pyproject 未声明 Requires-Python，不把环境建议写成硬下限。
    if (feature == "sam2" and (sys.platform != "linux" or sys.version_info[:2] < (3, 10))) or (feature == "sam3d" and sys.platform != "linux"):
        return {"status": "blocked", "feature": feature, "requirements": requirements,
                "install": [], "replace": [], "notes": notes,
                "reason": "SAM2 path requires Linux/Python >=3.10; SAM3D requires Linux / SAM2 路径需要 Linux/Python >=3.10，SAM3D 需要 Linux"}
    if feature == "sam3d":
        # Do not mutate the environment when the complete inference plan is unresolved.
        # 完整推理安装计划尚未确定时，不先修改环境再报告失败。
        return {"status": "blocked", "feature": feature, "requirements": requirements,
                "install": [], "replace": [], "notes": notes, "source_metadata_verified": False,
                "pending_source_dependencies": ["pytorch3d", "spconv", "utils3d", "MoGe"],
                "reason": "SAM3D inference-only native dependencies and checkpoint access are not yet verified / SAM3D 仅推理原生依赖及权重访问尚未验证"}
    if feature in ("sam2", "roma"):
        if source_requirement is None:
            return {"status": "needs_source", "feature": feature, "requirements": [],
                    "install": [], "replace": [], "notes": notes,
                    "reason": "Resolve the selected source metadata before installing / 安装前需要解析所选源码的实际依赖声明"}
        requirements = [source_requirement]
    print("WAPR_OPTIONAL / 可选功能", feature, notes, flush=True)
    result = install_requirements(requirements, allow_replacement=allow_replacement, check_only=check_only)
    result.update(feature=feature, notes=notes)
    # A dry-run cannot import packages that its plan has not installed yet.
    # dry-run 尚未安装计划中的新包，不能用导入失败覆盖有效安装计划。
    if check_only and result.get("install"):
        if result["status"] == "ready":
            result["status"] = "installable"
        return result
    if result["status"] in ("ready", "installed") and feature in ("det2d", "qwen"):
        # Importing Transformers alone can succeed with Torch models disabled.
        # 仅导入 Transformers 可能成功，但其 Torch 模型实际上已被禁用。
        try:
            transformers = importlib.import_module("transformers")
            if feature == "det2d":
                config = transformers.BertConfig(hidden_size=16, num_hidden_layers=1,
                                                 num_attention_heads=2, intermediate_size=32,
                                                 vocab_size=32)
                model = transformers.BertModel(config)
                import torch
                model.eval()
                with torch.no_grad():
                    output = model(input_ids=torch.zeros((1, 2), dtype=torch.long))
                if tuple(output.last_hidden_state.shape) != (1, 2, 16):
                    raise RuntimeError("Unexpected BERT output / BERT 输出异常")
                for name in ("get_extended_attention_mask", "get_head_mask", "invert_attention_mask"):
                    if not callable(getattr(model, name, None)):
                        raise RuntimeError("Missing BERT API / 缺少 BERT API: " + name)
            else:
                model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")
                # Dummy placeholders exist when Transformers disables the Torch backend.
                # Transformers 禁用 Torch 后端时可能暴露虚拟占位类。
                if not model_class.__module__.startswith("transformers.models.qwen2_5_vl"):
                    raise RuntimeError("Qwen2.5-VL Torch implementation is unavailable / Qwen2.5-VL Torch 实现不可用")
                getattr(transformers, "AutoProcessor")
            result["api_verified"] = True
        except Exception as error:
            result.update(status="blocked", api_verified=False,
                          reason="Optional API/import check failed / 可选 API/导入检查失败: " + str(error))
    return result
