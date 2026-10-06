<a id="top"></a>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">中文</a></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/wapr-banner.png" alt="WAPR" width="960"/>
</p>

<h1 align="center">WAPR · Wide-Angle Pose Refinement</h1>

<p align="center"><strong>ECCV 2026 · A foundation model for unseen-object 6D pose estimation</strong></p>

<p align="center">Yulin Wang* · Mengting Hu* · Hongli Li · Jianghao Zhou · Chen Luo<br />
*Equal contribution. Affiliations are listed on the project page.</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=en">Project page</a> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/?lang=en">Docs website</a> · <a href="https://link.springer.com/chapter/10.1007/978-3-032-37383-0_13">Paper</a> · <a href="https://wangyulin-seu.github.io/WAPR/pages/poster/wapr-eccv2026.jpg">Poster</a></p>

<p align="center"><strong>Large-angle correction · 12 initial hypotheses by default · Batched multi-object inference</strong></p>

<p align="center"><a href="#results">Gallery</a> &nbsp; · &nbsp; <a href="#awards">Awards</a> &nbsp; · &nbsp; <a href="#pipeline">Method</a> &nbsp; · &nbsp; <a href="#start">Get started</a> &nbsp; · &nbsp; <a href="#citation">Citation & license</a></p>

---

WAPR estimates and refines 6D poses of objects unseen during pose-model training, without per-object fine-tuning. Supply **RGB-D, camera intrinsics, a metric mesh, and a mask or box**; the optional 2D front end can find candidate regions first.

<a id="results"></a>
## 01 · Gallery

### 01 / Wide-angle refinement: three objects, three methods

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/wide/tudl_dragon.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/tudl_dragon.jpg" alt="TUD-L dragon: WAPR / MegaPose / FoundationPose, five updates" width="960"/></a>
</p>

<p align="center">The TUD-L dragon starts at <strong>100°</strong>, with all five updates shown. Below: the LM-O driller starts at <strong>80°</strong>, and the YCB-V clamp at <strong>120°</strong>. All three methods use the same initial rotation with translation fixed to ground truth. The 100°/120° cases exceed WAPR’s 90° training perturbation range; these individual examples are not aggregate accuracy.</p>

| LM-O driller | YCB-V clamp |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/wide/lmo_driller.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/lmo_driller.jpg" alt="LM-O driller: three-method rotation comparison" width="420"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/wide/ycbv_clamp.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/ycbv_clamp.jpg" alt="YCB-V clamp: three-method rotation comparison" width="420"/></a> |
| WAPR: 80° → 4° | WAPR: 120° → 2° |

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/docs/pose.html?lang=en#wide-angle-comparison">Comparison protocol and results</a></p>


---

### 02 / 6D pose estimation across seven BOP datasets

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=en#showcase"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/bop-seven-poses.jpg" alt="6D pose estimation across seven BOP datasets" width="1200"/></a>
</p>

<p align="center">Pose localization examples on LM-O, T-LESS, TUD-L, IC-BIN, YCB-V, HB and ITODD. Colored contours and 3D bounding boxes show the predicted poses.</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=en#showcase">Explore the interactive results and ground-truth comparison</a></p>


---

### 03 / Pose tracking across larger frame intervals

<p align="center">
  <video src="assets/readme/demo/tracking/predicted_init/track_mustard_easy_00_02_s32.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">Mustard tracking comparison</video>
</p>

<p align="center"><strong><a href="assets/readme/demo/tracking/predicted_init/track_mustard_easy_00_02_s32.mp4">▶ Play the mustard comparison</a></strong> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-ycbineoat.html?lang=en">Website player and protocol</a></p>

<p align="center">Both methods initialize from the same predicted mask and update poses every 32 frames. This mustard sequence does not reinitialize poses. Tracking can lose the target; these recordings do not guarantee recovery on every sequence.</p>

<p align="center"><strong>Cracker box · stride 1</strong></p>

<p align="center">
  <video src="assets/readme/demo/tracking/predicted_init/track_cracker_box_reorient_s1.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">Cracker-box tracking comparison</video>
</p>

<p align="center"><a href="assets/readme/demo/tracking/predicted_init/track_cracker_box_reorient_s1.mp4">▶ Play the cracker-box comparison</a></p>

<p align="center"><strong>Sugar box · stride 32</strong></p>

<p align="center">
  <video src="assets/readme/demo/tracking/recover_sugar_box1_s32.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">Independent DINOv2 recovery comparison</video>
</p>

<p align="center"><a href="assets/readme/demo/tracking/recover_sugar_box1_s32.mp4">▶ Play the sugar-box comparison</a> · Supplied initial mask with independent DINOv2 recovery; each method uses its own state and scorer.</p>


---

### 04 / TACO: roller and wooden box

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/figures/taco_pose_compare.mp4"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/taco_first_pair.png" alt="TACO: WAPR estimates beside motion-capture reference poses" width="960"/></a>
</p>

<p align="center"><strong><a href="https://wangyulin-seu.github.io/WAPR/pages/docs/figures/taco_pose_compare.mp4">▶ Play the TACO video</a></strong> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-taco.html?lang=en">Data and evaluation</a></p>

<p align="center">Frames 80–109: WAPR on the left and TACO motion-capture reference on the right. Image-contour overlap and 6D pose error are different metrics.</p>


---

### 05 / ROBI: positive and negative reflective-screw predictions

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=en#robi"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/robi/chrome/nearest_comparison.jpg" alt="ROBI: WAPR, AAE, PPF and Line2D; blue positives and red negatives" width="850"/></a>
</p>

<p align="center">Each annotation selects its nearest saved candidate by symmetry-aware ADD from all saved predictions. Blue denotes ADD &lt; 0.1d; red denotes failures. This is a post-hoc nearest-candidate comparison, not one-to-one recall of normal outputs. Click to inspect each instance’s error and status.</p>


---

### 06 / White wedges: linked image and 3D point cloud

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=en#wedge"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/wedge_en.jpg" alt="White wedges in a bowl: predicted poses and interactive point cloud" width="960"/></a>
</p>

<p align="center"><strong><a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=en#wedge">Open the wedge 3D viewer</a></strong>. HCCEPose frame 000003 shows nine saved candidates; select an object to highlight its pose. This frame has no ground-truth poses.</p>


---

### 07 / Reconstructed mesh for cross-scene pose estimation

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-reconstruct.html?lang=en#cross-scene-recorded-result"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/docs/figures/cross_scene_flow_en.svg" alt="Source image, reconstructed mesh, cross-scene 2D detection and 6D pose" width="960"/></a>
</p>

<p align="center">Scene A image → reconstructed mesh → 2D detection in scene B → WAPR 6D pose. Blue highlights the highest-scoring prediction; gray marks the lower-scoring results.</p>

| 2D detection | 6D pose and reconstructed mesh |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/cross/detect.png"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/cross/detect.png" alt="Cross-scene 2D detection" width="360"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/cross/pose.png"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/cross/pose.png" alt="Cross-scene 6D pose and reconstructed mesh" width="572"/></a> |

<p align="center">Source: the first YCBInEOAT mustard0 frame. Target: YCB-V scene 50, frame 1130. Target annotations are excluded from prediction. The reconstructed source mesh is used for pose estimation in the target frame; see the linked experiment for its configuration.</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-reconstruct.html?lang=en#cross-scene-recorded-result">Cross-scene experiment and details</a></p>

---

### 08 / Robot simulation: table and wrist cameras

<p align="center">
  <video src="assets/readme/demo/robot/bottle_cameras_en.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">Table and wrist camera pose tracking in robot simulation</video>
</p>

<p align="center"><strong><a href="assets/readme/demo/robot/bottle_cameras_en.mp4">▶ Play the two-camera simulation</a></strong> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-robot.html?lang=en">Website player and experiment details</a></p>

<p align="center">Top: table camera on the left, wrist camera on the right. Bottom: their respective fields of view. Red contours show estimated poses; green shows simulator ground truth.</p>

<p align="center">A saved temporal-tracking experiment; the wrist view can lose the target when visibility is limited. Blue side-grasp lines are visualizations, not new control commands at every frame.</p>

---

<a id="awards"></a>

## 02 · 2025 challenge certificates

The certificates included in this repository are shown below; click to read the originals. BPC Best One-shot Solution / 4th Place is from WAPR, and Second Place from FRTPose-WAPR. BOP certificates name FRTPose-WAPR system submissions with different 2D detectors.

| BPC · Best One-shot / 4th Place | BPC · Second Place |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bpc-oneshot.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bpc-oneshot.jpg" alt="BPC · Best One-shot / 4th Place" width="420"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bpc-second.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bpc-second.jpg" alt="BPC · Second Place" width="420"/></a> |


| BOP · Tracks 1–2 · Overall / Default | BOP · Track 8 |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bop-t12-overall-default.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bop-t12-overall-default.jpg" alt="BOP · Tracks 1–2 · Overall / Default" width="420"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bop-t8.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bop-t8.jpg" alt="BOP · Track 8" width="420"/></a> |


| BOP · Track 1 · Fast | BOP · Tracks 1–2 · Default |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bop-t1-fast.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bop-t1-fast.jpg" alt="BOP · Track 1 · Fast" width="420"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bop-t12-default.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bop-t12-default.jpg" alt="BOP · Tracks 1–2 · Default" width="420"/></a> |


| BOP · Track 10 · Single-view / Default | BOP · Track 10 · Fast |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bop-t10-single-default.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bop-t10-single-default.jpg" alt="BOP · Track 10 · Single-view / Default" width="420"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/awards/bop-t10-fast.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/awards/bop-t10-fast.jpg" alt="BOP · Track 10 · Fast" width="420"/></a> |


[Award sources and system descriptions](https://wangyulin-seu.github.io/WAPR/pages/?lang=en#awards)

---

<a id="pipeline"></a>
## 03 · Method overview

> **Region → 12 hypotheses → WAPR × 3 → SAPR × 2 → WBPS → 6D pose**

WBPS selects the pose within each instance’s group; its public score is `(100 - max(between_group)) / 200` over that complete group.

This is the current default estimation recipe, distinct from the five-update rotation comparison above. WAPR is trained with initial rotation perturbations up to **90°**; SAPR handles smaller corrections, and WBPS ranks the candidates. Rendering and inference batch supplied instances while keeping each object's scoring group independent. Settings live in [`wapr/recipe.py`](wapr/recipe.py).

A box can seed translation without segmentation. For seven-channel WAPR, the missing observed mask is zero; box depth and the network cue are separate. [Choosing mask or box input](https://wangyulin-seu.github.io/WAPR/pages/docs/pose.html?lang=en#mask-flexibility).

The four [SA6D-trained checkpoints](https://wangyulin-seu.github.io/WAPR/pages/docs/data-resources.html?lang=en#weights) are masked WAPR, unmasked WAPR, SAPR, and WBPS. Optional reconstruction examples prepare a mesh when CAD is unavailable; segmentation, mesh generation and robot planning are separate components, not outputs of the WAPR refiner itself.

Generated-mesh alignment defaults to gray geometry for WAPR/SAPR refinement and WBPS scoring. DINOv2 then ranks the retained refined views using the original mesh appearance; that appearance is also preserved for display. [Reconstruction recipe](https://wangyulin-seu.github.io/WAPR/pages/docs/applications-reconstruct.html?lang=en#reconstruct-align).

[2025 BOP/BPC awards and system results](https://wangyulin-seu.github.io/WAPR/pages/?lang=en#awards) · [Backend measurements and timing scope](https://wangyulin-seu.github.io/WAPR/pages/docs/pose-backend-scaling.html?lang=en). See each comparison for its device, input configuration and timing scope.

---

<a id="start"></a>
## 04 · Get started

This repository provides WAPR inference source and examples. Obtain model weights and sample inputs separately; prepare TensorRT engines on the target GPU. Follow the installation guide for supported environments.

For local wheel builds and release requirements, see [wheel build instructions](wheel_build/README.md). The PyPI package is not published yet.

For source use, follow the **[installation guide](https://wangyulin-seu.github.io/WAPR/pages/docs/install-guide.html?lang=en)** and **[2D detection and 6D pose setup](https://wangyulin-seu.github.io/WAPR/pages/docs/install-core.html?lang=en)**. Linux, Python 3.10 and CUDA toolkit 12.8 are the reference environment. Prepare GPU packages for your device; the installer preserves installed torch/torchvision versions and does not pin ordinary dependencies. Other environments require validation. After preparing the source environment and exporting its engines, begin with:

```bash
python examples/02_one_category_one_instance.py
```

Weights and samples use [one shared path registry](https://wangyulin-seu.github.io/WAPR/pages/docs/install-core.html?lang=en#resource-paths), including resources stored on another disk.

Read [the first-pose walkthrough](https://wangyulin-seu.github.io/WAPR/pages/docs/pose-one.html?lang=en) alongside this script; example 01 is not a prerequisite. Set `visualize = True` in [`wapr/recipe.py`](wapr/recipe.py) to save the example's pose overlay. Example 02 uses a supplied annotated visible mask, not automatic 2D detection.

| Your task | Start here |
| --- | --- |
| Pose from a supplied region | [02 · One object](examples/02_one_category_one_instance.py) |
| Find objects, then estimate their poses | [04 · Known counts](examples/04_6d_localization.py) / [05 · Unknown counts](examples/05_bop_6d_detection.py) |
| Use your own RGB-D and CADs | [06 · Custom scene](examples/06_custom_scene.py) |
| Track one or several objects | [08 · Single instance](examples/08_ycbineoat_one_instance.py) / [09 · TACO](examples/09_taco_many_instances.py) |
| Reconstruct a mesh and reuse it | [11 · Reconstruction](examples/11_reconstruct_object.py) / [12 · Cross-scene pose](examples/12_cross_scene_pose.py) |

[All 17 examples](https://wangyulin-seu.github.io/WAPR/pages/docs/index-examples.html?lang=en) · [Input conventions](https://wangyulin-seu.github.io/WAPR/pages/docs/data-own.html?lang=en) · [BOP export](examples/07_write_bop_pose_csv.py) · [Extension setup](https://wangyulin-seu.github.io/WAPR/pages/docs/install-extensions.html?lang=en)

*Practical scope:* unseen objects still require a metric mesh at inference. Depth, mesh scale and region quality affect alignment; symmetry and occlusion can leave ambiguity. Scores are ranking signals, not calibrated success probabilities.

See the [Docs website](https://wangyulin-seu.github.io/WAPR/pages/docs/?lang=en) for complete usage and input conventions.

---

<a id="citation"></a>
## 05 · Citation and license

```bibtex
@InProceedings{wang2026wapr,
    author    = {Wang, Yulin and Hu, Mengting and Li, Hongli and Zhou, Jianghao and Luo, Chen},
    title     = {WAPR: A Foundation Model for Wide-Angle Refinement in Unseen Object Pose Estimation},
    booktitle = {ECCV},
    year      = {2026},
    pages     = {221-239},
    volume    = {17016}
}
```

[Authorship and rights](AUTHORS.md) · [yulinwang@seu.edu.cn](mailto:yulinwang@seu.edu.cn).

Code: [LGPL-2.1-only](LICENSE). Four first-party checkpoints: [CC BY-ND 4.0](WEIGHTS_LICENSE.txt), allowing commercial use with its attribution and redistribution conditions; modified weights may not be publicly redistributed under this license. Third-party components retain their own terms: [notices](THIRD_PARTY_NOTICES.txt) · [acknowledgments](https://wangyulin-seu.github.io/WAPR/pages/docs/credits.html?lang=en).

Enterprise versions and customization: [shopedataset@gmail.com](mailto:shopedataset@gmail.com).

[中文版本](README.zh-CN.md#citation) · [Back to top](#top)
