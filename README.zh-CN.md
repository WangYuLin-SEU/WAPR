<a id="top"></a>

<p align="center"><a href="README.md">English</a> · <strong>中文</strong></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/wapr-banner.png" alt="WAPR" width="960"/>
</p>

<h1 align="center">WAPR · 广角位姿修正</h1>

<p align="center"><strong>ECCV 2026 · 面向未见物体 6D 位姿估计的基础模型</strong></p>

<p align="center">Yulin Wang* · Mengting Hu* · Hongli Li · Jianghao Zhou · Chen Luo<br />
*共同一作。作者单位见宣传页。</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=zh">宣传页</a> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/?lang=zh">文档网站</a> · <a href="https://link.springer.com/chapter/10.1007/978-3-032-37383-0_13">论文</a> · <a href="https://wangyulin-seu.github.io/WAPR/pages/poster/wapr-eccv2026.jpg">海报</a></p>

<p align="center"><strong>大角度修正 · 默认 12 个初始候选 · 多物体批量推理</strong></p>

<p align="center"><a href="#results">效果展示</a> &nbsp; · &nbsp; <a href="#awards">竞赛奖状</a> &nbsp; · &nbsp; <a href="#pipeline">方法概览</a> &nbsp; · &nbsp; <a href="#start">快速开始</a> &nbsp; · &nbsp; <a href="#citation">引用与许可</a></p>

---

WAPR 无需逐物体微调，即可估计和修正位姿模型训练时未见过的物体的 6D 位姿。输入为 **RGB-D、相机内参、米制网格，以及 mask 或包围盒**；也可先用可选的 2D 前端寻找候选区域。

<a id="results"></a>
## 01 · 效果展示

### 01 / 广角位姿修正：三个物体、三种方法

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/wide/tudl_dragon.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/tudl_dragon.jpg" alt="TUD-L dragon: WAPR / MegaPose / FoundationPose, five updates" width="960"/></a>
</p>

<p align="center">TUD-L 恐龙从 <strong>100°</strong> 开始，展示五次更新。下方为从 <strong>80°</strong> 开始的 LM-O 电钻与从 <strong>120°</strong> 开始的 YCB-V 夹钳。三种方法使用相同初始旋转，平移固定为真值。100°／120° 是超过 WAPR 90° 训练扰动范围的单独案例，不代表汇总精度。</p>

| LM-O 电钻 | YCB-V 夹钳 |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/wide/lmo_driller.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/lmo_driller.jpg" alt="LM-O driller: three-method rotation comparison" width="420"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/wide/ycbv_clamp.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/ycbv_clamp.jpg" alt="YCB-V clamp: three-method rotation comparison" width="420"/></a> |
| WAPR: 80° → 4° | WAPR: 120° → 2° |

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/docs/pose.html?lang=zh#wide-angle-comparison">对照协议与完整结果</a></p>


---

### 02 / 七个 BOP 数据集的 6D 位姿估计

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=zh#showcase"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/bop-seven-poses.jpg" alt="七个 BOP 数据集的 6D 位姿估计" width="1200"/></a>
</p>

<p align="center">LM-O、T-LESS、TUD-L、IC-BIN、YCB-V、HB 和 ITODD 的位姿定位示例。彩色轮廓与 3D 包围盒展示预测位姿。</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=zh#showcase">查看交互结果与真值对照</a></p>


---

### 03 / 跨较大帧间隔的位姿跟踪

<p align="center">
  <video src="assets/readme/demo/tracking/predicted_init/track_mustard_easy_00_02_s32.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">芥末瓶跟踪对照视频</video>
</p>

<p align="center"><strong><a href="assets/readme/demo/tracking/predicted_init/track_mustard_easy_00_02_s32.mp4">▶ 播放芥末瓶对照视频</a></strong> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-ycbineoat.html?lang=zh">网页播放与实验说明</a></p>

<p align="center">两种方法由同一份预测掩码初始化，每隔 32 帧更新位姿；这段芥末瓶序列不重新初始化位姿。跟踪允许中途丢失，视频中的案例不能作为所有序列的恢复保证。</p>

<p align="center"><strong>饼干盒 · 步长 1</strong></p>

<p align="center">
  <video src="assets/readme/demo/tracking/predicted_init/track_cracker_box_reorient_s1.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">饼干盒跟踪对照视频</video>
</p>

<p align="center"><a href="assets/readme/demo/tracking/predicted_init/track_cracker_box_reorient_s1.mp4">▶ 播放饼干盒对照视频</a></p>

<p align="center"><strong>糖盒 · 步长 32</strong></p>

<p align="center">
  <video src="assets/readme/demo/tracking/recover_sugar_box1_s32.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">DINOv2 独立补救对照视频</video>
</p>

<p align="center"><a href="assets/readme/demo/tracking/recover_sugar_box1_s32.mp4">▶ 播放糖盒对照视频</a> · 共享给定初始掩码，双方独立执行 DINOv2 补救，使用各自状态和评分器。</p>


---

### 04 / TACO：滚筒和木盒

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/figures/taco_pose_compare.mp4"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/taco_first_pair.png" alt="TACO: WAPR estimates beside motion-capture reference poses" width="960"/></a>
</p>

<p align="center"><strong><a href="https://wangyulin-seu.github.io/WAPR/pages/docs/figures/taco_pose_compare.mp4">▶ 播放 TACO 视频</a></strong> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-taco.html?lang=zh">数据与评估说明</a></p>

<p align="center">第 80–109 帧：左侧为 WAPR，右侧为 TACO 动捕参考。图像轮廓重合与 6D 位姿误差是不同指标。</p>


---

### 05 / ROBI：反光螺丝的正负样本对照

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=zh#robi"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/robi-four-methods.jpg" alt="ROBI: WAPR, AAE, PPF and Line2D; blue positives and red negatives" width="1200"/></a>
</p>

<p align="center">使用全部保存候选，为每个真值选择对称 ADD 最小的预测。蓝色为 ADD &lt; 0.1d，红色为未达阈值；这是事后最近候选对照，不能读成正常输出的一对一召回率。点击图片可逐实例查看误差和正负状态。</p>


---

### 06 / 白色三角块：照片与 3D 点云联动

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=zh#wedge"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/wedge_zh.jpg" alt="White wedges in a bowl: predicted poses and interactive point cloud" width="960"/></a>
</p>

<p align="center"><strong><a href="https://wangyulin-seu.github.io/WAPR/pages/?lang=zh#wedge">打开三角块 3D 查看器</a></strong>。HCCEPose 第 000003 帧，展示九个保存的候选；点击物体可联动选择。该帧没有位姿真值。</p>


---

### 07 / 重建网格用于跨场景位姿估计

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-reconstruct.html?lang=zh#cross-scene-recorded-result"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/docs/figures/cross_scene_flow_zh.svg" alt="Source image, reconstructed mesh, cross-scene 2D detection and 6D pose" width="960"/></a>
</p>

<p align="center">场景 A 的源图 → 重建网格 → 场景 B 的 2D 检测 → WAPR 6D 位姿。蓝色标出最高分预测，灰色为其余较低分结果。</p>

| 2D 检测 | 6D 位姿与重建网格 |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/cross/detect.png"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/cross/detect.png" alt="Cross-scene 2D detection" width="360"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/pages/demo/cross/pose.png"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/cross/pose.png" alt="Cross-scene 6D pose and reconstructed mesh" width="572"/></a> |

<p align="center">源图为 YCBInEOAT mustard0 首帧，目标为 YCB-V 场景 50、第 1130 帧。目标帧的标注类别、框和掩码不参与预测；源图重建出的网格用于目标帧位姿估计，具体配置见链接中的实验说明。</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-reconstruct.html?lang=zh#cross-scene-recorded-result">跨场景实验与完整说明</a></p>

---

### 08 / 机器人仿真：桌面与腕部双相机

<p align="center">
  <video src="assets/readme/demo/robot/bottle_cameras.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">桌面与腕部双相机位姿跟踪仿真视频</video>
</p>

<p align="center"><strong><a href="assets/readme/demo/robot/bottle_cameras.mp4">▶ 播放双相机仿真视频</a></strong> · <a href="https://wangyulin-seu.github.io/WAPR/pages/docs/applications-robot.html?lang=zh">网页播放与实验说明</a></p>

<p align="center">左上为桌面相机，右上为腕部相机；下方显示两台相机各自的视野。红色轮廓为估计位姿，绿色为仿真真值。</p>

<p align="center">保存的时序跟踪实验；腕部视野受限时可能丢失目标。蓝色侧抓线用于展示，不代表每帧重新发送控制指令。</p>

---

<a id="awards"></a>

## 02 · 2025 竞赛奖状

下方展示仓库收录的奖状，点击可查看原图。BPC 的最佳单次方案／第 4 名来自 WAPR；第二名来自 FRTPose-WAPR。BOP 奖状对应搭配不同 2D 检测器的 FRTPose-WAPR 系统提交。

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


[奖项来源与系统说明](https://wangyulin-seu.github.io/WAPR/pages/?lang=zh#awards)

---

<a id="pipeline"></a>
## 03 · 方法概览

> **区域 → 12 个候选 → WAPR × 3 → SAPR × 2 → WBPS → 6D 位姿**

WBPS 在每个实例的组内选位姿；公开分数取该完整组的 `(100 - max(between_group)) / 200`。

这是当前默认的位姿估计配置，与上方五次更新的旋转对照实验不同。WAPR 使用高达 **90°** 的初始旋转扰动训练；SAPR 进一步修正小误差，WBPS 对候选评分。渲染与推理跨实例批量执行，各物体的评分组保持独立。配置见 [`wapr/recipe.py`](wapr/recipe.py)。

只有框时也可初始化平移；七通道 WAPR 用全零观测掩码表示缺少分割，框内深度与网络提示分开处理。详见[掩码与包围盒输入选择](https://wangyulin-seu.github.io/WAPR/pages/docs/pose.html?lang=zh#mask-flexibility)。

四份[基于 SA6D 训练的权重](https://wangyulin-seu.github.io/WAPR/pages/docs/data-resources.html?lang=zh#weights)分别为带 mask WAPR、不带 mask WAPR、SAPR 和 WBPS。没有 CAD 时，可选重建示例可先准备网格；分割、网格生成与机器人规划是独立组件，并不是 WAPR 修正网络本身的输出。

生成网格的对齐默认用灰色几何进行 WAPR/SAPR 修正与 WBPS 评分；随后由 DINOv2 用原网格外观比较保留的修正后视角，原外观也保留给展示。详见[重建参数配置](https://wangyulin-seu.github.io/WAPR/pages/docs/applications-reconstruct.html?lang=zh#reconstruct-align)。

[2025 BOP/BPC 奖项与系统结果](https://wangyulin-seu.github.io/WAPR/pages/?lang=zh#awards) · [推理后端实测与计时范围](https://wangyulin-seu.github.io/WAPR/pages/docs/pose-backend-scaling.html?lang=zh)。各项对照的设备、输入配置和计时范围见对应说明。

---

<a id="start"></a>
## 04 · 快速开始

本仓库提供 WAPR 推理源码与使用示例。模型权重和样本输入另行获取，TensorRT 引擎在目标 GPU 上生成。运行环境和安装步骤见安装文档。

本地 wheel 构建步骤与发行条件见 [wheel 构建说明](wheel_build/README.md)。PyPI 包尚未发布。

源码运行先按 **[安装引导](https://wangyulin-seu.github.io/WAPR/pages/docs/install-guide.html?lang=zh)** 和 **[2D 检测与 6D 位姿环境](https://wangyulin-seu.github.io/WAPR/pages/docs/install-core.html?lang=zh)** 准备。参考环境为 Linux、Python 3.10 和 CUDA toolkit 12.8。GPU 包按设备环境选择；安装脚本保留现有 torch/torchvision 版本，不固定普通依赖版本，其他环境需验证。源码环境与引擎准备好后，从单物体示例开始：

```bash
python examples/02_one_category_one_instance.py
```

权重与小样通过[统一的路径登记](https://wangyulin-seu.github.io/WAPR/pages/docs/install-core.html?lang=zh#resource-paths)读取，也可复用其他磁盘上的已有资源。

结合[首个位姿说明](https://wangyulin-seu.github.io/WAPR/pages/docs/pose-one.html?lang=zh)阅读该脚本，无需先运行示例 01。在 [`wapr/recipe.py`](wapr/recipe.py) 中设置 `visualize = True`，即可保存该示例的位姿叠加图。示例 02 使用给定的标注可见掩码，不执行自动 2D 检测。

| 你的任务 | 建议入口 |
| --- | --- |
| 已有区域，估计位姿 | [02 · 单个物体](examples/02_one_category_one_instance.py) |
| 先找物体，再估计位姿 | [04 · 数量已知](examples/04_6d_localization.py) / [05 · 数量未知](examples/05_bop_6d_detection.py) |
| 接入自己的 RGB-D 与 CAD | [06 · 自定义场景](examples/06_custom_scene.py) |
| 跟踪单个或多个物体 | [08 · 单实例](examples/08_ycbineoat_one_instance.py) / [09 · TACO](examples/09_taco_many_instances.py) |
| 重建网格，并在其他场景复用 | [11 · 重建](examples/11_reconstruct_object.py) / [12 · 跨场景位姿](examples/12_cross_scene_pose.py) |

[全部 17 个示例](https://wangyulin-seu.github.io/WAPR/pages/docs/index-examples.html?lang=zh) · [输入约定](https://wangyulin-seu.github.io/WAPR/pages/docs/data-own.html?lang=zh) · [BOP 导出](examples/07_write_bop_pose_csv.py) · [扩展环境](https://wangyulin-seu.github.io/WAPR/pages/docs/install-extensions.html?lang=zh)

*使用边界：*未见物体在推理时仍需提供米制网格。深度、网格尺度与区域质量会影响对齐，对称与遮挡可能带来歧义。模型分数用于排序，不是经过校准的成功概率。

完整用法与输入约定见[文档网站](https://wangyulin-seu.github.io/WAPR/pages/docs/?lang=zh)。

---

<a id="citation"></a>
## 05 · 引用与许可

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

[作者与权益声明](AUTHORS.md) · [yulinwang@seu.edu.cn](mailto:yulinwang@seu.edu.cn)。

代码采用 [LGPL-2.1-only](LICENSE)；四份第一方权重采用 [CC BY-ND 4.0](WEIGHTS_LICENSE.txt)，允许商用，但须遵守署名及再发行条件，本许可不允许公开再发行修改版权重。第三方组件仍适用各自条款：[来源与许可声明](THIRD_PARTY_NOTICES.txt) · [致谢](https://wangyulin-seu.github.io/WAPR/pages/docs/credits.html?lang=zh)。

企业版本与定制需求：[shopedataset@gmail.com](mailto:shopedataset@gmail.com)。

[English version](README.md#citation) · [返回顶部](#top)
