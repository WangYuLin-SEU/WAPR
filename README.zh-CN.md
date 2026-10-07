<a id="top"></a>

<p align="center"><a href="README.md"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/language-en-link-equal.svg" width="78" height="28" alt="English"/></a> &nbsp; <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/language-zh-active-equal.svg" width="78" height="28" alt="中文"/></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/wapr-banner-readme.webp" alt="WAPR" width="960"/>
</p>

<h1 align="center">WAPR · 广角位姿修正</h1>

<p align="center"><strong><code>ECCV 2026</code></strong></p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/?lang=zh">项目宣传页</a></p>

<h3 align="center">面向未见物体 6D 位姿估计的基础模型</h3>

<p align="center">
  Yulin&nbsp;Wang<sup>*</sup> &nbsp; · &nbsp; Mengting&nbsp;Hu<sup>*</sup> &nbsp; · &nbsp; Hongli&nbsp;Li &nbsp; · &nbsp; Jianghao&nbsp;Zhou &nbsp; · &nbsp; Chen&nbsp;Luo
</p>
<p align="center"><sub><sup>*</sup> 共同一作。作者单位见宣传页。</sub></p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/?lang=zh"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/website.svg" width="18" height="18" alt=""/>&nbsp;文档网站</a> &nbsp; <a href="https://link.springer.com/chapter/10.1007/978-3-032-37383-0_13"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/paper.svg" width="18" height="18" alt=""/>&nbsp;论文</a> &nbsp; <a href="https://wangyulin-seu.github.io/WAPR/poster/wapr-eccv2026.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/poster.svg" width="18" height="18" alt=""/>&nbsp;海报</a></p>

<p align="center"><strong>大角度修正 · 默认 12 个初始候选 · 多物体批量推理</strong></p>

<p align="center"><a href="#awards"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/award.svg?v=color1" width="18" height="18" alt=""/>&nbsp;竞赛奖状</a> &nbsp; <a href="#results"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/gallery.svg?v=color1" width="18" height="18" alt=""/>&nbsp;效果展示</a> &nbsp; <a href="#pipeline"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/method.svg?v=color1" width="18" height="18" alt=""/>&nbsp;方法概览</a> &nbsp; <a href="#start"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/start.svg?v=color1" width="18" height="18" alt=""/>&nbsp;快速开始</a> &nbsp; <a href="#citation"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/citation.svg?v=color1" width="18" height="18" alt=""/>&nbsp;引用与许可</a></p>

---

WAPR 修正位姿模型训练时未见过的物体的 6D 位姿。输入为 RGB-D、相机内参、米制网格，以及 mask 或框。训练覆盖高达 90° 的候选旋转误差。默认流程使用 12 个候选、三次 WAPR 更新、两次 SAPR 更新，再由 WBPS 排序。推理时仍需提供米制网格。

**功能亮点**

- **广角位姿修正：**训练覆盖高达 90° 的旋转扰动，支持 mask 或框，不需要逐物体微调位姿模型。
- **RGB-D 批量位姿估计：**RTX 5090、OpenGL + TensorRT FP16、每实例 12 个候选，预热后位姿阶段约 88–90 实例/秒；不含 2D 检测与 NMS。[测量条件](https://wangyulin-seu.github.io/WAPR/docs/pose-backend-scaling.html?lang=zh#backend-scaling-instances)。
- **大运动 6D 位姿跟踪：**步长 32 的芥末瓶对照中，WAPR 的 ADD 达标率为 100%、平均 ADD 为 4.6 mm；FoundationPose 为 68.2% 和 36.3 mm，均不重新初始化位姿。这是此序列的结果。[视频与协议](https://wangyulin-seu.github.io/WAPR/docs/applications-ycbineoat.html?lang=zh)。
- **位姿评分与可选跟踪补救：**WBPS 对候选位姿排序；独立 DINOv2 补救案例展示候选搜索与评分，并保留失败结果。[跟踪说明](https://wangyulin-seu.github.io/WAPR/docs/pose-track.html?lang=zh)。
- **四份权重：**带掩码 WAPR、无掩码 WAPR、SAPR 和 WBPS；示例涵盖定位、检测、跟踪、重建及机器人仿真。位姿推理仍需米制网格。

<a id="awards"></a>

## <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/award.svg?v=color1" width="26" height="26" alt=""/> 2025 竞赛奖状

下方展示仓库收录的奖状，点击可查看证书。BPC 的最佳单次方案／第 4 名来自 WAPR；第二名来自 FRTPose-WAPR。BOP 奖状对应搭配不同 2D 检测器的 FRTPose-WAPR 系统提交。

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/?lang=zh#awards"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/awards-bpc-bop-readme.webp" alt="八张 BOP 与 BPC 竞赛证书" width="1400"/></a></p>


[奖项来源与系统说明](https://wangyulin-seu.github.io/WAPR/?lang=zh#awards)

---

<a id="results"></a>
## <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/gallery.svg?v=color1" width="26" height="26" alt=""/> 效果展示

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/rotate.svg?v=color1" width="22" height="22" alt=""/> 广角位姿修正：三个物体、三种方法

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/demo/wide/tudl_dragon.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/tudl_dragon-readme.webp" alt="TUD-L dragon: WAPR / MegaPose / FoundationPose, five updates" width="960"/></a>
</p>

<p align="center">TUD-L 恐龙从 <strong>100°</strong> 开始，展示五次更新。下方为从 <strong>80°</strong> 开始的 LM-O 电钻与从 <strong>120°</strong> 开始的 YCB-V 夹钳。三种方法使用相同初始旋转，平移固定为真值。100°／120° 是超过 WAPR 90° 训练扰动范围的单独案例，不代表汇总精度。</p>

| LM-O 电钻 | YCB-V 夹钳 |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/demo/wide/lmo_driller.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/lmo_driller-readme.webp" alt="LM-O driller: three-method rotation comparison" width="420"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/demo/wide/ycbv_clamp.jpg"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/wide/ycbv_clamp-readme.webp" alt="YCB-V clamp: three-method rotation comparison" width="420"/></a> |
| WAPR: 80° → 4° | WAPR: 120° → 2° |

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/docs/pose.html?lang=zh#wide-angle-comparison">对照协议与完整结果</a></p>


---

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/pose.svg?v=color1" width="22" height="22" alt=""/> 七个 BOP 数据集的 6D 位姿估计

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/?lang=zh#showcase"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/bop-seven-poses-readme.webp" alt="七个 BOP 数据集的 6D 位姿估计" width="1200"/></a>
</p>

<p align="center">LM-O、T-LESS、TUD-L、IC-BIN、YCB-V、HB 和 ITODD 的位姿定位示例。彩色轮廓与 3D 包围盒展示预测位姿。</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/?lang=zh#showcase">查看交互结果与真值对照</a></p>


---

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/tracking.svg?v=color1" width="22" height="22" alt=""/> 跨较大帧间隔的位姿跟踪

<p align="center">
  <a id="animation-track_mustard_easy_00_02_s32" href="#animation-track_mustard_easy_00_02_s32"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/track_mustard_easy_00_02_s32-readme.webp" alt="芥末瓶跟踪对照视频" width="800"/></a>
</p>


<p align="center">两种方法由同一份预测掩码初始化，每隔 32 帧更新位姿；这段芥末瓶序列不重新初始化位姿。跟踪允许中途丢失，视频中的案例不能作为所有序列的恢复保证。</p>

<p align="center"><strong>饼干盒 · 步长 1</strong></p>

<p align="center">
  <a id="animation-track_cracker_box_reorient_s1" href="#animation-track_cracker_box_reorient_s1"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/track_cracker_box_reorient_s1.gif" alt="饼干盒跟踪对照视频" width="800"/></a>
</p>


<p align="center"><strong>糖盒 · 步长 32</strong></p>

<p align="center">
  <a id="animation-recover_sugar_box1_s32" href="#animation-recover_sugar_box1_s32"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/recover_sugar_box1_s32.gif" alt="DINOv2 独立补救对照视频" width="800"/></a>
</p>

<p align="center">共享给定初始掩码，双方独立执行 DINOv2 补救，使用各自状态和评分器。</p>


---

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/multi.svg?v=color1" width="22" height="22" alt=""/> TACO：滚筒和木盒

<p align="center">
  <a id="animation-taco_pose_compare" href="#animation-taco_pose_compare"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/taco_pose_compare-readme.webp" alt="TACO: WAPR estimates beside motion-capture reference poses" width="672"/></a>
</p>


<p align="center">第 80–109 帧：左侧为 WAPR，右侧为 TACO 动捕参考。图像轮廓重合与 6D 位姿误差是不同指标。</p>


---

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/industrial.svg?v=color1" width="22" height="22" alt=""/> ROBI：反光螺丝的正负样本对照

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/?lang=zh#robi"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/robi-four-methods-readme.webp" alt="ROBI: WAPR, AAE, PPF and Line2D; blue positives and red negatives" width="1200"/></a>
</p>

<p align="center">使用全部保存候选，为每个真值选择对称 ADD 最小的预测。蓝色为 ADD &lt; 0.1d，红色为未达阈值；这是事后最近候选对照，不能读成正常输出的一对一召回率。点击图片可逐实例查看误差和正负状态。</p>


---

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/wedge.svg?v=color1" width="22" height="22" alt=""/> 白色三角块：6D 位姿预测

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/?lang=zh#wedge"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/wedge_pose_zh-readme.webp" alt="White wedges in a bowl: predicted 6D poses" width="480"/></a>
</p>

<p align="center"><strong><a href="https://wangyulin-seu.github.io/WAPR/?lang=zh#wedge">打开三角块 3D 查看器</a></strong>。HCCEPose 第 000003 帧，展示九个保存的候选；点击物体可联动选择。该帧没有位姿真值。</p>


---

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/reconstruct.svg?v=color1" width="22" height="22" alt=""/> 重建网格用于跨场景位姿估计

<p align="center">
  <a href="https://wangyulin-seu.github.io/WAPR/docs/applications-reconstruct.html?lang=zh#cross-scene-recorded-result"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/docs/figures/cross_scene_flow_zh.svg" alt="Source image, reconstructed mesh, cross-scene 2D detection and 6D pose" width="960"/></a>
</p>

<p align="center">场景 A 的源图 → 重建网格 → 场景 B 的 2D 检测 → WAPR 6D 位姿。蓝色标出最高分预测，灰色为其余较低分结果。</p>

| 2D 检测 | 6D 位姿与重建网格 |
| :---: | :---: |
| <a href="https://wangyulin-seu.github.io/WAPR/demo/cross/detect.png"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/cross/detect-readme.webp" alt="Cross-scene 2D detection" width="360"/></a> | <a href="https://wangyulin-seu.github.io/WAPR/demo/cross/pose.png"><img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/demo/cross/pose-readme.webp" alt="Cross-scene 6D pose and reconstructed mesh" width="572"/></a> |

<p align="center">源图为 YCBInEOAT mustard0 首帧，目标为 YCB-V 场景 50、第 1130 帧。目标帧的标注类别、框和掩码不参与预测；源图重建出的网格用于目标帧位姿估计，具体配置见链接中的实验说明。</p>

<p align="center"><a href="https://wangyulin-seu.github.io/WAPR/docs/applications-reconstruct.html?lang=zh#cross-scene-recorded-result">跨场景实验与完整说明</a></p>

---

### <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/robot.svg?v=color1" width="22" height="22" alt=""/> 机器人仿真：桌面与腕部双相机

<p align="center">
  <video src="assets/readme/demo/robot/bottle_cameras.mp4" autoplay loop muted controls playsinline preload="metadata" width="960">桌面与腕部双相机位姿跟踪仿真视频</video>
</p>

<p align="center"><strong><a href="assets/readme/demo/robot/bottle_cameras.mp4">▶ 播放双相机仿真视频</a></strong> · <a href="https://wangyulin-seu.github.io/WAPR/docs/applications-robot.html?lang=zh">网页播放与实验说明</a></p>

<p align="center">左上为桌面相机，右上为腕部相机；下方显示两台相机各自的视野。红色轮廓为估计位姿，绿色为仿真真值。</p>

<p align="center">保存的时序跟踪实验；腕部视野受限时可能丢失目标。蓝色侧抓线用于展示，不代表每帧重新发送控制指令。</p>

---

<a id="pipeline"></a>
## <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/method.svg?v=color1" width="26" height="26" alt=""/> 方法概览

> **区域 → 12 个候选 → WAPR × 3 → SAPR × 2 → WBPS → 6D 位姿**

WBPS 在每个实例的组内选位姿；公开分数取该完整组的 `(100 - max(between_group)) / 200`。

这是当前默认的位姿估计配置，与上方五次更新的旋转对照实验不同。WAPR 使用高达 **90°** 的初始旋转扰动训练；SAPR 进一步修正小误差，WBPS 对候选评分。渲染与推理跨实例批量执行，各物体的评分组保持独立。配置见 [`wapr/recipe.py`](wapr/recipe.py)。

只有框时也可初始化平移；七通道 WAPR 用全零观测掩码表示缺少分割，框内深度与网络提示分开处理。详见[掩码与包围盒输入选择](https://wangyulin-seu.github.io/WAPR/docs/pose.html?lang=zh#mask-flexibility)。

四份[基于 SA6D 训练的权重](https://wangyulin-seu.github.io/WAPR/docs/data-resources.html?lang=zh#weights)分别为带 mask WAPR、不带 mask WAPR、SAPR 和 WBPS。没有 CAD 时，可选重建示例可先准备网格；分割、网格生成与机器人规划是独立组件，并不是 WAPR 修正网络本身的输出。

生成网格的对齐默认用灰色几何进行 WAPR/SAPR 修正与 WBPS 评分；随后由 DINOv2 用原网格外观比较保留的修正后视角，原外观也保留给展示。详见[重建参数配置](https://wangyulin-seu.github.io/WAPR/docs/applications-reconstruct.html?lang=zh#reconstruct-align)。

[2025 BOP/BPC 奖项与系统结果](https://wangyulin-seu.github.io/WAPR/?lang=zh#awards) · [推理后端实测与计时范围](https://wangyulin-seu.github.io/WAPR/docs/pose-backend-scaling.html?lang=zh)。各项对照的设备、输入配置和计时范围见对应说明。

---

<a id="start"></a>
## <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/start.svg?v=color1" width="26" height="26" alt=""/> 快速开始

复用兼容的 Linux 或 Windows NVIDIA GPU 环境及支持 CUDA 的 PyTorch。OGL 需要 CUDA toolkit 和 C++ 编译器；Windows 还需准备 MSVC C++ 构建工具。安装 WAPR 0.0.5、准备核心依赖，再导出随包示例：

```bash
python -m pip install -U wapr==0.0.5
python -m wapr.bootstrap
python -c "from wapr.bootstrap import export_examples; export_examples('wapr_examples')"
python wapr_examples/02_one_category_one_instance.py
```

Windows 未安装 TensorRT 时使用 PyTorch 后端；TensorRT 后端需要 TensorRT 10。macOS 不支持 NVIDIA CUDA 推理。检测、重建和机器人功能在首次使用时准备，仅导入示例不会安装这些功能。示例 01、08、09、12 会先准备检测依赖，再使用对应库；Ultralytics 不属于基础安装依赖。随包 SAM3D 环境方案与机器人依赖面向 Linux。SAM3D 可能需要独立环境及用户自己的模型访问权限；准备过程更换已有依赖前需确认。系统前提与源码使用见[安装引导](https://wangyulin-seu.github.io/WAPR/docs/install-guide.html?lang=zh)，扩展功能见[可选功能说明](https://wangyulin-seu.github.io/WAPR/docs/install-extensions.html?lang=zh)。资源默认使用用户缓存；运行前设置绝对路径 `WAPR_CACHE_DIR` 可更换位置。示例 02 使用给定掩码，不需要自动 2D 检测；位姿叠加图默认开启，可通过 `wapr/recipe.py` 中的 `visualize` 开关控制。

| 你的任务 | 建议入口 |
| --- | --- |
| 已有区域，估计位姿 | [02 · 单个物体](examples/02_one_category_one_instance.py) |
| 先找物体，再估计位姿 | [04 · 数量已知](examples/04_6d_localization.py) / [05 · 数量未知](examples/05_bop_6d_detection.py) |
| 接入自己的 RGB-D 与 CAD | [06 · 自定义场景](examples/06_custom_scene.py) |
| 跟踪单个或多个物体 | [08 · 单实例](examples/08_ycbineoat_one_instance.py) / [09 · TACO](examples/09_taco_many_instances.py) |
| 重建网格，并在其他场景复用 | [11 · 重建](examples/11_reconstruct_object.py) / [12 · 跨场景位姿](examples/12_cross_scene_pose.py) |

[全部 17 个示例](https://wangyulin-seu.github.io/WAPR/docs/index-examples.html?lang=zh) · [输入约定](https://wangyulin-seu.github.io/WAPR/docs/data-own.html?lang=zh) · [BOP 导出](examples/07_write_bop_pose_csv.py) · [扩展环境](https://wangyulin-seu.github.io/WAPR/docs/install-extensions.html?lang=zh)

*使用边界：*未见物体在推理时仍需提供米制网格。深度、网格尺度与区域质量会影响对齐，对称与遮挡可能带来歧义。模型分数用于排序，不是经过校准的成功概率。

完整用法与输入约定见[文档网站](https://wangyulin-seu.github.io/WAPR/docs/?lang=zh)。

---

<a id="citation"></a>
## <img src="https://raw.githubusercontent.com/WangYuLin-SEU/WAPR/website/pages/assets/readme/icons/citation.svg?v=color1" width="26" height="26" alt=""/> 引用与许可

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

代码采用 [LGPL-2.1-only](LICENSE)；四份第一方权重采用 [CC BY-ND 4.0](WEIGHTS_LICENSE.txt)，允许商用，但须遵守署名及再发行条件，本许可不允许公开再发行修改版权重。第三方组件仍适用各自条款：[来源与许可声明](THIRD_PARTY_NOTICES.txt) · [致谢](https://wangyulin-seu.github.io/WAPR/docs/credits.html?lang=zh)。

企业版本与定制需求：[shopedataset@gmail.com](mailto:shopedataset@gmail.com)。

[English version](README.md#citation) · [返回顶部](#top)
