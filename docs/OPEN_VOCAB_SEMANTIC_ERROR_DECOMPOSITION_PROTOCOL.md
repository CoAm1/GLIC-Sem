# FAST-LIVO2 + Gaussian-LIC2 开放词汇语义误差分解与优化协议

版本：v1.0（2026-08-03）  
状态：实验前预注册  
适用数据：MCD `ntu_day_02`，暂不包含 HKU 数据  

## 1. 研究目标

本项目的最终目标是构建一条增量式语义 Gaussian 建图链路：FAST-LIVO2 输出图像、`T_world_camera` 和当前帧世界坐标 LiDAR 点云；Gaussian-LIC2 在服务器完成视锥裁剪、nearest-Z、彩色 Gaussian 初始化和增量几何/光度优化；SAM1 与 OpenCLIP 提供开放词汇二维语义；独立语义头把二维语言特征融合到 Gaussian 地图，并允许建图后输入任意文本查询。

当前实验不直接追求完整序列上的最高数值，而是回答一个更基础的问题：现有开放词汇结果差，主要损失发生在 SAM 分区、OpenCLIP 表征、PCA128 压缩，还是二维到三维的融合与渲染阶段。只有先确定主要误差来源，后续修改 loss、PCA 维数或 3D 融合才有科学依据。

本轮允许验证的结论仅限于：

1. 在固定的 60 帧 MCD 窗口内，各阶段造成的相对语义性能损失；
2. Head-only 语义训练是否保持既有几何和外观；
3. 某一项单变量修改是否在相同输入、划分和评价规则下改善结果。

本轮不允许宣称：完整 MCD 序列已经有效、跨场景泛化已经成立、稠密二维 mIoU 已经获得，或系统已经达到实时开放词汇语义建图。

## 2. 系统职责边界

### 2.1 FAST-LIVO2 前端

前端只负责输出：

- 原始或无损 PNG 图像；
- `T_world_camera`，格式为 `qw qx qy qz tx ty tz`；
- 当前观测时间窗、运动补偿后、转换到世界坐标的 LiDAR 点云。

前端不重复执行固定相机分辨率下的 z-buffer。服务器 `projectCloudToFrame()` 已经将 `T_world_camera` 取逆为 `T_camera_world`，完成相机坐标变换、视锥裁剪和每像素 nearest-Z。

### 2.2 Gaussian-LIC2 后端

后端负责：

- LiDAR 点投影、可见性筛选与稀疏深度构建；
- 彩色 Gaussian 初始化；
- 增量光度和几何优化；
- 独立语义头优化与语言特征渲染；
- PLY、RGB、深度、alpha 和语言分数输出。

当前默认采用 Head-only：语义反向传播只能更新语义参数，位置、尺度、旋转、不透明度和颜色全部冻结或 detach。True Joint 仅作为失败对照，不作为默认路线。

### 2.3 开放词汇教师

固定教师为 SAM1 ViT-H + OpenCLIP ViT-B/16。SAM1 产生区域，OpenCLIP 为每个区域产生 512 维归一化语言特征。固定的 PCA128 基底将语言特征压缩后供增量语义头学习；查询时使用同一基底重建至 512 维并与文本特征比较。

SAM1 只负责提出区域，不负责判断开放词汇类别；OpenCLIP 负责区域和文本之间的语义相似度。MCD 标签不得参与 SAM、OpenCLIP、PCA 拟合或 3D mapper。

## 3. 第一性原理因果链

固定相机图像经过以下链路：

```text
图像
  -> SAM1 区域
  -> 512-D OpenCLIP 区域特征
  -> 固定 PCA128 编码/重建
  -> 2D 到 3D Head-only 融合与语言渲染
  -> 文本查询分类
```

定义四个评价阶段：

### S0：SAM Oracle

在每个评价帧上，使用该帧的 MCD 稀疏参考标签，为每个 SAM 区域赋予区域内的多数参考类别。它只用于测量 SAM 区域对参考类别边界的理论上限。

这是使用评价标签构造的 oracle，绝不是可部署方法，也不能与正式方法结果混写。未被 SAM 覆盖的有效参考像素预测为 unknown。S0 主要报告 macro-IoU、balanced accuracy、覆盖率、区域纯度和区域碎片度；不把 one-hot oracle 的 AP 当作有意义的连续分数指标。

### S1：Full Teacher

保持同一组 SAM 区域，直接使用完整 512 维 OpenCLIP 特征与冻结文本提示计算余弦分数。它测量在 SAM 边界固定后，OpenCLIP 是否能识别目标类别。

### S2：PCA Teacher

保持 S1 的区域和特征，只加入固定 PCA128 编码与重建：

```text
z = (f - mean) @ basis
f_hat = normalize(mean + z @ basis.T)
```

使用 `f_hat` 与完整文本特征计算余弦分数。S1 与 S2 的差异只能来自 PCA，不得改变 SAM 区域、提示词、图像、阈值或聚合方式。

### S3：3D Head-only

从冻结几何的 Gaussian 地图渲染 PCA128 语言特征，使用与 S2 完全相同的重建和文本查询方式。S2 与 S3 的差异来自二维到三维关联、可见性、跨帧融合、语义头拟合和渲染。

定义阶段损失：

```text
Delta_CLIP = mIoU(S0) - mIoU(S1)
Delta_PCA  = mIoU(S1) - mIoU(S2)
Delta_3D   = mIoU(S2) - mIoU(S3)
```

同时报告绝对差值、相对保留率和 bootstrap 置信区间，避免只依据单个点估计判断。

## 4. 固定数据、划分与隔离规则

### 4.1 预注册窗口

- 序列：MCD `ntu_day_02`；
- 帧范围：1800–1859，共 60 帧；
- 图像分辨率：640×480；
- 训练关键帧规则：`frame % 5 == 4`；
- 训练关键帧：1804、1809、1814、1819、1824、1829、1834、1839、1844、1849、1854、1859；
- 其他帧先作为 held-out 候选；
- 若候选帧使用的 MCD LiDAR 标签 scan 与任一训练帧相同，则从严格 held-out 中剔除；
- 当前严格 held-out 为 36 帧。

### 4.2 标签使用边界

MCD 投影标签是稀疏 LiDAR 参考，不是稠密二维人工真值。类别 0 是 ignore/unknown，类别 11 是 other-noise；主评价只使用类别 1–10 且参考置信度不低于 0.35 的像素。

标签只允许进入：

- S0 SAM Oracle；
- 最终 evaluator、混淆矩阵和可视化；
- 训练集上的固定阈值选择；
- 误差诊断统计。

标签禁止进入：

- SAM mask 生成或筛选；
- OpenCLIP crop、prompt 或特征生成；
- PCA 基底拟合；
- Gaussian 初始化、语义训练或可见性权重；
- held-out prompt、阈值或超参数选择。

### 4.3 固定资产

- 教师：SAM1 ViT-H + OpenCLIP ViT-B/16；
- 教师目录：`mcd_1800_1859_sam1_single_teacher_v1`；
- 固定 PCA：`universal_pca128_text0.25_demo01_20260727`；
- PCA `basis.f32` 当前 SHA-256：`ca180ea4ff4d89a95311d3e00a495d9490308ed2fe46fe432e8da0720eab3069`；
- 文本模板：`a photo of a {query}`；
- 类别提示组在首次 held-out 评价前冻结；
- 聚合主结果使用 raw score，不使用逐帧 min-max 和 29×29 smoothing。

固定 PCA 由 demo01 特征与通用文本锚点拟合，不使用 MCD 图像或标签。实验输出必须记录 PCA、prompt、mapper 二进制和输入 manifest 的哈希。

## 5. 评价域与指标

### 5.1 主评价域：完整参考域

对所有满足类别和置信度条件的参考像素评价。若某阶段没有 SAM 覆盖、没有有效 alpha 或没有预测，则显式预测 unknown，作为错误计入。该域用于阶段间主结论，防止通过丢弃困难像素提高 mIoU。

### 5.2 辅助域：共同有效交集

只在 S1、S2、S3 都具有有效预测的像素交集上评价，用于区分“覆盖率下降”和“有效区域分类错误”。它只能作为诊断，不能代替完整参考域主结果。

### 5.3 必报指标

- macro-IoU（类别 1–10 等权）；
- macro-AP（S1–S3 使用未阈值化连续分数）；
- balanced accuracy / macro recall；
- 整体 pixel accuracy；
- 参考域覆盖率和 alpha 覆盖率；
- 每类 IoU、precision、recall、AP；
- 混淆矩阵；
- 每帧指标与时间顺序曲线。

类别没有参考正样本时不把其 IoU/AP 人为记为 0，也不静默删除；必须记录为 N/A，并同时报告参与宏平均的类别数。

### 5.4 统计方法

像素不是独立样本。置信区间必须按连续时间块或 LiDAR scan block bootstrap，禁止把数百万像素当作独立样本计算极小标准误。默认使用 2000 次 block bootstrap，报告 95% CI。

涉及随机初始化的 3D 训练至少运行种子 3407、3408、3409。二维确定性诊断运行一次，但必须验证重复运行哈希一致。最终比较报告均值、标准差、配对差值和置信区间，而不只报告最好 seed。

## 6. 预注册门槛与决策树

### Gate A：工程完整性

必须全部通过：

- 60 个 teacher NPZ、60 个 PCA target、60 个 segmentation 一一对应；
- 图像、参考标签、scan alignment 与帧号一致；
- PCA、prompt 和输入哈希一致；
- evaluator 单元测试通过；
- MCD 标签路径不出现在 teacher/PCA/mapper 的输入清单中；
- Git 提交不含模型、数据、PLY、渲染结果或大文件。

### Gate B：SAM 区域上限

若 S0 macro-IoU 低于 0.50，或主要小类的 oracle IoU 接近 0，则当前 SAM 区域本身不足。下一步只研究 mask 粒度、多尺度区域、区域去重和小物体保留，不修改 PCA 或 3D loss。

### Gate C：完整语言教师

若 S1 保留不到 S0 macro-IoU 的 70%，主要瓶颈是区域 crop、OpenCLIP 表征或 prompt。下一步只在训练关键帧调试 crop 上下文、背景抑制、尺度和 prompt；冻结后再评 held-out。

### Gate D：PCA 压缩

PCA128 需同时满足：

- S2 至少保留 S1 macro-IoU 的 95%；
- 全部区域 top-1 agreement 不低于 0.90；
- Full Teacher margin ≥0.01 的区域 top-1 agreement 不低于 0.95；
- 文本分数 MAE 不高于 0.02。

若失败，先做纯二维 PCA128/PCA256 对照，不运行新的 3D 训练。只有 PCA256 显著改善且代价可接受时才改变语义通道数。

### Gate E：二维到三维保持率

S3 应至少保留 S2 macro-IoU 的 90%，且参考域覆盖率下降不超过 5 个百分点。若失败，先运行冻结 Geometry PLY 的确定性几何回投影基线：将 S2 区域特征按可见性写回 Gaussian，再渲染评价。若回投影基线好而 SGD Head-only 差，问题在优化；两者都差则问题在关联、遮挡、视角一致性或地图覆盖。

### Gate F：几何保护

相对 Geometry baseline，Head-only 必须满足：

- held-out PSNR 降低不超过 0.2 dB；
- held-out Depth-L1 恶化不超过 2%；
- Gaussian 数量变化不超过 5%；
- 几何字段、颜色字段和非语义训练状态的哈希保持不变，或逐字段差异在明确的浮点容差内。

True Joint 已出现明显几何破坏，只保留为负对照。在语义链通过 Gate B–E 前，不调整联合 loss 系数。

## 7. 实施顺序

### Phase 1：基础诊断代码

实现统一 evaluator，一次输出 S0–S3 的完整参考域、共同有效域、覆盖率、每类指标、阶段 drop、每帧 CSV 和机器可读 JSON。所有路径通过配置传入，不在代码中写死服务器绝对路径。

同时实现：

- artifact manifest 与 SHA-256 审计；
- disjoint-scan split 验证；
- SAM Oracle 区域多数票；
- full/PCA region score 生成；
- 3D score 与 alpha 读取；
- temporal-block bootstrap；
- 合成数据单元测试；
- Git staged-file 模型/大文件拦截器。

### Phase 2：60 帧二维误差分解

先运行 S0、S1、S2，不启动 Gaussian 训练。根据 Gate B–D 确定主要二维瓶颈。禁止同时改变 SAM、prompt 和 PCA。

### Phase 3：60 帧三维对照

若二维 Gate 通过，复用完全相同的 Geometry PLY，运行：

1. 几何回投影控制；
2. Head-only，seed 3407；
3. Head-only，seed 3408；
4. Head-only，seed 3409。

每次运行前后记录 PLY 几何字段哈希、二进制哈希、显存、耗时和退出码。

### Phase 4：单变量优化

每轮只允许修改一个因果模块，并保留上一轮作为 A/B：

- SAM 失败：区域粒度或多尺度；
- OpenCLIP 失败：crop/context/prompt；
- PCA 失败：128 与 256 维；
- 3D 失败：可见性、法线角权重、置信度、长期证据或 loss；
- 几何保护失败：检查 detach、优化器参数集合和共享 backward。

一次修改只有在预注册主指标改善、置信区间支持、覆盖率没有作弊性下降且几何 Gate 仍通过时才保留。否则回退该修改，但保留失败记录。

### Phase 5：扩展验证

60 帧 Gate 全部通过后，才扩展到 501 帧和完整 4558 帧。至少增加一个不同场景/时段的 MCD 序列后，才讨论跨场景开放词汇能力。完整序列不是自动的科学证明；仍需保持无泄漏划分和固定 prompt。

## 8. 代码与 Git 交付规则

Git 只提交：

- C/C++、CUDA、Python、Shell 源码；
- YAML/JSON 配置模板，但不含本机密钥和绝对私有路径；
- 单元测试；
- 技术文档与空目录占位文件。

Git 禁止提交：

- `*.pth`、`*.pt`、`*.ckpt`、`*.safetensors`、`*.onnx`、`*.engine`；
- rosbag、图像数据集、PCD、PLY、NPZ、二进制 target；
- 编译产物、Docker layer、日志、渲染图、实验输出；
- SSH key、token、密码和机器凭据；
- 单文件超过 10 MiB 的未知资产。

开发使用独立分支 `feature/open-vocab-error-decomposition`。提交前必须运行 staged-file 审计，只显式 `git add` 本轮源码、配置、测试和文档。服务器实验输出放在 `/mnt/data/tyh/Gaussian-LIC-experiments`，不放进源码仓库。

## 9. 对抗式审查清单

每轮结论前必须主动尝试推翻结果：

1. held-out 是否与训练关键帧共享 LiDAR scan；
2. MCD 标签是否意外进入 teacher、PCA 或 mapper；
3. prompt 或阈值是否看过 held-out 后才修改；
4. 是否只在有效 alpha 像素评价而隐藏覆盖率失败；
5. min-max、smoothing 或 threshold 是否改变了方法排序；
6. 类别不平衡是否让 accuracy 掩盖小类失败；
7. 是否把像素当独立样本夸大显著性；
8. 是否只报告最好 seed 或最好帧；
9. PCA basis、图像分辨率、prompt 与二进制哈希是否一致；
10. query 改变后原始 PLY 的哈希和 mtime 是否不变；
11. Head-only 是否真的没有更新几何、颜色和 densification 状态；
12. 可视化是否使用相同色表、阈值、裁剪和帧选择规则；
13. 当前 MCD 标签是否只能支持“稀疏投影语义一致性”而非“稠密二维真值 mIoU”；
14. 60 帧单窗口是否被错误外推为完整场景和跨场景结论。

任何一项失败，本轮对应结论降级为工程现象，不作为科学结论。

## 10. 当前已知事实与立即动作

已有 60 帧 A/B 表明 Head-only 能保持 Geometry 的 RGB 与深度，而 True Joint 明显破坏几何；但 Head-only 的严格 held-out macro-IoU 仅为 0.0663，小类 vehicle、bike 和 barrier 仍接近失败。因此当前最优先任务不是继续调联合 loss，而是完成 S0–S3 误差分解。

现有 PCA 诊断中，全区域 Full/PCA top-1 agreement 为 0.7138，margin ≥0.01 时为 0.9403，语言相关性 MAE 为 0.0190。它满足 MAE 门槛，但未满足本协议的 top-1 agreement 门槛，PCA 是一个真实嫌疑点；然而在获得 S0、S1、S2 的同域 mIoU 前，不能断言 PCA 是最大瓶颈。

此外，当前 SAM1 teacher 的 `report.json` 沿用了“`SAM3 references are pseudo labels`”警告文本，与实际教师不一致。基础代码必须将模型身份写入结构化 metadata，并通过测试防止再次混淆 SAM1/SAM3 实验来源。
