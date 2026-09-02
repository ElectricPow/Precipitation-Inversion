# 第二阶段：GR 稀疏反射率到 DPR 反射率实验路线

> 更新日期：2026-09-02  
> 当前状态：Stage 2第一版已完成3N/3V、距离、局部密度、Fill-45、E2插值输入、E3-W1.5/W1.25以及Stage 3串联实验。证据表明，当前“共享骨干+支持域/dBZ双头”把标定、配准、支持域恢复、空间补全和强回波长尾混在一个目标中，已成为串联性能的主要瓶颈。`S2-R0-DecompositionAudit`已经完成并据此启动 **Stage 2-v2任务分解路线**。非部署空间恢复对照`S2-R1-O-DPRSparseValue`已经完成60轮训练、完整validation和冻结Stage 1串联；其锚点与gap恢复较好，但outside、强回波和最终串联仍不足。严格受控的下一项`S2-R1-P-PartialConv`现已完成代码、配置、固定验收门槛和真实数据烟雾测试，等待三卡正式训练。  
> 本文是[《降水反演任务拆解与实验路线》](./降水反演任务拆解与实验路线.md)中“阶段二”的详细实施文档。

## 0. 当前执行路线：Stage 2-v2 任务分解

> **版本效力声明**：本节是2026-09-01起的当前实验路线，覆盖文末旧的“下一步”建议。后面第1—17节保留Stage 2第一版的数据契约、已完成实验和历史结论，不应再被解读为当前待执行顺序。

### 0.1 纠正“可观测性不足”的解释

GR稀疏、与DPR同体素重叠少，是地基雷达观测方式与当前加工产品的固有条件，而不是“等一份更好数据再解决”的外部借口。已有可观测性审计的正确用法是：

1. 将不同可观测区域显式定义为不同子任务，而不是继续使用同一解码器和全局平均损失；
2. 在直接观测区学习传感器数值域标定和有限位移；
3. 在邻近观测可达区学习局部空间传播和结构补全；
4. 在远离直接观测的区域明确建模不确定性，不把单一平滑均值当成“被精确恢复的真实回波”。

验证集38条轨道共有`1,496,179`个DPR有效反射率体素，其可观测性分解为：

| 区域 | 体素数 | 占DPR目标比例 | 子任务含义 |
|---|---:|---:|---|
| `Q11 / overlap` | 486,178 | 32.49% | GR和DPR同体素都有反射率：数值域标定与局部配准 |
| `gap_proxy` | 514,776 | 34.41% | DPR有值、GR同体素无值，但GR插值支持可达：邻域补全 |
| `outside_proxy` | 495,225 | 33.10% | DPR有值且GR插值支持也不可达：强先验、不确定性较高的远程恢复 |

原始GR–DPR在`Q11`上的`Pearson r=0.7315`、`RMSE=6.094 dBZ`、`Bias=-1.346 dBZ`。这说明已被GR直接观测的区域具有明显信息，但不代表`0.7315`是整个Stage 2的性能上限，也不代表其余区域可以用相同方式处理。

### 0.2 当前W1.25的区域误差预算

W1.25是Stage 2第一版的当前较好中间模型，完整验证集总体dBZ指标为`RMSE=5.3085 dBZ`、`r=0.7020`、`CCC=0.6816`和`Bias=-0.1489 dBZ`。将误差按任务区域分解后：

| 区域 | DPR目标占比 | dBZ RMSE | dBZ Pearson r | 总平方误差贡献 | Support Recall |
|---|---:|---:|---:|---:|---:|
| `Q11 / overlap` | 32.5% | 3.940 | 0.841 | 17.9% | 0.946 |
| `gap_proxy` | 34.4% | 4.657 | 0.779 | 26.5% | 0.871 |
| `outside_proxy` | 33.1% | 6.881 | 0.428 | 55.6% | 0.232 |
| `DPR >= 35 dBZ` | 8.1% | 10.623 | 0.171 | 32.6% | — |

W1.25的整体support指标为`CSI=0.5763`、`Recall=0.6840`和`Precision=0.7854`（`FAR=0.2146`）。强回波区还有`MAE=8.08 dBZ`、`Bias=-7.37 dBZ`和`r=0.171`的明显压缩。

因此，当前Stage 2并非“所有子任务都一样差”：

- `Q11`的标定已明显优于直接复制GR，但仍需审计小范围错位；
- `gap_proxy`的邻域恢复有效，但还未达到给串联留出足够裕度的程度；
- `outside_proxy`仅占三分之一目标，却贡献55.6%的dBZ平方误差，是最大空间瓶颈；
- 强回波仅占8.1%，却贡献32.6%的dBZ平方误差，是最大强度瓶颈；
- 当前四类Patch采样比例并不等于体素损失中的强回波比例，W1.25也只小幅改变了总回归权重，因而不足以单独解决长尾问题。

### 0.3 Stage 2-v2的任务与模型分解

不再把Stage 2简化为“一个双头U-Net同时猜support和dBZ”，而将其拆成可独立验收的五个能力：

1. **标定与有限配准**：在高可信`Q11`及其局部邻域中，分离GR→DPR数值域残差与有界水平位移；
2. **DPR支持域/回波发生恢复**：独立预测DPR回波发生概率，不再只在共享解码器最后接一个`1×1×1`头；
3. **直接观测区的条件反射率回归**：在有可信锚点的地方预测校准残差，并保留观测信息；
4. **邻近缺口与远离观测区的条件补全**：使用显式mask、最近距离、局部密度和三维上下文，分区域恢复，不再全局一次求平均损失；
5. **置信度融合与不确定性**：融合支持域概率、dBZ条件均值、方差/分位数和观测锚点，远离观测时允许模型表达“不确定”。

建议的概率分解为：

\[
p(S,Z\mid X,O)
=p(S\mid X,O)\,p(Z\mid S=1,X,O),
\]

其中：

- `X`是部署时真正可得的GR稀疏dBZ、原生有效mask、距离、高度等特征；
- `O`是由GR-only信息得到的可观测状态/置信度；
- `S`是DPR回波支持域；
- `Z`是`S=1`时的DPR反射率；
- `Q11/Q01`等使用DPR标签才能完整定义的分区只能用于训练loss元数据和评价，绝不能当作推理输入。

初始架构原则是：高度60层仍不下采样；优先使用mask-aware/partial convolution，避免标准化后占位`0`被普通卷积当成真实观测；support与条件dBZ至少使用独立解码器；直接观测区预测残差，补全区预测条件分布；所有复杂机制都必须通过前一个子任务审计后才能引入。

### 0.4 R0–R6渐进实验路线

#### `S2-R0-DecompositionAudit`：任务分解与误差上限审计（当前）

不训练新模型，固定数据split、W1.25 Stage 2检查点、封版Stage 1和已选定的support阈值，只在完整validation上完成：

1. **区域误差预算**：统计`Q11/gap_proxy/outside_proxy`以及强回波在dBZ MAE/RMSE/Bias/r、support指标和总平方误差中的贡献；
2. **局部位移oracle**：在可配置的水平搜索半径内计算最佳局部匹配，比较exact与relaxed/oracle指标，量化“错位”理论上能够解释多少误差；
3. **分区真值替换的冻结Stage 1串联oracle**：严格拆为两类单因素反事实实验。`value oracle`在全局真实DPR support下只替换一个区域的dBZ；`support oracle`固定Stage 2稠密dBZ，只在一个区域纠正support。value按`Q11/Q01/gap/outside/strong-tail`统计，support另加入`Q10/Q00`以审计假阳性；二者都报告最终降水`RMSE/r/CCC/Bias`，不能把彼此重叠区域的闭合比例相加。

R0的输出应包含可复现的JSON/CSV、总结Markdown和图表，并记录检查点hash、split、阈值、样本/体素数。R0不用test集选方案，不在局部位移oracle上训练模型，也不将oracle指标宣称为可部署性能。

#### `S2-R1-MaskedPretrain/Oracle-DPRSparse`：纯空间恢复能力

这一阶段有两个互补实验：

- **GR人工遮挡自监督**：对真实GR有效体素人工隐藏仰角圈、扇区、连续空间块或随机点，只在“隐藏前实际有值”的位置考查重建，对比普通卷积、mask-aware convolution和partial convolution；
- **`Oracle-DPRSparse`**：使用GR原生物理值mask作为稀疏几何，仅在`gr_value_mask & M_DPR`位置保留同体素真实DPR dBZ作为稀疏锚点，其余DPR信息隐藏，再恢复完整DPR support和dBZ。它去除了GR→DPR传感器数值域转换误差，专门测量当前稀疏几何下的空间重建上限。

`Oracle-DPRSparse`使用了部署时不可得的真实DPR值，因而是**非部署的空间重建上限/可选预训练任务**，不是新输入方案，也不是本次R0“现有checkpoint分区真值替换”oracle。

#### `S2-R2-OverlapCalibrationAlignment`：重叠区标定与配准

先在`Q11`训练低容量、按高度条件化的残差标定器；仅当R0的局部位移oracle证明错位有明确收益时，再引入受限的逐高度水平位移/可变形采样。位移必须有幅度上限、空间平滑和近似恒等约束，不改变高度坐标，不允许模型无限制“搬运”强回波。

#### `S2-R3-SupportOnly`：独立支持域恢复

冻结或固定R1/R2的稀疏特征，使用独立解码器预测DPR回波发生。在基本二值support外，将训练集分布固定后的10/15/20/30/35 dBZ等级建模为有序嵌套support，使支持域分支同时学会弱/中/强回波范围。选模不只看BCE，而看分区域CSI/POD/FAR/PR-AUC/Brier和多尺度FSS。

#### `S2-R4-ConditionalCompletion`：条件dBZ补全

第一步使用oracle support，单独学习“已知有DPR回波时，dBZ是多少”。`Q11/gap/outside`分别计算“加权和/本区域有效权重”再组合，不得让数量多、容易或误差大的单一区域隐式支配全部梯度。逐步比较：

1. mask-aware 3D U-Net粗场；
2. 置信度引导的邻域传播/残差细化；
3. 有序dBZ等级分类+等级内连续残差；
4. `outside`区的分位数或异方差输出；
5. 完成区域解耦后，再将Balanced MSE作为强回波长尾单因素消融。

#### `S2-R5-StagedFusion`：渐进融合与低学习率联合

按下列顺序把oracle中间量逐个换成预测量：

```text
oracle support + 固定/真值锚点
→ predicted support + 固定/真值锚点
→ predicted support + predicted calibration/alignment
→ 全模块低学习率联合微调
```

联合前先记录每个子任务对共享参数的梯度范数和余弦：只在训练速率/梯度尺度失衡被确认时测GradNorm，只在负余弦冲突频繁发生时测PCGrad，不在无证据时同时叠加两者。

#### `S2-R6-CascadeGate`：重新进入串联

仅当R1–R5的子任务门槛逐项通过后，才重新接入原始封版Stage 1：

1. 先测oracle support串联，隔离dBZ值场误差；
2. 再测predicted support串联，量化support带来的额外差距；
3. 只在Stage 2物理任务不退化时，增加小权重最终降水损失；
4. 不立即解冻Stage 1，避免再次把已练好的反射率→降水映射带偏。

### 0.5 验收门槛与分支决策

下列是本项目的预注册工程门槛，不是气象学通用标准。所有比较使用相同split、完整38条validation轨道、完全相同评价mask，并优先查看逐轨配对bootstrap置信区间。

| 关口 | 当前参考线 | 进入后续的条件 |
|---|---|---|
| R0局部配准 | raw GR/DPR support exact CSI 0.2916；局部oracle 0.3014 | 原预注册门槛仍以dBZ `r`绝对提升≥0.02或RMSE下降≥3%为准；当前support CSI只提供辅助证据，不能单独宣告通过。若后续实现位移分支，必须是有界局部对应且在R1/R2基础任务之后受控消融 |
| R1空间恢复 | 普通卷积+相同人工mask | mask-aware/partial convolution必须在held-out真实点上显著降低误差，且不靠提高平滑度换取 |
| R2 `Q11` | W1.25：RMSE 3.940、r 0.841 | 低容量标定/配准至少超过其中一项，另一项不退化，Bias更接近0 |
| R3 support | CSI 0.5763、Recall 0.6840、Precision 0.7854 | CSI/FSS提升不能伴随不可接受的FAR增长；`outside` Recall须明显超过0.232 |
| R4 `gap` | RMSE 4.657、r 0.779 | RMSE下降或r提升且强度/结构指标不退化 |
| R4 `outside` | RMSE 6.881、r 0.428 | 均值预测改善的同时，分位数/方差校准有效，不允许只以过度平滑降低MAE |
| R4强回波 | RMSE 10.623、Bias -7.37、r 0.171 | `>=35 dBZ`的RMSE和负Bias显著收窄，且中等反射率和support不退化 |
| R6 oracle support串联 | RMSE 3.9180 mm/h、r 0.4713 | 必须显著超过W1.25 oracle-support基线，否则不用support错误解释值场瓶颈 |
| R6 predicted support串联 | RMSE 3.9884 mm/h、r 0.4538 | 与oracle-support的差距持续缩小；最终同口径目标为超过师兄约`r=0.68`，并将工程目标设为`r>=0.70`以留出波动裕度 |

R0的分区真值替换并不要求某个区域达到预设的单一数字才能继续；它首先用`ΔRMSE/Δr/ΔCCC`和可关闭的降水平方误差份额排序子任务。原则上先处理“真值替换收益大且现有输入仍有物理信息”的区域，而不是先处理体素数最多或视觉最显眼的区域。

### 0.6 R0当时的严格实施边界（已完成）

本次只实现与运行：

```text
src/precipitation_inversion/data/stage2_subtask_masks.py
src/precipitation_inversion/inference/stage2_oracles.py
src/precipitation_inversion/metrics/stage2_decomposition.py
src/precipitation_inversion/metrics/stage2_local_shift.py
scripts/analyze_stage2_task_decomposition.py
scripts/analyze_stage2_local_shift_audit.py
scripts/evaluate_stage2_stage1_cascade.py --r0-decomposition-oracles
tests/test_stage2_subtask_masks.py
tests/test_stage2_decomposition_metrics.py
tests/test_stage2_decomposition_audit.py
tests/test_stage2_local_shift_audit.py
tests/test_stage2_oracles.py
```

输入固定为：已有的validation patch/whole-track索引、W1.25最佳检查点、封版Stage 1 epoch 22、现有规范化统计和validation阈值。输出固定写入新的R0分析目录，不覆盖任何已有训练/评价结果。

R0运行期间明确不混入R1人工遮挡训练、`Oracle-DPRSparse`训练、partial convolution、独立support解码器、可学习位移、Balanced MSE或GradNorm/PCGrad。现在R0的三部分均已跑满38条validation并得到`formal_result=true`，因此已按其结论单独启动0.10节的`S2-R1-O-DPRSparseValue`；其他模块仍未同时叠加。

截至2026-09-01，R0静态部分已经在完整38条validation轨道上复算完成。输出位于
`outputs/stage2_r0_decomposition_audit/static/`，并通过以下一致性检查：

- `Q11 + Q01 = 全部DPR回波目标`；
- `gap + outside = Q01`；
- `Q11 + gap + outside`覆盖全部support假阴性；
- `Q10 + Q00`覆盖全部support假阳性；
- 输入文件数与`val.json`中的38条轨道严格一致，且拒绝把test集或截断结果当作正式R0结论。

旧的全验证聚合逐高度位移oracle显示60层中58层以零位移为最佳，但该口径会把不同轨道、不同局部的相反位移互相抵消，因此不能据此关闭局部配准。新实现使用相同Stage 2 `occupancy_domain`、固定目标域、`64×49`非重叠水平窗口和`±2`格搜索，在完整38轨上得到：

- exact `(0,0)` pooled support CSI：`0.291614`；
- 全验证一个shift：`0.291614`，没有收益；
- 逐轨一个shift：`0.291975`；
- 逐轨逐高度shift：`0.294899`；
- 局部窗口逐高度oracle：`0.301430`，绝对提升`0.009816`、相对提升约`3.37%`；
- 38轨中35轨同时出现相反方向局部shift并具有抵消证据；但5144个有效窗口×高度的增益中位数为0，约49.96%严格大于0。

这表明“统一全局平移”不是答案，局部错位真实存在但其**理论support上限增益仍属中等**。因此当前不立即实现复杂可学习形变；有界局部对应保留为R2/R5候选，优先级仍要与冻结Stage 1区域value/support Oracle的最终降水闭合比例一起决定。

冻结Stage 1分区Oracle是计算成本较高的最后一部分，由同一个串联评价脚本通过`--r0-decomposition-oracles`开启。它已经跑满38条validation轨道，并保存value/support闭合比例、support概率校准、多dBZ阈值CSI/FSS、echo top/base、反射率质心和CFAD；checkpoint、index、threshold和有序sample ID的SHA256均通过校验，最终汇总已标记`formal_result=true`。1轨smoke结果即便显式允许，也永远不能成为正式结论。

### 0.7 R0输出位置与当前结论边界

- 完整静态误差预算及图：`outputs/stage2_r0_decomposition_audit/static/`；
- 完整局部有限位移审计：`outputs/stage2_r0_local_shift_audit/`；
- 完整冻结串联Oracle：`outputs/stage2_r0_decomposition_audit/cascade_oracles/`；
- 三部分最终汇总：`outputs/stage2_r0_decomposition_audit/final/`；
- test集尚未访问；`static_formal_result=true`且最终`formal_result=true`。

### 0.8 方法选择的论文依据

- 星地雷达对比应先建立公共采样体积/质量控制，再分析反射率偏差，支持R0/R2将空间配准与数值标定分开：[Warren 等，2018](https://amt.copernicus.org/articles/11/5223/2018/)、[Three-way calibration checks，2022](https://amt.copernicus.org/articles/15/915/2022/)、[Zhu 等，2023](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022WR033719)；
- Partial Convolution只用当前有效元素计算并随层更新mask，是R1/R4处理非规则稀疏缺口的可检验候选：[Liu 等，2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html)；
- 学习型邻域传播可在粗估计后传播稀疏锚点信息，是R4的候选而非R0默认模块：[Cheng 等，2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.html)；
- Balanced MSE用于连续长尾回归，只在区域解耦后对强回波进行受控消融：[Ren 等，2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html)；
- GradNorm和PCGrad分别面向多任务训练速率失衡与梯度冲突，支持R5的“先审计、后引入”原则：[Chen 等，2018](https://proceedings.mlr.press/v80/chen18a.html)、[Yu 等，2020](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。

### 0.9 R0正式运行顺序

R0不训练模型。局部位移审计已完成；如需复现，运行：

```bash
python scripts/analyze_stage2_local_shift_audit.py \
  --workers 4 \
  --window-scan 64 \
  --window-ray 49 \
  --max-shift 2 \
  --output-dir outputs/stage2_r0_local_shift_audit \
  --overwrite
```

冻结Stage 1区域Oracle使用单GPU串行评价，而不是DDP训练。将`CUDA_VISIBLE_DEVICES=4`中的`4`替换为运行时空闲卡：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_stage2_stage1_cascade.py \
  --stage1-checkpoint outputs/ablations/stage1_i_g002_t3d/best.pt \
  --stage2-run W1p25 \
    outputs/stage2_ablations/four_channel_distance_intensity_w1p25/best_dbz.pt \
    outputs/stage2_ablations/four_channel_distance_intensity_w1p25/analysis/validation_candidates/reflectivity/support_threshold.json \
  --split val \
  --output-dir outputs/stage2_r0_decomposition_audit/cascade_oracles \
  --device cuda:0 \
  --stage1-batch-size 1 \
  --stage2-batch-size 1 \
  --num-workers 0 \
  --save-orbits 0 \
  --bootstrap-replicates 2000 \
  --no-gr-interp \
  --r0-decomposition-oracles \
  --overwrite
```

完成后将三部分严格汇总；只有此时`formal_result`才应为`true`：

```bash
python scripts/analyze_stage2_task_decomposition.py \
  --stage2-evaluation-dir outputs/stage2_ablations/four_channel_distance_intensity_w1p25/analysis/validation_candidates/reflectivity \
  --patch-index metadata/stage2_patch_indices/val.json \
  --alignment-dir outputs/stage2_alignment_audit \
  --local-shift-audit outputs/stage2_r0_local_shift_audit/summary.json \
  --oracle-audit outputs/stage2_r0_decomposition_audit/cascade_oracles/r0_decomposition_oracles.json \
  --output-dir outputs/stage2_r0_decomposition_audit/final \
  --overwrite
```

### 0.10 `S2-R1-O-DPRSparseValue`实现契约（待正式训练）

这个实验不是可部署模型，而是把第一版Stage 2中的多个困难拆开后的**空间补全上限对照**。它保留当前GR的稀疏观测几何，但将锚点值替换成同体素真实DPR dBZ，从而暂时移除GR→DPR传感器数值域差异：

```text
anchor_mask = gr_value_mask AND dpr_valid

完整轨道真实DPR dBZ
  -> 仅anchor_mask=1处保留
  -> 使用train-only DPR逐高度mean/std标准化
  -> 其余位置填标准化0.0

模型输入：(B,4,64,64,60)
  channel 0 = dbz_dpr_sparse_anchor_standardized
  channel 1 = dpr_sparse_anchor_mask
  channel 2 = dpr_sparse_anchor_distance_scaled
  channel 3 = height_scaled

单头高度保持3D U-Net
  -> predicted_dpr_standardized：(B,1,64,64,60)
```

最近锚点距离先在完整轨道、每个高度独立计算水平Chebyshev距离，再截断并缩放到`[0,1]`；距离0表示当前格就是锚点，1表示超过最大搜索距离、轨道外halo或水平padding。高度仍不做下采样和padding。

完整DPR support不进入模型输入，也没有support输出头。它只定义：

```text
M_dbz = core_mask AND dpr_valid
```

损失是在`M_dbz=1`处计算的标准化dBZ Smooth-L1，`beta=0.2`；延续W1.25的物理dBZ监督权重：`<25/25–35/>=35 dBZ`分别为`1.0/1.10/1.25`。各batch和DDP进程最终都按实际权重和归约，不按batch数量平均。为了避免此前“余弦调度尚未进入低学习率阶段便早停”，本实验固定训练60轮并关闭早停；仍覆盖保存`best.pt/last.pt`，不可变检查点只保存`epoch_0009/0019/0029/0039/0049/0059.pt`。

由于该模型没有support头，`dpr_count=0`的纯背景Patch会产生严格零损失，保留第一版的20%背景配额只会浪费优化步。R1-O将背景采样权重设为0，并把W1.25三个有DPR目标层的相对比例`0.10:0.30:0.40`归一化为`0.125:0.375:0.50`；每轮总Patch步数仍不变。这是删除support任务后的必要配套，不是额外模型因素。

正式validation评价包含总体、逐高度、Q11/Q01、gap/outside、锚点/非锚点、`<15/15–25/25–35/>=35 dBZ`，并保存15/25/35 dBZ嵌套事件FSS、CFAD和反射率质心。由于真实DPR support被固定，support CSI和echo top/base会平凡地等于真值，不得解读成模型学会了支持域恢复。冻结串联只比较：

```text
true DPR dBZ + true support -> frozen Stage 1
R1-O完成dBZ   + true support -> frozen Stage 1
```

因此它回答的是“当前稀疏几何在去除传感器数值差异后，普通3D U-Net能否补全DPR值场并保住最终降水信息”，而不是可部署性能。

实现文件为：

```text
configs/stage2_r1_o_dpr_sparse_value.yaml
src/precipitation_inversion/models/stage2_completion_unet3d.py
src/precipitation_inversion/losses/stage2_completion_losses.py
src/precipitation_inversion/training/stage2_completion_engine.py
src/precipitation_inversion/inference/stage2_completion_sliding_window.py
scripts/train_stage2_r1_oracle_sparse_value.py
scripts/evaluate_stage2_r1_oracle_sparse_value.py
scripts/evaluate_stage2_r1_oracle_sparse_cascade.py
scripts/launch_stage2_r1_ddp.sh
tests/test_stage2_r1_oracle_sparse_value.py
```

结果决策分支预先固定为：

1. 若非锚点、outside、`>=35 dBZ`和冻结串联均显著改善，说明空间补全可行；下一项进入GR人工遮挡自监督，比较普通卷积与Partial Convolution，再进入R2低容量数值标定。
2. 若锚点区好、非锚点尤其outside仍差，说明瓶颈在稀疏几何传播；先在同一Oracle输入上比较mask-aware/Partial Convolution或受约束邻域传播，不急于重新混入传感器域转换。
3. 若连锚点区都不能拟合，先排查模型、标准化、损失和调度；不能据此宣称数据不可恢复。
4. 若dBZ指标改善但冻结串联仍无明显改善，优先检查强尾部、CFAD和垂直结构，而不是马上恢复support双任务。

### 0.11 `S2-R1-P-PartialConv`严格受控实验

R1-O完整validation表明，普通卷积已经能较好恢复直接锚点和邻近gap，但远离锚点的outside及强回波仍差：

| 区域 | R1-O RMSE（dBZ） | Bias（dBZ） | Pearson r |
|---|---:|---:|---:|
| 稀疏DPR锚点 | 0.991 | 0.078 | 0.991 |
| gap proxy | 2.772 | -0.035 | 0.927 |
| outside proxy | 6.847 | -1.482 | 0.460 |
| 全部非锚点 | 5.187 | -0.744 | 0.731 |
| DPR `>=35 dBZ` | 9.533 | -5.034 | 0.235 |

R1-O送入冻结Stage 1后的最终降水相关性为`r=0.62345`，因此下一项先检查普通卷积是否错误地把“标准化0占位”当成真实输入，而不是马上进入GR到DPR数值标定。

R1-P与R1-O保持完全相同的数据、标签、split、seed、采样器、W1.25 Smooth-L1损失、优化器、60轮余弦调度、checkpoint间隔和完整整轨评价。唯一模型因素是输入stem：

```text
inputs: (B,4,64,64,60)

channel 0 稀疏DPR锚点dBZ ─┐
channel 1 锚点mask        ─┴─ 3D PartialConv值分支 ─┐
                                                     ├─ 融合 -> 原R1-O U-Net主干
channel 2 最近锚点距离    ─┐                       │
channel 3 高度            ─┴─ 普通3D卷积几何分支 ──┘

output: standardized DPR dBZ (B,1,64,64,60)
```

PartialConv先把mask为0处的稀疏值严格置零，再按卷积窗口内的真实有效邻居数量重新归一化；无有效邻居的窗口输出0并保持传播mask为0。距离与高度是稠密物理几何信息，只通过普通卷积分支，不与锚点mask相乘。融合后仍沿用原来仅下采样水平维、保持60层高度不变的编码器/解码器和单dBZ输出头。新模型为`1,988,209`个参数，R1-O为`1,986,481`个，只增加`1,728`个（约`0.087%`），因此不是容量膨胀实验。

验收门槛在正式训练前固定如下：

1. outside的RMSE至少下降3%，或Pearson r至少增加0.02；
2. 全部非锚点的RMSE与r必须同时改善，排除只拟合锚点；
3. gap RMSE退化不得超过2%；
4. `>=35 dBZ` RMSE至少下降5%，且绝对Bias继续靠近0；
5. 冻结串联最终降水r至少比R1-O增加0.02；
6. 最终降水dR/dz相关性不得下降。

实现文件为：

```text
configs/stage2_r1_p_partial_conv.yaml
src/precipitation_inversion/models/stage2_partial_completion_unet3d.py
scripts/train_stage2_r1_oracle_sparse_value.py
scripts/evaluate_stage2_r1_oracle_sparse_value.py
scripts/evaluate_stage2_r1_oracle_sparse_cascade.py
scripts/compare_stage2_r1_partial_conv.py
tests/test_stage2_r1_partial_conv.py
```

训练结束后仍自动运行完整validation和冻结Stage 1串联；随后由`compare_stage2_r1_partial_conv.py`一次性应用上述固定门槛。若outside主门槛未通过，则停止连续尝试更多稀疏卷积变体；若通过且其它门槛没有明显冲突，再决定进入R2低容量数值标定或先处理强回波传播。

---

> **历史记录说明**：以下第1—17节记录Stage 2第一版从数据审计到W1.25和冻结串联的完整演进。其中的已完成结果仍是Stage 2-v2的对照基线；但“首轮/第一版/紧接着实现”等措辞属于当时语境，不再指示当前任务。

## 1. 先给出结论

阶段二不能定义成普通的“稀疏 GR dBZ 到 DPR dBZ 的逐点回归”，而应定义成一个联合任务：

1. **DPR 产品支持域/回波位置预测**：预测哪些三维格点应该有 DPR 有效反射率；
2. **条件反射率回归**：在真实 DPR 有效回波位置上预测 DPR dBZ；
3. **GR–DPR 观测域校准**：在 GR 和 DPR 同时有值的位置保证数值转换能力；
4. **稀疏恢复与小范围位移学习**：在 GR 没有直接数值、DPR 有值的位置学习空间填充。

形式化表示为：

\[
Z_{GR}^{sparse},\ M_{GR},\ H
\longrightarrow
\left\{
\begin{aligned}
\hat p_{DPR} &= P(M_{DPR}=1),\\
\hat Z_{DPR} &= Z_{DPR}\mid M_{DPR}=1.
\end{aligned}
\right.
\]

这里的“密集 DPR 反射率”是**相对 GR 更密集**，不是要让全部三维格点都输出反射率。DPR 产品本身也只在一部分格点保留反射率。

因此，当前最先要做的不是直接搭建大模型，而是：

> 先完成 GR 原始存储状态、DPR 支持域、GR–DPR 空间偏移和可恢复性审计，再建立简单校准/插值基线，最后训练双输出头 3D U-Net。

## 2. 这一阶段究竟在学什么

### 2.1 数值转换

GR 和 DPR 都以 dBZ 表示反射率，但它们的频段、观测方向、波束体积、衰减订正、时间和空间配准都不同。因此，即使在同一格点，两者也不应被要求完全相等。

阶段二首先需要学习：

\[
Z_{GR}\longrightarrow Z_{DPR}.
\]

### 2.2 支持域转换

GR 的直接反射率比 DPR 稀疏得多。只对 GR 已有数值做一个 MLP 或线性校准，输出仍然只在原来的 GR 位置有值，无法满足阶段二的目标。

因此还必须学习：

\[
M_{GR}^{sparse}\longrightarrow M_{DPR},
\]

以及在新增支持域上恢复反射率。

### 2.3 它还包含“删除”而非只有“填充”

全数据集中也存在大量“GR 有值、DPR 没有有效反射率”的位置。这可能由弱 GR 回波、两类仪器的灵敏度差异、时空错位或质量控制产生。

因此模型实际上必须同时学会：

- 保留和校准可靠的 GR 回波；
- 在有信息的邻域内填充 DPR 回波；
- 把一部分 GR-only 回波抑制为 DPR 无回波；
- 容忍并矫正小范围空间错位。

## 3. PPT 中可以确认的旧流程

PPT 的第一张实质内容页给出了：

```text
GR Raw (0.5°)
→ Correction
→ 3D Rasterization at Beam Center
→ Multi-GR Combination
→ ZRH Model
→ ZRH Sparse PrecipRate
→ 3D Linear Interpolation 或 UNet Model
→ Retrieved PrecipRate
```

图中能够直接确认的是：有限宽度的原始波束体积被表示到 beam-center 格点后，垂直剖面只留下离散的倾斜波束中心轨迹，它们之间形成了结构性空洞。

导师提到的“等距扇形圈被压缩到同心圆、不同仰角之间留空”可以作为对数据生成流程的工作解释；但 PPT 本身直接证明的是“波束中心栅格化”，并没有给出更详细的数学压缩方法。

另外，PPT 中旧路线更准确的描述是：

> 先用 Z–R 关系把稀疏 GR 反射率转为稀疏降水率，然后用线性插值或 U-Net 恢复降水率。

因此，后续复现师兄方法时应按这一顺序实现，不应将它和本阶段的“先恢复 DPR 反射率，再进入阶段一”路线混为同一方法。

## 4. 数据证据：稀疏恢复是主任务而非附带问题

### 4.1 全数据集统计

已有 `metadata/manifests/dataset_summary.json` 覆盖 254 个 NC 文件，共 `268,857,120` 个三维格点。下表的“有效反射率”使用当前严格口径：有限且不是 `-9999.9` 一类填充码。

| 数据或空间关系 | 格点数 | 比例/含义 |
|---|---:|---:|
| GR 直接有效 | 5,113,752 | 全域 1.902% |
| GR 插值有效 | 13,586,745 | 全域 5.054% |
| DPR 反射率有效 | 9,762,539 | 全域 3.631% |
| GR 与 DPR 同格点有效 | 2,864,798 | 仅覆盖 DPR 目标的 29.345% |
| DPR 有效但 GR 直接缺测 | 6,897,741 | 占 DPR 目标的 70.655% |
| DPR 有效且在 GR 插值支持域内 | 6,193,992 | 占 DPR 目标的 63.447% |
| DPR 有效但 GR 插值也覆盖不到 | 3,568,547 | 占 DPR 目标的 36.553% |
| GR 有效但 DPR 无有效反射率 | 2,248,954 | 占 GR 有效点的 43.98% |
| DPR 有效且位于 CFB 以下 | 1,941,424 | 占 DPR 目标的 19.89% |

这些数字表明：

- 只用 GR–DPR 同时有值的 29.345% 训练，只能学到校准，无法学到主要的稀疏恢复任务；
- 70.655% 的 DPR 目标需要模型在同格点没有直接 GR 值的情况下恢复；
- 36.553% 的 DPR 目标连现有插值产品都没有覆盖，这些点的可恢复性是阶段二的主要上限风险；
- 模型不能只“扩张”GR 回波，还必须学会抑制大量 GR-only 点，否则虚警会很高。

### 4.2 DPR 的“缺测”不等于卫星没有扫描

在全部 254 个文件中：

```text
dpr_reflectivity_valid == (pre_dpr > 0)
```

在每一个体素上都完全相等。而 `pre_dpr` 的原生有效域中，96.297% 是有效的零降水。

因此，`dbz_dpr=-9999.9` 不能简单解释为“卫星没扫到”。在这个加工数据集里，`dbz_dpr` 的有效性实际上就是 DPR 产品保留的正降水/回波支持域。

这也解释了为什么阶段二必须有支持域分类头。这个分类头是在 GR-only 输入上学习 DPR 产品支持域，不是把 DPR mask 作为输入，因而不构成输入泄漏。

## 5. GR 缺测的真正难点：原文件实际上有三种存储状态

### 5.1 示例文件的三态编码

对示例文件 `018026` 的 `dbz_gr_sparse`：

| 原始状态 | 数量 | 占全部格点 |
|---|---:|---:|
| NetCDF 原生 mask/NaN | 1,047,276 | 59.77% |
| 非 mask，但值为 `-9999.9` | 653,363 | 37.29% |
| 有限物理 dBZ | 51,601 | 2.94% |
| 合计 | 1,752,240 | 100% |

同一文件中：

- `dbz_gr_interp` 的三类比例是 15.18% / 78.42% / 6.40%；
- `dbz_dpr` 没有原生 mask，94.56% 为 `-9999.9`，5.44% 为物理 dBZ。

因此，不能先把所有状态都折叠成一个 `NaN`，再试图区分缺测原因。一旦折叠，原始存储中仍然存在的线索就已经丢失。

### 5.2 新旧代码对这两类缺测的处理相互冲突

当前项目的 `masks.py::to_float_array` 会将：

```text
NetCDF mask / NaN / Inf / <= -9990
```

全部统一转换为 `NaN`。这对阶段一是一个保守的有效数值口径，但会丢失阶段二需要审计的原始编码状态。

师兄的 `/storage/GR_DPR_3D/DatasetNew.py` 则：

- 只使用 NetCDF 原生 mask 构建 `X_mask`；
- 把所有 `<9 dBZ` 的 GR 数值（包括 `-9999.9`）设为 0；
- 但不因此把 `X_mask` 改为无效。

旧代码的注释将“非原生 mask 的廉线”解释为雷达扫描范围，这为两种编码可能分别表示“覆盖外”和“覆盖内无保留数值/波束间空洞”提供了证据，但还不足以证明 `-9999.9` 只有一种物理含义。

更重要的是，旧加载器默认 `maskSGRWithRMask=True`，会使用 DPR 降水 mask 去修改 GR 输入 mask。这个信息在部署时不可得，阶段二绝对不能复用该默认行为，否则会发生 DPR 标签泄漏。

### 5.3 建议保留的原始状态 mask

阶段二读取器应在进行任何数值替换前保留：

```text
gr_native_mask       = NetCDF原生mask/NaN的位置
gr_sentinel_mask     = ~gr_native_mask & (raw_value <= -9990)
gr_value_mask        = ~gr_native_mask & finite(raw_value) & (raw_value > -9990)
```

三者互斥。在未得到数据生产者明确回答前，它们应使用**存储语义名称**，不应直接重命名为：

```text
true_not_scanned / elevation_gap / no_echo
```

还可构建：

```text
gr_interp_value_mask
gap_proxy     = ~gr_value_mask & gr_interp_value_mask
outside_proxy = ~gr_interp_value_mask
```

但 `gap_proxy` 只能说明“当前插值算法能覆盖的空洞”，不能当作仰角间空洞的真实标签。

## 6. 可恢复性分区

定义：

```text
M_GR  = gr_value_mask
M_DPR = dbz_dpr物理dBZ有效mask
```

在可靠的标签域和非 padding 输出核心内，必须将体素划分为：

| 分区 | 定义 | 学习含义 |
|---|---|---|
| `Q11` | `M_GR & M_DPR` | 同格点观测域校准 |
| `Q01` | `~M_GR & M_DPR` | 稀疏填充，阶段二的核心难点 |
| `Q10` | `M_GR & ~M_DPR` | 抑制 GR-only 回波、学习错位 |
| `Q00` | `~M_GR & ~M_DPR` | 大量背景，防止虚警，但不能主导损失 |

`Q01` 还必须再按下列方式分组评价：

- 原生 mask 与 `-9999.9` 存储状态；
- `gap_proxy` 与 `outside_proxy`；
- 到最近 GR 物理值的距离；
- 局部 GR 观测密度；
- 高度、反射率强度和降水类型。

对于远离任何 GR 直接观测，甚至连 `dbz_gr_interp` 也覆盖不到的 DPR 回波，模型没有足够的确定性信息去精确还原。这些区域中的输出很可能主要来自高度先验和降水结构先验，而不是被“恢复”的真实细节。这是信息不可辨识问题，不能只靠增大模型解决。

## 7. 第一版 Stage 2 数据契约

### 7.1 Patch 几何

继续沿用阶段一已验证的方案：

```text
每个轨道原始形状：          (nscan, 49, 60)
32条非重叠输出核心：       (32, 49, 60)
左右各16条scan上下文：       (64, 49, 60)
nray从49补齐到64：              (64, 64, 60)
batch + channel first：            (B, C, 64, 64, 60)
```

高度 60 层不做 padding，也不在模型内下采样。损失只作用于非重叠 32-scan 核心和原始 49 条横轨波束，不作用于 halo、轨道边界伪造上下文或 nray padding。

### 7.2 输入通道

先完成三态审计。如果证明原生 mask 在全数据集中具有稳定且部署时可重现的含义，推荐的第一版输入为：

| 通道 | 形状 | 数值处理 |
|---|---|---|
| `gr_dbz_standardized` | `(1,64,64,60)` | 只对 `gr_value_mask=True` 的物理 dBZ 逐高度标准化；其余位置填 0 |
| `gr_value_mask` | `(1,64,64,60)` | 有限物理 dBZ 为 1，其余为 0 |
| `gr_native_available` | `(1,64,64,60)` | 原生 NetCDF 未 mask 为 1，原生 mask/NaN 为 0 |
| `height_scaled` | `(1,64,64,60)` | 0.125–14.875 km 线性缩放到 `[-1,1]` |
| **合计** | **`(4,64,64,60)`** | batch 后为 **`(B,4,64,64,60)`** |

`gr_sentinel_mask` 可由 `gr_native_available & ~gr_value_mask` 推出，无需再增加一个完全冗余通道。

如果审计后发现原生 mask 编码在不同文件间不稳定，则回退为：

```text
GR标准化dBZ + gr_value_mask + height_scaled
→ (B,3,64,64,60)
```

并把 `dbz_gr_interp` 的有效 mask、最近直接观测距离和局部观测密度作为后续受控消融，而不伪装成真实缺测类别。

### 7.3 标准化和填充

GR 的均值和标准差只使用训练集中 `gr_value_mask=True` 的点按高度拟合：

\[
Z'_{GR}(z)=\frac{Z_{GR}(z)-\mu_{GR}(z)}{\sigma_{GR}(z)}.
\]

处理规则是：

- 不要把 `-9999.9` 参与均值/标准差拟合；
- 第一版保留所有物理 dBZ，包括负 dBZ 和 `<9 dBZ` 弱回波；
- 师兄方法中的 `<9 dBZ → 0` 作为后续阈值消融，不作为新基线的默认处理；
- 缺测和 padding 位置的反射率通道在标准化空间填 `0`，它表示该高度训练集平均值占位，不是物理 `0 dBZ`；
- 缺测和 padding 位置的 mask 通道填 0；
- 高度通道仍保留每层坐标，水平 padding 是否有效由 `core_mask` 和输入 mask 决定。

DPR 反射率标签也只用训练集有效 DPR 点逐高度标准化。如果阶段二沿用相同的 train split，应校验后直接复用阶段一的 DPR 标准化统计，这样阶段二输出可以直接接入封板的阶段一模型。

### 7.4 标签和 Dataset 返回值

第一版 Dataset 应返回：

| 字段 | 形状 | 含义 |
|---|---|---|
| `inputs` | `(C,64,64,60)` | 前述 GR-only 输入 |
| `target_dbz` | `(1,64,64,60)` | 逐高度标准化 DPR dBZ，mask 外填 0 |
| `target_valid` | `(1,64,64,60)` | DPR 反射率支持域 0/1 目标 |
| `occupancy_domain_mask` | `(1,64,64,60)` | 非 padding 核心且 `pre_dpr` 原生有效的支持域分类区域 |
| `regression_mask` | `(1,64,64,60)` | `core & target_valid` |
| `overlap_mask` | `(1,64,64,60)` | `core & Q11` |
| `dpr_only_mask` | `(1,64,64,60)` | `core & Q01` |
| `gr_only_mask` | `(1,64,64,60)` | `core & Q10` |
| `below_cfb_target_mask` | `(1,64,64,60)` | CFB 以下 DPR 目标，只用于质量分组/后续消融 |
| `core_mask` | `(1,64,64,60)` | 非重叠输出核心的真实几何范围 |

`pre_dpr`、`cfb`、`typePrecip`、DPR mask 和 DPR dBZ 只能用于构建训练标签、质量分组或评价，不能进入 Stage 2 模型输入。

CFB 以下的 DPR 反射率先保留，因为阶段一模型实际会把这些反射率当作上下文，而且这些点占 DPR 目标的 19.89%。但必须单独报告 CFB 上下结果；如果证明 CFB 以下质量显著差，再进行弱权重消融，不在首版主动删除。

## 8. 第一版模型：保持高度的双头 3D U-Net

继续使用阶段一已经验证的高度保持编解码骨干：

```text
inputs:                    (B,C,64,64,60)
共享3D U-Net解码特征:     (B,16,64,64,60)

occupancy_head:
    1×1×1 Conv3d
    → dpr_valid_logits:      (B,1,64,64,60)

reflectivity_head:
    1×1×1 Conv3d
    → dpr_dbz_prediction:    (B,1,64,64,60)
```

编码和解码只对 `nscan/nray` 使用 `(2,2,1)` 下/上采样，高度始终保持 60 层。

首版不加 attention、GAN、扩散模型、可变形卷积或 STN。普通 3D U-Net 的局部三维感受野本身就可以学习一定范围的位移和补全。只有数据审计证明存在稳定、显著且普通 U-Net 无法处理的偏移时，才有必要增加显式配准模块。

训练时不能用硬阈值后的 `occupancy_head` 输出去门控反射率损失，否则早期的支持域误判会切断反射率头的梯度。反射率头应始终用真实 DPR mask 接受监督；硬阈值只在推理组合输出时使用。

## 9. 损失函数

### 9.1 DPR 支持域损失

如果对全体素直接使用普通 BCE，大量背景会将模型推向“全部预测为无回波”。第一版使用正负类分别求均值的平衡 BCE：

\[
L_{occ}=
\frac{1}{2}\operatorname{mean}_{M_{DPR}=1}BCE
+
\frac{1}{2}\operatorname{mean}_{M_{DPR}=0}BCE.
\]

计算范围是 `occupancy_domain_mask`，其外不能自动当成负样本。这比直接使用约 26:1 的极端 `pos_weight` 更容易控制过度填充和虚警。

若第一版输出的回波区域过碎，再作为单因素消融增加小权重 Soft-Dice，而不从首轮就混入多种分类损失。

### 9.2 DPR 反射率损失

仅在真实 DPR 有效点上使用 masked Smooth-L1，但将同格点校准和稀疏填充分别求均值：

\[
L_Z=
\frac{1}{2}L_{Q11}^{Huber}
+
\frac{1}{2}L_{Q01}^{Huber}.
\]

这样可避免数量更多的 `Q01` 完全淹没观测域校准任务。`Q01` 内部的不同缺测代理类别第一版先分组报告，不在未知物理含义时人为设置监督权重。

第一版总损失为：

\[
L=\lambda_{occ}L_{occ}+\lambda_ZL_Z.
\]

因为两项内部都已做平衡归约，可从 `lambda_occ=1, lambda_Z=1` 开始，但必须日志记录未加权的两项数值和梯度尺度，再决定是否调整。

在 DDP 下，各区域的损失实现应保留分子和有效计数，对空区域安全返回与计算图相连的 0，并正确处理多 rank 间有效数不同的归约，避免某个 GPU 恰好没有 `Q11` 时改变总权重。

### 9.3 后续只做受控增量

根据诊断结果再逐一测试：

- 支持域过碎：小权重 Soft-Dice 或多尺度支持域损失；
- 强回波系统性偏弱：按 DPR dBZ 训练集频率做有上限的强度权重；
- 小位移导致逐点误差很大：小权重 FSS/多尺度结构项；
- 基线稳定后再考虑水平/垂直梯度项。

不建议首先加 TV loss，因为它可能进一步平滑强对流核心；也不建议首先使用 GAN/全局直方图匹配，因为它们可能产生“整体分布像 DPR，但位置是幻觉”的回波。

## 10. Patch 索引与采样

阶段二索引至少应记录：

```text
gr_value_count
dpr_count
overlap_count
dpr_only_count
gr_only_count
interp_supported_dpr_count
strong_dpr_count
```

训练不能只选择 GR 和 DPR 同时有效的格点或 Patch，否则模型永远学不会填充。也不能只保留 DPR-positive Patch，否则模型无法学会降低 FAR。

建议按 Patch 分层采样：

- 含 `Q11` 的校准 Patch；
- 含较多 `Q01` 的填充 Patch；
- 含 `Q10` 的抑制/错位 Patch；
- 含高 dBZ 核心的长尾 Patch；
- 保留一定比例纯背景 Patch；
- 对“输入上下文内完全无 GR 值，但 DPR 有回波”的 Patch 保留评价和少量训练，但单独标记为无信息上限对照，不让它们主导训练。

验证和测试必须 `positive_only=False`，遍历所有 Patch，并使用非重叠 core 重建完整轨道。

## 11. 空间偏移如何处理

目前 GR 和 DPR 已在形式上映射到同一 `(nscan,49,60)` 网格，但这不意味强回波核心在物理上必然完全重合。卫星过境时间差、风暴移动、波束体积和栅格化都可能产生偏移。

第一步应通过审计量化：

- 同格点指标；
- 在水平 `1/2/3` 格邻域中的最佳匹配指标；
- 按高度计算 GR/DPR mask 和强回波的二维互相关最佳位移；
- 偏移随高度、轨道、降水类型和事件的分布。

然后再区分：

1. **接近固定、各轨道一致的偏移**：先尝试数据层简单对齐；
2. **随高度稳定变化的偏移**：普通 3D U-Net 可能通过三维上下文学习；
3. **随事件随机变化的偏移**：如果没有时间/风场信息，确定性模型无法精确解决；
4. **普通 U-Net 训练集能拟合、验证集仍系统性模糊**：才考虑小权重多尺度损失、可变形卷积或显式位移分支。

邻域放宽指标只是诊断，正式选模仍必须考虑精确 DPR 网格上的输出，不能通过“在旁边一格也算对”掩盖系统性配准问题。

## 12. 基线、模型与消融顺序

### 12.1 S2-0：数据和可恢复性审计

完成全数据集三态编码、四分区、逐高度稀疏度、最近观测距离、局部密度、插值可达性和空间偏移审计。

### 12.2 B0：无 GR 数值先验基线

只用高度，或只用 `GR mask + height`，预测 DPR 回波概率和条件高度均值。

它用来判断后续模型是否真正使用了 GR 反射率，还是只学会高度先验和 mask 形态。

### 12.3 B1：同格点观测域校准

仅在 `Q11` 上比较：

1. GR 原值直接作为 DPR 预测；
2. 全局线性校准；
3. 分高度线性校准；
4. 分高度单调/分位数映射；
5. 如果有必要，再使用小型 MLP。

如果校准后连 `Q11` 都不能稳定优于原始 GR，应先排查配准、时间差和变量口径，不宜用更大网络掩盖数据问题。

### 12.4 B2：师兄 GR 插值基线

比较：

```text
dbz_gr_interp
dbz_gr_interp + 分高度线性/分位数校准
```

`dbz_gr_interp` 是派生产品，不是新观测。它是阶段二必须超过的传统基线，但不是稀疏模型的默认输入。

### 12.5 S2-E0：双头稀疏 3D U-Net

核心输入是稀疏 GR 反射率、其原始状态 mask 和高度；输出 DPR 支持域与 DPR dBZ。

### 12.6 受控消融

只在前一个实验给出明确诊断依据后测试：

| 实验 | 唯一改动 | 要回答的问题 | 当前状态 |
|---|---|---|---|
| `S2-E1-D` | 在3V上加最近 GR 观测距离 | 显式稀疏几何是否有用 | 已测，成为当前平衡主模型 |
| `S2-E1-rho` | 在3V上加5×5局部观测密度 | 局部密度是否比距离更有效 | 已测，未超过距离模型 |
| `S2-E2` | 加 `dbz_gr_interp + interp_mask` | 师兄插值是否提供有用的覆盖代理，以及是否导致平滑 | 已测，保留Pareto候选但不取代D |
| `S2-E3-W1.5/W1.25` | 在D上增加有上限的 dBZ 强度权重 | 是否恢复强回波尾部 | 已测，W1.25作为Stage 3较好预训练起点 |
| `S2-E4` | 增加小权重 Dice/FSS/多尺度结构项 | 是否改善支持域连续性和小偏移 |
| `S2-E5` | gated/partial convolution | 普通卷积是否确实无法处理结构性空洞 |
| `S2-E6` | 显式位移/可变形模块 | 仅当审计证明存在稳定系统偏移时测试 |

一次只改变一个因素。首轮不使用大型 attention、GAN 或扩散模型。

### 12.7 E0/3N/3V/E1/E2的实际执行结论

完整验证集结果表明，3N删除`gr_value_mask`后明显退化；3V保留`gr_value_mask`、删除`gr_native_available`后，支持域CSI/F1/FSS超过E0，且dBZ MAE没有实质退化。因此后续以3V为母实验，不再沿用3N分支。

以3V为母实验后，距离模型`S2-3V-D`的输入为：

```text
S2-3V-D
  = dbz_gr_sparse_standardized
  + gr_value_mask
  + gr_nearest_distance_scaled
  + height_scaled
  -> (B,4,64,64,60)
```

`gr_nearest_distance_scaled`在每个高度层内单独计算水平Chebyshev最近观测距离，截断到8格并缩放到`[0,1]`。它只使用GR物理值mask，不使用DPR。在完整验证集上，D的联合最优检查点达到`CSI=0.5771`、`Recall=0.6893`、`FSS-r2=0.8582`、`MAE=3.8408 dBZ`、`RMSE=5.3521 dBZ`和`r=0.6966`，因而作为当前平衡主模型。

局部密度与Fill-45采样实验都没有超过D。E2在3V上同时增加`dbz_gr_interp_standardized`和`gr_interp_value_mask`，输入为`(B,5,64,64,60)`。三个代表性检查点的完整val结果为：

| 检查点 | CSI | Recall | FAR | FSS-r2 | MAE (dBZ) | RMSE (dBZ) | Bias (dBZ) | Pearson r | CCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D `best_joint` | **0.5771** | **0.6893** | 0.2200 | 0.8582 | 3.8408 | **5.3521** | **-0.2594** | 0.6966 | 0.6749 |
| E2 `best_dbz` | 0.5727 | 0.6592 | **0.1865** | 0.8552 | **3.7971** | 5.3659 | -0.5608 | **0.6982** | **0.6756** |
| E2 `best_support` | 0.5719 | 0.6733 | 0.2084 | **0.8603** | 3.9093 | 5.5954 | -1.3867 | 0.6830 | 0.6380 |

E2 `best_dbz`的低MAE和低FAR是真实的Pareto改善，但Recall下降约0.030，CSI和FSS也未超过D。分区结果显示，它在插值可达的`gap_proxy`中将MAE从3.568降至3.399 dBZ，但在`outside_proxy`中将MAE从5.021提高至5.116 dBZ，且Recall约从0.239降至0.161。`best_support`也没有挽回这一问题。因此E2实验到此结束，不继续组合“距离+插值”六通道；E2 `best_dbz`仅作为低MAE/低虚警候选保留。

### 12.8 已完成的强回波权重实验：`S2-E3-W1.5/W1.25`

E3以D为父实验，从相同随机初始化开始训练。split、seed、Sampler、四通道输入及顺序、网络、支持域损失、优化器和训练策略全部不变，唯一变量是`M_dbz=1`体素的反射率回归权重。权重依据标准化前的物理DPR dBZ构造：

\[
w(Z)=
\begin{cases}
1.00, & Z<25\\
1.25, & 25\le Z<35\\
1.50, & Z\ge35.
\end{cases}
\]

加权dBZ损失使用有效权重总和归约：

\[
L_{dbz}^{E3}=
\frac{\sum_i M_{dbz,i}w_i\,
\operatorname{SmoothL1}(\widehat Z'_{DPR,i},Z'_{DPR,i};\beta=0.2)}
{\sum_i M_{dbz,i}w_i}.
\]

`M_support`、`y_support`、`M_dbz`和标签标准化都不改。权重是由训练标签派生的loss元数据，不会拼到模型输入，推理时也不需要DPR。

W1.5后补充的W1.25降低了尾部权重幅度。其`best_dbz.pt`在完整38条val轨道上达到：

| 检查点 | CSI | Recall | FAR | FSS-r2 | MAE (dBZ) | RMSE (dBZ) | Bias (dBZ) | Pearson r | CCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 0.5771 | **0.6893** | 0.2200 | **0.8582** | 3.8408 | 5.3521 | -0.2594 | 0.6966 | 0.6749 |
| W1.25 `best_dbz` | 0.5763 | 0.6840 | **0.2146** | 0.8569 | **3.8067** | **5.3085** | **-0.1489** | **0.7020** | **0.6816** |

W1.25形成小幅Pareto改善，但在`>=35 dBZ`上仍有`MAE=8.08 dBZ`、`Bias=-7.37 dBZ`和`r=0.171`的明显强尾部压缩。因此它作为Stage 3当前最好的预训练接口起点，但不将小幅反射率改善解读为已经封版的最终链路，也不继续搜索邻近W权重。

## 13. 评价口径

### 13.1 DPR 支持域

主指标：

- Precision；
- Recall/POD；
- FAR；
- CSI；
- F1；
- PR-AUC；
- Brier score 和概率校准；
- 预测/真实支持域体素数比。

Accuracy 和 ROC-AUC 不作为主指标，因为负类占比极高。支持域概率阈值只允许在验证集选择，不能使用测试集调参。

还要在 10/15/20/30 dBZ 等阈值上统计邻域 CSI/FSS，区分“位置略有偏移”和“完全没有恢复回波”。

### 13.2 反射率数值

至少报告 MAE、RMSE、Bias、Pearson r 和 CCC，并分开：

- `Q11`：同格点校准能力；
- `Q01`：稀疏填充能力；
- `Q10`：支持域抑制和虚警；
- oracle mask 下的 dBZ：使用真实 DPR mask，隔离反射率数值误差；
- predicted mask 下的 dBZ：使用预测支持域，反映完整部署效果。

必须按高度、DPR dBZ 强度、最近 GR 距离、局部稀疏度、CFB 上下、原始存储状态代理、层云/对流和轨道进行分组。

### 13.3 空间和垂直结构

- 多尺度 FSS；
- 回波顶高度误差；
- 强回波中心位置和质心偏移；
- CFAD（反射率-高度联合频率分布）；
- 平均垂直廓线及水平/垂直剖面；
- exact 与邻域放宽指标的差距。

### 13.4 冻结阶段一的串联评价

阶段二的中间 dBZ 指标好不代表最终降水一定好。每个正式 Stage 2 模型都必须在同一冻结阶段一模型上比较：

```text
1. 真实DPR dBZ + 真实DPR mask
   → 阶段一理想输入上限

2. Stage2预测dBZ + 真实DPR mask
   → oracle-mask，单独隔离反射率数值误差

3. Stage2预测dBZ + Stage2预测mask
   → 完整可部署串联效果

4. 稀疏GR/插值GR基线
   → 与新模型比较
```

阶段二输出先转换为阶段一的 `DPR standardized + valid mask + height`契约，再输入当前封板模型。支持域硬阈值只在验证集选定。

最终降水不仅报告总体 MAE/RMSE/r，还要继续报告 5–10、10–30、强降水尾部、层云/对流、高度和 `dR/dz` 指标。

### 13.5 全冻结串联实测结论

封版Stage 1为`outputs/ablations/stage1_i_g002_t3d/best.pt`，best epoch 22。在完整38条val轨道和1,195,966个可靠正降水体素上：

| 串联输入 | RMSE (mm/h) | Pearson r | dR/dz r |
|---|---:|---:|---:|
| 真实DPR dBZ + 真实mask | 2.2262 | 0.8637 | 0.7023 |
| 师兄插值GR | 4.0753 | 0.4280 | 0.0456 |
| D预测dBZ + 真实mask | 4.0004 | 0.4419 | 0.0615 |
| D预测dBZ + 预测mask | 4.0755 | 0.4246 | 0.0582 |
| W1.25预测dBZ + 真实mask | 3.9180 | 0.4713 | 0.0688 |
| W1.25预测dBZ + 预测mask | 3.9884 | 0.4538 | 0.0654 |

W1.25部署串联相对师兄插值只有小幅改善，与真实DPR输入上限相差巨大。更关键的是，使用真实mask时已降至`r=0.4713`，因此主要损失不是support硬阈值，而是Stage 2 dBZ值场、强尾部和三维结构与Stage 1训练域不匹配。

该结果触发Stage 3接口适配与直接反演路线，具体顺序见[第三阶段实验路线](./第三阶段串联适配、联合优化与直接反演实验路线.md)。

## 14. 何时应向师兄索取原始 GR 体扫和几何数据

导师建议先用当前数据建模是合理的，因为可以先用实验定量判断问题是否真的受制于数据生成方式。但若出现以下现象，应停止继续堆叠模型复杂度：

- `outside_proxy` 区域包含大量 DPR 回波，而稀疏模型和插值模型都不优于高度先验；
- 误差随到最近 GR 观测的距离快速恶化；
- 提高支持域 Recall 必然伴随 FAR 大幅增加；
- beam-gap 代理区域长期显著差于其他区域，现有变量无法解释；
- relaxed 邻域指标远好于 exact 指标，但偏移方向随轨道/事件随机变化；
- 增大模型后训练集继续改善，完整验证集却不再改善；
- 未见站点/事件的泛化显著差于已见几何。

届时应优先索取：

```text
原始极坐标体扫或未压缩波束数据
雷达站ID和站点经纬度/高度
每个体扫的方位角、仰角、距离和波束宽度
真实扫描覆盖标志和质量控制标志
GR与DPR的实际时间差
数据生产时对NaN和-9999.9的确切定义
```

这样向师兄申请原始数据时，可以给出“哪一类格点、占比多少、损失多少”的实验证据，而不是只依据定性猜测。

## 15. 第一版历史代码实施顺序（已完成）

> 本节只用于追溯第一版代码是如何建立的；当前R0实施范围以0.6节为准。

### 15.1 紧接着实现

第一步先建立原始编码与可恢复性审计，不修改阶段一的默认 mask 语义：

```text
src/precipitation_inversion/data/stage2_masks.py
scripts/analyze_stage2_gr_dpr_alignment.py
tests/test_stage2_masks.py
tests/test_stage2_alignment_audit.py
```

`stage2_masks.py` 应从 NetCDF 原始 masked array 中保留 `native/sentinel/value` 三态，而不在阶段二未验证前改变全项目共用 `masks.py` 的行为。

审计脚本至少输出：

- train/val/test 的三态数量和比例；
- `Q11/Q01/Q10/Q00` 与逐高度统计；
- `gap_proxy/outside_proxy` 中的 DPR 回波比例；
- 最近 GR 距离和局部密度分组；
- exact 与 1/2/3 格邻域匹配；
- 按高度的互相关最佳位移；
- JSON/CSV 统计、PNG 图表和可复现的配置信息。

### 15.2 审计后依次实现

```text
scripts/evaluate_stage2_baselines.py
scripts/fit_stage2_normalization_stats.py
scripts/build_stage2_patch_index.py
src/precipitation_inversion/data/stage2_patch_dataset.py
tests/test_stage2_patch_dataset.py

src/precipitation_inversion/models/stage2_unet3d.py
src/precipitation_inversion/losses/stage2_losses.py
src/precipitation_inversion/metrics/stage2_reflectivity.py
src/precipitation_inversion/training/stage2_engine.py

configs/stage2_sparse_dual_head.yaml
scripts/train_stage2_unet3d.py
scripts/evaluate_stage2_unet3d.py
scripts/visualize_stage2_predictions.py
scripts/evaluate_stage2_stage1_cascade.py
```

每个模块都应有对应单元测试，并继续使用当前按日期/轨道隔离的固定 split。

## 16. 阶段门槛

在进入更复杂模型或阶段三前，一个 Stage 2 候选模型至少应同时满足：

1. `Q11` 上的校准稳定优于直接复制 GR；
2. `Q01` 上的 dBZ 指标优于现有插值+校准基线；
3. 支持域 CSI/F1/FSS 提升时没有伴随无法接受的 FAR 增长；
4. 完整轨道和测试集改善，不只是随机 Patch 改善；
5. 接入同一冻结阶段一模型后，最终降水指标优于师兄插值/ZRH 基线；
6. 逐轨配对 bootstrap 置信区间支持改善，而不是由少数轨道贡献。

可在第一轮基线完成后预注册项目内的实用改进门槛，例如相对当前最强插值基线：

```text
CSI 绝对提升约 0.02
Q01 dBZ RMSE 下降约 3%
串联降水 RMSE 下降约 3% 或 Pearson r 提升约 0.01
```

这些只是本项目用于避免过度解读小波动的候选标准，不是气象学的通用标准。最终仍应结合置信区间、强回波、对流和稀疏区域是否退化来判断。

实际结果中，W1.25部署串联相对师兄插值达到了小幅的r改善，但仅有`r=0.4538`，远低于真实DPR输入的`0.8637`，且dR/dz降至`0.0654`。因此W1.25可用作Stage 3预训练起点，但Stage 2独立硬串联不能作为最终部署方案。

## 17. 第一版历史计划路线（已由R0–R6替代）

> 以下流程图记录Stage 2第一版到Stage 3早期串联的已执行路径，不是当前待执行路线。当前顺序为0.4节的`S2-R0`→`S2-R6`。

```text
S2-0  三态mask/稀疏性/偏移/可达性审计
  ↓
B0    高度与mask先验基线
  ↓
B1    Q11上的GR→DPR分高度校准
  ↓
B2    dbz_gr_interp + 校准传统基线
  ↓
S2-E0/3N/3V 选定3V母实验
  ↓
E1-D/局部密度/Fill-45 选定距离D平衡主模型
  ↓
E2    插值辅助输入已结束，不进入六通道组合
  ↓
E3-W1.5/W1.25 已完成，选W1.25作为Stage 3预训练起点
  ↓
冻结Stage 1做oracle-mask/预测mask完整val串联评价（已完成）
  ↓
确认独立硬串联严重退化，停止独立Stage 2调权
  ↓
移交Stage 3：C0接口审计→C1/C2单边冻结→较好P版→D0直接多头→有条件C3
  ↓
与Stage 3最小实验并行，用outside_proxy定量证据询问原始GR体扫与几何信息
```

这条路线将“两类雷达的数值差异”、“回波支持域差异”、“仰角/波束间空洞”、“真正无覆盖”和“空间错位”尽可能拆开诊断。只有这样，后续模型的改进才能被解释，也才能判断瓶颈究竟来自网络、损失还是原始观测信息不足。
