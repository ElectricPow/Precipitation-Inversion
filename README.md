# Precipitation Inversion

基于地基雷达（Ground Radar, GR）稀疏三维反射率反演稠密三维降水率的研究项目。截至2026-09-02，仓库已完成Stage 1高度保持3D U-Net、物理dR/dz与降水类型辅助任务，以及Stage 2三态缺测审计、双头3D U-Net、多卡训练、整轨评价和多组输入/损失消融。四通道距离+W1.25虽然是第一版Stage 2最强起点，但冻结串联仅达到`RMSE=3.9180 mm/h、r=0.4713`，表明Stage 2值场是主要瓶颈。当前已转入Stage 2-v2任务分解：非部署上限`S2-R1-O-DPRSparseValue`已完成60轮训练并达到冻结串联`r=0.62345`，但outside和强回波恢复仍不足；下一项严格受控实验`S2-R1-P-PartialConv`仅将稀疏值输入stem替换为3D Partial Convolution，现已完成代码、固定验收门槛、单元测试和真实CUDA烟雾测试，等待三卡正式训练。

## 1. 研究任务

数据集将多个地基雷达观测、GPM 卫星 DPR 产品和气象背景场配准到同一个三维卫星条带网格。当前拟研究的主要映射为：

```text
GR 稀疏三维反射率 + 观测掩码 + 可用气象背景场
                         ↓
                 三维降水反演模型
                         ↓
              稠密三维降水率 pre_dpr
```

- 主要输入：地基雷达稀疏反射率 `dbz_gr_sparse`；
- 辅助输入候选：谱宽 `sw_gr_sparse*`、气压 `p`、温度 `t`、比湿 `q`、高度和观测掩码；
- 参考标签：GPM DPR 反演的三维降水率 `pre_dpr`；
- 质量控制信息：`cfb`、`binRealSurface`、`flagPrecip` 等。

`pre_dpr` 是卫星遥感反演产品，并非无误差的物理真值，因此本文称其为参考标签或伪真值。

最终任务拆为三个阶段：

1. `dbz_dpr → pre_dpr`：先学习卫星DPR反射率到卫星降水率的条件强度映射；
2. `dbz_gr_sparse → DPR反射率域`：再学习地基雷达稀疏反射率到DPR反射率域的分布映射；
3. 比较两阶段串联适配、局部联合优化和直接GR→rain共享多头模型，再按受控消融结果决定是否加入气象背景场。

阶段一使用卫星变量`dbz_dpr`学习“反射率到降水率”；当前Stage 2则已改为输入最终部署可得的`dbz_gr_sparse`及GR自身派生几何信息，同时预测DPR支持域和DPR dBZ。这样继续保持“跨雷达域转换”与“反射率到降水率”两个问题的可分析性。

师兄此前采用的主要路线是：

```text
稀疏 GR 反射率 → 空间插值 → 稠密反射率 → 降水反演
```

这一流程便于使用常规稠密卷积网络，但插值容易混合邻近观测、削弱局地极值；再叠加降水标签严重的零膨胀和长尾分布，模型容易给出偏平滑的预测。本项目后续重点是探索能够保留观测掩码、空间稀疏性和强降水尾部的反演方法。

## 2. 数据集

### 2.1 存储位置

数据集位于实验室服务器：

```text
/storage/GR_DPR_3D/GRToDPRRes_V07_Pct_V1.2.1_sw_260412
```

当前共 254 个 NetCDF 文件，总量约 23 GB。数据量较大且不属于代码仓库，因此不会上传到 GitHub。

本文使用以下文件作为主要分析样本：

```text
2A.GPM.DPR.V9-20211125.20170501-S071945-E085219.018026.V07A.nc
```

### 2.2 网格结构

该样本的主要三维变量形状为：

```text
(nscan, nray, z) = (596, 49, 60)
```

| 维度 | 含义 | 样本长度 |
|---|---|---:|
| `nscan` | 沿卫星飞行方向的扫描线 | 596 |
| `nray` | 每条扫描线的横轨波束 | 49 |
| `z` | 垂直高度层，间隔 0.25 km | 60 |

高度从 0.125 km 延伸至 14.875 km。`596×49` 不是规则经纬度矩形，而是沿卫星轨道弯曲的条带网格，每根垂直廓线的位置由 `lat` 和 `lon` 给出。

### 2.3 主要变量

| 类别 | 变量 | 形状 | 单位 | 研究角色 |
|---|---|---:|---|---|
| 地基雷达 | `dbz_gr_sparse` | `(nscan,nray,z)` | dBZ | 稀疏反射率主输入 |
| 地基雷达 | `dbz_gr_sparse_min/max` | `(nscan,nray,z)` | dBZ | 格点内反射率范围 |
| 插值产品 | `dbz_gr_interp` | `(nscan,nray,z)` | dBZ | 传统插值基线 |
| 地基雷达 | `sw_gr_sparse*` | `(nscan,nray,z)` | m/s | 谱宽辅助信息候选 |
| 星载 DPR | `dbz_dpr` | `(nscan,nray,z)` | dBZ | 跨传感器对比/辅助监督候选 |
| 星载 DPR | `pre_dpr` | `(nscan,nray,z)` | mm/h | 核心参考标签 |
| 星载 DPR | `nsrr_dpr`、`srr_dpr` | `(nscan,nray)` | mm/h | 近地面和地表降水率 |
| 质量控制 | `cfb`、`binRealSurface` | `(nscan,nray)` | — | 杂波底与真实地表位置 |
| 分类/标志 | `typePrecip`、`flagPrecip` | `(nscan,nray)` | — | 分类型评价和质量筛选 |
| 气象背景场 | `p`、`t`、`q` | `(nscan,nray,z)` | hPa/K/kg·kg⁻¹ | 环境辅助输入候选 |

更完整的物理解释见[数据集说明](./数据集说明.md)，所有变量、属性、统计量和代表值见[单样本变量与数值分析](./NC样本变量与数值分析.md)。

## 3. 阶段研究进度（截至2026-09-01）

### 3.1 完成数据定位和资料梳理

- 确认数据已迁移至当前服务器的 `/storage/GR_DPR_3D`；
- 核对数据集规模为 254 个 `.nc` 文件、约 23 GB；
- 阅读导师项目介绍 PPT、师兄变量说明文档和两份已有代码；
- 明确数据是将 GR、DPR 与气象背景场配准后的公共三维数据，而非单一雷达原始扫描文件。

### 3.2 完成变量与物理意义学习

- 理解 `nscan/nray/z` 三个维度及卫星条带网格；
- 区分地基雷达 GR 数据、星载 DPR 数据、气象背景场和质量控制变量；
- 明确缺测、有效无降水和弱回波不能使用同一种数值语义处理；
- 明确 GR 与 DPR 即使索引已经配准，仍会因频率、扫描几何、波束体积、衰减和时间差存在系统差异；
- 明确 `pre_dpr` 只应作为参考标签，`cfb` 以下等不可信区域需在训练和评价中屏蔽；
- 明确部署时不可获得的 DPR 变量不能直接作为最终模型输入，避免信息泄漏。

### 3.3 完成单个真实样本的全变量分析

已读取示例 `.nc` 文件中的全部 23 个变量，记录其维度、形状、数据类型、属性、有效比例、分位数和少量代表值。单样本得到的阶段性结果如下。

#### GR 稀疏性与插值覆盖

- `dbz_gr_sparse` 有效格点：51,601，占全部三维格点的 2.9449%；
- `dbz_gr_interp` 有效格点：112,087，占 6.3968%；
- 插值新增 60,486 个有值格点；
- 在原本已有 GR 观测的位置，插值值基本保持原值，新增区域则来自邻近观测推断，不是独立观测。

#### GR 与 DPR 的同格点差异

在双方均有效的格点上：

| 对比 | 共同有效数 | 偏差 GR−DPR | MAE | RMSE | 相关系数 |
|---|---:|---:|---:|---:|---:|
| 稀疏 GR vs DPR（所有高度） | 37,221 | -1.4811 dBZ | 3.8428 | 5.5642 | 0.7760 |
| 插值 GR vs DPR（所有高度） | 75,976 | -1.8818 dBZ | 4.4159 | 6.3877 | 0.7156 |
| 稀疏 GR vs DPR（1.875 km） | 1,810 | -0.4749 dBZ | 4.1805 | 5.9794 | 0.7248 |

这些结果只描述一个轨道样本，不能直接作为全数据集结论，但说明“完成网格配准”不等于“两类雷达数值完全一致”。

#### 标签零膨胀与强降水长尾

- `pre_dpr` 有效标签占 99.5014%；
- 有效标签中 94.5361% 为 0；
- 正降水只占有效标签的 5.4639%；
- `>10 mm/h` 的格点仅占有效标签的 0.1091%；
- `>20 mm/h` 的格点仅占 0.0168%；
- 最大降水率为 62 mm/h。

该分布解释了普通 MSE/MAE 容易被无降水和弱降水主导、强降水预测容易被平滑的问题。后续实验需要同时报告分阈值和分降水类型指标，而不能只报告总体误差。

#### 地表杂波和降水类型

- `cfb` 高度中位数为 1.625 km；文件在 `cfb` 以下仍可能保存数值，但这些数值不应默认作为可靠标签；
- 对流样本的正降水上尾显著强于层云样本；
- 总体指标可能掩盖对流性强降水上的性能不足，因此应按 `typePrecip` 分组评价。

### 3.4 完成师兄代码解读和服务器适配

#### `zrh_nc_to_rain.py`

该脚本使用60个高度层各自的参数执行：

\[
R=\exp(\mathrm{dBZ}\cdot w_z+b_z)
\]

本周完成：

- 梳理命令行参数、权重加载、NetCDF复制、分块转换和临时文件保护流程；
- 将默认数据路径、权重路径和输出路径适配到当前服务器；
- 使用 NumPy读取本项目仅含 `weight/bias` 的小型权重文件，避免安装不必要的 PyTorch 运行时；
- 创建项目虚拟环境和固定依赖；
- 使用真实样本完成单文件端到端临时转换验证。

验证结果：由 `dbz_gr_sparse` 生成的变量形状为 `(596,49,60)`，单位为 `mm/h`，满足 `[0,70) dBZ` 条件的有效转换格点为 49,742。

#### `plot_2km_zrh_4files.py`

该脚本对比六个场的约2 km水平分布与同一A–B垂直剖面：

```text
dbz_gr_sparse             | dbz_gr_interp
rain_rate_zrh_gr_sparse   | rain_rate_zrh_gr_interp
dbz_dpr                   | pre_dpr
```

本周完成：

- 梳理数据读取、ZRH计算、剖面选择、大圆距离计算和3×2面板绘图流程；
- 修正旧服务器默认路径并移除 PyTorch 依赖；
- 补充 Matplotlib/Pillow 环境；
- 使用真实样本生成并校验临时 PNG。

测试选择最接近2 km的 `z[7]=1.875 km` 层和第351扫描行，输出图像尺寸为 `2256×2441`，PNG 完整性检查通过。

### 3.5 本周阶段性认识

1. 任务本质不是普通稠密图像回归，而是带观测掩码、标签掩码和多传感器偏差的三维稀疏到稠密反演；
2. 插值可以提高覆盖率，但不能视作新增观测，训练时仍需保留原始观测掩码；
3. 模型效果不能只由网络结构决定，样本不平衡、损失权重、杂波屏蔽和数据划分同样关键；
4. 强降水占比极低，应关注条件误差、阈值命中率和对流样本表现；
5. 在研究新模型前，需要先建立统一、可复现的传统 Z–R 和插值基线。

### 3.6 完成阶段一高度保持3D U-Net首轮训练

阶段一把每条长度不同的轨道切成“非重叠32扫描线输出核心 + 左右各16扫描线上下文”的窗口。原始窗口`(64,49,60)`只在横轨方向补到64，形成模型批次`(B,C,64,64,60)`；高度60层始终不裁剪、不补齐，也不在U-Net中下采样。模型只沿扫描和横轨方向用`(2,2,1)`缩放，最后输出`(B,1,64,64,60)`，评价时只保留中心核心和真实49条横轨波束。

首轮三组实验除下表所列输入区别外，使用相同的数据划分、随机种子、网络、优化器和训练策略：

| 实验 | 输入通道及形状 | CFB处理 | 主损失 |
|---|---|---|---|
| E0 baseline | `dbz_dpr标准化值 + 有效性mask + 高度`，`(B,3,64,64,60)` | 保留CFB以下有效DPR反射率 | `pre_positive_qc`内的`log1p` masked Smooth L1，`beta=0.2` |
| E1 mask CFB | 同E0，`(B,3,64,64,60)` | CFB以下输入置为`dbz=0, valid=0` | 与E0完全相同 |
| E2 CFB distance | E0三通道 + 裁剪后的有符号CFB距离，`(B,4,64,64,60)` | 距离定义为`clip((z-z_cfb)/2 km,-1,1)` | 与E0完全相同 |

三组实验的主监督和模型选择口径没有变化：仅在同时满足DPR反射率有效、`pre_dpr>0`、且位于CFB及其上方的`pre_positive_qc`体素上训练和验证。标签先做`log1p(R)`，损失只在mask内归约；MAE、RMSE、Bias、R²和Pearson相关系数再变回`mm/h`空间计算。因此首轮是“可靠正降水条件下的强度回归”，尚不是含无降水识别的全空间降水检测任务。

### 3.7 E0/E1/E2完整验证结果与分析

以下结果来自各实验`best.pt`在同一完整验证集上的299个Patch、1,195,966个可靠正降水体素；没有用6条固定测试轨道选择模型。

| 实验 | best epoch | MAE (mm/h) | RMSE (mm/h) | Bias (mm/h) | R² | Pearson r |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 31 | 0.489750875 | **2.347860068** | -0.094303728 | **0.717183289** | **0.848496524** |
| E1 | 42 | 0.504331201 | 2.478820312 | -0.103190132 | 0.684753161 | 0.831565221 |
| E2 | 34 | 0.489888098 | 2.405469622 | -0.097537388 | 0.703134040 | 0.842449410 |

完整验证集总体上E1和E2都弱于E0：E1 RMSE比E0高约5.58%，E2高约2.45%。分层结果进一步说明误差主要来自低层和极端长尾：

- 0–2 km RMSE分别为E0 `4.9414`、E1 `5.3481`、E2 `5.1676 mm/h`；屏蔽CFB以下输入没有改善低层边界，反而进一步损伤上下文。
- `>=30 mm/h`仅有4,624个体素，约占验证支持的0.39%，却贡献E0总平方误差的68.16%；其E0 RMSE为`31.173 mm/h`、Bias为`-16.429 mm/h`，强降水低估是下一轮损失设计的首要目标。
- E2在部分高层以及`5–30 mm/h`中强雨分箱优于E0。6条固定测试轨道上E2也优于E0（RMSE `1.57768`对`1.63468 mm/h`，r `0.91748`对`0.90823`），但样本太少且属于测试集，不能据此调参或覆盖完整验证结论。

E1从训练输入额外删除了1,036,674个有效DPR反射率体素，约占15.90%。被删除值统一编码为`dbz=0, valid=0`，使“CFB以下低置信回波”和“传感器真正缺测”在模型输入中无法区分。因此E1实际检验的是删除低层上下文，而没有检验低置信数据能否作为弱证据使用。

E2也不能简单解释为“第四通道稀释了前三通道”。它只给网络增加160个参数，约占总参数的0.008%，不是容量上的显著改变；更可能的问题是该通道在空间中稠密、绝对尺度较大、大量值裁剪饱和，并与已有高度通道高度共线，容易在优化早期形成捷径。E2仍保留了局部收益，因此暂不否定CFB几何信息，但下一轮不直接以该四通道方案为主线。

仓库保留了早期拟定的E3/E4配置（从E1或E2继续叠加逆频率高度权重、较强强度权重和/或弱CFB监督），但它们没有正式训练。由于其父实验E1/E2总体已退化，而且E3/E4一次改变多个因素，当前不再优先启动，改用下述从E0逐项增加因素的E0-N系列。

### 3.8 第二轮E0-N/I/W实验

为避免同时改变过多因素，第二轮全部从E0三通道结构出发，并使用同一网络、划分、种子和训练配置：

| 实验 | 相对E0的唯一或组合改动 | 输入形状 | 训练损失变化 | 状态 |
|---|---|---:|---|---|
| E0-N | 只更换`dbz_dpr`输入归一化统计 | `(B,3,64,64,60)` | 无 | 已完成，best epoch 38 |
| E0-N-I | E0-N + 按真实降水强度加权 | `(B,3,64,64,60)` | `<1/1–5/5–10/10–30/>=30 mm/h`权重为`1/1/1.5/2/3` | 已完成，best epoch 32 |
| E0-N-W | E0-N + CFB下方原生正降水弱监督 | `(B,3,64,64,60)` | CFB下第1/2层权重为`0.1/0.05` | 已完成，best epoch 35 |
| E0-N-IW | E0-N同时加入I和W | `(B,3,64,64,60)` | 强度权重与弱CFB权重相乘 | 条件待测 |

旧文件`stage1_positive_qc.json`在拟合输入统计时使用了标签QC条件，导致0.125和0.375 km两层无样本，标准化器也会把这两层的原生有效DPR输入当作无效值。新文件`stage1_dbz_valid.json`把输入与标签口径完全解耦：

- 输入归一化只使用175个训练文件中`dbz_dpr`自身全部有限、非fill值，共6,822,356个，60个高度层都有统计；最低两层分别有131,972和172,153个值；
- 标签、可靠主损失和主评价仍使用`pre_positive_qc`，共5,481,557个训练体素；
- 标签QC不再反向决定输入统计，但不会扩大可靠标签范围；E0-N的输入仍是三通道`(B,3,64,64,60)`。

I和W都通过`loss_weights`作用于同一个masked Smooth L1加权平均，归一化分母是当前local batch内被选体素的权重和。正式配置每卡batch为1，所以语义仍是“每Patch先归一、DDP再等权平均各rank梯度”，不是跨GPU使用一个全局体素分母。W只选择CFB下方第1、2个高度层中同时具有原生正`pre_dpr`和原生有效DPR回波的体素；它们不会并入`reliable_loss_mask`。W/IW在整轨推理中的近CFB `output_mask` 只由核心几何、原生DPR有效性、有效CFB和配置层数决定，不查看真实降水标签。完整验证、`best.pt`选择及各实验主表仍只看可靠mask，CFB以下原生正降水另列为诊断，避免通过改变评价口径制造“提升”。

注意：W实验日志的`valid_voxels`包含弱监督，而`metrics.rain.all.count`只包含可靠体素；加权后的loss数值也不能与E0的未加权loss直接比大小，应比较同一可靠mask上的MAE/RMSE/Bias/R²/r。

验证流程也已修正：DDP训练仍使用各rank等步数的`even_batches=True`，而验证使用互不重复且允许不等长的分片，并通过已同步模型的裸模块前向，因此299个验证Patch既不丢失也不补重复。评估器新增CFB以下原生正降水独立诊断、逐文件/轨道宏平均，以及以完整轨道为抽样单位、固定种子的bootstrap置信区间；这些结果均与主可靠指标分开保存。

四个可直接比较的模型在同一完整验证集、1,195,966个可靠正降水体素上的结果如下。N仅改变输入归一化后RMSE轻微退化；I明显改善RMSE、R²和相关性；W与E0总体接近，不能证明CFB下弱监督带来稳定收益。

| 实验 | MAE (mm/h) | RMSE (mm/h) | Bias (mm/h) | R² | Pearson r |
|---|---:|---:|---:|---:|---:|
| E0 | **0.489751** | 2.347860 | -0.094304 | 0.717183 | 0.848497 |
| E0-N | 0.489903 | 2.363538 | -0.107568 | 0.713394 | 0.851559 |
| E0-N-I | 0.495106 | **2.129368** | **+0.000455** | **0.767372** | **0.876012** |
| E0-N-W | 0.497695 | 2.349883 | -0.053538 | 0.716696 | 0.848683 |

### 3.9 物理垂直梯度dR/dz评价

统一实现先把模型的`log1p`输出还原为非负物理降水率`R (mm/h)`，再在原生高度坐标上计算相邻层向上差分：

\[
\frac{dR}{dz}\bigg|_{k+1/2}=\frac{R_{k+1}-R_k}{z_{k+1}-z_k}
\]

输入60层降水率得到59层梯度，单位为`mm h^-1 km^-1`。只有相邻两个端点都属于可靠主评价mask时才计入，因而不会跨缺测、padding或W的CFB弱监督区域。当前口径是“连续可靠正降水体素内部的条件梯度”，不包含有效零雨以及雨顶/雨底发生边界，也不能直接与PPT中未明确支持域的“所有样本dR/dz”数值比较。

训练后自动分析会保存总体MAE/RMSE/Bias/R²/Pearson r、预测/标签平均绝对梯度幅值比、符号一致率，以及逐高度、相对CFB距离、目标强度、降水类型和逐轨结果。跨实验比较还会核对完整pair mask的SHA-256指纹，并以整条轨道为单位执行配对bootstrap。

四个模型已经在同一完整验证集上重新推理。38条轨道、437个Patch共产生1,113,507个可靠相邻正降水梯度对，四者pair mask的SHA-256指纹完全一致：

| 实验 | dR/dz MAE | dR/dz RMSE | Bias | Pearson r | 绝对梯度幅值比 | 符号一致率 |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.6300 | 2.5613 | +0.0605 | 0.5873 | 0.8379 | 0.9046 |
| E0-N | **0.6122** | 2.3924 | +0.0517 | 0.6511 | 0.8211 | **0.9075** |
| E0-N-I | 0.6355 | 2.3881 | **+0.0229** | 0.6540 | **0.8837** | 0.9030 |
| E0-N-W | 0.6167 | **2.3626** | +0.0381 | **0.6612** | 0.8533 | 0.9069 |

I在主降水任务上仍是明确最优模型：降水RMSE为2.1294 mm/h、Pearson r为0.8760，相对E0和W的RMSE均低约9.3%。其梯度幅值比也最接近1，但仍有11.6%的平均幅值收缩；W的梯度RMSE和相关性略优，却以主降水性能明显退化为代价。因此下一步不叠加W或改变网络，而以I为父实验，只增加权重0.02的物理梯度辅助项`G`。

`I+G-0.02`保持I的三通道输入、归一化、强度权重、网络、划分、种子、优化器和选模规则不变。主项仍为带I强度权重的log空间Smooth-L1；G先按`R=expm1(clamp(logR,min=0))`恢复物理降水率，再在真实高度上计算`(B,1,64,64,59)`的相邻层梯度，使用`beta_G=1.0`的无额外权重Smooth-L1。总损失为`L=L_I+0.02L_G`，G只使用相邻两端均为`reliable_loss_mask=true`的pair，不会纳入弱CFB、缺测、padding或雨顶/雨底边界。

### 3.10 Stage 2独立实验结论

Stage 2使用高度保持的双头3D U-Net：四通道距离模型D输入为`[sparse_dbz_std, gr_value_mask, nearest_distance_scaled, height_scaled]`，batch形状是`(B,4,64,64,60)`；共享解码器输出`support_logits`和标准化DPR dBZ，形状均是`(B,1,64,64,60)`。

E2在3V上增加`dbz_gr_interp_standardized + gr_interp_value_mask`，输入增至`(B,5,64,64,60)`。相同完整val的38条轨道结果为：

| 检查点 | CSI | Recall | FAR | FSS-r2 | MAE (dBZ) | RMSE (dBZ) | Bias (dBZ) | Pearson r | CCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D `best_joint` | **0.5771** | **0.6893** | 0.2200 | 0.8582 | 3.8408 | **5.3521** | **-0.2594** | 0.6966 | 0.6749 |
| E2 `best_dbz` | 0.5727 | 0.6592 | **0.1865** | 0.8552 | **3.7971** | 5.3659 | -0.5608 | **0.6982** | **0.6756** |
| E2 `best_support` | 0.5719 | 0.6733 | 0.2084 | **0.8603** | 3.9093 | 5.5954 | -1.3867 | 0.6830 | 0.6380 |

E2 `best_dbz`形成了低MAE、低FAR的Pareto候选，但Recall相对D下降约0.030，CSI、FSS和RMSE也没有全面超过D；`best_support`仍未挽回这一问题。E2在插值可达的`gap_proxy`内改善，在插值不可达的`outside_proxy`内退化，显示模型可能借助插值覆盖形成保守捷径。因此D `best_joint`继续作为当前平衡主模型，E2实验结束，不继续构建“距离+插值”六通道模型。

后续W1.5与W1.25实验已完成。当前用作Stage 3串联起点的W1.25 `best_dbz.pt`在相同完整val上达到`CSI=0.5763`、`Recall=0.6840`、`FSS-r2=0.8569`、`MAE=3.8067 dBZ`、`RMSE=5.3085 dBZ`和`r=0.7020`。它相对D只形成小幅Pareto改善，仍在`>=35 dBZ`上出现约`-7.37 dBZ`偏差，因此不再继续搜索邻近强度权重。W1.25的加权dBZ损失为：

\[
L_{dbz}^{E3}=
\frac{\sum_i M_{dbz,i}w_i\operatorname{SmoothL1}(\widehat Z'_i,Z'_i;0.2)}
{\sum_i M_{dbz,i}w_i}.
\]

### 3.11 Stage 2→Stage 1全冻结串联结论

使用同一封版Stage 1 `I+G-0.02+T3D` epoch 22，对完整38条val轨道、1,195,966个可靠正降水体素做了统一串联评价：

| Stage 1输入 | RMSE (mm/h) | Pearson r | CCC | dR/dz r |
|---|---:|---:|---:|---:|
| 真实DPR dBZ + 真实mask | 2.2262 | 0.8637 | 0.8522 | 0.7023 |
| 师兄插值GR | 4.0753 | 0.4280 | 0.3638 | 0.0456 |
| D预测dBZ + 真实mask | 4.0004 | 0.4419 | 0.3669 | 0.0615 |
| D预测dBZ + 预测mask | 4.0755 | 0.4246 | 0.3589 | 0.0582 |
| W1.25预测dBZ + 真实mask | 3.9180 | 0.4713 | 0.3888 | 0.0688 |
| W1.25预测dBZ + 预测mask | 3.9884 | 0.4538 | 0.3796 | 0.0654 |

`true DPR -> Stage 1`与原Stage 1完整val几乎逐位一致，可基本排除串联标准化、通道顺序和滑窗实现错误。主要退化在“Stage 2预测dBZ + 真实mask”中已经发生，预测mask只带来次要附加损失。因此当前停止独立Stage 2调权，转入[第三阶段串联适配、联合优化与直接反演路线](./第三阶段串联适配、联合优化与直接反演实验路线.md)。

## 4. 项目结构

```text
precipitation-inversion/
├── README.md
├── 第三阶段串联适配、联合优化与直接反演实验路线.md # Stage 3预注册路线
├── 20260408雷达降水廓线反演-20260818.pptx  # 导师项目介绍
├── variables_schema-段晨阳.docx             # 师兄变量说明
├── ZRH_37refine.pth                         # 60层ZRH参数
├── zrh_nc_to_rain.py                        # 反射率批量转换为ZRH降水率
├── plot_2km_zrh_4files.py                   # 2km平面与A-B剖面对比绘图
├── plot_nc_sample_diagnostics.py            # 单样本全变量与专题诊断绘图
├── scripts/
│   ├── build_dataset_manifest.py          # 全数据集文件级审计与清单
│   ├── make_dataset_splits.py             # 按日期分组的无泄漏数据划分
│   ├── fit_normalization_stats.py         # 仅用训练集拟合分高度归一化量
│   ├── build_stage1_sample_index.py       # 构建阶段一正降水体素索引
│   ├── build_stage1_patch_index.py        # 构建3D核心+上下文窗口索引
│   ├── check_distributed_runtime.py       # 低显存NCCL/DDP通信自检
│   ├── launch_stage1_ddp.sh               # 共享服务器安全的单机多卡入口
│   ├── launch_stage1_ablation_suite.sh    # 消融命令预览/顺序启动与输出防覆盖
│   ├── train_stage1_unet3d.py             # 单卡/DDP训练、验证和checkpoint
│   ├── plot_stage1_training_history.py    # 逐epoch曲线、分箱及泛化分析
│   ├── plot_stage1_stratified_metrics.py  # 高度/CFB/类型/轨道宏指标分析
│   ├── backfill_stage1_drdz.py            # 旧实验统一回填dR/dz后再比较
│   ├── compare_stage1_drdz.py             # 多实验统一dR/dz与逐轨配对比较
│   ├── visualize_stage1_test_predictions.py # best.pt固定测试轨道诊断
│   ├── evaluate_stage1_unet3d.py          # Patch指标与完整轨道评估
│   ├── train_stage2_unet3d.py             # Stage 2单卡/DDP训练与checkpoint
│   ├── train_stage2_r1_oracle_sparse_value.py # R1-O非部署稀疏DPR值场补全训练
│   ├── evaluate_stage2_unet3d.py          # Stage 2完整val/test整轨评价
│   ├── evaluate_stage2_r1_oracle_sparse_value.py # R1-O整轨dBZ/物理结构评价
│   ├── evaluate_stage2_r1_oracle_sparse_cascade.py # R1-O真实support冻结串联
│   ├── compare_stage2_r1_partial_conv.py # R1-P相对R1-O的预注册门槛审计
│   ├── evaluate_stage2_stage1_cascade.py  # 全冻结整轨串联与Stage 3 C0严格2×2审计
│   ├── visualize_stage2_stage1_cascade.py # 公共地理范围、QC底图和support轮廓可视化
│   ├── launch_stage2_ddp.sh               # Stage 2单机多卡启动器
│   ├── launch_stage2_r1_ddp.sh            # R1-O/R1-P单机多卡启动器
│   ├── train_stage3_cascade.py            # C1-O冻结S2、仅适配S1的单卡/DDP训练
│   ├── evaluate_stage3_cascade.py         # 读取C1元数据并复用统一整轨串联评价
│   └── launch_stage3_c1_ddp.sh            # C1-O单机多卡启动器
├── configs/
│   ├── stage1_unet3d.yaml                 # 第一版通用模型和训练参数
│   ├── stage1_ablation_e0_baseline.yaml   # 已训练：三通道基线
│   ├── stage1_ablation_e1_mask_cfb.yaml   # 已训练：屏蔽CFB以下输入
│   ├── stage1_ablation_e2_cfb_distance.yaml # 已训练：增加CFB距离通道
│   ├── stage1_ablation_e3_weighted_from_e1.yaml # 暂缓：E1旧多因素分支
│   ├── stage1_ablation_e3_weighted_from_e2.yaml # 暂缓：E2旧多因素分支
│   ├── stage1_ablation_e4_weak_from_e1.yaml     # 暂缓：E1旧弱监督分支
│   ├── stage1_ablation_e4_weak_from_e2.yaml     # 暂缓：E2旧弱监督分支
│   ├── stage1_ablation_e0_n_dbz_valid.yaml  # 已训练：输入归一化解耦
│   ├── stage1_ablation_e0_n_i_intensity.yaml # 已训练：再加强度权重
│   ├── stage1_ablation_e0_n_w_weak_cfb.yaml  # 已训练：再加弱CFB监督
│   ├── stage1_ablation_e0_n_iw_combined.yaml # 条件待训练：组合I与W
│   ├── stage1_ablation_e0_n_i_g_drdz_002.yaml # 已训：I+0.02物理G母实验
│   ├── stage1_ablation_smoke.yaml          # 快速数据/训练链路测试配置
│   ├── stage2_unet3d_4ch_distance.yaml     # 已训D：当前平衡主模型
│   ├── stage2_unet3d_5ch_interp.yaml       # 已训E2：插值值+mask消融
│   ├── stage2_unet3d_4ch_distance_intensity_w1p5.yaml # 已训E3-W1.5：强回波加权
│   ├── stage2_unet3d_4ch_distance_intensity_w1p25.yaml # 已训W1.25：Stage 3当前串联起点
│   ├── stage2_r1_o_dpr_sparse_value.yaml  # 已训R1-O：普通卷积空间恢复上限
│   ├── stage2_r1_p_partial_conv.yaml       # 待训R1-P：稀疏值PartialConv严格对照
│   └── stage3_c1_o_freeze_s2_train_s1.yaml # C1-O：冻结W1.25并低学习率适配T3D
├── src/precipitation_inversion/data/
│   ├── masks.py                           # GR/DPR/降水/cfb统一mask规则
│   ├── splits.py                          # 分组平衡划分核心逻辑
│   ├── nc_reader.py                       # 按变量/扫描区间读取NC并派生mask
│   ├── transforms.py                      # 在线统计、标准化与降水可逆变换
│   ├── dataset.py                         # 紧凑索引及PyTorch Dataset
│   ├── samplers.py                        # 文件/块级洗牌及DDP批次分片
│   ├── patch_dataset.py                   # 3D U-Net核心+halo Patch Dataset
│   ├── stage2_masks.py                    # Stage 2原生missing/sentinel/物理值三态
│   ├── stage2_geometry.py                 # 最近GR距离与局部密度
│   ├── stage2_patch_dataset.py            # Stage 2多通道双目标Patch数据
│   ├── stage2_samplers.py                 # Stage 2四分层DDP采样
│   └── stage3_patch_dataset.py            # C1-O对齐Patch、oracle support与防泄漏打包
├── src/precipitation_inversion/inference/
│   ├── sliding_window.py                  # Stage 1中心核心裁剪与整轨重建
│   ├── stage2_sliding_window.py           # Stage 2双头整轨重建
│   └── stage2_stage1_cascade.py           # dBZ物理转换与冻结串联接口
├── src/precipitation_inversion/models/
│   ├── blocks3d.py                        # 保持高度的各向异性3D残差块
│   ├── unet3d.py                          # 阶段一高度保持3D U-Net
│   ├── stage2_unet3d.py                   # Stage 2 support+dBZ双头U-Net
│   ├── stage2_completion_unet3d.py        # R1-O单dBZ头高度保持U-Net
│   └── stage2_partial_completion_unet3d.py # R1-P稀疏PartialConv+稠密几何双分支
├── src/precipitation_inversion/losses/
│   ├── masked_losses.py                   # mask内Smooth L1/MSE/MAE
│   └── stage2_losses.py                   # 支持域+dBZ加权双损失
├── src/precipitation_inversion/metrics/
│   └── regression.py                      # 流式log/物理空间回归指标
├── src/precipitation_inversion/training/
│   ├── engine.py                          # Stage 1 AMP/DDP epoch循环
│   ├── stage2_engine.py                   # Stage 2双任务训练引擎
│   └── stage2_completion_engine.py        # R1-O单任务AMP/DDP训练引擎
├── tests/
│   ├── test_masks.py                      # mask规则单元测试
│   ├── test_splits.py                     # 划分确定性和无泄漏测试
│   ├── test_nc_reader.py                  # 选择读取、维度和mask语义测试
│   ├── test_transforms.py                 # 分块统计等价性和变换可逆性测试
│   ├── test_dataset.py                    # 索引、缓存和DataLoader测试
│   ├── test_samplers.py                   # 采样完整性、确定性及DDP分片测试
│   ├── test_patch_dataset.py              # Patch形状、padding和halo测试
│   ├── test_sliding_window.py             # 核心拼接与整轨推理测试
│   ├── test_unet3d.py                     # 高度保持和模型梯度测试
│   ├── test_masked_losses.py              # mask归约和梯度隔离测试
│   ├── test_regression_metrics.py         # 指标、强度分箱和流式合并测试
│   ├── test_distributed_configuration.py  # DDP设备绑定及配置安全检查
│   ├── test_training_visualization.py     # 训练JSONL解析与图表输出测试
│   ├── test_prediction_visualization.py   # 固定抽样、指标及诊断图测试
│   ├── test_normalization_stats_script.py # 输入统计与标签QC解耦测试
│   ├── test_stratified_visualization.py   # 分层/CFB/轨道宏图表测试
│   ├── test_drdz_comparison.py            # dR/dz跨实验支持域与比较测试
│   └── test_training_engine.py            # 参数更新和checkpoint恢复测试
├── metadata/manifests/
│   ├── dataset_manifest.csv              # 254个文件的统计清单
│   ├── dataset_summary.json              # 全数据集汇总
│   └── failed_files.csv                  # 读取失败记录
├── metadata/splits/
│   ├── train_files.txt                   # 训练集NC路径
│   ├── val_files.txt                     # 验证集NC路径
│   ├── test_files.txt                    # 测试集NC路径
│   ├── split_manifest.csv                # 带split字段的完整清单
│   └── split_summary.json                # 划分平衡性与泄漏检查
├── metadata/normalization/
│   ├── stage1_positive_qc.json           # 首轮：在标签QC网格拟合的旧统计
│   └── stage1_dbz_valid.json             # 新一轮：全部有限训练DPR输入统计
├── metadata/stage1_indices/              # 本机生成，Git忽略
│   ├── train.npy / val.npy / test.npy    # 6字节/体素的内存映射索引
│   └── train.json / val.json / test.json # 索引来源、哈希和文件范围
├── metadata/stage1_patch_indices/        # 本机生成，Git忽略
│   ├── train.npy / val.npy / test.npy    # 非重叠nscan核心窗口索引
│   └── train.json / val.json / test.json # halo、padding和文件范围
├── requirements-zrh.txt                     # 转换脚本依赖
├── requirements-plot.txt                    # 绘图完整依赖
├── requirements-training.txt                # 阶段一PyTorch依赖
├── 数据集说明.md                            # 数据结构与物理含义
├── NC样本变量与数值分析.md                 # 单样本全变量统计
├── 降水反演任务拆解与实验路线.md             # 三阶段任务拆解和实验清单
├── 模型数据处理流程与张量形状.md             # 模型输入处理、变量和shape查询
├── 运行ZRH转换脚本.md                       # 转换脚本使用说明
├── 运行2km_ZRH绘图脚本.md                  # 六场对比绘图脚本使用说明
└── 运行NC单样本诊断绘图.md                 # 全变量诊断脚本使用说明
```

`.venv/`、`outputs/`、NetCDF 数据和临时文件不纳入版本控制。

## 5. 环境配置

服务器当前验证环境：

- Python 3.10.12
- NumPy 1.26.4
- netCDF4 1.6.2
- Matplotlib 3.8.4
- Pillow 12.3.0
- PyTorch 2.8.0+cu128

创建并激活环境：

```bash
cd /home/koujizhi/projects/precipitation-inversion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
python -m pip install -r requirements-training.txt
```

每次新开终端都需要重新执行：

```bash
source /home/koujizhi/projects/precipitation-inversion/.venv/bin/activate
```

## 6. 运行方法

### 6.1 查看 ZRH 转换参数

```bash
python zrh_nc_to_rain.py --help
```

只转换稀疏 GR 反射率：

```bash
python zrh_nc_to_rain.py --variables dbz_gr_sparse
```

不传参数时会处理全部 254 个文件和五类反射率变量，并在项目 `outputs/` 下生成大量数据。正式运行前应确认磁盘空间和实验范围。

### 6.2 绘制一个样本

```bash
python plot_2km_zrh_4files.py --count 1
```

默认绘制按文件名排序后的前四个文件：

```bash
python plot_2km_zrh_4files.py
```

图片默认保存到：

```text
outputs/plots_2km_zrh
```

详细说明见[ZRH转换脚本运行说明](./运行ZRH转换脚本.md)和[2km绘图脚本运行说明](./运行2km_ZRH绘图脚本.md)。

### 6.3 生成单样本全变量诊断报告

```bash
python plot_nc_sample_diagnostics.py
```

默认按文件名排序处理数据集前20个样本，每个样本生成23张逐变量图、8张总览/专题图、统计CSV和31页PDF。处理前10个样本：

```bash
python plot_nc_sample_diagnostics.py --count 10
```

已有完整样本结果默认跳过，异常中断且没有完成标记的样本会在下次运行时自动重建；需要主动更新完整结果时添加 `--overwrite`：

```bash
python plot_nc_sample_diagnostics.py --overwrite
```

也可以只分析一个指定文件：

```bash
python plot_nc_sample_diagnostics.py --input-file /path/to/sample.nc
```

批处理默认记录单样本异常并继续处理后续文件，最终统一汇总；调试时可使用 `--fail-fast` 在首个异常处停止。

详细说明见[单样本诊断绘图运行说明](./运行NC单样本诊断绘图.md)。

### 6.4 生成全数据集审计清单

默认扫描全部254个NC文件，保持源数据只读：

```bash
python scripts/build_dataset_manifest.py --workers 2
```

调试时只检查前10个文件：

```bash
python scripts/build_dataset_manifest.py \
  --count 10 \
  --output-dir /tmp/precipitation-manifest-debug \
  --overwrite
```

正式输出位于 `metadata/manifests/`，包含文件级CSV、全数据集JSON汇总和失败文件记录。默认使用单进程；在共享存储上不宜盲目设置过多 `--workers`。

### 6.5 生成训练、验证和测试划分

默认以完整日期为不可分割组，使用10,000次固定种子候选搜索，尽量平衡GR覆盖、正降水、强降水尾部和降水类型：

```bash
python scripts/make_dataset_splits.py --overwrite
```

也可生成按时间连续切分的分布偏移对照组：

```bash
python scripts/make_dataset_splits.py \
  --strategy chronological \
  --output-dir /tmp/precipitation-splits-chronological \
  --overwrite
```

正式平衡划分位于 `metadata/splits/`。同一日期的所有NC文件只会出现在一个集合中。

### 6.6 选择性读取NC样本

`read_nc_sample` 只读取显式请求的变量，并可以统一裁剪连续的 `nscan` 区间：

```python
from precipitation_inversion.data.nc_reader import read_nc_sample

sample = read_nc_sample(
    "/path/to/sample.nc",
    variables=("z", "dbz_dpr", "pre_dpr", "cfb"),
    scan_slice=slice(100, 132),
)

dbz = sample.variables["dbz_dpr"]
positive = sample.masks["pre_positive_native"]
trainable = sample.masks["pre_positive_qc"]
```

默认返回 `float32`，缺测保留为NaN。`pre_valid_native` 表示原始有效降水标签，`pre_valid_qc` 还会剔除 `cfb` 以下的杂波区域。

### 6.7 拟合阶段一归一化统计量

统计脚本仅读取`train_files.txt`中的175个训练文件，并与`split_manifest.csv`交叉检查，遇到验证或测试文件会直接报错。第二轮阶段一实验使用以下命令；`dbz_dpr`输入口径和`pre_dpr`标签/高度权重参考口径是两个独立参数：

```bash
python scripts/fit_normalization_stats.py \
  --variables dbz_dpr \
  --selection-mask variable_valid \
  --height-loss-selection-mask pre_positive_qc \
  --output metadata/normalization/stage1_dbz_valid.json \
  --overwrite
```

`variable_valid`表示每个输入变量分别使用自身全部有限、非fill值拟合每层均值和标准差，不再让标签的降水/CFB条件删除有效输入。`height-loss-selection-mask`只额外记录可靠标签的逐层数量，供未来高度权重使用，绝不能把输入计数误当作标签计数。当前正式文件已生成：`dbz_dpr`共6,822,356个值、60层均非空；独立的`pre_positive_qc`高度参考仍为5,481,557个。

调试时可限制文件数并写到临时位置：

```bash
python scripts/fit_normalization_stats.py \
  --variables dbz_dpr \
  --selection-mask variable_valid \
  --height-loss-selection-mask pre_positive_qc \
  --max-files 2 \
  --output /tmp/stage1-normalization-smoke.json \
  --overwrite
```

`--max-files` 产物只用于检查统计脚本。其 `processed_file_count < validated_file_count`，`Stage1PatchDataset` 会明确拒绝将它用于训练，避免将小样本调试统计误当正式归一化。

首轮E0/E1/E2保留`metadata/normalization/stage1_positive_qc.json`以保证历史实验可复现。该旧文件把输入统计限制在5,481,557个`pre_positive_qc`标签体素，最低两层输入统计为`null`；后续E0-N系列统一改用`stage1_dbz_valid.json`。

### 6.8 构建并读取阶段一体素索引（前期体素级管线）

按训练、验证、测试文件清单生成索引：

```bash
python scripts/build_stage1_sample_index.py --split all
```

索引只保留同时满足 `pre_positive_qc`、DPR反射率有效且 `dbz_dpr/p/t/q` 均为有限值的体素。每条记录仅保存 `file_id/scan/ray/level`，占6字节；三组真实数据索引共约45 MiB，属于可再生文件，因此不提交Git。

```python
from precipitation_inversion.data.dataset import Stage1IntensityDataset

dataset = Stage1IntensityDataset(
    "metadata/stage1_indices/train.json",
    "metadata/normalization/stage1_positive_qc.json",
)
item = dataset[0]
# features: 标准化后的 dbz_dpr/p/t/q 和缩放高度，共5维
# target: log(1 + pre_dpr)
```

Dataset以内存映射方式读取索引，并为每个DataLoader进程维护独立的NetCDF文件LRU缓存。索引按文件连续排列；后续训练采样器宜按文件块洗牌，再在块内洗牌，避免全局随机访问反复淘汰缓存。

该体素级接口用于早期数据验证和MLP/逐点模型扩展，不是当前3D U-Net的正式训练入口；当前模型使用6.10节的Patch索引和`Stage1PatchDataset`。

### 6.9 使用文件块批采样器

阶段一训练不应再向DataLoader传入全局 `shuffle=True`。使用批采样器后，文件顺序、文件内块顺序和块内样本都会随epoch确定性地洗牌，但每个batch仍只读取一个NC文件：

```python
import torch

from precipitation_inversion.data.dataset import Stage1IntensityDataset
from precipitation_inversion.data.samplers import FileBlockBatchSampler

dataset = Stage1IntensityDataset(
    "metadata/stage1_indices/train.json",
    "metadata/normalization/stage1_positive_qc.json",
)
sampler = FileBlockBatchSampler(
    dataset,
    batch_size=256,
    block_size=4096,  # 16个batch组成一个局部洗牌块
    seed=2026,
    drop_last=True,
)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=2,
)

for epoch in range(100):
    sampler.set_epoch(epoch)
    for batch in loader:
        pass
```

设置`batch_sampler`后不能再同时设置DataLoader的`batch_size/shuffle/drop_last`。采样器支持`num_replicas/rank`；若DDP进程组已初始化，则会自动读取world size和当前rank。默认`even_batches=True`会舍弃至多`world_size-1`个全局batch，使所有rank执行相同步数，适合需要同步反向传播的DDP训练。正式验证入口会显式设置`even_batches=False`并使用无DDP前向collective的已同步裸模型，从而保留每个Patch且不重复；不要把训练采样策略直接套到完整验证。

### 6.10 构建3D U-Net Patch并重建完整轨道

默认使用32个扫描线的非重叠输出核心，以及左右各16个扫描线的上下文：

```bash
python scripts/build_stage1_patch_index.py --split all
```

原始窗口 `(64,49,60)` 只在横轨高端补齐为 `(64,64,60)`，不增加虚假高度层。
Dataset的三个输入通道是标准化DPR反射率、显式有效性mask和缩放至 `[-1,1]`
的高度坐标：

```python
from precipitation_inversion.data.patch_dataset import Stage1PatchDataset

train_dataset = Stage1PatchDataset(
    "metadata/stage1_patch_indices/train.json",
    "metadata/normalization/stage1_dbz_valid.json",
    positive_only=True,
)
item = train_dataset[0]
print(item["inputs"].shape)     # (3,64,64,60)
print(item["target"].shape)     # (1,64,64,60)
print(item["loss_mask"].shape)  # (1,64,64,60)
```

E0-N系列中三通道依次为：按高度层标准化的`dbz_dpr`、DPR输入有效性mask、缩放到`[-1,1]`的物理高度。缺测反射率只在记录`valid=0`后填成标准化空间的0；横轨和轨道边界padding同样以数值0、mask false填充，不会进入损失。`target`为`log1p(pre_dpr)`，无监督位置以0占位，但必须由`loss_mask=false`区分，不能把占位0理解为无降水标签。

训练时 `positive_only=True` 过滤没有正降水loss体素的核心；逐epoch快速选模也可保持这一设置，而训练后独立完整诊断与整轨测试必须使用 `positive_only=False`，才能覆盖全部Patch、原始 `nscan` 以及只含CFB下原生标签的区域。高度保持3D U-Net只在扫描和横轨
方向以 `(2,2,1)` 下采样，模型输出 `(B,1,64,64,60)` 后，可使用
`predict_full_orbit` 自动取每个窗口的中心32扫描线并拼成 `(nscan,49,60)`：

```python
from precipitation_inversion.inference.sliding_window import predict_full_orbit

rain = predict_full_orbit(model, test_dataset, file_id=0, device="cuda")
```

### 6.11 训练和评估高度保持3D U-Net

配置文件 `configs/stage1_unet3d.yaml` 使用JSON兼容的YAML 1.2语法，因此当前环境
不安装PyYAML也能直接由标准库读取。单GPU训练：

```bash
python scripts/train_stage1_unet3d.py --device cuda:0
```

多GPU训练前先在服务器监视器 `http://211.86.155.236:8081/` 或终端确认空闲卡：

```bash
nvidia-smi
nvidia-smi pmon -c 1
```

启动脚本要求显式声明当时确认空闲的物理GPU，防止共享服务器上误占他人正在使用的卡。下面的`GPU_ID_1,GPU_ID_2`只是待替换占位符，不代表固定推荐卡号；进程内会重新映射为`cuda:0`、`cuda:1`：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID_1,GPU_ID_2 \
STAGE1_NUM_GPUS=2 \
STAGE1_MASTER_PORT=29519 \
scripts/launch_stage1_ddp.sh
```

同一服务器上的不同分布式任务必须使用不同的 `STAGE1_MASTER_PORT`。正式训练前可先
运行低显存通信自检；它检查进程绑定、NCCL collective和DDP梯度同步，不读取NC数据：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID_1,GPU_ID_2 .venv/bin/torchrun \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29518 \
  scripts/check_distributed_runtime.py
```

`data.batch_size` 是每张GPU的batch大小，当前为1。有效全局batch为
`每卡batch × GPU数 × accumulation_steps`；两卡为2、八卡为8。第一轮实验先保持
学习率 `1e-4`，不要因GPU数自动线性放大；得到稳定曲线后再单独比较学习率。训练采样器保持各rank等步数，最多舍弃`world_size-1`个全局batch以避免反向collective死锁；验证采用不等长、互斥分片并绕过DDP前向collective，完整保留299个Patch。指标在所有rank上聚合，checkpoint仅由rank 0写入。

断点恢复和两批次烟雾测试：

```bash
python scripts/train_stage1_unet3d.py \
  --resume outputs/stage1_unet3d/last.pt \
  --device cuda:0

python scripts/train_stage1_unet3d.py \
  --smoke-test \
  --output-dir outputs/stage1_unet3d_smoke \
  --device cuda:0
```

训练入口每个epoch记录log空间及mm/h空间的MAE、RMSE、Bias、R²、Pearson相关系数，
并按目标降水率 `<1/1–5/5–10/10–30/≥30 mm/h` 分箱。组合损失还会分别记录I主项、原始G、`0.02×G`、G占总目标比例和可靠梯度pair数；两项分别按体素权重和pair数归约，不混用分母。默认仍以验证集物理空间RMSE选择 `best.pt`，并覆盖更新 `last.pt`；不可变的 `epoch_XXXX.pt` 每10轮才保存一次（零基epoch为 `0009/0019/0029/...`），避免逐轮快照耗尽磁盘。Stage 2的 `best_joint.pt`、`best_support.pt`、`best_dbz.pt` 同样按各自验证最优覆盖更新，不受周期快照设置影响。

完整训练正常结束后，只有DDP rank 0会自动执行训练历史、完整验证和固定测试轨道三类后处理。配置由`postprocessing`控制；测试可视化默认固定随机种子2026，从有正降水监督的测试轨道里抽取6条完整轨道。所有结果写入本次命令实际指定的output目录：

```text
outputs/<experiment>/analysis/
├── README.md
├── training_history/
│   ├── training_overview.png
│   ├── loss_components.png
│   ├── validation_intensity_bins.png
│   ├── generalization_gap.png
│   ├── epoch_metrics.csv
│   ├── summary.json
│   └── summary.md
├── full_validation/
│   ├── metrics.json                     # 遍历437个Patch，主指标仍为可靠mask
│   └── stratified/
│       ├── metrics_by_height.csv/png
│       ├── metrics_by_cfb_distance.csv
│       ├── metrics_by_precipitation_type.csv
│       ├── drdz_summary.md
│       ├── drdz_overall.csv
│       ├── drdz_by_height.csv/png
│       ├── drdz_by_cfb_distance.csv
│       ├── drdz_by_target_intensity.csv
│       ├── drdz_by_precipitation_type.csv
│       ├── drdz_grouped.png
│       ├── drdz_metrics_by_file.csv
│       ├── drdz_filewise_macro_bootstrap.csv
│       ├── below_cfb_native_positive.csv/png
│       ├── metrics_by_file.csv
│       ├── filewise_macro_bootstrap.csv/png
│       └── summary.json/md
└── test_predictions/
    ├── aggregate_diagnostics.png
    ├── summary.json
    ├── summary.md
    └── sample_XX/
        ├── diagnostics.png
        ├── metrics.json
        └── prediction_and_target.npz
```

`diagnostics.png`包括约2 km平面降水图、自动选择的A–B剖面、预测误差、正降水
分布、`log1p`相关性和分高度RMSE/Bias/Pearson相关系数。自动后处理失败默认不会删除
已经训练好的checkpoint；如希望后处理失败也令训练命令返回非零状态，可设置
`postprocessing.fail_on_error=true`。调试训练时可传入 `--skip-postprocessing`。

`full_validation/metrics.json`的主指标始终使用`reliable_loss_mask`。CFB以下原生正降水只进入`diagnostics.below_cfb_native_positive`；逐文件结果、轨道等权宏平均和95% bootstrap置信区间写入`filewise`，bootstrap默认以完整文件/轨道为重采样单位、种子2026、重复2,000次。物理`dR/dz`默认启用并写入`patch_evaluation.metrics.physical_drdz`；只在临时调试时才使用`--no-physical-drdz`关闭。

已有训练也可随时手动重跑：

```bash
python scripts/plot_stage1_training_history.py outputs/stage1_unet3d_3gpu

CUDA_VISIBLE_DEVICES=GPU_ID python scripts/visualize_stage1_test_predictions.py \
  outputs/stage1_unet3d_3gpu/best.pt \
  --output-dir outputs/stage1_unet3d_3gpu/analysis/test_predictions \
  --sample-count 6 --seed 2026 --device cuda:0

python scripts/evaluate_stage1_unet3d.py \
  outputs/stage1_unet3d_3gpu/best.pt \
  --split val --stratified --device cuda:0 \
  --bootstrap-seed 2026 --bootstrap-replicates 2000 \
  --output outputs/stage1_unet3d_3gpu/analysis/full_validation/metrics.json

python scripts/plot_stage1_stratified_metrics.py \
  outputs/stage1_unet3d_3gpu/analysis/full_validation/metrics.json \
  --output-dir outputs/stage1_unet3d_3gpu/analysis/full_validation/stratified
```

历史E0/N/I/W的`metrics.json`生成于物理`dR/dz`功能之前，不能直接传给比较脚本。先确认一张真正空闲的物理GPU并执行统一回填入口；下例中的`GPU_ID`需要替换，进程内仍使用映射后的`cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID \
python scripts/backfill_stage1_drdz.py --device cuda:0
```

该入口会按E0、N、I、W顺序加载各自`best.pt`，重新遍历完整验证集、生成分层图，最后自动调用比较脚本。若中途因GPU占用失败，再次执行时会复用已经成功且带支持域指纹的报告，只继续未完成项；需要全部强制重算时添加`--force`。详细日志分别写入各实验的`analysis/full_validation/drdz_backfill.log`。

只有四份完整报告都已生成时，才可单独使用下面的比较入口核对评价协议、逐体素支持域指纹和逐轨样本数，再生成统一表格、图和相对E0的配对bootstrap：

```bash
python scripts/compare_stage1_drdz.py \
  --run E0=outputs/ablations/stage1_e0_baseline/analysis/full_validation/metrics.json \
  --run N=outputs/ablations/stage1_e0_n_dbz_valid/analysis/full_validation/metrics.json \
  --run I=outputs/ablations/stage1_e0_n_i_intensity/analysis/full_validation/metrics.json \
  --run W=outputs/ablations/stage1_e0_n_w_weak_cfb/analysis/full_validation/metrics.json \
  --baseline E0 \
  --output-dir outputs/ablations/stage1_drdz_comparison
```

只要任一输入不是完整验证集，或四组的高度网格、mask定义、pair指纹、逐高度/逐轨支持数不同，比较脚本就会拒绝排序，避免把W的弱监督体素或不完整批次混入结论。

Patch评估及一条完整轨道重建：

```bash
python scripts/evaluate_stage1_unet3d.py \
  outputs/stage1_unet3d/best.pt \
  --split test \
  --device cuda:0

python scripts/evaluate_stage1_unet3d.py \
  outputs/stage1_unet3d/best.pt \
  --split test \
  --full-orbits 1 \
  --device cuda:0
```

完整轨道模式把非重叠核心重建为 `(nscan,49,60)`，在 `pre_positive_qc` 上计算
条件强度指标，并把预测、标签和评价mask保存为压缩NPZ。

### 6.12 复现第二轮E0-N/I/W消融

启动器默认只打印将要执行的命令，不会占用GPU。先预览四组配置、输出目录和端口：

```bash
STAGE1_ABLATION_PHASE=e0n \
scripts/launch_stage1_ablation_suite.sh
```

正式复现前再次检查GPU和端口，并把下面的`GPU_ID_1,GPU_ID_2`替换成当时确实空闲的物理卡编号、把进程数改成编号数量。若从头复现实验，仍应先单独运行E0-N，以隔离新归一化的影响：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID_1,GPU_ID_2 \
STAGE1_NUM_GPUS=2 \
STAGE1_EXECUTE=1 \
STAGE1_ABLATIONS=e0_n \
STAGE1_ABLATION_MASTER_PORT_BASE=29820 \
scripts/launch_stage1_ablation_suite.sh
```

随后可分别复现强度权重I与弱CFB监督W。逗号分隔的实验会在同一组GPU上顺序执行，不会同时抢占显存：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID_1,GPU_ID_2 \
STAGE1_NUM_GPUS=2 \
STAGE1_EXECUTE=1 \
STAGE1_ABLATIONS=e0_n_i,e0_n_w \
STAGE1_ABLATION_MASTER_PORT_BASE=29830 \
scripts/launch_stage1_ablation_suite.sh
```

组合项IW仍遵循“只有I和W各自都带来可复现收益时才运行”的预注册条件；当前W没有明确总体收益，因此IW暂缓：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID_1,GPU_ID_2 \
STAGE1_NUM_GPUS=2 \
STAGE1_EXECUTE=1 \
STAGE1_ABLATIONS=e0_n_iw \
STAGE1_ABLATION_MASTER_PORT_BASE=29840 \
scripts/launch_stage1_ablation_suite.sh
```

输出依次位于`outputs/ablations/stage1_e0_n_dbz_valid`、`stage1_e0_n_i_intensity`、`stage1_e0_n_w_weak_cfb`和`stage1_e0_n_iw_combined`。启动器遇到非空输出目录会拒绝覆盖；需要续训时应单独调用训练脚本并显式传`--resume`，不要删除已有结果。若希望一次顺序运行全部四组，可在确认资源和时间后使用`STAGE1_ABLATION_PHASE=e0n`与`STAGE1_EXECUTE=1`，但不推荐跳过前述逐步判定。

### 6.13 运行严格I+G-0.02消融

启动ID为`e0_n_i_g`，默认输出到`outputs/ablations/stage1_e0_n_i_g_drdz_002`。下例的GPU编号仍必须替换为启动当时确认空闲的三张物理卡：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID_1,GPU_ID_2,GPU_ID_3 \
STAGE1_NUM_GPUS=3 \
STAGE1_EXECUTE=1 \
STAGE1_ABLATIONS=e0_n_i_g \
STAGE1_ABLATION_MASTER_PORT_BASE=29860 \
scripts/launch_stage1_ablation_suite.sh
```

训练必须从头开始，不能用I的checkpoint初始化，否则会把“训练轮数/初始化方式”混入G这一单因素。训练结束后自动生成完整验证集物理梯度报告。只比较I和I+G时要显式把I设为baseline：

```bash
python scripts/compare_stage1_drdz.py \
  --run I=outputs/ablations/stage1_e0_n_i_intensity/analysis/full_validation/metrics.json \
  --run I+G=outputs/ablations/stage1_e0_n_i_g_drdz_002/analysis/full_validation/metrics.json \
  --baseline I \
  --output-dir outputs/ablations/stage1_i_vs_ig_drdz_002
```

正式接受门槛预先固定为：降水RMSE不高于2.151 mm/h、Pearson r不低于约0.874；同时dR/dz RMSE低于2.363、r高于0.661，幅值比继续接近1，并核对低于2 km、CFB上方0–0.5 km和4–6 km分组没有局部恶化。

### 6.14 `typePrecip`三维形态辅助任务（已实现，待顺序实验）

这一轮不把卫星`typePrecip`作为输入。模型输入仍为
`(B,3,64,64,60)`：标准化DPR反射率、DPR有效性mask和高度；Dataset另外返回
`type_target=(B,64,64)`与`type_loss_mask=(B,64,64)`，只监督非重叠核心区、类别为
1/2/3且该廓线至少存在一个原生DPR回波的位置。`-1111`无降水、`-9999`缺测、halo和padding均使用
`ignore_index=-100`排除，类别标签不会进入三通道输入。

实验必须依次执行并在每一步后判定：

1. `analyze_type_precip_structure.py`完成train/val/test类别审计，保存CFAD、垂直廓线、水平纹理、回波顶、亮带代理、强回波质心轨迹以及代表性scan-z/ray-z剖面；
2. 冻结同一个`I+G-0.02 best.pt`主干，分别训练Pool探针与Ordered-3D探针。Pool只做高度均值/最大池化；Ordered-3D使用水平、垂直和两个方向的水平—高度耦合卷积，将`(B,16,64,64,60)`学习压缩为`(B,8,64,64,15)`，再按固定高度顺序折叠为`(B,120,64,64)`；
3. 只有Ordered-3D的Macro-F1、balanced accuracy和对流召回明确优于Pool，并且高度打乱造成合理退化时，才训练严格多任务`I+G-0.02+T3D`。

正式T3D输出为`rain=(B,1,64,64,60)`和`type_logits=(B,3,64,64)`。总损失为
`L=L_I+0.02L_G+0.01L_type`，其中类别权重由完整训练集核心廓线的频次自动计算为归一化
`1/sqrt(count)`；`type_logits`不反馈给降水头，`best.pt`仍只按验证集降水RMSE选择。因此T3D相对
I+G只增加辅助头和分类监督，没有改变输入、雨量标签、主损失、G、划分、种子或优化器。训练后会自动保存分类混淆矩阵、Macro-F1、balanced accuracy、逐类召回、高度倒序/随机打乱及低中高层遮挡诊断；所有雨量分箱同时新增CCC，避免Pearson相关性掩盖尺度和偏差问题。

## 7. 当前限制与待确认事项

1. 已完成按日期隔离的划分，但相邻多日是否属于同一天气过程尚无事件级标注；
2. 尚未从数据生成代码确认 GR–DPR 最大时间匹配误差；
3. 尚未确认多雷达重叠区域的合并权重，以及均值是在 dBZ 空间还是线性反射率空间计算；
4. 尚未确认 `dbz_gr_interp` 的具体插值算法、搜索半径和边界处理；
5. 已完成3D U-Net学习模型首轮实验，但传统分层Z–R和插值路线尚未在完全相同的验证mask上形成统一定量基线；
6. 当前Stage1只评价可靠正降水条件下的强度，不能由现有MAE/RMSE推断无降水检测、降水覆盖范围或虚警能力；标签为DPR遥感产品，CFB以下原生正值置信度更低，只能作为弱监督候选和独立诊断；
7. E0/E1/E2/N/I/W各只有一个随机种子，且正式配置`deterministic=false`。完整验证集的逐轨bootstrap只描述固定权重下的轨道差异，不能替代训练随机种子的重复实验；
8. E0-N、E0-N-I和E0-N-W已经完成正式训练；其中只有I在完整验证集条件强度指标上形成明确总体收益，N轻微退化、W总体持平。E0-N-IW尚未训练，不应预先假设I与W组合后仍会受益；
9. 当前权重读取器针对仓库提供的`ZRH_37refine.pth`固定结构，不用于加载任意PyTorch checkpoint；
10. 已在RTX 4090 D服务器上完成两卡NCCL collective、DDP反向传播以及真实NC数据/完整3D U-Net的训练与验证冒烟测试。受管执行环境建立`TCPStore`时曾报告hostname解析警告并使初始化延迟约40秒，但最终通信正确；若用户终端长时间停顿，应检查`/etc/hosts`、主机名解析、本地socket权限和端口冲突；
11. “非等长、不重复”验证分片已经过分片完整性单测，并在E0-N/I/W正式多卡训练中得到相同的1,195,966个可靠验证体素；训练后完整诊断则遍历包含无正降水核心在内的437个唯一Patch；
12. 尚未在八卡同时空闲时做八进程实测。代码路径与两卡相同，但不能预先假定任何卡空闲；正式启动前应检查目标GPU，并先用对应进程数运行`check_distributed_runtime.py`。
13. Stage 1当前封版为`I+G-0.02+T3D` epoch 22；G只监督连续可靠正降水内部的垂直差分，不能解决有效零雨、雨顶/雨底或CFB以下区域的检测问题。
14. Stage 2当前各模型仍只有一个训练随机种子；W1.25只是Stage 3的较好预训练起点，不是已形成可独立部署链路的Stage 2封版结论。
15. Stage 2的`dbz_gr_interp`插值算法、搜索半径和边界处理仍未从原始生成代码确认；E2在`outside_proxy`的退化也说明不应将该插值产品当成新观测。
16. W1.25全冻结部署串联在完整val上只达到`RMSE=3.9884 mm/h`、`r=0.4538`和`dR/dz r=0.0654`，而真实DPR输入为`2.2262/0.8637/0.7023`。独立硬串联未通过Stage 3门槛。
17. W1.25在`outside_proxy`的dBZ r约0.428、support Recall约0.232，已显示当前压缩GR数据的可观测性上限；Stage 3适配实验应与原始GR体扫和几何信息询问并行。
18. `S3-C1-O-S1Adapt`已实现：Stage 2严格`eval()+no_grad()+requires_grad=False`；Stage 1从T3D epoch 22初始化，只优化rain共享干路与rain head，类型头保留但冻结；目标仍为`I+0.02G`，使用真实DPR support隔离dBZ接口误差。
19. C1内部DataLoader搬运`(B,6,64,64,60)`打包张量，其中前4通道仅供Stage 2，后2通道为oracle support和Stage 1高度副本；两个U-Net并不接收6通道。Stage 1实际输入始终是`[predicted_dbz_std_s1,true_support,height]`的`(B,3,64,64,60)`。
20. C1检查点只保存可训练Stage 1权重，冻结Stage 2路径及SHA-256写入metadata；始终保存`best.pt/last.pt`，周期文件只在`0009/0019/...`写入。正式训练后自动执行完整val串联评价和统一轨道可视化，不访问test。
21. C1完整38轨oracle-support结果为`RMSE=3.9269 mm/h、r=0.4723`，相对匹配F0-PT的`3.9180/0.4713`未达到`RMSE<3.800`或`r>0.481`门槛；弱降水MAE与dR/dz RMSE改善，但Bias更负、CCC下降且中强降水没有恢复，因此不进入C1-P。
22. `S3-C2-O-S2TaskAware`已实现并完成训练/完整val：冻结原始T3D epoch 22，只解冻W1.25 Stage 2的最高分辨率decoder与reflectivity head，共7025个参数；support head固定但其BCE仍通过固定head权重约束共享decoder。
23. C2恢复完整2076个Stage 2训练Patch及原四类采样，而非只取正降水Patch。复合损失严格为`L_dbz^W1.25+L_support+lambda_R*(I+0.02G)`；`lambda_R`由train batch梯度范数自动固定，不访问val/test。
24. C2关闭早停并完整执行20轮余弦调度；保存`best_rain/best_joint/best_dbz/best_support/last`，周期文件仍只写`0009/0019`。训练后先为适配后的support头重新在val选阈值，再自动执行Stage 2物理评价、完整38轨串联评价和统一可视化。
25. C2的train-only梯度审计得到`lambda_R=0.67919`，但最佳rain检查点仍出现在epoch 1；其oracle-support为`RMSE=3.9195、r=0.4715`，相对匹配F0仅`RMSE +0.0015、r +0.0002`，没有达到继续试验门槛。predicted-support为`3.9848/0.4545`，dR/dz `r=0.0692`，因此停止C2-P和双U-Net联合解冻。
26. `S3-D0-DirectMultiHead`已实现：输入严格为`[GR标准化dBZ, GR值mask, 最近GR距离, 高度]`的`(B,4,64,64,60)`；共享高度保持3D U-Net同时输出`rain_log1p`、标准化DPR dBZ和support logits，三个输出均为`(B,1,64,64,60)`。DPR dBZ、DPR support和`pre_dpr`只在标签侧出现。
27. 首个D0-H实验从W1.25 `best_dbz`初始化全部共享层和两个物理头，只新增并训练17参数的`1×1×1` rain head；完整38轨oracle-support为`RMSE=4.1538、r=0.3728`，相对F0明显退化，说明冻结W1.25表示不能由线性头充分读取降水。
28. `S3-D0-D-RainPrimary`从D0-H epoch 2 `best_rain.pt`恢复完整三头状态，冻结stem/encoder，解冻完整decoder与三头，共589011个可训练参数。损失严格为`(I+0.02G)+lambda_phys*(support+dbz_W1.25)`；`lambda_phys`由train-only共享decoder梯度审计选择，使物理/降水梯度比为0.25，rain权重固定1。真实CPU烟雾审计得到`lambda_phys≈0.1072`，正式三卡会重新聚合审计。

## 8. 下一阶段计划

1. 在确认三张GPU空闲后正式训练`S3-D0-D-RainPrimary`，完整跑20轮余弦调度，不用早停；
2. 训练前自动在train batch上固定`lambda_phys`，不访问val/test；训练后只在完整38轨val选择D0 support阈值；
3. 同时报告`D0 rain+真实support`诊断口径和`D0 rain+预测support`部署口径，并与DPR-oracle、F0、C2、D0-H做同口径强度分箱、类型、高度、CFB和dR/dz比较；
4. 若RainPrimary不能至少超过F0的`RMSE=3.9180、r=0.4713`，则停止D0结构/损失权重扩张，转向原始GR极坐标体扫、真实几何和压缩数据可观测性；
5. C1/C2均失败，因此当前不启动C1-P、C2-P或`S3-C3-PartialJoint`；
6. Stage 3开发全程只使用完整val，只在锁定1至2个最终候选后访问test。详细冻结矩阵、张量契约、门槛和停止规则见[第三阶段路线](./第三阶段串联适配、联合优化与直接反演实验路线.md)。

## 9. 参考资料

- [数据集结构与物理意义](./数据集说明.md)
- [单个NetCDF样本全变量与数值分析](./NC样本变量与数值分析.md)
- [ZRH批量转换运行说明](./运行ZRH转换脚本.md)
- [2km水平图与垂直剖面运行说明](./运行2km_ZRH绘图脚本.md)
- [单样本全变量诊断绘图运行说明](./运行NC单样本诊断绘图.md)
- [降水反演任务拆解与实验路线](./降水反演任务拆解与实验路线.md)
- [第二阶段GR稀疏反射率到DPR反射率实验路线](./第二阶段GR稀疏反射率到DPR反射率实验路线.md)
- [第三阶段串联适配、联合优化与直接反演实验路线](./第三阶段串联适配、联合优化与直接反演实验路线.md)
- [模型数据处理流程、变量与张量形状](./模型数据处理流程与张量形状.md)
- `variables_schema-段晨阳.docx`：原始变量说明
- `20260408雷达降水廓线反演-20260818.pptx`：项目背景与任务介绍
