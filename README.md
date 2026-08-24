# Precipitation Inversion

基于地基雷达（Ground Radar, GR）稀疏三维反射率反演稠密三维降水率的研究项目。本仓库当前处于数据理解、基线代码梳理与可视化验证阶段，目标是为后续毕业设计中的模型构建和实验评估建立可靠的数据基础。

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

## 3. 本周研究进度（2026-08-17—2026-08-23）

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

## 4. 项目结构

```text
precipitation-inversion/
├── README.md
├── 20260408雷达降水廓线反演-20260818.pptx  # 导师项目介绍
├── variables_schema-段晨阳.docx             # 师兄变量说明
├── ZRH_37refine.pth                         # 60层ZRH参数
├── zrh_nc_to_rain.py                        # 反射率批量转换为ZRH降水率
├── plot_2km_zrh_4files.py                   # 2km平面与A-B剖面对比绘图
├── plot_nc_sample_diagnostics.py            # 单样本全变量与专题诊断绘图
├── scripts/
│   ├── build_dataset_manifest.py          # 全数据集文件级审计与清单
│   └── make_dataset_splits.py             # 按日期分组的无泄漏数据划分
├── src/precipitation_inversion/data/
│   ├── masks.py                           # GR/DPR/降水/cfb统一mask规则
│   └── splits.py                          # 分组平衡划分核心逻辑
├── tests/
│   ├── test_masks.py                      # mask规则单元测试
│   └── test_splits.py                     # 划分确定性和无泄漏测试
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
├── requirements-zrh.txt                     # 转换脚本依赖
├── requirements-plot.txt                    # 绘图完整依赖
├── 数据集说明.md                            # 数据结构与物理含义
├── NC样本变量与数值分析.md                 # 单样本全变量统计
├── 降水反演任务拆解与实验路线.md             # 三阶段任务拆解和实验清单
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

创建并激活环境：

```bash
cd /home/koujizhi/projects/precipitation-inversion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
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

## 7. 当前限制与待确认事项

1. 已完成按日期隔离的划分，但相邻多日是否属于同一天气过程尚无事件级标注；
2. 尚未从数据生成代码确认 GR–DPR 最大时间匹配误差；
3. 尚未确认多雷达重叠区域的合并权重，以及均值是在 dBZ 空间还是线性反射率空间计算；
4. 尚未确认 `dbz_gr_interp` 的具体插值算法、搜索半径和边界处理；
5. 尚未完成统一的定量基线评估和学习模型实验；
6. 当前权重读取器针对仓库提供的 `ZRH_37refine.pth` 固定结构，不用于加载任意 PyTorch checkpoint。

## 8. 下一阶段计划

1. 建立统一的NC变量读取、mask应用、切块和归一化数据管线；
2. 建立面向PyTorch的阶段一 DPR反射率–降水率数据集，且只用训练集计算统计量；
3. 建立稀疏 Z–R、插值 Z–R 和简单稠密网络等可复现基线；
4. 同时采用总体误差、正降水误差、分阈值指标和分降水类型指标；
5. 研究长尾处理方法，例如分层采样、强降水加权和兼顾连续值与降水发生的联合目标；
6. 在基线充分验证后，再比较显式掩码、部分卷积、稀疏卷积或其他稀疏到稠密模型路线。

## 9. 参考资料

- [数据集结构与物理意义](./数据集说明.md)
- [单个NetCDF样本全变量与数值分析](./NC样本变量与数值分析.md)
- [ZRH批量转换运行说明](./运行ZRH转换脚本.md)
- [2km水平图与垂直剖面运行说明](./运行2km_ZRH绘图脚本.md)
- [单样本全变量诊断绘图运行说明](./运行NC单样本诊断绘图.md)
- [降水反演任务拆解与实验路线](./降水反演任务拆解与实验路线.md)
- `variables_schema-段晨阳.docx`：原始变量说明
- `20260408雷达降水廓线反演-20260818.pptx`：项目背景与任务介绍
