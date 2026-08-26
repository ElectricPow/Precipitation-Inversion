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
│   ├── make_dataset_splits.py             # 按日期分组的无泄漏数据划分
│   ├── fit_normalization_stats.py         # 仅用训练集拟合分高度归一化量
│   ├── build_stage1_sample_index.py       # 构建阶段一正降水体素索引
│   ├── build_stage1_patch_index.py        # 构建3D核心+上下文窗口索引
│   ├── check_distributed_runtime.py       # 低显存NCCL/DDP通信自检
│   ├── launch_stage1_ddp.sh               # 共享服务器安全的单机多卡入口
│   ├── train_stage1_unet3d.py             # 单卡/DDP训练、验证和checkpoint
│   ├── plot_stage1_training_history.py    # 逐epoch曲线、分箱及泛化分析
│   ├── visualize_stage1_test_predictions.py # best.pt固定测试轨道诊断
│   └── evaluate_stage1_unet3d.py          # Patch指标与完整轨道评估
├── configs/
│   └── stage1_unet3d.yaml                 # 第一版模型和训练参数
├── src/precipitation_inversion/data/
│   ├── masks.py                           # GR/DPR/降水/cfb统一mask规则
│   ├── splits.py                          # 分组平衡划分核心逻辑
│   ├── nc_reader.py                       # 按变量/扫描区间读取NC并派生mask
│   ├── transforms.py                      # 在线统计、标准化与降水可逆变换
│   ├── dataset.py                         # 紧凑索引及PyTorch Dataset
│   ├── samplers.py                        # 文件/块级洗牌及DDP批次分片
│   └── patch_dataset.py                   # 3D U-Net核心+halo Patch Dataset
├── src/precipitation_inversion/inference/
│   └── sliding_window.py                  # 中心核心裁剪与完整轨道重建
├── src/precipitation_inversion/models/
│   ├── blocks3d.py                        # 保持高度的各向异性3D残差块
│   └── unet3d.py                          # 阶段一高度保持3D U-Net
├── src/precipitation_inversion/losses/
│   └── masked_losses.py                   # mask内Smooth L1/MSE/MAE
├── src/precipitation_inversion/metrics/
│   └── regression.py                      # 流式log/物理空间回归指标
├── src/precipitation_inversion/training/
│   └── engine.py                          # AMP/DDP epoch循环和checkpoint
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
│   └── stage1_positive_qc.json           # 阶段一训练集分高度统计量
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

默认仅读取 `train_files.txt` 中的175个文件，并与 `split_manifest.csv` 交叉检查，遇到验证或测试文件会直接报错：

```bash
python scripts/fit_normalization_stats.py --overwrite
```

默认在 `pre_positive_qc` 网格上按60个高度层分别统计 `dbz_dpr/p/t/q` 的数量、均值、标准差和范围。调试时可限制文件数：

```bash
python scripts/fit_normalization_stats.py \
  --max-files 2 \
  --output /tmp/stage1-normalization-smoke.json \
  --overwrite
```

正式结果保存在 `metadata/normalization/stage1_positive_qc.json`。统计共使用5,481,557个训练网格；由于 `cfb` 近地面杂波质控，0.125 km和0.375 km两层没有可训练正降水样本，其统计量以 `null` 保存。

### 6.8 构建并读取阶段一体素索引

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

设置 `batch_sampler` 后不能再同时设置DataLoader的 `batch_size/shuffle/drop_last`。采样器支持 `num_replicas/rank`；若DDP进程组已初始化，则会自动读取world size和当前rank。默认 `even_batches=True`，会舍弃至多 `world_size-1` 个全局batch，使所有rank执行相同步数，避免同步训练挂起。

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
    "metadata/normalization/stage1_positive_qc.json",
    positive_only=True,
)
item = train_dataset[0]
print(item["inputs"].shape)     # (3,64,64,60)
print(item["target"].shape)     # (1,64,64,60)
print(item["loss_mask"].shape)  # (1,64,64,60)
```

训练时 `positive_only=True` 过滤没有正降水loss体素的核心；验证和整轨测试必须使用
`positive_only=False`，以完整覆盖原始 `nscan`。高度保持3D U-Net只在扫描和横轨
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

启动脚本要求显式声明物理GPU，防止共享服务器上误占他人正在使用的卡。例如用物理
5、6号卡做两卡训练（进程内分别映射为 `cuda:0`、`cuda:1`）：

```bash
CUDA_VISIBLE_DEVICES=5,6 \
STAGE1_MASTER_PORT=29519 \
scripts/launch_stage1_ddp.sh
```

八卡均确认空闲时才启动八卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
STAGE1_NUM_GPUS=8 \
STAGE1_MASTER_PORT=29519 \
scripts/launch_stage1_ddp.sh
```

同一服务器上的不同分布式任务必须使用不同的 `STAGE1_MASTER_PORT`。正式训练前可先
运行低显存通信自检；它检查进程绑定、NCCL collective和DDP梯度同步，不读取NC数据：

```bash
CUDA_VISIBLE_DEVICES=5,6 .venv/bin/torchrun \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29518 \
  scripts/check_distributed_runtime.py
```

`data.batch_size` 是每张GPU的batch大小，当前为1。有效全局batch为
`每卡batch × GPU数 × accumulation_steps`；两卡为2、八卡为8。第一轮实验先保持
学习率 `1e-4`，不要因GPU数自动线性放大；得到稳定曲线后再单独比较学习率。训练与
验证采样器都保证各rank步数相同，最多舍弃 `world_size-1` 个batch以避免collective
死锁；训练指标会在所有rank上聚合，checkpoint仅由rank 0写入。

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
并按目标降水率 `<1/1–5/5–10/10–30/≥30 mm/h` 分箱。默认以验证集物理空间RMSE
选择 `best.pt`，同时保存 `last.pt` 和逐epoch checkpoint。

完整训练正常结束后，只有DDP rank 0会自动执行两个后处理脚本。配置由
`postprocessing` 控制，默认固定随机种子2026，从测试集中有正降水监督的轨道里抽取
6条完整轨道。所有结果写入本次命令实际指定的output目录：

```text
outputs/<experiment>/analysis/
├── README.md
├── training_history/
│   ├── training_overview.png
│   ├── validation_intensity_bins.png
│   ├── generalization_gap.png
│   ├── epoch_metrics.csv
│   ├── summary.json
│   └── summary.md
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

已有训练也可随时手动重跑：

```bash
python scripts/plot_stage1_training_history.py outputs/stage1_unet3d_3gpu

CUDA_VISIBLE_DEVICES=7 python scripts/visualize_stage1_test_predictions.py \
  outputs/stage1_unet3d_3gpu/best.pt \
  --output-dir outputs/stage1_unet3d_3gpu/analysis/test_predictions \
  --sample-count 6 --seed 2026 --device cuda:0
```

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

## 7. 当前限制与待确认事项

1. 已完成按日期隔离的划分，但相邻多日是否属于同一天气过程尚无事件级标注；
2. 尚未从数据生成代码确认 GR–DPR 最大时间匹配误差；
3. 尚未确认多雷达重叠区域的合并权重，以及均值是在 dBZ 空间还是线性反射率空间计算；
4. 尚未确认 `dbz_gr_interp` 的具体插值算法、搜索半径和边界处理；
5. 尚未完成统一的定量基线评估和学习模型实验；
6. 当前权重读取器针对仓库提供的 `ZRH_37refine.pth` 固定结构，不用于加载任意 PyTorch checkpoint。
7. 已在物理5、6号RTX 4090 D上完成两卡NCCL collective、DDP反向传播以及真实NC
   数据/完整3D U-Net的训练与验证冒烟测试。当前受管执行环境在建立 `TCPStore` 时会
   重复报告hostname解析警告并使初始化延迟约40秒，但最终通信正确；若用户终端也出现
   长时间停顿，应检查 `/etc/hosts`、主机名解析、本地socket权限和端口冲突。
8. 尚未在八卡同时空闲时做八进程实测；代码路径与两卡相同，但正式启动前仍需确认
   所有目标GPU空闲，并先用8进程运行 `check_distributed_runtime.py`。

## 8. 下一阶段计划

1. 正式运行第一版训练并检查学习曲线、过拟合和强降水分箱指标；
2. 建立现有分层Z–R方法在相同测试mask上的定量指标并与3D U-Net比较；
3. 比较Smooth L1、MSE、MAE和训练集频率拟合的强降水加权损失；
4. 增加预测—标签平面图、垂直剖面和强降水误差诊断；
5. 第一版稳定后再加入 `p/t/q` 做消融，随后进入阶段二GR→DPR分布映射。

## 9. 参考资料

- [数据集结构与物理意义](./数据集说明.md)
- [单个NetCDF样本全变量与数值分析](./NC样本变量与数值分析.md)
- [ZRH批量转换运行说明](./运行ZRH转换脚本.md)
- [2km水平图与垂直剖面运行说明](./运行2km_ZRH绘图脚本.md)
- [单样本全变量诊断绘图运行说明](./运行NC单样本诊断绘图.md)
- [降水反演任务拆解与实验路线](./降水反演任务拆解与实验路线.md)
- [模型数据处理流程、变量与张量形状](./模型数据处理流程与张量形状.md)
- `variables_schema-段晨阳.docx`：原始变量说明
- `20260408雷达降水廓线反演-20260818.pptx`：项目背景与任务介绍
