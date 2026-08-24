# `plot_nc_sample_diagnostics.py` 运行说明

该脚本把 GR–DPR NetCDF 样本的变量统计和物理分析转换为图表。它默认按文件名排序处理数据集前20个样本，也可以通过 `--count` 指定数量，或通过 `--input-file` 只分析一个文件。脚本逐文件读取并释放内存，不修改源数据。

## 默认输入与输出

默认输入目录：

```text
/storage/GR_DPR_3D/GRToDPRRes_V07_Pct_V1.2.1_sw_260412
```

默认取按文件名排序后的前20个 `.nc` 文件。

默认输出根目录：

```text
/home/koujizhi/projects/precipitation-inversion/outputs/nc_sample_diagnostics
```

脚本会在输出根目录下按每个输入文件名创建独立子目录。已有完整结果默认跳过，添加 `--overwrite` 才会重新生成。若目录来自异常中断且没有完成标记，重新运行时会自动重建该样本。

## 运行

```bash
cd /home/koujizhi/projects/precipitation-inversion
source .venv/bin/activate
python plot_nc_sample_diagnostics.py
```

处理前10个样本：

```bash
python plot_nc_sample_diagnostics.py --count 10
```

重新生成已有结果：

```bash
python plot_nc_sample_diagnostics.py --overwrite
```

只分析指定样本（此时 `--count` 不生效）：

```bash
python plot_nc_sample_diagnostics.py \
  --input-file /storage/GR_DPR_3D/GRToDPRRes_V07_Pct_V1.2.1_sw_260412/example.nc \
  --height-km 2.0
```

只生成指定专题：

```bash
python plot_nc_sample_diagnostics.py \
  --count 2 \
  --sections overview gr_interp gr_dpr rain_tail \
  --output-dir outputs/selected_diagnostics
```

可选专题名称：

```text
overview variables gr_interp gr_dpr strong_window
cfb precip_types rain_tail zrh
```

查看全部参数：

```bash
python plot_nc_sample_diagnostics.py --help
```

批处理默认会在单个样本失败后继续处理后续文件，并在最后汇总失败项。如需调试时立即停止：

```bash
python plot_nc_sample_diagnostics.py --count 20 --fail-fast
```

## 输出内容

```text
<样本名>/
├── 00_dataset_overview.png
├── variables/                  # 23个变量各一张诊断图
├── comparisons/                # 7张专题对比图
├── statistics.csv              # 作图所依据的变量统计
├── figure_manifest.md          # 图表说明和相对路径预览
└── diagnostics.pdf             # 全部31张图的多页PDF
```

逐变量图根据维度自动选择画法：

- 一维变量：实际值折线、直方图、有效率和数值摘要；
- 二维变量：卫星条带空间图、分布、扫描行实际值和摘要；
- 三维变量：1.875 km水平图、A–B垂直剖面、分布和垂直有效率/分位数。

专题图包括：

1. 稀疏GR与插值GR；
2. GR与DPR反射率；
3. 强降水中心7×7实际数值窗口；
4. `cfb`杂波底；
5. 层云、对流、其他和无降水类型；
6. `pre_dpr`零膨胀和强降水尾部；
7. 稀疏/插值ZRH与DPR参考降水率。

## 已完成的真实样本测试

- 完整生成31张PNG，23张逐变量图和8张总览/专题图；
- PNG全部通过Pillow完整性检查；
- `diagnostics.pdf`包含31页；
- 输出总量约8.6 MB；
- `statistics.csv`与已有分析报告中的关键数值一致；
- 实际水平层为最接近2 km的 `z[7]=1.875 km`；
- A–B剖面选择第351扫描行。

单个样本完整运行约需1分钟；批量运行时间近似随 `--count` 线性增加，具体取决于服务器负载和输出分辨率。
