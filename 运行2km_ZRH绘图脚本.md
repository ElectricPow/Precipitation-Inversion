# `plot_2km_zrh_4files.py` 运行说明

## 默认配置

- 输入目录：`/storage/GR_DPR_3D/GRToDPRRes_V07_Pct_V1.2.1_sw_260412`
- ZRH 权重：项目目录下的 `ZRH_37refine.pth`
- 图片目录：项目目录下的 `outputs/plots_2km_zrh`
- 默认图片数量：按文件名排序后的前 4 个 NetCDF 文件

脚本直接读取原始 NetCDF 文件，并在内存中根据 `dbz_gr_sparse` 和 `dbz_gr_interp` 计算 ZRH 降水率，不会修改原始数据，也不要求提前运行 `zrh_nc_to_rain.py`。

## 激活环境

每次新开终端后执行：

```bash
cd /home/koujizhi/projects/precipitation-inversion
source .venv/bin/activate
```

## 运行方式

使用默认配置绘制前 4 个文件：

```bash
python plot_2km_zrh_4files.py
```

先只绘制第 1 个文件进行检查：

```bash
python plot_2km_zrh_4files.py --count 1
```

指定输出目录和文件数量：

```bash
python plot_2km_zrh_4files.py \
  --count 10 \
  --output-dir /home/koujizhi/projects/precipitation-inversion/outputs/my_plots
```

查看全部参数：

```bash
python plot_2km_zrh_4files.py --help
```

输出文件名形如：

```text
plot_2km_zrh_2A.GPM.DPR....V07A.png
```

脚本使用无图形界面的 `Agg` 后端，适合直接在服务器终端运行。重跑时，同名 PNG 会被覆盖。

## 环境依赖

绘图所需依赖记录在 `requirements-plot.txt`：

- NumPy 1.26.4
- netCDF4 1.6.2
- Matplotlib 3.8.4
- Matplotlib 自动安装的 Pillow 等依赖

若以后重建虚拟环境：

```bash
cd /home/koujizhi/projects/precipitation-inversion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
```

ZRH 权重使用项目中已经验证的 NumPy 读取逻辑，不需要安装 PyTorch。

## 已验证结果

使用排序后的第一个真实样本测试时：

- 使用高度层：`z[7] = 1.875 km`，即最接近 2 km 的数据层
- A–B 剖面扫描行：351
- 输出 PNG：`2256 × 2441` 像素，RGBA 格式
- PNG 可被 Pillow 正常读取并通过完整性检查
