# `zrh_nc_to_rain.py` 运行说明

## 已配置的默认路径

- 输入数据：`/storage/GR_DPR_3D/GRToDPRRes_V07_Pct_V1.2.1_sw_260412`
- Z–R 关系权重：项目目录下的 `ZRH_37refine.pth`
- 输出数据：项目目录下的 `outputs/zrh_GRToDPRRes_V07_Pct_V1.2.1_sw_260412`

输入目录和权重路径已经存在。原始数据挂载在 `/storage`，脚本只读原始文件，结果写入项目目录。

## 激活环境并运行

每次新开终端后执行：

```bash
cd /home/koujizhi/projects/precipitation-inversion
source .venv/bin/activate
```

看到终端提示符前出现 `(.venv)` 后，可以查看参数：

```bash
python zrh_nc_to_rain.py --help
```

使用全部默认配置运行：

```bash
python zrh_nc_to_rain.py
```

默认会处理目录中的全部 `.nc` 文件，并为文件中存在的以下五个反射率变量生成对应的 ZRH 降水率：

- `dbz_gr_sparse`
- `dbz_gr_sparse_min`
- `dbz_gr_sparse_max`
- `dbz_gr_interp`
- `dbz_dpr`

输出变量名形如 `rain_rate_zrh_gr_sparse`，单位为 `mm/h`。已有输出默认跳过；只有明确添加 `--overwrite` 才会覆盖。

例如，只转换稀疏地基雷达反射率，并减小每次处理的扫描数：

```bash
python zrh_nc_to_rain.py \
  --variables dbz_gr_sparse \
  --chunk-scans 64
```

## 环境说明

项目虚拟环境位于 `.venv`，当前使用：

- Python 3.10.12
- NumPy 1.26.4
- netCDF4 1.6.2

脚本已改用 NumPy 完成 `R = exp(dBZ × weight[z] + bias[z])`。提供的 `.pth` 文件仅包含两个长度为 60 的 float64 数组，因此不再需要安装约 192 MB 的 PyTorch CPU 运行时。

若以后需要重建环境，可执行：

```bash
cd /home/koujizhi/projects/precipitation-inversion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-zrh.txt
```

## 运行前注意

当前输入目录包含 254 个 NetCDF 文件，原始数据总量约 23 GB。完整批处理会运行较长时间并产生大量输出；正式运行前可先确认项目所在磁盘的剩余空间。脚本按扫描块处理数组，不会一次把完整数据集读入内存。

运行中若异常退出，尚未完成的文件会使用 `.partial` 后缀；脚本通常会自动清理该临时文件。若提示已有 `.partial` 文件，应先检查它是否来自已经终止的旧进程，再手动处理。
