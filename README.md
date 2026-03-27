# 风浪异常识别基线项目

本项目面向西太平洋海区的风浪异常识别任务，使用风场、波浪场和 IBTrACS 台风轨迹数据，构建了一个从台风影响打标、数据预处理、样本索引构建、模型训练、推理到评估的完整流程。

当前默认任务是二分类分割：
- 输入：一段历史风场与波浪场序列
- 输出：目标时刻每个网格点是否受台风影响
- 默认标签：基于 IBTrACS `usa_r34` 风圈生成
- 训练用标签：`typhoon_affected_soft`
- 评估用标签：`typhoon_affected`

## 项目概览

项目主要流程如下：
1. 根据 IBTrACS 轨迹和 `usa_r34` 风圈生成台风影响标签。
2. 将原始月度 NetCDF 数据预处理为年度缓存文件。
3. 构建训练、测试、验证所需的滑动时间窗索引。
4. 训练双分支 ConvLSTM U-Net 模型。
5. 对指定年份进行推理。
6. 输出像素级和目标级评估指标。

## 目录结构

```text
.
|-- data/
|   |-- YYYY/YYYYMM/
|   |   |-- data_stream-oper_stepType-instant.nc
|   |   `-- data_stream-wave_stepType-instant.nc
|   `-- IBTrACS.WP.v04r01.nc
|-- labels_r34/
|-- outputs/
|   |-- cache/
|   |-- index/
|   |-- predictions/
|   |-- reports/
|   `-- train/
|-- scripts/
|   `-- build_typhoon_r34_labels.py
|-- src/wave_anomaly/
|-- preprocess.py
|-- build_index.py
|-- train.py
|-- predict.py
|-- evaluate.py
`-- configs/default.yaml
```

## 数据说明

### 原始输入数据

项目默认读取 `data/YYYY/YYYYMM/` 下的月度数据：
- 风场文件：`data_stream-oper_stepType-instant.nc`
- 波浪文件：`data_stream-wave_stepType-instant.nc`

默认研究海区：
- 纬度：`0-60N`
- 经度：`100-180E`

### IBTrACS 台风轨迹文件

`data/IBTrACS.WP.v04r01.nc` 不是规则的经纬度网格，而是“台风列表 + 每条台风轨迹点”的结构。主要字段包括：
- `storm`：台风编号索引
- `date_time`：该台风沿时间展开的轨迹点索引
- `lat/lon`：每个时刻的台风中心位置
- `iso_time`：时间字符串
- `usa_r34`
- `usa_roci`
- `usa_rmw`

标签脚本默认使用 `usa_r34` 四象限风圈，并同时生成：
- 二值标签：按象限半径判断网格点是否受台风影响
- 软标签：按高斯形式 `exp(-d^2 / (2 sigma^2))` 生成，默认让 `r34` 边界对应软标签值 `0.5`

## 模型说明

模型实现在 [model.py](d:/Competition/CSSOIE2026/Wave-Anomaly-Detection/src/wave_anomaly/model.py)。

当前模型为双分支 `ConvLSTM U-Net`：
- 风场分支编码历史风场特征
- 波浪分支编码历史波浪特征
- 每个尺度使用 ConvLSTM 建模时间信息
- 两个分支的特征通过 `concat + 1x1 conv` 融合
- 最终输出目标时刻的单通道概率图

默认输入通道：
- 风场：`u10`、`v10`、`ws`
- 波浪：`mwd_sin`、`mwd_cos`、`mwp`、`swh`

默认损失函数：
- `BCEWithLogits + Dice`
- 正样本权重：`pos_weight = 8.0`
- 默认损失权重：`bce_weight = 1.0`，`dice_weight = 1.0`
- 无效标签区域通过 `loss_mask` 屏蔽，不参与损失计算
- 训练默认使用软标签，评估和最终汇报仍使用二值标签

## 环境安装

推荐 Python 版本：`3.10+`

安装依赖：

```bash
pip install -r requirements.txt
```

如果本地已经有能读取 NetCDF 的 conda 环境，也可以直接使用。当前项目开发时使用过名为 `ocean-swinlstm` 的环境。

如果你希望把离线日志之后再同步到 Weights & Biases (`wandb`)，可以先登录：

```bash
wandb login
```

## 完整使用流程

### 1. 根据 IBTrACS 生成台风标签

在 `oper` 网格上生成 2016-2024 年的年度标签：

```bash
python scripts/build_typhoon_r34_labels.py --start-year 2016 --end-year 2025 --grid oper --output-dir labels_r34
```

如果还需要 `wave` 网格版本：

```bash
python scripts/build_typhoon_r34_labels.py --start-year 2016 --end-year 2025 --grid wave --output-dir labels_r34
```

输出文件形式：
- `labels_r34/oper/typhoon_r34_mask_oper_YYYY.nc`
- `labels_r34/wave/typhoon_r34_mask_wave_YYYY.nc`

### 2. 预处理年度缓存

```bash
python preprocess.py --config configs/default.yaml
```

预处理阶段会执行以下操作：
- 统一时间、经纬度和变量名
- 裁剪到目标海区
- 将波浪场插值到风场网格
- 构造额外特征，如风速、波向正余弦
- 填补缺失值
- 将标签对齐到处理后的目标网格
- 输出训练缓存到 `outputs/cache/aligned_YYYY/`
- 统计训练集归一化参数并保存到 `outputs/cache/stats.json`



### 3. 构建滑动时间窗索引

```bash
python build_index.py --config configs/default.yaml
```

输出：
- `outputs/index/train_index.csv`
- `outputs/index/test_index.csv`
- `outputs/index/val_index.csv`
- `outputs/index/all_index.csv`

### 4. 训练模型

```bash
python train.py --config configs/default.yaml
```

当前默认配置已经开启 `wandb`，可以直接使用脚本启动训练：

```bash
bash scripts/train_wandb.sh
```

如果想指定其他配置文件：

```bash
bash scripts/train_wandb.sh configs/default.yaml
```

训练阶段会输出：
- `outputs/train/best.ckpt`
- `outputs/train/last.ckpt`
- `outputs/train/history.csv`
- `outputs/train/test_threshold_scan.csv`
- `outputs/train/summary.json`

### 5. 按年份推理

例如对 2024 年进行推理：

```bash
python predict.py --config configs/default.yaml --checkpoint outputs/train/best.ckpt --year 2024
```

输出：
- `outputs/predictions/prediction_2024.nc`
- `outputs/predictions/prediction_2024_summary.csv`

### 6. 评估结果

评估测试集：

```bash
python evaluate.py --config configs/default.yaml --split test
```

评估验证集：

```bash
python evaluate.py --config configs/default.yaml --split val
```

评估结果保存在 `outputs/reports/`，包括：
- 各年份指标文件
- 汇总指标文件
- PR 曲线
- 阈值扫描曲线

## 默认配置

默认配置文件为 [default.yaml](d:/Competition/CSSOIE2026/Wave-Anomaly-Detection/configs/default.yaml)。

当前主要默认值：
- 处理网格：`oper`
- 标签目录：`labels_r34/oper`
- 训练年份：`2016-2023`
- 测试年份：`2024`
- 验证年份：`2025`
- 历史长度：`8`
- 预测偏移：`0`
- 批大小：`1`
- 训练轮数：`50`
- 学习率：`3e-4`
- 损失类型：`bce_dice`

`wandb` 相关默认值：
- `wandb.enabled: true`
- `wandb.project: wave-anomaly-detection`
- `wandb.mode: offline`

如果要自定义实验名，建议至少修改：

```yaml
wandb:
  enabled: true
  project: wave-anomaly-detection
  run_name: exp001
```

如果本地没有 `2025` 数据，建议先修改 `val_years`，再运行完整流程。

## 输出文件格式

### 年度缓存文件

`outputs/cache/aligned_YYYY/` 目录包含：
- `wind.npy`
- `wave.npy`
- `label.npy`
- `quality_mask.npy`
- `time.npy`
- `latitude.npy`
- `longitude.npy`
- `meta.json`

### 推理结果文件

`outputs/predictions/prediction_YYYY.nc` 包含：
- `probability(time, latitude, longitude)`
- `binary_prediction(time, latitude, longitude)`
- `label(time, latitude, longitude)`

## 评估指标

像素级指标：
- `precision`
- `recall`
- `f1`
- `iou`
- `dice`
- `csi`
- `pod`
- `far`
- `accuracy`
- `pr_auc`

目标级指标：
- `object_csi`
- `object_pod`
- `object_far`

目标级评估基于连通域分析，受以下参数控制：
- `min_area`
- `connectivity`

## 测试

运行测试：

```bash
pytest
```

当前包含的测试文件：
- `tests/test_model.py`
- `tests/test_metrics.py`
- `tests/test_preprocessing.py`

## 备注

- 当前数据中的 IBTrACS `time` 字段可能带有异常纳秒偏移，打标时建议优先使用 `iso_time`。
- 标签值 `-1` 表示该区域无有效标签，训练时会被 `loss_mask` 忽略。
- 默认训练流程使用 `oper` 网格作为统一目标网格，波浪场会先插值到该分辨率上。
