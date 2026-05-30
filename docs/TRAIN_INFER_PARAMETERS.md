# Train And Infer Parameters

这份文档只面向当前主线：

- 训练入口：`train.py`
- 推理入口：`infer.py`
- colab普通训练配置：`configs/colab_bolt.yml`
- 普通训练配置：`configs/project.yml`
- 蒸馏训练配置：`configs/project_distill.yml`

## 配置来源

当前数据配置已经改成：

- `dataset.root`
- `dataset.class_summary_csv`
- 其他数据路径使用相对 `dataset.root` 的相对子路径

当前 `num_classes` 不需要手写。
程序会从 `dataset.class_summary_csv` 读取类别列表，并自动同步 `dataset.classes` 和 `num_classes`。

## 训练时通常要改的参数

主要改 `configs/colab_bolt.yml`、`configs/project.yml` 或 `configs/project_distill.yml`。

### 1. 数据根目录

最常改的是：

```yaml
dataset:
  root: ./datasets/ 或 `数据存放路径`
```

只要你的目录结构不变，下面这些相对子路径一般不用改：

- `class_summary_csv: manifests/class_summary.csv`
- `train_images: images`
- `train_annotations: labels`
- `train_split_csv: manifests/trainval_split_90_10/train_split.csv`
- `val_images: images`
- `val_annotations: labels`
- `val_split_csv: manifests/trainval_split_90_10/val_split.csv`

### 2. 训练输出目录

```yaml
output_dir: ./outputs/project 或 ./outputs/colab_bolt
```

蒸馏训练对应：

```yaml
output_dir: ./outputs/project_distill
```

### 3. Student 初始化权重

这是训练开始前加载到 student 上的初始化权重，不是断点续训：

```yaml
student_checkpoint: ./pretrain/RTv4-M-hgnet.pth
```

如果你想换成别的初始化权重，可以改这里。

### 4. Batch Size / Worker / 图像尺寸

显存不够时优先改这些：

```yaml
train_dataloader:
  batch_size: 2/4
  num_workers: 2/4

val_dataloader:
  batch_size: 2/4
  num_workers: 2/4

train_transforms:
  image_size: [1024, 1024] 或 image_size: [640, 640]

val_transforms:
  image_size: [1024, 1024] 或 image_size: [640, 640]
```

建议：

- 显存紧张：先降 `batch_size`
- 还不够：再降 `image_size`
- CPU 很强：再提高 `num_workers`

### 5. 训练轮数和学习率

普通训练看：

```yaml
train:
  epochs:
  optimizer:
    lr:
  scheduler: warmup_cosine
```

蒸馏训练看：

- `stages[].epochs`
- `stages[].optimizer.lr`
- `stages[].scheduler`

### 6. Polygon 样本策略

当前默认：

```yaml
dataset:
  polygon_policy: exclude_sample
```

这会跳过带 polygon 的 LabelMe 样本。
如果你想把 polygon 转成 bbox，可以改成：

```yaml
polygon_policy: convert_to_bbox
```

### 7. 蒸馏专用参数

只在 `configs/project_distill.yml` 下需要关注：

- `teacher_model.dinov3_repo_path`
- `teacher_model.dinov3_weights_path`
- `stages`
- `RTv4Criterion.weight_dict.loss_distill`

## 训练命令

普通训练：

```bash
python train.py configs/project.yml --device cuda
或
python train.py configs/colab_bolt.yml --device cuda
```

蒸馏训练：

```bash
python train.py configs/project_distill.yml --device cuda
```

如果要覆盖 worker：

```bash
python train.py configs/project_distill.yml --device cuda --num-workers 8
```

## 断点续训

当前支持断点续训。

续训时不要改 `student_checkpoint`，而是使用：

```bash
python train.py configs/project_distill.yml --device cuda --resume outputs/project_distill/latest.pt
```

`latest.pt` 会恢复：

- 模型参数
- optimizer
- scheduler
- scaler
- `stage_index`
- `stage_epoch`
- `global_epoch`

## 推理时通常要改的参数

推理主要改命令行参数，不一定需要改 YAML。

### 1. 必须指定训练后的权重

推理应该使用你训练产出的权重，例如：

- `outputs/project/best_map50.pt`
- `outputs/project/latest.pt`
- `outputs/project_distill/best_map50.pt`
- `outputs/project_distill/latest.pt`

示例：

```bash
python infer.py configs/colab_bolt.yml --checkpoint outputs/project/best_map50.pt --input /path/to/images --device cuda
```

不要把官方初始化权重直接当最终推理权重使用。

### 2. 输入路径

```bash
--input /path/to/image_or_dir
```

支持：

- 单张图片
- 图片目录

目录递归扫描时加：

```bash
--recursive
```

### 3. 推理输出目录

```bash
--output-dir outputs/project_infer
```

会输出：

- `predictions.jsonl`
- `run_summary.json`
- `failures.json`
- `visualizations/`

### 4. 设备

```bash
--device cuda
```

或：

```bash
--device cuda:0
```

### 5. 分数阈值和 topk

```bash
--score-threshold 0.25
--topk 100/300
```

### 6. 是否保存可视化

默认会保存可视化图。

如果不需要：

```bash
--no-save-vis
```

## 推理说明

- 推理只使用 student，不使用 teacher
- `infer.py` 默认会拦截类别头不匹配的 checkpoint
- 如果目录里有坏图，程序会跳过并把失败信息写入 `failures.json`

## 云端环境最常改的地方

如果切到云端 GPU，最常改的是：

1. `dataset.root`
2. `output_dir`
3. `student_checkpoint`
4. `teacher_model.dinov3_repo_path`
5. `teacher_model.dinov3_weights_path`
6. `batch_size`
7. `num_workers`
8. `--device cuda`

一般不需要手改：

- `num_classes`
- `dataset.classes`

因为它们已经由 `class_summary.csv` 自动决定。
