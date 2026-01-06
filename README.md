## 目录结构（核心）
- `data_process/`：原始数据放在 `data_process/origin/<Dataset>` 下，运行对应生成脚本产出统一格式数据
- `generated_datasets/<dataset_name>/datasets.h5`：数据生成后的位置（默认在仓库根目录）
- `configs`：统一配置文件存储位置
- `run_models.py`：主入口，读取配置、训练、验证、测试、保存模型与结果
- `unified_dataloader.py`：统一数据加载器；训练集在加载时按给定比例随机掩码，验证/测试使用预生成的 holdout 掩码
- `modeling/` 与 `baselines/`：模型实现与基线
- `utils.py`：日志、早停、模型保存/恢复、评估指标等工具

## 数据与结果路径说明
- 数据集处理后的存储位置：`generated_datasets/<dataset_name>/datasets.h5`
- 训练权重保存位置：`NIPS_results/<model_type>_<dataset_name>/models/<时间戳>/checkpoints.ckpt`

##### 数据准备
进入data_process目录，运行数据生成文件，生成数据文件会存储在根目录generated_datasets下：
```bash
cd data_process
```
生成文件的参数主要包含：
- `--file_path`: 原始数据集存储位置
- `--artificial_missing_rate`: 人工设置缺失值比例
- `--seq_len`: 时间序列长度
- `--dataset_name`: 数据集名称
- `--saving_path`: 生成数据文件存储位置，默认值为generated_datasets

例如处理ETTh1数据集：
```bash
python generate_ETT_hour_dataset.py --dataset_name ETTh1 --file_path ./origin/ETT-small/ETTh1.csv  --artificial_missing_rate 0.1
```

生成后：
- 数据会保存到 `generated_datasets/ETTh1_seq_len24_rate0.1/datasets.h5`
- 配置文件中 `[dataset]` 的 `dataset_name` 需设为 `ETTh1_seq_len24_rate0.1`，`seq_len` 需与生成脚本一致（此处为 24）
- `feature_num` 请与转换后数据的特征维度一致（ETTh1 示例通常为 7）

##### 运行
进入根目录，运行run_models.py文件，运行参数主要包含：
- `--config_path`: 配置文件路径
- `--test_mode`: 是否测试模式，默认值为False

训练：
```bash
python run_models.py --config_path configs/test.ini
```

测试：
将检查点位置添加到配置文件中的[test]模块下的model_path 参数中
```bash
python run_models.py --config_path configs/test.ini --test_mode
```
