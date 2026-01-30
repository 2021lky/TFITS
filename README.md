# TFITS: Time Series Imputation via Dual-Perspective Fusion of Temporal and Feature Views
Manuscript ID: IEEE LATAM Submission ID: 9732

Authors:
- Junfeng Yuan
- Kangyan Li
- Baofu Wu
- Jian Wan
- Jilin Zhang
- Yuyu Yin
- Yan Zeng

Affiliation:
- Hangzhou Dianzi University

## Description
#### Directory Structure
- `data_process/`：Raw data should be placed under `data_process/origin/<Dataset>`, Some datasets are provided as zip archives.
- `generated_datasets/<dataset_name>/datasets.h5`：Location of processed data (default is at the repository root).
- `configs`：Stores unified model configuration files.
- `run_models.py`：Main entry point for reading configurations, training, validation, testing, and saving models and results.
- `unified_dataloader.py`：Unified data loader. The training set is randomly masked at a given ratio during loading, while validation/testing use pre-generated masks.
- `modeling/` 与 `baselines/`：Implementations of models and baselines.
- `utils.py`：Utility functions for logging, early stopping, model saving/loading, evaluation metrics, etc.

#### Data and Result Paths
- Processed dataset location: `generated_datasets/<dataset_name>/datasets.h5`
- Saved model weights: `NIPS_results/<model_type>_<dataset_name>/models/<时间戳>/checkpoints.ckpt`

## Execution
#### Environment Setup
Create a new conda environment:
```bash
conda create --name TFITS python=3.9
```
Activate the conda environment:
```bash
conda activate IFITS
```
Install required dependencies:
```bash
pip install -r requirements.txt
```
#### Data Preparation
- Navigate to the `data_process` directory.
- Enter the `origin` folder and extract the dataset archives.
- Run the data generation script. The generated data files will be stored under `generated_datasets/` in the root directory:
```bash
cd data_process
```
Key command parameters include:
- `--file_path`: Path to the raw dataset.
- `--artificial_missing_rate`: Artificially set missing rate.
- `--seq_len`: Time series length (default values are pre-configured).
- `--dataset_name`: Name of the dataset.
- `--saving_path`: Storage path for generated data files (default: `generated_datasets`).

Example for processing the ETTh1 dataset:
```bash
python generate_ETT_hour_dataset.py --dataset_name ETTh1 --file_path ./origin/ETT-small/ETTh1.csv  --artificial_missing_rate 0.1
```

After generation:
- Data will be saved to  `generated_datasets/ETTh1_seq_len24_rate0.1/datasets.h5`
- In the configuration file, the `[dataset]` section should set `dataset_name` to `ETTh1_seq_len24_rate0.1`, and`seq_len` must match the generation script(here: 24)
- `feature_num` should match the feature dimension of the transformed data (e.g., typically 7 for ETTh1).

#### Running
Navigate to the root directory and run `run_models.py`. Key parameters include:
- `--config_path`: Path to the configuration file.
- `--test_mode`: Whether to run in test mode (default: False).

Training:
```bash
python run_models.py --config_path configs/test.ini
```

Testing:
Add the checkpoint path to the model_path parameter under the `[test]` section in the configuration file.
```bash
python run_models.py --config_path configs/test.ini --test_mode
```
