# citation-prediction

## Prerequisites
- Linux
- Python 3.7
- PyTorch 1.10.0+cu111

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/citation-prediction.git
cd citation-prediction
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

## Dataset
We provide two datasets for author citation prediction. The one is to predict author citations is _2016_ and another is to predict author citations in _2022_. The two datasets contain different authors. The datasets can be downloaded from [BaiduPan](https://pan.baidu.com/s/1O4Jr2NWGKLelnhQBjL50Zw?pwd=g5uk) with password g5uk. Please put the _data_ folder into the project directory.

## How to run
```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used

# processing: set pred_year = 2016/2022 in process.py
python process.py   

# ARIMA: set pred_year = 2016/2022 below __main__ function
python arima.py

# regressor: set pred_year = 2016/2022 below __main__ function
python regressor.py

# LSTM: set pred_year = 2016/2022 below __main__ function
python lstm.py

# EvolveGCN
cd evolvegcn
python run_exp_inf.py --config_file ./experiments/parameters_inf_2016.yaml
python run_exp_inf.py --config_file ./experiments/parameters_inf_2022.yaml
```

### Results 

Evaluation metrics: RSME

|       | 2016 | 2022 |
|-------|-------|-----|
| ARIMA  |  |  |
