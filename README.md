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
We provide two datasets for author citation prediction. The one is to predict author citations is _2016_ and another is to predict author citations in _2022_. The two datasets contain different authors. The datasets can be downloaded from [BaiduPan](https://pan.baidu.com/s/1O4Jr2NWGKLelnhQBjL50Zw?pwd=g5uk) with password g5uk or [data-2016](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/influence-prediction/author-influence-prediction/data/2016.zip) & [data-2022](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/influence-prediction/author-influence-prediction/data/2022.zip). Please put the _data_ folder into the project directory.

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
| ARIMA  | 1225 | 23920 |
| LR  | 562 | 22057 |
| GBRT | 553 | 21777 |
| LSTM | 1034 | 25409 |
| EvolveGCN | 969 | 22841 |

## References
🌟 If you find our work helpful, please leave us a star and cite our paper.
```
@inproceedings{zhang2024oag,
  title={OAG-bench: a human-curated benchmark for academic graph mining},
  author={Fanjin Zhang and Shijie Shi and Yifan Zhu and Bo Chen and Yukuo Cen and Jifan Yu and Yelin Chen and Lulu Wang and Qingfei Zhao and Yuqing Cheng and Tianyi Han and Yuwei An and Dan Zhang and Weng Lam Tam and Kun Cao and Yunhe Pang and Xinyu Guan and Huihui Yuan and Jian Song and Xiaoyan Li and Yuxiao Dong and Jie Tang},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6214--6225},
  year={2024}
}
```
