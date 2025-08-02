# HCSeer2
A Deep Learning-Based Multi-Scale Modeling Framework for Predicting Cold and Hot Spots of Variants in the Human Exome



## Prerequisites
To run this project, you need the following prerequisites:
- Python 3.9
- PyTorch 1.13.1+cu117
- Other required Python libraries (please refer to requirements.txt)

You can install all the required packages using the following command:
```
conda create -n pytorch python=3.9.16
conda activate pytorch
```
```python
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
```python
pip install -r requirements.txt 
```

## Model Training
To train the Translatomer model, use the following command:
```
python HCSeer2_train_predict.py [options]

[options]:
- --run_type  run type. Default = 'train'.
- --seed  Random seed for training. Default value: 42.
- --save_path  Path to the model checkpoint. Default = 'checkpoints'.
- --assembly  Genome assembly for training data. Default = 'hg38'.
- --model-type  Type of the model to use for training. Default = 'TransModel'.
```
Example to run the codes:
```
nohup python train_all_11fold.py --run_type predict --sequence_data "E:\冷热点预 测课题_E盘分部\预测原始数据\predict_data_chrY_seq.txt" --input_feature_data "E:\冷热点预测课题_E盘分部\预测原始数据\predict_data_chrY_feature.txt" --checkpoint "E:\Python项目文件夹\Translatomer\lightning_logs\version_178\checkpoints\epoch=31-step=62176.ckpt" --predict_result_path "E:\冷热点预测课题_E盘分部\predict_result\predict_result_chrY.txt" 
```

