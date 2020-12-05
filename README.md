# group_reid

## Installation
We use python 3.7 and pytorch=1.0.1 torchvision=0.2.1

## Data preparation
All experiments are done on CSG. Please download CUHK-SYSU.

Original dataset webpage: [CUHK-SYSU Person Search Dataset](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) or [JDE dataset zoo](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)


### Usage
To train and evaluate the model, please run

    sh run_train_test.sh

### Performance
|Task|Rank1 | mAP | Link |
|-----|------|-----|------|
|Group re-id| 65.7%|65.7%| [[Google]](https://drive.google.com/file/d/1j6r4-Fu2FyfE5LHeWrTcFm3xl92t8Lnp/view?usp=sharing)|
|Person re-id|69.4%|68.5%| [[Baidu]](https://pan.baidu.com/s/1Ifgn0Y_JZE65_qSrQM2l-Q)|

### Acknowledgements
Our code is developed based on Video-Person-ReID (https://github.com/jiyanggao/Video-Person-ReID). 
