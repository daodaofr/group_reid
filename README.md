# group_reid

## Installation
We use python 3.7 and pytorch=1.0.1 torchvision=0.2.1

## Data preparation
All experiments are done on CSG. Please download CUHK-SYSU.
Original dataset webpage: [CUHK-SYSU Person Search Dataset](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) or [JDE dataset zoo](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)


### Usage
To train the model, please run

    sh run_hypergraphsage_part.sh

### Performance
Normaly the model achieves 85.8%  mAP and 89.5% rank-1 accuracy. According to my training log, the best model achieves 86.2% mAP and 90.0% top-1 accuracy. This may need adjustion in hyperparameters.

### Acknowledgements
Our code is developed based on Video-Person-ReID (https://github.com/jiyanggao/Video-Person-ReID). 
