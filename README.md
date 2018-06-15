# Medical Images Classcification


This is a group assignment of Media and Cognition base on [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/). The model takes a chest X-ray image as input and outputs the probability of each disease.

## Dataset

The [ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) comprises 112,120 frontal-view chest X-ray images of 30,805 unique patients with 14 disease labels. To evaluate the model, we randomly split the dataset into training (70%), validation (10%) and test (20%) sets, following the work in paper. Partitioned image names and corresponding labels are placed under the directory [labels](./ChestX-ray14/labels).

## Prerequisites

- Python 3.5+
- [PyTorch](http://pytorch.org/) and its dependencies
- GTX 1080Ti or more 

## Usage

1. Clone this repository.

2. Download images of ChestX-ray14 from this [website](https://nihcc.app.box.com/v/ChestXray-NIHCC) and decompress them to your path.

3. Modify the `pathDirData` in `main.py`  `Multi_Classes_Main.py`  `DML_Main.py`

4. Specify one or multiple GPUs and run

   `python main.py` or `python Multi_Classes_Main.py` or `python DML_Main.py`

5. Deep Mutual Learning needs 2* GTX 1080Ti or more 

## Comparsion

`report.pdf` is our report including more details.

Comparing with the original CheXNet, the average AUROC of our model is almost the same.

|     Pathology      | [CheXNet(baseline)](https://arxiv.org/abs/1711.05225) | [Deep Mutual Learning](https://arxiv.org/abs/1706.00384) | Multi-Classes |Image preprocessing | 
| :----------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :---------------------: |
|    AUROC Mean      |                  0.810                   |                  0.811                   |                  0.8154                  |         0.7847          |
|    Atelectasis     |                  0.768                   |                  0.779                   |                  0.7723                  |         0.7509          |
|    Cardiomegaly    |                  0.881                   |                  0.874                   |                  0.8869                  |         0.8599          |
|      Effusion      |                  0.832                   |                  0.826                   |                  0.8233                  |         0.8152          |
|    Infiltration    |                  0.703                   |                  0.709                   |                  0.7020                  |         0.6889          |
|        Mass        |                  0.824                   |                  0.813                   |                  0.8277                  |         0.7909          |
|       Nodule       |                  0.753                   |                  0.765                   |                  0.7783                  |         0.7100          |
|     Pneumonia      |                  0.729                   |                  0.735                   |                  0.7216                  |         0.7142          |
|    Pneumothorax    |                  0.841                   |                  0.864                   |                  0.8565                  |         0.8467          |
|   Consolidation    |                  0.751                   |                  0.754                   |                  0.7395                  |         0.7238          |
|       Edema        |                  0.847                   |                  0.848                   |                  0.8481                  |         0.8353          |
|     Emphysema      |                  0.902                   |                  0.911                   |                  0.9245                  |         0.8545          |
|      Fibrosis      |                  0.824                   |                  0.814                   |                  0.8176                  |         0.7948          |
| Pleural Thickening |                  0.768                   |                  0.771                   |                  0.7805                  |         0.7365          |
|       Hernia       |                  0.913                   |                  0.887                   |                  0.9373                  |         0.8635          |

## Contributions

The work was finished by 618clw collaboratively. Letters c means Yifan Chen, while l represents Zheyi Li, and w stands for Boyu Wang.

## Our Group

We are students of Media and Cognition Class, Department of Electronic Engineering
