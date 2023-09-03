# Revealing the Dark Side of Non-Local Attention in Single Image Super-Resolution
This project is for DTFN introduced in the following paper "Revealing the Dark Side of Non-Local Attention in Single Image Super-Resolution".

The code is test on Ubuntu 16.04 environment (Python3.6, PyTorch >= 1.1.0) with Nvidia 3090 GPUs. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)

## Introduction

Single Image Super-Resolution (SISR) aims to reconstruct a high-resolution image from its corresponding low-resolution input. A common technique to enhance the reconstruction quality is Non-Local Attention (NLA), which leverages self-similar texture patterns in images. However, we reveal a novel finding that contradicts the prevailing wisdom: NLA can sometimes harm rather than help SISR, and even produce distorted textures. For example, when dealing with severely degrade textures, NLA may generate unrealistic textures due to the inconsistency of non-local texture patterns. This problem is overlooked by existing works, which only measure the average reconstruction quality of the whole image, without considering the potential risks of using NLA. To address this issue, we propose a novel perspective for evaluating the reconstruction quality of NLA, by focusing on the sub-pixel level that matches the pixel-wise fusion manner of NLA. From this perspective, we provide the approximate reconstruction performance upper bound of NLA, which guides us to design a concise yet effective Texture-Fidelity Strategy (TFS) to mitigate the degradation caused by NLA. Moreover, the proposed TFS can be conveniently integrated into existing NLA-based SISR models as a general building block. Based on the TFS, we develop a Deep Texture-Fidelity Network (DTFN), which achieves state-of-the-art performance for SISR.

## Train
### Prepare training data 

1. Download the training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Unzip the training data into the folder '../SrTrainingData'.

3. Place Set5 dataset in '../SrTrainingData/SrBenchmark'.
   
4. Specify '--dir_data' based on the HR and LR images path. 

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. Cd to 'src', run the following script to train models.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # Example X2 SR
    python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 2 --rgb_range 1 --chunk_size 128 --n_hashes 3 --save_models --lr 1e-4 --decay 300-600-900-1200 --epochs 1500 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 256 --reduction 4 --res_scale 0.1 --batch_size 16 --model DTFN --scale 2 --patch_size 96 --save DTFN_x2 --data_train DIV2K

    # Example X3 SR
    python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 2 --rgb_range 1 --chunk_size 128 --n_hashes 3 --save_models --lr 1e-4 --decay 50-100-150-200 --epochs 200 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 256 --reduction 4 --res_scale 0.1 --batch_size 16 --pre_train ../experiment/DTFN_x2/model/DTFN_x2.pt --model DTFN --scale 3 --patch_size 144 --save DTFN_x3 --data_train DIV2K 

    # Example X4 SR
    python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 2 --rgb_range 1 --chunk_size 128 --n_hashes 3 --save_models --lr 1e-4 --decay 50-100-150-200 --epochs 200 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 256 --reduction 4 --res_scale 0.1 --batch_size 16 --pre_train ../experiment/DTFN_x2/model/DTFN_x2.pt --model DTFN --scale 4 --patch_size 192 --save DTFN_x4 --data_train DIV2K 
    ```

## Test
### Quick start
1. Download the pre-trained DTFN with scale factor 3 from [BaiduYun](https://pan.baidu.com/s/1CB876XTZuD4qqtM706nOEA?pwd=9s8h) or [GoogleDrive](https://drive.google.com/file/d/1sZLrWuzTi1U28HwNXdXUr6SJzzoKGzMV/view?usp=sharing) and place it in '/experiment/DTFN_x3/model'.
2. Place the original test sets (e.g., Set5, other test sets are available from [GoogleDrive](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo) ) in '../SrBenchmark'.
3. Cd to 'src', run the following scripts.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # Test on Set5, Set14, B100, Urban100, Manga109 datasets.
    # Example X3 SR
    python main.py --dir_data ../../ --data_test Set5+Set14+B100+Urban100+Manga109 --n_GPUs 2 --rgb_range 1 --save_models --save_results --n_resgroups 10 --n_resblocks 4 --n_feats 256 --res_scale 0.1 --model DTFN --pre_train ../experiment/DTFN_x3/model/DTFN_x3.pt --save Temp --data_range 1-800/1-5 --scale 3 --test_only --reduction 4 --chunk_size 128 --n_hashes 4 --chop
    ```

## Results
Visual results on Urban100 and Manga109 with scale factor 4 are available from [BaiduYun](https://pan.baidu.com/s/1ssfR3NWm-CwrRvWc9B8Cdw?pwd=f07o) or [GoogleDrive](https://drive.google.com/file/d/1aXt4Y2h9JrNldSW0lMgaj9-A8LI0ihJd/view?usp=sharing).

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.