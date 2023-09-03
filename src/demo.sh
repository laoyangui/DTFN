#!/bin/bash

#Training on scale factor 2, 3 and 4.
python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 1 --rgb_range 1 --chunk_size 128 --n_hashes 3 --save_models --lr 1e-4 --decay 300-600-900-1200 --epochs 1500 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 256 --reduction 4 --res_scale 0.1 --batch_size 16 --model DTFN --scale 2 --patch_size 96 --save DTFN_x2 --data_train DIV2K
python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 1 --rgb_range 1 --chunk_size 128 --n_hashes 3 --save_models --lr 1e-4 --decay 50-100-150-200 --epochs 200 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 256 --reduction 4 --res_scale 0.1 --batch_size 16 --pre_train ../experiment/DTFN_x2/model/DTFN_x2.pt --model DTFN --scale 3 --patch_size 144 --save DTFN_x3 --data_train DIV2K
python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 1 --rgb_range 1 --chunk_size 128 --n_hashes 3 --save_models --lr 1e-4 --decay 50-100-150-200 --epochs 200 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 256 --reduction 4 --res_scale 0.1 --batch_size 16 --pre_train ../experiment/DTFN_x2/model/DTFN_x2.pt --model DTFN --scale 4 --patch_size 192 --save DTFN_x4 --data_train DIV2K

# Testing on Set5, Set14, B100, Urban100, Manga109 datasets with scale factor 3.
python main.py --dir_data ../../ --data_test Set5+Set14+B100+Urban100+Manga109 --n_GPUs 1 --rgb_range 1 --save_models --save_results --n_resgroups 10 --n_resblocks 4 --n_feats 256 --res_scale 0.1 --model DTFN --pre_train ../experiment/DTFN_x3/model/DTFN_x3.pt --save DTFN_x3_results --data_range 1-800/1-5 --scale 3 --test_only --reduction 4 --chunk_size 128 --n_hashes 4 --chop
