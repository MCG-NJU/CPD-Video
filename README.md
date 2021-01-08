# Learning Spatiotemporal Features via Video and Text Pair Discrimination

An official PyTorch implementation of [CPD](https://arxiv.org/abs/2001.05691).

## Overview

In this repo, we release codes of our CPD model. It contains two parts: pre-training spatiotemporal features on uncurated web dataset Instagram-300k and finetuning pre-trained models on downstream datasets UCF101 and HMDB51 for evaluation. We also provide [weights](https://drive.google.com/file/d/1itG7_rdSMwVRmdiD9BJwkfzz1eV1AZ_i/view?usp=sharing) of CPD model pre-trained on Instagram-300k dataset.

## Requriments

The code is built with following libraries:

- Python==3.6
- PyTorch==1.0
- [transformers](https://github.com/huggingface/transformers)
- TensorboardX
- tqdm
- scikit-learn
- pyav

## Data Preparation

For the pre-training stage, videos of the Instagram-300k dataset should be prepared. It is a subset of [OmniSource](https://arxiv.org/abs/2003.13042) dataset. Please follow the [instruction](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/omnisource/README.md) to obtain those data. The video ids of Instagram-300k are in `./datasets/ins_list/ins_300k.json`. These videos should be copied to one independent folder. For storage efficiency, we trim each video to 20s from the middle of it and resize the shorter side to 256. So the processed video is named by `trimmed_resize_{video id}.mp4`. The data directory should be structured like this:

```
.
|-- Instagram-300k
    trimmed_resize_{video id 1}.mp4
    trimmed_resize_{video id 2}.mp4
    ...
```

For evaluation on downstream datasets, videos of [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) can be directly downloaded from their official website. The downloaded data directory needs no further change for this repo.

## Pre-training on Instagram-300k

Pre-training CPD model on Instagram-300k dataset:

```bash
python main.py --video_path <DATA_PATH> --result <RESULT_PATH> \
               --dataset_file ./datasets/ins_list/ins_300k.json \
               --dataset ins --phase pretraining \
               --n_epochs 300 --ft_bert_ep 150 \
               --batch_size 256 --n_threads 16 \
               --learning_rate 0.1 --weight_decay 1e-4 \
               --stride_size 4 --sample_duration 32 \
               --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
```

`<DATA_PATH>` is your data directory and `<RESULT_PATH>` is the path to store checkpoints and training log. More parameters and their meaning can be found in `opts.py`. We do experiments on 8 V100(32G) GPUs.

## Evaluation on Downstream Datasets

Finetuning on split 1 of UCF101 dataset:

```bash
python main.py --video_path <DATA_PATH> --result <RESULT_PATH> \
               --dataset_file ./datasets/ucf_list/ucf101_01.json \
               --dataset ucf101 --phase finetuning \
               --pretrain_path <PRETRAIN_WEIGHT_PATH> \
               --n_finetune_classes 101 --n_epochs 100 \
               --batch_size 128 --n_threads 16 \
               --lr_patience 5 --dp 0.8 --lr_factor 0.1 \
               --learning_rate 0.02 --weight_decay 1e-4 \
               --stride_size 4 --sample_duration 64 \
               --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
```

`<PRETRAIN_WEIGHT_PATH>` is the path of your pre-trained weight. You can pre-train a model by yourself or use our provided [model](https://drive.google.com/file/d/1itG7_rdSMwVRmdiD9BJwkfzz1eV1AZ_i/view?usp=sharing). Since the UCF101 dataset has 3 splits, you should change the `--dataset_file` and run it 3 times.

After finetuning on the UCF101 dataset, we can test the performance on the validation set using 3 crops, 10 clips test setting. The testing scripts are as follow:

```bash
dataset='./datasets/ucf_list/ucf101_01.json'
pred_path=<RESULT_PATH>
model='save_100.pth'
python main.py --video_path <DATA_PATH> --dataset_file $dataset --result $pred_path --dataset ucf101 --phase finetuning --n_finetune_classes=101 --batch_size 512 --n_threads 32 --no_train --no_val --test --resume_path $pred_path/$model --stride_size 4 --sample_duration 64 --sample_size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
python evaluation/evaluate_ucf101.py --dataset $dataset --pred_results $pred_path
```

Finetuning on split 1 of HMDB51 dataset:

```bash
python main.py --video_path <DATA_PATH> --result <RESULT_PATH> \
               --dataset_file ./datasets/hmdb_list/hmdb51_1.json \
               --dataset hmdb51 --phase finetuning \
               --pretrain_path <PRETRAIN_WEIGHT_PATH> \
               --n_finetune_classes 51 --n_epochs 100 \
               --batch_size 128 --n_threads 16 \
               --lr_patience 5 --dp 0.8 --lr_factor 0.1 \
               --learning_rate 0.02 --weight_decay 1e-4 \
               --stride_size 4 --sample_duration 64 \
               --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
```

HMDB51 dataset also has 3 splits, so 3 times runnings are needed. Also, testing scripts are as follow:

```bash
dataset='./datasets/hmdb_list/hmdb51_1.json'
pred_path=<RESULT_PATH>
model='save_100.pth'
python main.py --video_path <DATA_PATH> --dataset_file $dataset --result $pred_path --dataset hmdb51 --phase finetuning --n_finetune_classes=51 --batch_size 512 --n_threads 32 --no_train --no_val --test --resume_path $pred_path/$model --stride_size 4 --sample_duration 64 --sample_size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
python evaluation/evaluate_ucf101.py --dataset $dataset --pred_results $pred_path
```

Average finetuning results on 3 splits of UCF101 and HMDB51 datasets:

|                   | UCF101 top-1 acc. | HMDB51 top-1 acc. |
| :---------------: | :---------------: | :---------------: |
| CPD (3D ResNet-50) |       92.8        |       63.6        |

Note that we choose the last checkpoint for each training and the same finetuning schedule for different datasets by default. Selecting the best-performing epoch and carefully adjusting the finetuning hyper-parameters (e.g. lr, wd ...) can get further performance improvements.

## Contact

For any questions, please feel free to reach Tianhaolee@outlook.com .

## Citation
Please consider citing our paper in your publications if the project helps your research.
```
@article{cpd2020,
      title={Learning Spatiotemporal Features via Video and Text Pair Discrimination}, 
      author={Tianhao Li and Limin Wang},
      journal={arXiv preprint arXiv:2001.05691},
      year={2020}
}
```
