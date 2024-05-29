
<div align="center">
<h1>Exploring Intrinsic Dimension for Vision-Language Model Pruning</h1>
</div>

This is the official implementation of ICML'24 paper [Exploring Intrinsic Dimension for Vision-Language Model Pruning](https://openreview.net/forum?id=xxL7CEWuxz&noteId=dIPRrajDnh). 

## 🏃‍♂️ TL;DR
The [Intrinsic Dimension (ID)](https://proceedings.neurips.cc/paper/2019/hash/cfcce0621b49c983991ead4c3d4d3b6b-Abstract.html) of vision representations spans a wider and greater range than that of language representations, which we attribute to the heightened sensitivity of vision models to pruning. In contrast, language models exhibit greater robustness despite containing more redundant weights.

![Example Image](ID.png)

## 🔨 Installation
This code is tested on `Pytorch==1.11.0`, `cuda==11.5`, and `python==3.9.0`. Install the dependencies with:
```bash
conda install --yes --file requirements.txt
```

## 📐 Evaluation of IDs
* **CPU Mode**
    ```bash
    python ComputeID.py -n 2000 --Path ID/Blip_coco --cpu
    ```
* **GPU Mode**
    ```bash
    python ComputeID.py -n 2000 --Path ID/Blip_coco --gpu 0
    ```
## :scissors: Pruning
### Image Captioning on the COCO Caption Dataset with BLIP

* **Dataset & Annotation**
    1. Download the COCO2014 dataset and unzip it under the `datasets` folder. Update the `image_root` in [config](./configs/caption_coco.yaml).
    2. Download all-in-one annotations from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `coco/annotation` folder, and update the `annotation` in [config](./configs/caption_coco.yaml).

* **Pruning**
    1. Download the uncompressed model from [this link](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth) and place it in the `pretrained` folder. Update the `pretrained` in [config](./configs/caption_coco.yaml).
    2. To prune BLIP by 80% and add Intrinsic Dimension to the importance score metric during pruning, run:
        ```bash
        python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 train_caption.py --final_threshold 0.2 --model_dir coco/PLATON80 --pruner_name PLATON --useID
        ```

* **Evaluation**
    1. Place the pruned model in the `output` folder and update the `--pretrained` in the scripts. 
    2. To evaluate the pruned model, run:
        ```bash
        python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 train_caption.py --pruner_name PLATON --pruned output/pruned_model_path --evaluate
        ```

### Visual Reasoning on the NLVR2 Dataset with BLIP

* **Dataset & Annotation**
    1. Download the NLVR2 dataset and unzip it under the `datasets` folder. Update the `image_root` in [config](./configs/nlvr.yaml).
    2. Download all-in-one annotations from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `nlvr/annotation` folder, and update the `annotation` in [config](./configs/nlvr.yaml).

* **Pruning**
    1. Download the uncompressed model from [this link](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth) and place it in the `pretrained` folder. Update the `pretrained` in [config](./configs/nlvr.yaml).
    2. To prune BLIP by 80% and add Intrinsic Dimension to the importance score metric during pruning, run:
        ```bash
        python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 train_nlvr.py --final_threshold 0.2 --model_dir nlvr/PLATON80 --pruner_name PLATON --useID
        ```

* **Evaluation**
    1. Place the pruned model in the `output` folder and update the `--pretrained` in the scripts.
    2. To evaluate the pruned model, run:
        ```bash
        python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 train_nlvr.py --pruner_name PLATON --pruned output/pruned_model_path --evaluate
        ```

## 💐 Acknowledgments
This code is built upon [IntrinsicDimDeep](https://github.com/ansuini/IntrinsicDimDeep), [BLIP](https://github.com/salesforce/BLIP) and [PLATON](https://github.com/QingruZhang/PLATON), and we sincerely appreciate their contributions.

## 🌸 Citation
If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{
wang2024exploring,
title={Exploring Intrinsic Dimension for Vision-Language Model Pruning},
author={Hanzhang Wang, Jiawen Zhang, and Qingyuan Ma},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=xxL7CEWuxz}
}
```
