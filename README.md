## _ViTAMINS_: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives

Official PyTorch implementation of **ViTAMINS**, a self-supervised learning method that integrates **synthetic hard negatives** into contrastive pretraining of vision transformers. Instead of relying only on the negatives available in the memory queue, ViTAMINS synthesizes harder negatives on-the-fly in representation space, using geometric, stochastic and adversarial perturbations of the hardest existing ones. This sharpens the contrastive task at **negligible extra cost**, without multi-crop augmentation or pixel-level reconstruction, and works with both ViT and Swin backbones.

> **ViTAMINS: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives**  
> Nikos Giakoumoglou, Andreas Floros, Kleanthis-Marios Papadopoulos, Tania Stathaki — **WACV 2027**  
> [Paper](https://arxiv.org/pdf/2609.01041)
---

## Overview

ViTAMINS integrates synthetic hard negatives into contrastive self-supervised pretraining of vision transformers. Hard negatives are synthesized on-the-fly in the embedding space with six strategies that span the canonical perturbation families: geometric, stochastic and adversarial. The method requires only simple modifications to existing contrastive frameworks, adds negligible overhead, and applies to both ViT and Swin backbones.

## Evaluation

ViTAMINS consistently learns stronger representations than contrastive, self-distillation, generative and JEPA baselines under matched architectures, without relying on multi-crop augmentation. The synthetic hard negatives add negligible overhead over standard contrastive training, yet ViTAMINS reaches the accuracy of much larger models: our ViT-B reaches 77.1% top-1 in 300 epochs, surpassing V-JEPA with ViT-L and matching LeJEPA with ViT-H at 7.3× fewer parameters. The learned features also serve as strong nearest-neighbour classifiers, improving k-NN accuracy by up to +11.3% over reproduced baselines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ad46d405-4c8d-4852-98f2-b8cdf9b8f800" alt="Params vs Top1" width="700">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3ac9fc6-c9ed-4536-8c32-b4cf9d9319e4" alt="Params vs Top1" width="1100">
</p>

The gains carry over to the more challenging dense prediction task. ViTAMINS achieves the best object detection and instance segmentation results on COCO and the best semantic segmentation on ADE20K among all compared models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b06f9e6f-a48a-4589-aad7-121802f5775a" alt="Dense transfer" width="700">
</p>


## Visualization

We visualize the learned patch representations using PCA on ViTAMINS encoder features. Without any labels, the features recover coherent object and background structures with clear object boundaries, suggesting that synthetic hard negatives encourage semantically structured representations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/43b018ca-d1a7-4be1-89c0-8cc295c53e8d" alt="PCA" width="700">
</p>

---

## Environment Setup

To set up a compatible environment (CUDA 11.7/11.8) on [PBS](https://en.wikipedia.org/wiki/Portable_Batch_System), follow these steps:

```bash
conda create -n vitamins -c conda-forge cudatoolkit=11.8 python=3.10.11
conda activate vitamins
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install timm==0.4.9 diffdist Pillow pyyaml yacs termcolor scipy numpy==1.21.5 opencv-python tqdm
```

## Unsupervised Pretraining

To do unsupervised pre-training of a ViT-Base model on ImageNet in an 8-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=12345 \
    main_pretrain.py \
    --cfg configs/vitamins/vit_base.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --batch-size 64 \
    --output [output folder] \
    --tag [tag folder]
```

To use BYOL, MoBY or DINO instead, swap the config file accordingly, e.g. `--cfg configs/dino/vit_base.yaml`.

### Submitting on PBS

Ready-made job scripts are provided for every method and architecture. Submit them **from the repository root**, since each script does `cd $PBS_O_WORKDIR`:

```bash
qsub scripts/vitamins/vit_base.pbs
```

Each script runs pre-training followed by linear evaluation, writes to `./logs/`, and requests the GPU count matching its architecture (4 GPUs at batch 128, or 8 GPUs at batch 64, for a total batch of 512 either way).

## Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=12345 \
    main_linear.py \
    --cfg configs/vitamins/vit_base.yaml \
    --data-path [your imagenet-folder with train and val folders] \
    --output [output folder] \
    --tag [tag folder]
```

Use the same `--cfg`, `--output`, and `--tag` as in pre-training. By default, this performs linear probing with frozen features (`LINEAR_EVAL.WEIGHTS freeze`). For full fine-tuning, set `LINEAR_EVAL.WEIGHTS finetune` in the config.


## Transfer Learning

To evaluate on downstream datasets (CIFAR-10, CIFAR-100, STL-10, Oxford Flowers102, Oxford Pets, Food101, Stanford Cars, Caltech101, DTD, FGVC Aircraft, SUN397, VOC2007, Places365), append `--opts DATA.DATASET <dataset>` to the linear evaluation command. For full fine-tuning, add `--opts LINEAR_EVAL.WEIGHTS finetune`.

## *k*-NN Classification, Image Retrieval, Video Object Segmentation

Use the evaluation scripts from the official [DINO repository](https://github.com/facebookresearch/dino).

---

### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

### Acknowledgments

This codebase is built upon [MoBY](https://github.com/SwinTransformer/TransformerSSL), [SynCo](https://github.com/giakoumoglou/synco), [DINO](https://github.com/facebookresearch/dino), [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We thank the authors for their excellent work and for making their code publicly available.

## Citation

If you find this repository useful, please consider giving it a star ⭐ and citing our paper:

```bibtex
@misc{giakoumoglou2026vitamins,
      title={{ViTAMINS}: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives}, 
      author={Nikos Giakoumoglou and Andreas Floros and Kleanthis-Marios Papadopoulos and Tania Stathaki},
      year={2026},
      eprint={2609.01041},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.01041}, 
}
```
