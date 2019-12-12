# Example code for DRIT (Lee et al., ECCV 2018) + PONO-MS (Li et al., NeurIPS 2019)
This code is based on the official [DRIT code base](https://github.com/HsinYingLee/DRIT). We appreciate the authors for sharing their awesome code.

## Usage

### Prerequisites
- Python >= 3.5
- Pytorch >= 0.4.0 and torchvision
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [Tensorflow](https://www.tensorflow.org/) (for tensorboard usage)
- DRIT authors provide a Docker file for building the environment based on CUDA 9.0, CuDNN 7.1, and Ubuntu 16.04 which we also keep in this code base.

## Datasets (instructions provided by Lee et al., ECCV 2018)
- Download the dataset using the following script.
```
bash ../datasets/download_dataset.sh dataset_name
```
- portrait: 6452 photography images from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 1811 painting images downloaded and cropped from [Wikiart](https://www.wikiart.org/).
- cat2dog: 871 cat (birman) images, 1364 dog (husky, samoyed) images crawled and cropped from Google Images.
- You can follow the instructions in CycleGAN [website](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to download the Yosemite (winter, summer) dataset and artworks (monet, van Gogh) dataset. For photo <-> artrwork translation, we use the summer images in Yosemite dataset as the photo images.

## Training Examples
- Cat <-> Dog
```
cd src
python train.py --dataroot ../datasets/cat2dog --name cat2dog_drit_pono_ms --arch drit_pono_ms --concat 0
tensorboard --logdir ../logs/cat2dog_drit_pono_ms
```
Results and saved models can be found at `../results/cat2dog_drit_pono_ms`.

- Portrait (photograpy <-> painting)
```
cd src
python train.py --dataroot ../datasets/portrait --name portrait_drit_pono_ms --arch drit_pono_ms --concat 0
tensorboard --logdir ../logs/portrait_drit_pono_ms
```
Results and saved models can be found at `../results/portrait_drit_pono_ms`.


## Paper
Please cite these papers if you find this code useful for your research.
```
@inproceedings{li2019positional,
  title={Positional Normalization},
  author={Li, Boyi and Wu, Felix and Weinberger, Kilian Q and Belongie, Serge},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1620--1632},
  year={2019}
}

@inproceedings{DRIT,
  author = {Lee, Hsin-Ying and Tseng, Hung-Yu and Huang, Jia-Bin and Singh, Maneesh Kumar and Yang, Ming-Hsuan},
  booktitle = {European Conference on Computer Vision},
  title = {Diverse Image-to-Image Translation via Disentangled Representations},
  year = {2018}
}
