We apply simple PONO and MS to the [CycleGAN and Pix2pix Official Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 


Based on this code, we only change [networks.py](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py) into networks_pono.py.

To run this code, please follow the original instruction and set `--netG restnetPONOMS_9block` for PONO-MS or `--netG resnetPONODMS_9blocks` for PONO-DMS. 

If you find this repo useful, please cite:
```
@inproceedings{li2019positional,
  title={Positional Normalization},
  author={Li, Boyi and Wu, Felix and Weinberger, Kilian Q and Belongie, Serge},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1620--1632},
  year={2019}
}

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
