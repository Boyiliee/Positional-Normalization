# Example code for MUNIT (Huang et al., ECCV 2018) + PONO-MS (Li et al., NeurIPS 2019)
This code is based on the official [MUNIT code base](https://github.com/NVlabs/MUNIT). We appreciate the authors for sharing their awesome code.


Based on this code, we change `networks.py` and `trainer.py` into `networks_pono.py` and `trainer_pono.py`. 
To use PONO-MS with MUNIT' in the paper, please follow the original instruction and set
```yaml
gen:
    arch: adain_ponoms
```
The detailed training instructions are coming soon.


## Citation
If you find this repo useful, please cite:
```
@inproceedings{li2019positional,
    title={Positional Normalization},
    author={Li, Boyi and Wu, Felix and Weinberger, Kilian Q and Belongie, Serge},
    booktitle={Advances in Neural Information Processing Systems},
    pages={1620--1632},
    year={2019}
}

@inproceedings{huang2018munit,
    title={Multimodal Unsupervised Image-to-image Translation},
    author={Huang, Xun and Liu, Ming-Yu and Belongie, Serge and Kautz, Jan},
    booktitle={ECCV},
    year={2018}
}
```
