# SCDM (IEEE TGRS 2024)
### [**Paper**](https://ieeexplore.ieee.org/document/10654291)

PyTorch codes for "[Spectral-Cascaded Diffusion Model for Remote Sensing Image Spectral Super-Resolution](https://ieeexplore.ieee.org/document/10654291)", **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, 2024.

## Abstract
> Hyperspectral remote sensing images (HSIs) have unique advantages in urban planning, precision agriculture, and ecology monitoring since they provide rich spectral information. However, hyperspectral imaging usually suffers from low spatial resolution and high cost, which limits the wide application of hyperspectral data. Spectral super-resolution provides a promising solution to acquire hyperspectral images with high spatial resolution and low cost, taking RGB images as input. Existing spectral super-resolution methods utilize neural networks following a single-shot framework, i.e., final results are obtained by one-stage spectral super-resolution, which struggles to capture and model the complex relationships between spectral bands. In this article, we propose a spectral-cascaded diffusion model (SCDM), a coarse-to-fine spectral super-resolution method based on the diffusion model. The diffusion model fits the real data distribution through stepwise denoising, which is naturally suitable for modeling rich spectral information. We cascade the diffusion model in the spectral dimension to gradually refine the spectral trends and enrich spectral information of the pixels. The cascade solves the highly ill-posed problem of spectral super-resolution step-by-step, mitigating the inaccuracies of previous single-shot approaches. To better utilize the potential of the diffusion model for spectral super-resolution, we design image condition mixture guidance (ICMG) to enhance the guidance of image conditions and progressive dynamic truncation (PDT) to limit cumulative errors in the sampling process. Experimental results demonstrate that our method achieves state-of-the-art performance in spectral super-resolution. 
## Pipeline  
 ![image](/figs/SCDM.png)
 
## Install
```
https://github.com/Mr-Bamboo/SCDM.git
```

## Environment
 > * CUDA 11.8
 > * Python >=3.7.0
 > * PyTorch >= 1.7.0
 > * GDAL


## Usage

### Settings
- We provide relevant tools for the [IEEE GRSS DFC 2018 dataset](https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/) in ```datasets/hyper.py```.
- The hyperparameter settings are located in ```def diffusion_defaults()``` within ```ddpm/script_utils.py```.

### Test
- For Stage 1: 
```
python scripts/test_hyper.py
```
- For Stage 2: 
```
python scripts/test_hyper_wo.py
```
### Train
```
python scripts/train_hyper.py
```


## Acknowledgments
Our SCDM mainly borrows from [DDPM](https://github.com/abarankab/DDPM) and [R2HGAN](https://github.com/liuliqin/R2HGAN-generate-HSI-from-RGB). Thanks for these excellent open-source works!

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: chenbowen@buaa.edu.cn

## Citation
If you find our work helpful in your research, please consider citing it. Your support is greatly appreciated! 😊

```
@ARTICLE{10654291,
  author={Chen, Bowen and Liu, Liqin and Liu, Chenyang and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Spectral-Cascaded Diffusion Model for Remote Sensing Image Spectral Super-Resolution}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  keywords={Superresolution;Diffusion models;Spatial resolution;Biological system modeling;Task analysis;Training;Image synthesis;Cascade-based methods;diffusion model;remote sensing;spectral super-resolution},
  doi={10.1109/TGRS.2024.3450874}}

```
