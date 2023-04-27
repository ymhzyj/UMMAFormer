<div align="center">

## **UMMAFormer**: A Universal Multimodal-adaptive Transformer Framework For Temporal Forgery Localization

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<!-- [![Conference](https://img.shields.io/badge/ACM%20MM-2022-orange)](https://2022.acmmm.org/)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/nku-shengzheliu/SER30K/blob/main/LICENSE) -->

</div>

This is the official repository of our Work submmited to ACM MM'23.
**Temporal Video Inpainting Localization(TVIL) dataset** and pytorch training/validation code for **UMMAFormer**.

<p align="center">
<img src="./figures/overview_frameworks.png" alt="drawing" width="70%" height="70%"/>
    <h4 align="center">Overview of UMMAFormer</h4>
</p>


## Abstract

The emergence of  artificial intelligence-generated content~(AIGC) has raised concerns about the authenticity of multimedia content in various fields. Existing research has limitations and is not widely used in industrial settings, as it is only focused on binary classification tasks of complete videos. We propose a novel universal transformer framework for temporal forgery localization (TFL) called UMMAFormer, which predicts forgery segments with multimodal adaptation. We also propose a Temporal Feature Abnormal Attention (TFAA) module based on temporal feature reconstruction to enhance the detection of temporal differences. In addition, we introduce a parallel cross-attention feature pyramid network (PCA-FPN) to optimize the Feature Pyramid Network (FPN) for subtle feature enhancement. To address the lack of available datasets, we introduce a novel temporal video inpainting localization (TVIL) dataset that is specifically tailored for video inpainting scenes. Our experiments demonstrate that our proposed method achieves state-of-the-art performance on benchmark datasets, Lav-DF, TVIL, Psynd surpassing the previous best results significantly.

<p align="center">
<img src="./figures/vilsamples.png" alt="drawing" width="100%" height="100%"/>
    <h4 align="center">Movitation of UMMAFormer</h4>
</p>


## TVIL dataset

### a. Data Download
If you need the TVIL dataset for academic purposes, please download the full data from [BaiduYun Disk](https://pan.baidu.com/s/1xWcrNL-lUiUSLklnozyQvQ) (code：95lo).

### b. Data Sources
The raw data is coming from [Youtube VOS 2018](https://codalab.lisn.upsaclay.fr/competitions/7685#participate-get_data).

<!-- 原图 -->

### c. Inpainting Methods
We use four different video inpainting methods to create new videos. They are [E2FGVI](https://github.com/MCG-NKU/E2FGVI), [FGT](https://github.com/hitachinsk/FGT), [FuseFormer](https://github.com/ruiliu-ai/fuseformer), and [STTN](https://github.com/researchmm/STTN), respectively. We used [XMEM](https://github.com/hkchengrex/XMem) to generate the inpainting mask.


<!-- 放图 -->

### d. Feature Extract
We also provided [TSN features](https://pan.baidu.com/s/1xWcrNL-lUiUSLklnozyQvQ) (code：95lo) as used in the paper, specifically extracted by [mmaction2](https://github.com/open-mmlab/mmaction2).




## Code

The code will be released after paper accept.



## TODO List
- [ ] Release full code.
- [x] Release TVIL datasets and TSN features.
- [ ] Release TSN features and BYOL-A features for Lav-DF and Psynd 
- [ ] Release our pre-trained model








## Acknowledgement

Thanks for the work of [Actionformer](https://github.com/happyharrycn/actionformer_release). My code is based on the implementation of them.