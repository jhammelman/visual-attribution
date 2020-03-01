# Pytorch Visual Attribution Methods

A collection of visual attribution methods for model interpretability

![](./images/tusker_saliency.png)

Including:
- [x] Vanilla Gradient Saliency
- [x] Grad X Input
- [x] Integrated Gradient
- [x] SmoothGrad
- [x] Guided Backpropagation
- [x] Excitation Backpropagation, Contrastive Excitation Backpropagation
- [x] GradCAM
- [x] DeepLIFT
- [ ] Deconv
- [ ] PatternNet, PatternLRP
- [ ] Real Time Saliency
- [ ] Occlusion
- [ ] Feedback
- [ ] Meaningful Perturbation

## Setup

### Prerequisities
- Linux
- NVIDIA GPU + CUDA (Current only support running on GPU)
- Python 3.x
- PyTorch version == 1.3 TODO check
- torchvision, skimage, matplotlib

modify torchvision:
Modify from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION to from PIL import Image, ImageOps, ImageEnhance, __version__ in functional.py

### Getting Started
- Clone this repo:
```bash
git clone git@github.com:yulongwang12/visual-attribution.git
cd visual-attribution
```
- Download pretrained weights
```bash
cd weights
bash ./download_patterns.sh  # for using PatternNet, PatternLRP
bash ./download_realtime_saliency.sh # for using Real Time Saliency
```
**Note**: I convert caffe bvlc_googlenet pretrained models in pytorch format (see `googlenet.py` and `weights/googlenet.pth`). 

### Visual Saliency Comparison
see notebook [saliency_comparison.ipynb](./notebooks/saliency_comparison.ipynb). If everything works, you will get the above image.

### Weakly Supervised Object Localization
TBD

## Citation
If you use our codebase or models in your research, please cite this project.
```bibtex
@misc{visualattr2018,
  author =       {Yulong Wang},
  title =        {Pytorch-Visual-Attribution},
  howpublished = {\url{https://github.com/yulongwang12/visual-attribution}},
  year =         {2018}
}
```
