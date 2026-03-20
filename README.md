# lpips-vgg

Lightweight LPIPS (Learned Perceptual Image Patch Similarity) implementation using VGG16.

## Install

pip install .

## Usage

```python


from lpips_vgg import LPIPS
import torch

loss_fn = LPIPS()

# example tensors (N, C, H, W) in range [0,1]
x = torch.randn(1, 3, 256, 256)
y = torch.randn(1, 3, 256, 256)

loss = loss_fn(x, y)
print(loss)

```

## Weights

The pretrained VGG16 weights are loaded automatically from:

lpips_vgg/lpips_weights/vgg.pth

Make sure this file exists inside the package.

## Notes

- Expects images normalized to [0,1]
- Uses torchvision VGG16 backbone trained using ImageNet (from torchvision.models import VGG16_Weights)
- Designed for internal / research use
