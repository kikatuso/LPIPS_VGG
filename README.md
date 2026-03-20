# lpips-vgg

Lightweight LPIPS (Learned Perceptual Image Patch Similarity) implementation using VGG.

## Install

pip install .

## Usage

from lpips_vgg import LPIPS
import torch

loss_fn = LPIPS()

# example tensors (N, C, H, W) in range [-1, 1]
x = torch.randn(1, 3, 256, 256)
y = torch.randn(1, 3, 256, 256)

loss = loss_fn(x, y)
print(loss)

## Weights

The pretrained VGG weights are loaded automatically from:

lpips_vgg/lpips_weights/vgg.pth

Make sure this file exists inside the package.

## Notes

- Expects images normalized to [0,1]
- Uses torchvision VGG backbone
- Designed for internal / research use