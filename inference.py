# Load images and run inference
# Add more images in the samples directory if you want to test BKinD on different samples.
import torch
import torchvision
import torchvision.transforms as transforms

from model.unsupervised_model import Model as BKinD
from model.kpt_detector import Model

import cv2
from PIL import Image
from utils import visualize_with_circles

import numpy as np
import os

## Input parameters #################################
nKps = 10
gpu = 0  # None if gpu is not available
resume = 'checkpoint/CalMS21/checkpoint.pth.tar'
sample_path = 'sample_images'
image_size = 256
#####################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize,])

## Load a model
model = Model(nKps)
if os.path.isfile(resume):
  if gpu is not None:
    model = model.cuda(gpu)
    loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(resume, map_location=loc)
  else:
    checkpoint = torch.load(resume)

  bkind_model = BKinD(nKps)

  bkind_model.load_state_dict(checkpoint['state_dict'])
  bkind_model_dict = bkind_model.state_dict()

  model_dict = model.state_dict()
  pretrained_dict = {k: v for k, v in bkind_model_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)

  print("=> loaded checkpoint '{}' (epoch {})"
        .format(resume, checkpoint['epoch']))
else:
  print("=> no checkpoint found at '{}'".format(resume))

## Inference on sample images
sample_files = sorted(os.listdir(sample_path))
for i in range(len(sample_files)):
  if 'output' not in sample_files[i]:
    with open(os.path.join(sample_path, sample_files[i]), 'rb') as f:
      im = Image.open(f)
      im = im.convert('RGB')

    im = torchvision.transforms.Resize((image_size, image_size))(im)
    im = transform(im)
    im = im.unsqueeze(0).cuda(gpu, non_blocking=True)

    output = model(im)

    pred_kps = torch.stack((output[0][0], output[0][1]), dim=2)
    pred_kps = pred_kps.data.cpu().numpy()

    im_with_kps = visualize_with_circles(im[0].data.cpu().numpy(), pred_kps[0]+1,
                                        output[5][0], mean=mean, std=std)
    im_with_kps = im_with_kps.astype('uint8')
    im_with_kps = cv2.cvtColor(im_with_kps, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(sample_path, 'output_'+str(i+1)+'.png'), im_with_kps)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', im_with_kps)

print("==> Output images saved in \'sample_images\' directory")