import os
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import tensorflow as tf
import torchvision.transforms as transforms

from tf_dovenet.network import att_unet
from tf_dovenet.srgan import Generator


# Image Compositon GAN Generator
class ICGEN():
    def __init__(self, models_dir='models'):
        self.device = "cuda" if tf.test.is_gpu_available() else "cpu"

        # set up parameters
        self.image_size = 224
        self.num_channels = 3
        self.batch_size = 1

        self.base_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        print(self.base_dir)

        # models directory
        self.models_dir = self.base_dir / models_dir
        (self.models_dir).mkdir(parents=True, exist_ok=True)

        urls = {"ICGEN" : "https://www.dropbox.com/sh/57bna1fo4pufu8a/AAD73v3tAJJQkQBefKlMiYT9a?dl=0",
        "SRGAN_G" : "https://www.dropbox.com/s/x9ks2hnc7rswexo/best_g.pth.tar?dl=0"}

        # image compositon generator
        self.model = att_unet(self.image_size, self.image_size, self.num_channels)

        zip_name = 'ICGEN.zip'
        model_name = 'gen.data-00000-of-00001'
        model_path = os.path.join(self.models_dir, model_name)
        zip_path = os.path.join(self.models_dir, zip_name)

        # download pretrained model
        if not os.path.exists(model_path):
            subprocess.call(['wget', urls["ICGEN"], '-O', zip_path])
            subprocess.call(['unzip', zip_path, '-d', self.models_dir])

        # load image compositon generator
        self.model.load_weights(os.path.join(self.models_dir, 'gen'))

        # SRGAN generator
        self.srgan_gen = Generator(scale_factor=4)

        sr_gen_name = os.path.split(urls["SRGAN_G"])[-1][:-5]
        sr_gen_path = os.path.join(self.models_dir, sr_gen_name)

        # download pretrained SRGAN generator model
        if not os.path.exists(sr_gen_path):
            subprocess.call(['wget', urls["SRGAN_G"], '-O', sr_gen_path])

        # load SRGAN generator
        srgan_checkpoint = torch.load(sr_gen_path, map_location=self.device)
        self.srgan_gen.load_state_dict(srgan_checkpoint["state_dict"])
        self.srgan_gen.to(self.device)

    # preprocess composite and mask images
    def preprocess(self, comp, mask):
        # composite

        # open image and convert to RGB channels
        comp = Image.open(comp).convert('RGB')
        # resize image
        comp = np.array(comp.resize([self.image_size, self.image_size]))
        # cast to tensor
        comp = tf.cast(comp, tf.float32)
        # normalize image
        comp = comp / 255.0
        # reshape image
        comp = tf.reshape(comp, (self.batch_size, self.image_size, self.image_size, self.num_channels))

        # mask

        # open image and convert to black and white image
        mask = Image.open(mask).convert('1')
        # resize image
        mask = np.array(mask.resize([224, 224])) * 1
        # cast to tensor
        mask = tf.cast(mask, tf.float32)
        # reshape image
        mask = tf.reshape(mask, (1, 224, 224, 1))

        # concatenate both composite and mask images
        inputs = tf.concat([comp, mask], 3)

        return inputs

    def harmonize(self, inputs):
        # harmonize image
        with tf.device(self.device):
            result = self.model(inputs).numpy()
            result = np.transpose(result, (0, 3, 1, 2))

        return result

    # upsample image using super resolution
    def upsample(self, lr_image):
        # convert to tensor
        lr_tensor = torch.from_numpy(lr_image)
        # convert LR image to HR image
        hr_tensor = torch.squeeze(self.srgan_gen(lr_tensor))
        # convert tensor to image
        to_image = transforms.ToPILImage()
        hr_image = to_image(hr_tensor)

        return hr_image
