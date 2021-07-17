import os
import glob
import numpy as np
from PIL import Image

import tensorflow as tf


# preprocess data
def load_and_preprocess_image(data, i_array, image_size=224, num_channels=3):
    # data loading
    comp_data = sorted(glob.glob(os.path.join(data, 'Composite')+'/*'))
    mask_data = sorted(glob.glob(os.path.join(data, 'Mask')+'/*'))
    real_data = sorted(glob.glob(os.path.join(data, 'Real')+'/*'))

    assert len(comp_data) > 0 and len(mask_data) > 0 and len(real_data) > 0

    c = 0
    for i in range(len(i_array)):
        path = 'f' + str(i_array[i]) + '_'
        path1 = 'f' + str(i_array[i]) + '.'

        for j in comp_data:
            if path in j:
                f = False
                comp_path = j
                break
            else:
                f = True
        for k in mask_data:
            if path in k:
                mask_path = k
                break
        for l in real_data:
            if path1 in l:
                real_path = l
                break

        if f:
            continue

        comp = Image.open(comp_path).convert('RGB')
        comp = np.array(comp.resize([image_size, image_size]))
        comp = tf.cast(comp, tf.float32)
        comp = comp / 255.0
        comp = tf.reshape(comp, (1, image_size, image_size, num_channels))

        real = Image.open(real_path).convert('RGB')
        real = np.array(real.resize([image_size, image_size]))
        real = tf.cast(real, tf.float32)
        real = real / 255.0
        real = tf.reshape(real, (1, image_size, image_size, num_channels))

        mask = Image.open(mask_path).convert('1')
        mask = np.array(mask.resize([image_size, image_size])) * 1
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, (1, image_size, image_size, 1))

        if c == 0:
            c = 1
            real_final = real
            comp_final = comp
            mask_final = mask
        else:
            real_final = tf.concat([real, real_final], axis = 0)
            comp_final = tf.concat([comp, comp_final], axis = 0)
            mask_final = tf.concat([mask, mask_final], axis = 0)

    inputs = tf.concat([comp_final,mask_final], 3)

    return inputs, comp_final, real_final, mask_final, f
