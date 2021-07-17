import numpy as np
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from .data import *
from .network import *


# training code
class Trainer():
    def __init__(self, data, batch_size=16, image_size=224, epochs=100, lambda_a=1.0, lambda_v=1.0, lambda_L1=100.0, lr=0.0002):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = 3
        self.epochs = epochs
        self.lambda_a = lambda_a
        self.lambda_v = lambda_v
        self.lambda_L1 = lambda_L1
        self.lr = lr

        self.generator = att_unet(self.image_size, self.image_size, self.num_channels)
        self.discriminator = NLayerDiscriminator()

        self.mae = MeanAbsoluteError()
        self.bce = BinaryCrossentropy(from_logits=True)

        self.optimizer = Adam(learning_rate=self.lr)

    def train(self):
        loss_gen = []
        loss_dis = []

        for e in range(self.epochs):
            print('Epoch', e)

            for i in range(1, 100):
                print(i)

                rand_nums = np.random.randint(1, 1600, size = self.batch_size)
                inputs, comp, real, mask, f = load_and_preprocess_image(data, rand_nums)

                with tf.GradientTape() as dis_tape, tf.GradientTape() as gen_tape:
                    output = self.generator(inputs)
                    fake_f = output * mask
                    cap = output * mask + comp * (1 - mask)
                    harmonized = output

                    fake_AB = harmonized
                    pred_fake, ver_fake = self.discriminator(fake_AB, mask)
                    global_fake = self.bce(tf.broadcast_to(tf.constant(0.0), shape=pred_fake.shape), pred_fake)
                    local_fake = self.bce(tf.broadcast_to(tf.constant(0.0), shape=ver_fake.shape), ver_fake)
                    loss_D_fake = global_fake + local_fake

                    real_AB = real
                    pred_real, ver_real = self.discriminator(real_AB, mask)
                    global_real = self.bce(tf.broadcast_to(tf.constant(1.0), shape=pred_fake.shape), pred_real)
                    local_real = self.bce(tf.broadcast_to(tf.constant(1.0), shape=ver_fake.shape), ver_real)
                    loss_D_real = global_real + local_real
                    loss_D_global = global_fake + global_real
                    loss_D_local = local_fake + local_real
                    loss_D = loss_D_fake + loss_D_real

                    pred_fake, ver_fake, featg_fake, featl_fake = self.discriminator(fake_AB, mask, feat_loss=True)
                    loss_G_global = self.bce(tf.broadcast_to(tf.constant(1.0), shape=pred_fake.shape), pred_fake)
                    loss_G_local = self.bce(tf.broadcast_to(tf.constant(1.0), shape=ver_fake.shape), ver_fake)
                    pred_real, ver_real, featg_real, featl_real = self.discriminator(real_AB, mask, feat_loss=True)

                    loss_G_GAN = self.lambda_a * loss_G_global + self.lambda_v * loss_G_local
                    loss_G_L1 = self.mae(real, output) * self.lambda_L1
                    loss_G = loss_G_GAN + loss_G_L1
                    print(loss_G, loss_D)

                dis_gradients = dis_tape.gradient(loss_D, self.discriminator.trainable_variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(dis_gradients, self.discriminator.trainable_variables))
                gen_gradients = gen_tape.gradient(loss_G, self.generator.trainable_variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(gen_gradients, self.generator.trainable_variables))

            loss_gen.append(loss_G)
            loss_dis.append(loss_D)
            print(loss_G, loss_D)
