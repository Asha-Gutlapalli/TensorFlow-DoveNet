import fire
from tf_dovenet.train import Trainer


# train model
def train(data, batch_size=16, image_size=224, epochs=100, lambda_a=1.0, lambda_v=1.0, lambda_L1=100.0, lr=0.0002):
    Trainer(data, batch_size, image_size, epochs, lambda_a, lambda_v, lambda_L1, lr).train()

if __name__ == '__main__':
    fire.Fire(train)
