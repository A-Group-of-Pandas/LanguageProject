import mygrad as mg
from mynn.layers import dense
from mygrad.nnet.initializers import glorot_normal
import numpy as np

# model.py
class Model:
    def __init__(self, descriptor_dims=512, image_embed_dims=200):
        self.dense1 = dense(descriptor_dims, image_embed_dims, weight_initializer = glorot_normal, bias=False)

    def __call__(self, descriptors): 
        batch = self.dense1(descriptors)
        return batch / mg.sqrt(mg.einsum("nd, nd -> n", batch, batch)).reshape(-1, 1)

    @property
    def parameters(self):
        return self.dense1.parameters

    def save_model(self, file_name='model_params.npy'):
        with open(file_name, 'wb') as f:
            np.save(f, self.parameters[0].data)

        save_confirmation_msg = 'save confirmed!'
        
        print(save_confirmation_msg)
        
    def load_model(self, file_name='model_params.npy'):
        self.parameters[0].data = np.load(file_name)
