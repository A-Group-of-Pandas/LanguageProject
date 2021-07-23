from coco import COCO
from model import Model
import numpy as np
import mygrad as mg

coco_dataset = COCO()

model = Model()
model.load_model('model_params.npy')
print('starting')
inference = coco_dataset.create_inference_dataset(model)
inference.save_database()
coco_dataset.glove_database.save_database()