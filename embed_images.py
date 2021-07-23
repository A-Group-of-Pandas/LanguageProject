from coco import COCO
from model import Model

coco_dataset = COCO(database_dir='database')

model = Model()
model.load_model('model_params.npy')
print('starting')
coco_dataset.vec2embed(model)
coco_dataset.save_database()