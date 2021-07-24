from model import Model
from coco import COCO
import mygrad as mg
import numpy as np

model = Model()
coco_dataset = COCO(database_dir='database')
model.load_model('model_params.npy')
val_inputs, val_captions, val_confusers = coco_dataset.generate_matrix(10)

with mg.no_autodiff:
    true_img_embedding = model(val_inputs)    
    confuser_img_embedding = model(val_confusers) 
    true_caption_embedding = val_captions
    #print()
    print(np.einsum("ni,ni -> n", true_img_embedding, val_captions)[0])
    true_similarity = np.sum(true_img_embedding*val_captions,axis=1)
    confuser_similarity = np.sum(confuser_img_embedding*val_captions,axis=1)
    print(true_similarity,confuser_similarity)
    print(mg.max(true_similarity),mg.max(confuser_similarity))
    num_correct = np.sum(true_similarity > confuser_similarity)
    acc = num_correct/true_similarity.shape[0]
    print(f'Val Accuracy: {acc}')
