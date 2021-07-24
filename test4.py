from numpy.core.numeric import indices
from coco import COCO
from GloveDatabase import GloveDatabase
from InferenceDatabase import InferenceDatabase
import numpy as np
import mygrad as mg
from model import Model

model = Model()
model.load_model('model_params.npy')
coco = COCO(database_dir='database')
glove =  GloveDatabase(database_path='database')
inference_database = InferenceDatabase()
inference_database.load_database()

anno = coco.annotation[32]
image_id = anno['image_id']
print(coco.image_data[image_id]['url'])
word_vec = glove.embed_caption('cat')
with mg.no_autodiff:
    image_ids, image_embeddings = inference_database.get_numpy()
    # image_embeddings = []
    # image_ids = []
    # for image_id, values in coco.image_data.items():
    #     image_ids.append(image_id)
    #     image_embeddings.append(model(values['feature_vector'][None,...]))
    # image_embeddings = np.concatenate(image_embeddings,axis=0)
    #print(image_embeddings.data)
    query_word_embedding = word_vec
    print(query_word_embedding.shape)
    dot_product = image_embeddings @ mg.tensor(query_word_embedding)
    argmax = np.argmax(dot_product)
    print(np.max(dot_product))
    print(coco.image_data[image_ids[argmax]]['url'])
    # indices = np.argsort(dot_product,axis=0)
    # print(dot_product[:5])
    # print(dot_product[-5:])
    # print(coco.image_data[indices[-1]]['url'])
