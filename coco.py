from InferenceDatabase import InferenceDatabase
from pathlib import Path
import json
import pickle
from typing import ValuesView
import mygrad as mg
from gensim.models import KeyedVectors
import string
import re
import numpy as np
from collections import Counter

# Dimension of word vectors


class COCO:
    # maybe store image id, caption id, captions, feature vectors here?
    def __init__(self, database_dir:str = None, glove_path: str = "glove.6B.200d.txt.w2v", coco_path: str =  "captions_train2014.json", feature_path: str = "resnet18_features.pkl") -> None:
        self.annotation = None
        self.image_data = None
        self.D = 200
        if database_dir is None:
            print('creating database')
            self.read_data(coco_path,glove_path,feature_path)
        else:
            print('loading database')
            self.load_database()
    
    def read_data(self, coco_path, glove_path, feature_path):
        self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)
        self.load_coco(coco_path)
        self.load_feature_vectors(feature_path)
    
    def load_coco(self, filename: str):
        with Path(filename).open() as f:
            coco_data = json.load(f)
        annotation = coco_data['annotations']
        captions = [caption_info['caption'] for caption_info in annotation]
        captions = self.embed_caption(captions)
        self.annotation = {caption_info['id'] : {'image_id': caption_info['image_id'], 'caption': caption} for caption_info,caption in zip(annotation,captions)}        
        image_data = coco_data['images']
        self.image_data = {image['id'] : {'url': image['coco_url'],'captions':[]} for image in image_data}
        # image_data stores url (ID) --> captions (empty)
        
        for i,captions in self.annotation.items():
            #print(captions['image_id'], len(self.image_data))
            self.image_data[captions['image_id']]['captions'].append(i)
            # image_data now stores url (ID) --> captions
    
    def load_feature_vectors(self, filename: str):
        with Path(filename).open('rb') as f:
            resnet18_features = pickle.load(f)
        # note: resnet model somehow doesn't have features for all images
        del_id = []
        for i in self.image_data.keys():
            if i in resnet18_features:
                self.image_data[i]['feature_vector'] = resnet18_features[i][0]
            else:
                del_id.append(i)

        for i in del_id:
            del self.image_data[i]

    def get_inference_database(self, model):
        image_data = {}
        image_data = {image_id : {} for image_id in self.image_data.keys()}
        with mg.no_autodiff:
            for image_id,values in self.image_data.items():
                vec = values['feature_vector']
                emb = model(vec[None,...])[0]
                image_data[image_id]['image_embed'] = emb
                image_data[image_id]['url'] = values['url']
        return InferenceDatabase(image_data)

    def save_database(self, dir_path: str = 'database'):
        with open(dir_path+'/image_data.txt','wb') as f:
            pickle.dump(self.image_data, f)
        with open(dir_path+'/annotation.txt','wb') as f:
            pickle.dump(self.annotation, f)

    def load_database(self, dir_path: str = 'database',glove_path:str = 'glove.6B.200d.txt.w2v'):
        with open(dir_path+'/image_data.txt','rb') as f:
            self.image_data = pickle.load(f)
        with open(dir_path+'/annotation.txt','rb') as f:
            self.annotation = pickle.load(f)
        self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)

    def get_data(self): 
        #image-ID -> shape-(512,) descriptor
        feature_values = [value['feature_vector'] for image_id,value in self.image_data.values()]
        captions = [value['captions'] for image_id,value in self.image_data.values()]
        return feature_values, captions

    def generate_matrix(self, triplet_size: int):
        image_ids = list(self.image_data.keys())
        #print(len(image_ids))
        chosen_image_ids = np.random.choice(image_ids,size=triplet_size,replace=True)
        #print(chosen_image_ids)
        chosen_confuser_ids = np.random.choice(image_ids,size=triplet_size,replace=True)
        chosen_images = [self.image_data[image_id]['feature_vector'] for image_id in chosen_image_ids]
        #print(self.image_data[198611]['feature_vector'])
        chosen_confusers = [self.image_data[image_id]['feature_vector'] for image_id in chosen_confuser_ids]
        chosen_captions = [self.annotation[np.random.choice(self.image_data[image_id]['captions'])]['caption'] for image_id in chosen_image_ids]
        #print('caption size: ' +str(self.annotation[658288]['caption']))
        return np.array(chosen_images), np.array(chosen_captions), np.array(chosen_confusers)

    def get_word_vector(self,word):
        if word in self.glove:
            return self.glove[word]
        return np.zeros(self.D)

    # text - specific caption to embed
    # captions - all captions in COCO dataset
    def embed_caption(self, captions):
        punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
        captions = [punc_regex.sub('', caption.lower()).split() for caption in captions]
        phrase_vector = np.zeros((len(captions),self.D))

        counter = Counter()
        for caption in captions:
            words = set(caption)
            for word in words:
                counter[word]+=1
        
        for i, caption in enumerate(captions):
            for word in caption:
                word_vector = self.get_word_vector(word)
                idf = np.log10(len(captions) / counter[word])
                phrase_vector[i] += idf * word_vector
        norm_vector = phrase_vector / np.sqrt(np.einsum("ij, ij -> i", phrase_vector, phrase_vector)).reshape(-1, 1)
        return norm_vector


# coco = COCO()
# coco.save_database()
