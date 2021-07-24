import pickle
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
import string
import re
import mygrad as mg

class GloveDatabase:
    def __init__(self, database_path:str = None, glove_path:str = 'glove.6B.200d.txt.w2v') -> None:
        self.D = 200
        self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)
        if database_path is None:
            self.counter = Counter()
            self.total_caption = 0
        else:
            print('load glove database')
            self.load_database(database_path)
    
    def save_database(self, dir_path:str = 'database'):
        with open(dir_path+'/glove_database.txt','wb') as f:
            pickle.dump([self.counter,self.total_caption],f)
    
    def load_database(self, dir_path:str = 'database'):
        with open(dir_path+'/glove_database.txt','rb') as f:
            self.counter,self.total_caption = pickle.load(f)

    def embed_caption(self, caption):
        punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
        caption = punc_regex.sub('', caption.lower()).split()
        phrase_vector = np.zeros((1,self.D))
        for word in caption:
            if word in self.counter:
                word_vector = self.get_word_vector(word)
                idf = np.log10(self.total_caption / self.counter[word])
                phrase_vector[0] += idf * word_vector
            else:
                print(f'cant find {word}')
        
        return (phrase_vector/np.sqrt(np.einsum("ij, ij -> i", phrase_vector, phrase_vector)).reshape(-1, 1))[0]
    
    # text - specific caption to embed
    # captions - all captions in COCO dataset
    def embed_captions(self, captions):
        punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
        captions2 = [punc_regex.sub('', caption.lower()).split() for caption in captions]
        phrase_vector = np.zeros((len(captions),self.D))
        self.total_caption = len(captions)
        self.counter = Counter()
        for caption in captions2:
            words = set(caption)
            for word in words:
                self.counter[word]+=1

        for i, caption in enumerate(captions):
            phrase_vector[i] = self.embed_caption(caption)
        return phrase_vector
    
    def get_word_vector(self,word):
        if word in self.glove:
            return self.glove[word]
        return np.zeros(self.D)