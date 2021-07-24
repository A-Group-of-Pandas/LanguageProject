from InferenceDatabase import InferenceDatabase
from coco import COCO
import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import Model
from GloveDatabase import GloveDatabase

class Query:
    def __init__(self) -> None:
        self.glove_database = GloveDatabase(database_path='database')
        self.inference_database = InferenceDatabase()
        self.inference_database.load_database()
    
    def download_image(self, img_url: str) -> Image:
        response = requests.get(img_url)
        return Image.open(io.BytesIO(response.content))

    def query_database(self, query, k=4):
        keys, image_embeddings = self.inference_database.get_numpy()
        #print(query)
        query_word_embedding = self.glove_database.embed_caption(query)
        
        #print(query_word_embedding)
        #print(image_embeddings.shape, query_word_embedding.shape)
        dot_product = image_embeddings @ query_word_embedding
        #print(dot_product.shape)
        #print(dot_product[0])
        # Get the indices for the maximum value. Make sure to get the corresponding image ids using these indices.
        """
        max_list = sorted(dot_product, reverse=True)[:k] 
        top_indices = dot_product.index(max_list)
        """
        
        sorted_dot_indxs = np.argsort(dot_product, axis=0)
        #print(dot_product[sorted_dot_indxs[:5]])
        #print(dot_product[sorted_dot_indxs[-5:]])
        sorted_dot_indxs = sorted_dot_indxs[-int(k):][::-1]
        fig = plt.figure()
        for i in sorted_dot_indxs:
            # get the image IDs
            top_ID = keys[i]
            #print(self.inference_database.image_data[top_ID]['image_embed'] @ query_word_embedding)
            #print(top_ID)
            # get url from COCO data
            url = self.inference_database.image_data[top_ID]["url"]
            # show image
            img = self.download_image(url)
            plt.imshow(img)
            plt.show()

