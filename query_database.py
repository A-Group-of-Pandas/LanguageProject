from coco import COCO
import io
import requests
from PIL import Image
from matplotlib.pyplot import pyplot as plt
import numpy as np
from model import Model

def download_image(img_url: str) -> Image:
    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))

def query_database(query, k=4):
    coco = COCO(database_dir='database')
    image_data = coco.image_data
    image_embeddings = np.array([data["image_embed"] for data in image_data.values()])
    query_word_embedding = coco.embed_caption([query])[0]
    dot_product = np.dot(image_embeddings, query_word_embedding)
    # Get the indices for the maximum value. Make sure to get the corresponding image ids using these indices.
    sorted_dot_indxs = dot_product.argsort()[-k:][::-1]
    fig = plt.figure()
    for i in sorted_dot_indxs:
        # get the image IDs
        top_ID = image_data.keys()[i]
        # get url from COCO data
        url = image_data[top_ID]["url"]
        # show image
        img = download_image(url)
        plt.imshow(img)

query = input("What would you like to search? ")
query_database(query)