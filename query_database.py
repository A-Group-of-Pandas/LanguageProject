from coco import embed_caption, COCO
import io
import requests
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np

def download_image(img_url: str) -> Image:
    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))

def query_database(query, k):
# 1. First, we embed the query in the same 
# way that we embedded the captions (using an IDF-weighted sum of GloVe embeddings).
# 2. Then we compute the dot product of this query’s embedding against all of our image embeddings in our database.
# 3. Return the top k image ids.
# 4. Use that to return the top k images. Maybe we write a function to convert the image ids back to image? idk
    coco = COCO()
    image_data = coco.image_data
    image_embeddings = [data["feature_vector"] for data in image_data.values()]
    query_word_embedding = embed_caption(query)
    dot_product = np.dot(image_embeddings, query_word_embedding)
    # Get the indices for the maximum value. Make sure to get the corresponding image ids using these indices.
    sorted_dot_indxs = dot_product.argsort()[-k:][::-1]
    for i in sorted_dot_indxs:
        # get the image IDs
        top_ID = image_data.keys()[i]
        # get url from COCO data
        url = image_data[top_ID]["url"]
        # show image
        img = download_image(url)
        imshow(img)
    """the top-k cosine-similarities points us to the top-k most relevant images to this query!
     We need image-IDs associated with these matches and then we can fetch their associated URLs fro
    m our COCO data. The code for downloading
    an image is quite simple thanks to the PIL and requests library! """