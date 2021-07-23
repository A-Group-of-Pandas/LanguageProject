import Model
import coco

class Embeddings:
    def __init__(self, vectors) -> None:
        coco = COCO(database_dir='database')
        image_data = coco.image_data
        image_embeddings = [data["feature_vector"] for data in image_data.values()]
        model = Model()
        model.load 
        image_embeddings = model(image_embeddings)
        self.database = image_embeddings