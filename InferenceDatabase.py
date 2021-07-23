import pickle
import numpy as np

class InferenceDatabase:
    def __init__(self, image_data = {}) -> None:
        self.image_data = image_data
        # map from image_id to image_url and image_embed

    def save_database(self, dir_path: str = 'database'):
        with open(dir_path+'/inference.txt','wb') as f:
            pickle.dump(self.image_data, f)

    def load_database(self, dir_path: str = 'database'):
        with open(dir_path+'/inference.txt','rb') as f:
            self.image_data = pickle.load(f)
    
    def get_numpy(self):
        values = np.array([data["image_embed"] for data in self.image_data.values()])
        keys = np.array(list(self.image_data.keys()))
        return keys, values