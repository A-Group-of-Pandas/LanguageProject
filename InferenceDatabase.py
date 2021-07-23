class InferenceDatabase:
    def __init__(self, image_data = {}) -> None:
        self.image_data = image_data
        # map from image_id to image_url and image_embed
