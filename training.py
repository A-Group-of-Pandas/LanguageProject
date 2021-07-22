from os import replace
from coco import embed_caption, COCO
import numpy as np
import mygrad as mg
from mynn.optimizers.sgd import SGD
from mygrad.nnet.initializers import glorot_normal
from mynn.layers.dense import dense
from mygrad.nnet.losses import margin_ranking_loss
from model import Model
import noggin as lognog


# embed the “true” image \-/
# embed the “confusor” image
# compute similarities (caption and good image, caption and bad image)
# compute loss and accuracy
# take optimization step
coco_dataset = COCO()
annotations = coco_dataset.annotation

print(f'annotations: {annotations}')

model = Model()
# Make sure the optimizer is SGD!!
optim = mg.SGD(model.parameters, learning_rate=0.1)


EPOCHS = 500
BATCH_SIZE = 5
MARGIN = 0.25

triplet_size = EPOCHS * BATCH_SIZE
inputs, captions, confusers = coco_dataset.generate_matrix(triplet_size)
loss_list = []

for EPOCH in range(EPOCHS):
    idxs = np.arange(triplet_size)
    np.random.shuffle(idxs)
    num_correct = 0
    
    for batch_cnt in range(0, triplet_size//BATCH_SIZE):
        
        batch_indices = idxs[batch_cnt*BATCH_SIZE : (batch_cnt + 1)*BATCH_SIZE]

        batch_input = inputs[batch_indices]
        confuser_input = confusers[batch_indices]
        
        # Getting the embeddings for both.
        true_img_embedding = model(batch_input)        
        confuser_img_embedding = model(confuser_input)   # pass in random image

        true_caption_embedding = captions[batch_indices]

        true_similarity = mg.dot(true_img_embedding, true_caption_embedding)
        confuser_similarity = mg.dot(confuser_img_embedding, true_caption_embedding)

        loss = margin_ranking_loss(true_similarity, confuser_similarity, margin=MARGIN)
        loss_list.append(loss)
        loss.backward()

        optim.step()

        num_correct += true_similarity > confuser_similarity + MARGIN
    
    acc = num_correct / BATCH_SIZE

    if EPOCH % 50 == 0:
            print(f'EPOCH: {EPOCH}')
            print(f'Accuracy: {acc}, Loss: {np.mean(loss_list)}')
