from os import replace
from coco import COCO
import numpy as np
import mygrad as mg
from mynn.optimizers.sgd import SGD
from mygrad.nnet.initializers import glorot_normal
from mynn.layers.dense import dense
from mygrad.nnet.losses import margin_ranking_loss
from model import Model


# embed the “true” image \-/
# embed the “confusor” image
# compute similarities (caption and good image, caption and bad image)
# compute loss and accuracy
# take optimization step
coco_dataset = COCO(database_dir='database')

annotations = coco_dataset.annotation

# print(f'annotations: {annotations}')

model = Model()
# Make sure the optimizer is SGD!!
optim = SGD(model.parameters, learning_rate=0.001, momentum=0.9)


EPOCHS = 500  # one just for testing
BATCH_SIZE = 32
MARGIN = 0.25

train_size = 30_000
val_size = 10_000
train_inputs, train_captions, train_confusers = coco_dataset.generate_matrix(train_size)
val_inputs, val_captions, val_confusers = coco_dataset.generate_matrix(val_size)
loss_list = []

for EPOCH in range(EPOCHS):
    idxs = np.arange(train_size)
    np.random.shuffle(idxs)
    num_correct = 0
    
    for batch_cnt in range(0, train_size//BATCH_SIZE):
        
        batch_indices = idxs[batch_cnt*BATCH_SIZE : (batch_cnt + 1)*BATCH_SIZE]
        # print(type(batch_indices))
        # print(len(inputs))
        batch_input = train_inputs[batch_indices]
        # print(batch_input.shape)
        confuser_input = train_confusers[batch_indices]
        
        # Getting the embeddings for both.
        true_img_embedding = model(batch_input) 
               
        confuser_img_embedding = model(confuser_input)   # pass in random image

        true_caption_embedding = train_captions[batch_indices]
        # (5,200) * (5,200)
        # (5)
        # (?,200)
        # print(true_caption_embedding.shape, true_img_embedding.shape)
        true_similarity = mg.einsum("ni,ni -> n", true_img_embedding, true_caption_embedding)
        #true_similarity = mg.matmul(true_img_embedding, true_caption_embedding)
        confuser_similarity = mg.einsum("ni,ni -> n", confuser_img_embedding, true_caption_embedding)
        #confuser_similarity = mg.matmul(confuser_img_embedding, true_caption_embedding)

        loss = margin_ranking_loss(true_similarity, confuser_similarity,1, margin=MARGIN)
        loss_list.append(loss)
        loss.backward()

        optim.step()

        num_correct = np.sum(true_similarity > confuser_similarity)
    
    acc = (num_correct / BATCH_SIZE) * 100

    if EPOCH % 50 == 0:
        print(f'EPOCH: {EPOCH}')
        print(f'Accuracy: {acc}%, Loss: {np.mean(loss_list)}')
        
        with mg.no_autodiff:
            true_img_embedding = model(val_inputs)    
            confuser_img_embedding = model(val_confusers) 
            true_caption_embedding = val_captions
            true_similarity = np.einsum("ni,ni -> n", true_img_embedding, val_captions)
            confuser_similarity = np.einsum("ni,ni -> n", confuser_img_embedding, val_captions)
            loss = margin_ranking_loss(true_similarity, confuser_similarity,1, margin=MARGIN)
            num_correct = np.sum(true_similarity > confuser_similarity)
            acc = num_correct/true_similarity.shape[0]
            print(f'Val Loss: {loss.item()}')
            print(f'Val Accuracy: {acc}')

model.save_model()
