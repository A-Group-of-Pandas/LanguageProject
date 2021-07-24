from coco import COCO
coco_dataset = COCO()
val_inputs, val_captions, val_confusers = coco_dataset.generate_matrix(10)
#print(val_captions)


print(coco_dataset.generate_matrix2(10))