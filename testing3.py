import cv2
import tensorflow as tf
CATEGORIES = ["A", "B", "C","D","E"]
def prepare(file):
    IMG_SIZE = 50
    IMG_SIZE_WIDTH=50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE_WIDTH))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE_WIDTH, 1)
model = tf.keras.models.load_model("CNN.model")
image = prepare("randomB.jpg") #your image path
prediction = model.predict([image])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])# -*- coding: utf-8 -*-

