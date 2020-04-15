import os
import cv2
import tensorflow as tf
try:
    from PIL import Image
except ImportError:
    import Image


import matplotlib.pyplot as plt
import numpy as np
UPLOAD_FOLDER = '/static/uploads/'
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


CATEGORIES = ["Black-grass","Charlock","Cleavers","Common Chickweed","Common wheat","Fat Hen","Loose Silky-bent","Maize","Scentless Mayweed","Shepherds Purse","Small-flowered Cranesbill","Sugar beet"]

def prepare(file):
    try:
        IMG_SIZE = 128
        IMG_SIZE_WIDTH=128
        img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img_array = img_array / 255.0
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE_WIDTH))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE_WIDTH, 1)

    except Exception as e:
        print(str(e))



def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    # text = pytesseract.image_to_string(Image.open(filename), config="-c tessedit"
    #                                                                 "_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #                                                                 " --psm 10"
    #                                                                 " -l osd"
    #                                                                 " ")  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image

    model = tf.keras.models.load_model("CNN.model")


    # print(image)
    # new code
    img = cv2.imread('./static/uploads/'+filename)
    plt.imshow(img)

    segmanted_img_name=filename[:-4]+"_segmanted"

    image_mask = create_mask_for_plant(img)
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)



    cv2.imwrite(os.getcwd() + UPLOAD_FOLDER+segmanted_img_name+".png", image_sharpen)

    image = prepare('./static/uploads/' + segmanted_img_name+".png")  # your image path segmented image path

    #
    assert not isinstance(image, type(None)), 'image not found'
    #image1=prepare(image)
    prediction = model.predict([image])
    prediction = list(prediction[0])
    text= CATEGORIES[prediction.index(max(prediction))]  # -*- coding: utf-8 -*-

    return text  # Then we will print the text in the image


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 40])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


def segment(img):
    # image_mask = create_mask_for_plant(img)
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)
    return image_sharpen