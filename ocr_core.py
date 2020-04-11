
import cv2
import tensorflow as tf
try:
    from PIL import Image
except ImportError:
    import Image
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


CATEGORIES = ["Black-grass","Charlock","Cleavers","Common Chickweed","Common wheat","Fat Hen","Loose Silky-bent","Maize","Scentless Mayweed","Shepherds Purse","Small-flowered Cranesbill","Sugar beet"]

def prepare(file):
    try:
        IMG_SIZE = 50
        IMG_SIZE_WIDTH=50
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
    image = prepare('./static/uploads/'+filename) # your image path
    print(image)
    assert not isinstance(image, type(None)), 'image not found'
    #image1=prepare(image)
    prediction = model.predict([image])
    prediction = list(prediction[0])
    text= CATEGORIES[prediction.index(max(prediction))]  # -*- coding: utf-8 -*-

    return text  # Then we will print the text in the image
