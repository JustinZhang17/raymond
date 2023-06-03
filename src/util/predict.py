import os
import tensorflow as tf
import numpy as np
import cv2 as cv
import sys

from train_binary import IMAGE_WIDTH as IMAGE_WIDTH_BINARY
from train_binary import IMAGE_HEIGHT as IMAGE_HEIGHT_BINARY
from train_multi import IMAGE_WIDTH as IMAGE_WIDTH_MULTI
from train_multi import IMAGE_HEIGHT as IMAGE_HEIGHT_MULTI

# True or 1= Has Tumor
# False or 0= No Tumor

def find(name, path):
    for root, _, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return False

if __name__ == '__main__':
    if (len(sys.argv) != 4):
        print("Please enter a ML model (any model in the models folder, ie. 'flair-set1-original-0.92-lofp.h5') and an image to predict (any image in the tests folder, ie. 'Original-Flair-HG0001-48-True.png')")
        print("Example: python predict.py flair-set1-original-0.92-lofp.h5 Original-Flair-HG0001-48-True.png")
        sys.exit(1)

    model_path = False
    image_path = False  

    for i, arg in enumerate(sys.argv):
        identification_type = sys.argv[1]
        if (i == 1 and arg.lower() not in ['set1', 'set2']):
            print("Invalid identification method. Try Again")
            print("Use 'set1' or 'set2'")
            print("set1 = binary classification (tumor or no tumor)")
            print("set2 = multiclass classification (no tumor, meningioma tumor, glioma tumor, pituitary tumor)")
            sys.exit(1)

        if (i == 2):
            model_path = find(arg, f'models/{identification_type}')
            if (not model_path):
                print("Invalid model name. Try Again")
                sys.exit(1)

        if (i == 3):
            image_path = find(arg, 'tests')
            if (not image_path):
                print("Invalid image name. Try Again")
                sys.exit(1)


    img = cv.imread(image_path)

    if(identification_type == 'set1'):
        resize = tf.image.resize(img, (IMAGE_HEIGHT_BINARY, IMAGE_WIDTH_BINARY))
    else:
        resize = tf.image.resize(img, (IMAGE_HEIGHT_MULTI, IMAGE_WIDTH_MULTI))

    model = tf.keras.models.load_model(model_path)

    if(identification_type == 'set1'):
        confidence = model.predict(np.expand_dims(resize/255, 0))[0][0]
    else:
        confidence = model.predict(np.expand_dims(resize/255, 0))[0]

    if(identification_type == 'set1'):
        if (confidence > 0.65):
            print(f'Tumor Detected: I am {confidence*100:.2f}% sure')
        elif(confidence < 0.35):
            print(f'This Brain is Healthy: I am {(1-confidence)*100:.2f}% sure')
        else:
            print(f'I am not sure what to think about this one. I am 50/50 on it. Here is my confidence level that it contains a tumor: {confidence*100:.2f}%')
    else:
        print(f'I am {confidence[0]*100:.2f}% sure there is a Glioma Tumor in this brain')
        print(f'I am {confidence[1]*100:.2f}% sure there is a Menigioma Tumor in this brain')
        print(f'I am {confidence[2]*100:.2f}% sure there is no tumor in this brain')
        print(f'I am {confidence[3]*100:.2f}% sure there is a Pituitary Tumor in this brain')