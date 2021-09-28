import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import cv2

img=load_img(r'C:\Users\Goutham N\Desktop\Mask detection\not_wearing mask.jpg',target_size=(224,224))
img=img_to_array(img)
img = preprocess_input(img)
#data = np.array(img)
data=np.expand_dims(img,axis=0)

model=load_model("mask_detector.model")
pred=model.predict(data)

mask = pred[0][0]
nomask = pred[0][1]


if(mask>nomask):
    print("mask detected")
else:
    print("no mask detected")






