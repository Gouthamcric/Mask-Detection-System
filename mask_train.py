from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

init_lr = 1e-4 #INITIAL LEARNING RATE(0.00001)
epochs = 20
bs = 32
#data preprocessing
DIRECTORY = r"C:\Users\Goutham N\Desktop\Mask detection\dataset"
CATEGORIES = ["with_mask","without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.20,stratify=labels)
#end of preprocessing

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#model building

convolutional_layer = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

model = convolutional_layer.output
model = AveragePooling2D(pool_size=(7, 7))(model)

#input layer
model = Flatten(name="flatten")(model)
#hidden layer
model = Dense(128, activation="relu")(model)
model = Dropout(0.5)(model) #to avoid overfitting of model
#output layer
model = Dense(2, activation="softmax")(model)

model = Model(inputs=convolutional_layer.input, outputs=model)


# loop over all layers in the convolutional layer and freeze them so they will not be updated during the training process
for layer in convolutional_layer.layers:
	layer.trainable = False

#compiling model
opt = Adam(lr=init_lr, decay=init_lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#training model
m = model.fit(aug.flow(trainX, trainY, batch_size=bs),steps_per_epoch=len(trainX)//bs,validation_data=(testX, testY),validation_steps=len(testX)//bs,epochs=epochs)

model.save("mask_detector.model", save_format="h5")
n=epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, n), m.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), m.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n), m.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n), m.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


