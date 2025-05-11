import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Setup data
path = "myData"
label_file = 'labels.csv'
batch_size_val = 32
steps_per_epoch_val = 1000
epochs_val = 10
imageDim = (32,32,3)
testRatio = 0.3         
validateRatio = 0.3     

#importing of the images
images = []
classNo = []
myList = os.listdir(path)
print("Total classes detected:",len(myList))
numberClasses = len(myList)
print("Importing Classes.......")
for class_id in range(numberClasses):
    imgList = os.listdir(f"{path}/{class_id}")
    for img_name in imgList:
        img = cv2.imread(f"{path}/{class_id}/{img_name}")
        img = cv2.resize(img,(32,32))
        images.append(img)
        classNo.append(class_id)
print(" ")
images = np.array(images)
classNo = np.array(classNo)

#spliting data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,test_size=validateRatio)

#preprocessing the images
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)        #convert to grayscale
    img = equalize(img)         #standardize the lighting in an image
    img = img/255.0              #to nomalize values between 0 and 1 instead of 0 to 255
    return img

X_train = np.array(list(map(preprocessing,X_train)))
X_validate = np.array(list(map(preprocessing,X_validate)))
X_test = np.array(list(map(preprocessing,X_test)))

# reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validate = X_validate.reshape(X_validate.shape[0],X_validate.shape[1], X_validate.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)

#One-hot encode labels
y_train = to_categorical(y_train,numberClasses)
y_validate = to_categorical(y_validate,numberClasses)
y_test = to_categorical(y_test,numberClasses)

#augmentation of images: to makeit more generic
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)

#convolution neural network model
def myModel():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberClasses, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define path to save/load model
model_path = "model_trained.keras"

# Load existing model if available
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Creating new model...")
    model = myModel()

# Train or continue training
history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_validate, y_validate),
    shuffle=True
)

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# Save model using Keras
model.save(model_path)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()