import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Setup data
path = "dataset/myData"  
label_file = 'labels.csv'  
batch_size_val = 64
steps_per_epoch_val = 260
epochs_val = 10
imageDim = (32,32,3)  
testRatio = 0.25         
validateRatio = 0.25     

# Importing images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total classes detected:", len(myList))
numberClasses = len(myList)
print("Importing Classes.......")
for x in range(0, len(myList)):
    imgList = os.listdir(f"{path}/{count}")
    for y in imgList:
        current = cv2.imread(f"{path}/{count}/{y}")
        images.append(current)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=validateRatio)

# Check shapes of data
print("Data Shapes")
print("Train", X_train.shape, y_train.shape)
print("Validation", X_validate.shape, y_validate.shape)
print("Test", X_test.shape, y_test.shape)

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equals to the number of labels in training set"
assert(X_validate.shape[0] == y_validate.shape[0]), "The number of images is not equals to the number of labels in validation set"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equals to the number of labels in test set"
assert(X_train.shape[1:] == (imageDim)), "The dimensions of the Training images are wrong"
assert(X_validate.shape[1:] == (imageDim)), "The dimensions of the Validation images are wrong"
assert(X_test.shape[1:] == (imageDim)), "The dimensions of the Test images are wrong"

# Read csv file
data = pd.read_csv(label_file)
print("data shape", data.shape, type(data))

# Preprocessing the images
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)       # Convert to grayscale
    img = equalize(img)        # Standardize lighting in image
    img = img / 255.0          # Normalize to values between 0 and 1
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validate = np.array(list(map(preprocessing, X_validate)))
X_test = np.array(list(map(preprocessing, X_test)))

# Reshaping images for input to CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Data augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# One-hot encoding of labels
y_train = to_categorical(y_train, numberClasses)
y_validate = to_categorical(y_validate, numberClasses)
y_test = to_categorical(y_test, numberClasses)

# Convolutional Neural Network model
def myModel():
    no_of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_Nodes = 500
    
    model = Sequential()
    model.add(Conv2D(no_of_Filters, size_of_Filter, input_shape=(imageDim[0], imageDim[1], 1), activation='relu'))
    model.add(Conv2D(no_of_Filters, size_of_Filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    
    model.add(Conv2D(no_of_Filters//2, size_of_Filter2, activation='relu'))
    model.add(Conv2D(no_of_Filters//2, size_of_Filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))   
    
    model.add(Flatten())
    model.add(Dense(no_of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberClasses, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training the model
if os.path.exists("model_trained.keras"):
    print("Loading saved model...")
    model = load_model("model_trained.keras")
else:
    print("No saved model found, training new model...")
    model = myModel()  
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val,
                    validation_data=(X_validate, y_validate),
                    shuffle=True)

# Plot training history
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Evaluate the model on test data
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Save model in .keras format
model.save("model_trained.keras")  

# Verify if model has been saved successfully
if os.path.exists("model_trained.keras"):
    print("Model saved successfully!")
else:
    print("Model save failed.")