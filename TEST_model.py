import cv2
import pickle
import numpy as np
import os

#parameter for cam
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

#setup camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

#import the trained model
pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)

def grayScale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayScale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(numberClasses):
    if numberClasses == 0:  return 'No parking'
    elif numberClasses == 1:    return 'Go over the roundabout'
    elif numberClasses == 2:    return 'Right-of-way at the next intersection'
    elif numberClasses == 3:    return 'Priority road'
    elif numberClasses == 4:    return 'Yield'
    elif numberClasses == 5:    return 'Stop'
    elif numberClasses == 6:    return 'No vehicles'
    elif numberClasses == 7:    return 'No entry'
    elif numberClasses == 8:    return 'Dangerous curve to left'
    elif numberClasses == 9:    return 'Dangerous curve to right'
    elif numberClasses == 10:    return 'Double curve'
    elif numberClasses == 11:    return 'Bumby road'
    elif numberClasses == 12:    return 'Slippery road'
    elif numberClasses == 13:    return 'Road narrowed on the right'
    elif numberClasses == 14:    return 'Road work'
    elif numberClasses == 15:    return 'Pedestrians'
    elif numberClasses == 16:    return 'Childen crossing'
    elif numberClasses == 17:    return 'Bicycles crossing'
    elif numberClasses == 18:    return 'Turn right ahead'
    elif numberClasses == 19:    return 'Turn left ahead'
    elif numberClasses == 20:    return 'Ahead only'
    elif numberClasses == 21:    return 'Go straight or right'
    elif numberClasses == 22:    return 'Go straight or left'
    elif numberClasses == 23:    return 'Go this way on the right'
    elif numberClasses == 24:    return 'Go this way on the left'
    elif numberClasses == 25:    return 'No left turn'
    elif numberClasses == 26:    return 'No right turn'
    elif numberClasses == 27:    return 'No automobiles'
    elif numberClasses == 28:    return 'No stopping'
    else:    return 'Unknown'

while True:
    #read image
    success, imgOrigin = cap.read()

    #proccess image
    img = np.asarray(imgOrigin)
    img = cv2.resize(img,(32,32))
    img = preprocessing(img)
    cv2.imshow("Proccess Image", img)
    img = img.reshape(1,32,32,1)
    cv2.putText(imgOrigin, "CLASS: ", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(imgOrigin,"PROBABILITY: ",(20,75), font, 0.75, (0,0,255),2,cv2.LINE_AA)

    #Predict image
    predicted = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityVal = np.amax(predicted)
    if probabilityVal > threshold:
        cv2.putText(imgOrigin,str(classIndex)+" "+str(getClassName(classIndex)),(120,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(imgOrigin,str(round(probabilityVal*100,2))+"%",(180,75),font,0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow("Result", imgOrigin)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break