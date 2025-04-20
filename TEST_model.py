import cv2
import pickle
import numpy as np
import os

folder_path = "test"
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

#load the train model
with open("model_trained.p",'rb') as pickle_in:
    model = pickle.load(pickle_in)

#preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def equalized(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalized(img)
    img = img/255
    return img

#Label decoding
def getClassName(numberClasses):
    class_name = [
        'No parking', 'Go over the roundabout', 'Right-of-way at the next itersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'No entry', 'Dangerous curve to left', 
        'Dangerous curve to right', 'Double curve', 'Bumby road', 'Slippery road', 'Road narrowed on the right', 'Road work', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Go this way on the right', 'Go this way on the left', 'No left turn',
        'No right turn', 'No automobiles', 'No stopping'
    ]
    return class_name[numberClasses]

#loop through all images in folder
for filename in sorted(os.listdir(folder_path)):
    if filename.lower().ednswith(('.jpg','.png','.jpeg')):
        path = os.path.join(folder_path,filename)
        imgOrigin = cv2.imread(path)

        if imgOrigin is None:
            print(f"Could not read {filename}")
            continue

        #resize image and preprocess
        img = cv2.resize(imgOrigin,(32,32))
        img = preprocessing(img)
        img = img.reshape(1,32,32,1)

        #predict
        predictions = model.predict(img)
        classIndex = int(np.argmax(predictions))
        probabilityVal = np.amax(predictions)

        #Annotate
        text_class = f"Class: {classIndex} - {getClassName(classIndex)}"
        text_proba = f"Probability: {round(probabilityVal * 10,2)}%"

        if probabilityVal > threshold:
            cv2.putText(imgOrigin, text_class, (10,30), font, 0.7,(0,255,0),2)
            cv2.putText(imgOrigin, text_proba, (10,60), font, 0.7, (255,0,0),2)

        #show the result
        cv2.imshow("Result", imgOrigin)
        print(text_class, "|", text_proba)

        key = cv2.waitKey(0)
        if key == 27:
            break
cv2.destroyAllWindows

