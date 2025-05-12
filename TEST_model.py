import cv2
import pickle
import numpy as np

# Camera config
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.5  
font = cv2.FONT_HERSHEY_SIMPLEX
from tensorflow.keras.models import load_model



model = load_model("model_trained.keras")

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Preprocessing
def grayScale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayScale(img)
    img = equalize(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = img / 255.0
    return img


# Class names
def getClassName(classIndex):
    classes = [
        'No parking', 'Go over the roundabout', 'Right-of-way at the next intersection',
        'Priority road', 'Yield', 'Stop', 'No vehicles', 'No entry',
        'Dangerous curve to left', 'Dangerous curve to right', 'Double curve',
        'Bumpy road', 'Slippery road', 'Road narrowed on the right', 'Road work',
        'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
        'Go this way on the right', 'Go this way on the left', 'No left turn',
        'No right turn', 'No automobiles', 'No stopping'
    ]
    return classes[classIndex] if 0 <= classIndex < len(classes) else 'Unknown'

# Main loop
try:
    while True:
        success, imgOriginal = cap.read()
        if not success:
            print("Cannot get pic from camera")
            break

        img = cv2.resize(imgOriginal, (32, 32))  
        img = preprocessing(img)

        imgInput = img.reshape(1, 32, 32, 1)  

        predictions = model.predict(imgInput)
        classIndex = int(np.argmax(predictions))
        probabilityVal = np.max(predictions)

        print(f"Predictions: {predictions}")  

        imgDisplay = (img * 255).astype(np.uint8)
        cv2.imshow("Process Image", imgDisplay)

        cv2.putText(imgOriginal, "CLASS:", (20, 35), font, 0.75, (0, 0, 255), 2)
        cv2.putText(imgOriginal, "PROBABILITY:", (20, 75), font, 0.75, (0, 0, 255), 2)

        if probabilityVal > threshold:
            className = getClassName(classIndex)
            cv2.putText(imgOriginal, f"{classIndex} {className}", (120, 35), font, 0.75, (0, 0, 255), 2)
            cv2.putText(imgOriginal, f"{round(probabilityVal*100,2)}%", (180, 75), font, 0.75, (0, 0, 255), 2)

        cv2.imshow("Result", imgOriginal)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stop program")
            break
except KeyboardInterrupt:
    print("Stop program")
finally:
    cap.release()
    cv2.destroyAllWindows()