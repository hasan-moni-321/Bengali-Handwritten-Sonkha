# Loading necessary library
import pickle 
import numpy as np
import cv2 as cv
import imageio
import pyttsx3

from keras.models import model_from_json

# for reaading text
engine = pyttsx3.init()

# load json and create model
json_file = open('/home/hasan/Bengali_Digit.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


width = 640
height = 480
threshold = .80


# Preprocessing image
def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img


# Capturing image using webcam
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    success, img_original = cap.read()
    img = np.asarray(img_original)
    img = cv.resize(img, (32,32), model='symmetric', preserve_range=True)
    img = preProcessing(img)
    cv.imshow("Webcam Imge", img)
    img = img.reshape(1,32,32,1)
    
    # predict
    class_index = int(model.predict_classes(img))
    #print(class_index)
    predictions = model.predict(img)
    #print(predictions)
    prob_value = np.amax(predictions)
    print(class_index, prob_value)
    
    if prob_value > threshold:
        class_number = str(class_index)
        engine.say(class_number)
        engine.runAndWait() 

        cv.putText(img_original, str(class_index)+ str(prob_value), 
                   (50,50), cv.FONT_HERSHEY_COMPLEX,
                   1, (0,0,255), 1)
    
    cv.imshow('Original Image', img_original)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
