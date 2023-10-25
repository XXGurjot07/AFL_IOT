import cv2
import tensorflow  as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import time


#Initialize model
model = tf.lite.Interpreter(model_path=r"/home/pi/Desktop/AFL Monitor/Models/tfiltemodelSN95sacc.tflite")
input_details = model.get_input_details()
output_details = model.get_output_details()

#print(input_details)
#print(output_details)

model.resize_tensor_input(input_details[0]['index'], (1, 227, 227, 3))
model.resize_tensor_input(output_details[0]['index'], (1, 2))
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

def inference(img, my_model):

    preds = ['Fire','no Fire']
    
    #img = cv2.imread(r"/home/pi/Desktop/AFL Monitor/new/4.jpg")
    i = cv2.resize(img,(227,227))
    i = i/255.0
    i = np.float32(i)
    
    my_model.set_tensor(input_details[0]['index'], [i])
    my_model.invoke()
    
    output = my_model.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)

    confidence = output[0][pred]
    output = preds[pred]
    
    #log = open(r'/home/pi/Desktop/AFL Monitor/new/logs.csv','a')
    #log.write(f'Time:{str(strftime("%Y-%m-%d %H:%M:%S"))},Output: {output}, Confidence: {confidence}\n')
    #log.close()
    
    return output,confidence    
    
    
    
    
    
    
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Camera Error')
        exit()

    i = 0
    out = "no Fire"

    while cap.isOpened():
    
        success, frame = cap.read()
        i+=1
    
        #cv2.imshow('Footage', frame)
    
        if i % 30 == 0:
            out,conf = inference(frame,model)
            i = 0
            print(out, conf)
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
    
        image = cv2.putText(frame, out, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Footage', image)
        cv2.waitKey(1)
    
        #sleep(0.625)

    
    cap.release()
    cv2.destroyAllWindows()