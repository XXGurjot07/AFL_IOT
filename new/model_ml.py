import cv2
import tensorflow  as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
#Initialize model
model = tf.lite.Interpreter(model_path=r"/home/pi/Desktop/AFL Monitor/Models/jul27net99.tflite")
input_details = model.get_input_details()
output_details = model.get_output_details()

#print(input_details)
#print(output_details)
#new input image size  =224*224 (1 Aug)
model.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
model.resize_tensor_input(output_details[0]['index'], (1, 2))
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

def inference(img, my_model):

    preds = ['Fire','no Fire']
    
    i = cv2.resize(img,(224,224))
#    i = i/255.0
    i = np.float32(i)
    
    my_model.set_tensor(input_details[0]['index'], [i])
    my_model.invoke()
    
    output = my_model.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)

    #confidence = output[0][pred]
    #output = preds[pred]
    confidence = output[0][0]
    #return output,confidence
    return confidence













cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Camera Error')
    exit()

out = "---"
    
while cap.isOpened():
    
    success, frame = cap.read()
    
    #cv2.imshow('Footage', frame)
    conf = inference(frame,model)
    #cv2.imshow('Footage', frame)
    #cv2.waitKey(1)
    
    print(conf)

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #org = (50, 50)
    #fontScale = 1
    #color = (255, 0, 0)
    #thickness = 2
    #image = cv2.putText(frame, out, org, font, fontScale, color, thickness, cv2.LINE    
cv2.destroyAllWindows()

