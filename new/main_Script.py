import time
import cv2
import tensorflow  as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import concurrent.futures
import serial
import skfuzzy as fuzzy
from skfuzzy import control
from skfuzzy.control import ControlSystem, ControlSystemSimulation, Rule
import RPi.GPIO as GPIO

model_arr = np.zeros(10)
fuzzy_arr = np.zeros(10)
net_arr = np.zeros(10)
return_arr = np.zeros(3)

def main():
    
    global run_inference, run_fuzzy
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
        i = i/255.0
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

    #Describing Fuzzy System            
    flame = control.Antecedent(np.arange(0,1024,1),'flame')
    sd = control.Antecedent(np.arange(0,1024,1), 'sd')
    fire = control.Consequent(np.arange(0,101,1), 'fire')

    #Defining Fuzzyficaion System (Membership Functions)
    flame['Negative'] = fuzzy.trapmf(flame.universe, [0,0,200,500])
    flame['Far'] = fuzzy.trapmf(flame.universe,[350,440,600,650])
    flame['Large'] = fuzzy.trapmf(flame.universe,[500,650,1023,1023])

    sd['Negative'] = fuzzy.trapmf(sd.universe, [0,0,250,500])
    sd['Low'] = fuzzy.trapmf(sd.universe, [400,400,600,700])
    sd['High'] = fuzzy.trapmf(sd.universe, [550,800,1023,1023])

    fire.automf(2, names=['No Fire','Fire'])

    #Describing Fuzzy Inference System using Rules
    rule1 = Rule(sd['Negative'] & flame['Negative'], fire['No Fire'])
    rule2 = Rule(sd['Negative'] & flame['Far'], fire['Fire'])
    rule3 = Rule(sd['Negative'] & flame['Large'], fire['Fire'])
    rule4 = Rule(sd['Low'] & flame['Negative'], fire['No Fire'])
    rule5 = Rule(sd['Low'] & flame['Far'], fire['Fire'])
    rule6 = Rule(sd['Low'] & flame['Large'], fire['Fire'])
    rule7 = Rule(sd['High'] & flame['Negative'], fire['No Fire'])
    rule8 = Rule(sd['High'] & flame['Far'], fire['Fire'])
    rule9 = Rule(sd['High'] & flame['Large'], fire['Fire'])

    rule_book = control.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9])
    system = ControlSystemSimulation(rule_book)

    #Defining Processes using Target functions
    def run_inference():
        
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
            cap.release()
            return conf
        
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #org = (50, 50)
            #fontScale = 1
            #color = (255, 0, 0)
            #thickness = 2
            #image = cv2.putText(frame, out, org, font, fontScale, color, thickness, cv2.LINE_AA)
           
        cap.release()    
        cv2.destroyAllWindows()

    def run_fuzzy():
        #Inputs from Sensors
        
        if __name__ == '__main__':
            ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            ser.flush()
                
        while True:
            if ser.in_waiting > 0:
                row_data = ser.readline().decode('ascii').rstrip()
                split = row_data.split(" ")
                data_list = list(map(int, split))
            
                flame_val = max(data_list[0:5])
                sd_val = data_list[5]
        
                #print(data_list, flame_val, sd_val)
        
                system.input['flame'] = flame_val
                system.input['sd'] =  sd_val
                system.compute()
            
                return system.output['fire']/100

    # One net inference for aproxx 4*10 = 45seconds (added processing delay)
    
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)

    while True:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = executor.submit(run_inference)
            f2 = executor.submit(run_fuzzy)
        
            print("Model Result = ",f1.result())
            print("Sensor Result = ",f2.result())
        
            time.sleep(2)
            
            model_arr = np.roll(model_arr,1)
            fuzzy_arr = np.roll(fuzzy_arr,1)
            net_arr = np.roll(net_arr,1)
            
            model_arr[0] = round(f1.result(),2)
            fuzzy_arr[0] = round(f2.result(),2)
            net_arr[0] = ((0.80*model_arr[0]) + (0.20*fuzzy_arr[0]))
            
            return_arr = [model_arr[0], fuzzy_arr[0], net_arr[0]]
            #(return_arr)
            sum = np.sum(net_arr)
            
            print("Model Array = ",model_arr)
            print("Fuzzy Array = ",fuzzy_arr)
            print("Net Array = ",net_arr)
            print("Sum = ", sum)
            print("========================================")
            
            #Choose factor
            if sum >= 7.0:
                GPIO.output(18, GPIO.HIGH)
            else:
                GPIO.output(18, GPIO.LOW)            

            return return_arr

while __name__=="__main__":
    print("Start")
    main()
    print("Finish")
















