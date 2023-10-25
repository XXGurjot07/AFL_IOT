import time
import serial
import concurrent.futures
import tflite_runtime.interpreter as tflite
from cv2 import resize as cv2resize
from cv2 import resize, imdecode
from cv2 import imread as cv2imread
import cameraThreading
from numpy import float32, argmax, arange, zeros, roll, sum, frombuffer, uint8
from skfuzzy import trapmf as fuzzytrapmf
from skfuzzy.control import ControlSystem, ControlSystemSimulation, Rule, Antecedent, Consequent

camera_1 = cameraThreading.Camera()

def main(method, camera):
    
    global run_inference, run_fuzzy
    
    #Initialize model
    model = tflite.Interpreter(model_path="models/july2022.tflite")
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    #new input image size  =224*224 (1 Aug)
    model.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
    model.resize_tensor_input(output_details[0]['index'], (1, 2))
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    #Describing Fuzzy System            
    flame = Antecedent(arange(0,1024,1),'flame')
    sd = Antecedent(arange(0,1024,1), 'sd')
    fire = Consequent(arange(0,101,1), 'fire')

    #Defining Fuzzyficaion System (Membership Functions)
    flame['Negative'] = fuzzytrapmf(flame.universe, [0,0,200,500])
    flame['Far'] = fuzzytrapmf(flame.universe,[350,440,600,650])
    flame['Large'] = fuzzytrapmf(flame.universe,[500,650,1023,1023])

    sd['Negative'] = fuzzytrapmf(sd.universe, [0,0,250,500])
    sd['Low'] = fuzzytrapmf(sd.universe, [400,400,600,700])
    sd['High'] = fuzzytrapmf(sd.universe, [550,800,1023,1023])

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

    rule_book = ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9])
    system = ControlSystemSimulation(rule_book)


    #Computing Functions
    def do_inference(img, my_model):

        preds = ['Fire','no Fire']
        
        i = cv2resize(img,(224,224))
        i = i/255.0
        i = float32(i)
        
        my_model.set_tensor(input_details[0]['index'], [i])
        my_model.invoke()
        
        output = my_model.get_tensor(output_details[0]['index'])
        pred = argmax(output)

        confidence = output[0][1]
        return confidence 
      
    def run_inference(camera):
        frame = camera.get_frame()
        npstring = frombuffer(frame, dtype= uint8)
        img = imdecode(npstring, 1)
        conf = do_inference(img, model)
        return conf
        

       

    def run_fuzzy():
        #Inputs from Sensors
        with serial.Serial('/dev/ttyS0', 9600, timeout=2) as ser:
            ser.flush()
                    
            while True:
                if ser.in_waiting > 0:
                    row_data = ser.readline().decode('ascii').rstrip()
                    split = row_data.split(" ")
                    data_list = list(map(int, split))
                
                    flame_val = max(data_list[0:5])
                    sd_val = data_list[5]
                    #print(data_list)
            
                    system.input['flame'] = flame_val
                    system.input['sd'] =  sd_val
                    system.compute()
                    
                    return system.output['fire']/100  

    
    #Program Flow
    if method != 0:
        print("Ran ML Alogrithm at Serial Location = ", method)
        #path for image
        path = f'static/image-sr{method}.jpg'
        img = cv2imread(path)
        conf = do_inference(img,model)
        return_arr = [method, conf, 0, 0]
        return return_arr
        
    
    #Defining Processes using Target functions
    #One net do_inference for aproxx 4*10 = 45seconds (added processing delay)
    model_arr = zeros(10)
    fuzzy_arr = zeros(10)
    net_arr = zeros(10)
    return_arr = zeros(4)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(run_inference, camera)
        f2 = executor.submit(run_fuzzy)
    
        
        
        #time.sleep(2)
            
        model_arr = roll(model_arr,1)
        fuzzy_arr = roll(fuzzy_arr,1)
        net_arr = roll(net_arr,1)
            
        model_arr[0] = round(f1.result(),5)
        print("Model Result = ", model_arr[0])
        print("Sensor Result = ", fuzzy_arr[0])

        #Complimentary Filter
        net_arr[0] = ((0.80*model_arr[0]) + (0.20*fuzzy_arr[0]))
        arrsum = sum(net_arr)
            
        return_arr = [0, model_arr[0], fuzzy_arr[0], net_arr[0]]
        print("Method = ", method)
        return return_arr


if __name__=="__main__":
    print("Start")
    i = main(method = 0, camera=camera_1)
    print(i)
    print("Finish")
















