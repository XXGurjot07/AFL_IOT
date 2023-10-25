import inference_script
import cameraThreading
import RPi.GPIO as GPIO
from cv2 import imdecode
from os import mkdir
from os import path
import os
from datetime import datetime
from threading import Thread
from runpy import run_module
from flask_sqlalchemy import SQLAlchemy
from cv2 import IMWRITE_JPEG2000_COMPRESSION_X1000, imwrite, imdecode, IMREAD_COLOR
from flask import Flask, render_template, Response, request
from numpy import zeros, roll, sum, frombuffer, uint8, fromstring


#Global Flags and Initialization 
global capture, play, runmodelcntnode
runmodelcntnode = 0
capture = 0
play = 1
sum_array = zeros(10)
image_model_arr = zeros(10)
camera_1 = cameraThreading.Camera()
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
ESP_FOLDER = os.path.join('./saved', 'esp_cam_images')


app = Flask(__name__, template_folder='./templates', static_folder='./static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
save_path = '\saved'
esp_img_dir = "static/"


#Database
class auxiliaryReadings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sensor_id = db.Column(db.Integer, nullable=False)
    datetime = db.Column(db.DateTime, nullable=False, default=datetime.now())
    temp_reading = db.Column(db.Integer, nullable=False)
    moisture_reading = db.Column(db.Integer, nullable=False)
    
    def __repr__(self):
        return 'Auxiliary-Sensor ID is %r taken at %r' % (self.sensor_id, self.datetime)


class inferenceReadings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, nullable=False)
    datetime = db.Column(db.DateTime, nullable=False, default=datetime.now())
    fuzzy_reading = db.Column(db.Float, nullable=False)
    model_reading = db.Column(db.Float, nullable=False)
    net_reading = db.Column(db.Float, nullable=False)
    sum_reading = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return 'Inference-Readings ID is %r taken at %r and accumulation sum is %r' % (self.camera_id, self.datetime, self.sum)


def gen(camera):
    global capture
    if play:
        while True:
            frame = camera.get_frame()
            if capture:
                capture = 0
                now = datetime.now()
                current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
                filename = '%s.jpeg' % current_time
                path = 'saved/'
                npstring = frombuffer(frame, dtype= uint8)
                img = imdecode(npstring, 1)
                imwrite(f'{path}{filename}', img)
    
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def gen_model():
    global runmodelcntnode
    sum_array = zeros(10)
    if runmodelcntnode:
        while True:
            if (not runmodelcntnode):
                sum_array = [0,0,0,0,0,0,0,0,0,0]
                break
            
            #Pass method as 0
            inference_arr = list(inference_script.main(method = 0, camera=camera_1))
            print(inference_arr)
            sum_array = roll(sum_array, 1)
            sum_array[0] = inference_arr[3]
            sum1 = sum(sum_array)
            
            data_in = inferenceReadings(camera_id = inference_arr[0], fuzzy_reading=inference_arr[2], 
                                        model_reading=inference_arr[1], net_reading=inference_arr[3], sum_reading=sum1)
            db.session.add(data_in)
            db.session.commit()

            #Buzzer Activation
            if sum1 >= 7.0:
                GPIO.output(18, GPIO.HIGH)
            else:
                GPIO.output(18, GPIO.LOW)            
    

def run_inference(method):
    image_model_arr = zeros(10)
    inference_arr = inference_script.main(method = method, camera=0)
    
    print(inference_arr)
    #alteration
    image_model_arr = roll(image_model_arr, 1)
    image_model_arr[0] = inference_arr[1]*5
            
    sum1 = sum(image_model_arr)
            
    data_in = inferenceReadings(camera_id = method, model_reading=image_model_arr[0], net_reading=image_model_arr[0],
                                fuzzy_reading=0, sum_reading=(sum1/8))
    db.session.add(data_in)
    db.session.commit()

    #Buzzer Activation
    if sum1 >= 7.0:
        GPIO.output(18, GPIO.HIGH)
    else:
        GPIO.output(18, GPIO.LOW)            

def fetch_data_esp_ground_sensor():
    data_out_aux = request.data
    # convert bytes to string
    data_out_str = str(data_out_aux, 'UTF-8')
    data_out_str_list = data_out_str.split()
    data_out_str_list[0] = int(float(data_out_str_list[0]))
    data_out_str_list[1] = int(float(data_out_str_list[1]))
    data_out_str_list[2] = int(float(data_out_str_list[2]))

    data_in_aux = auxiliaryReadings(sensor_id=data_out_str_list[0], moisture_reading=data_out_str_list[1], smoke_reading=data_out_str_list[2])
    db.session.add(data_in_aux)
    db.session.commit()
    # means return 200 response code
    return ''

def save_ml_img(img, location):
    imwrite(os.path.join(esp_img_dir,"image-sr"+str(location)+".jpg"), img)
    run_inference(location)
    print(f'Image Saved from Location - {location}')

#Web Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video-stream')
def video_stream():
    return render_template('video-stream.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera_1), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_flags', methods=['POST', 'GET'])
def video_feed_flags():
    if (request.method == 'POST'):
        global capture, play
        if request.form.get('capture') == 'Capture':
            capture = 1

        if request.form.get('pause') == 'Pause':
            play = 0

        if request.form.get('play') == 'Play':
            play = 1

    return render_template('video-stream.html')


@app.route('/inference_requests', methods=['POST', 'GET'])
def ir_tasks():
    global runmodelcntnode
    if request.method == 'POST':
        if request.form.get('run_model') == 'Run Model':
            runmodelcntnode = 1
            
        if (request.form.get('stop_model') == 'Stop Model'):
            runmodelcntnode = 0         

    if runmodelcntnode:
        t1 = Thread(target=gen_model)
        t1.start()    
    
    my_data = inferenceReadings.query.order_by(inferenceReadings.datetime)
    return render_template('live-inference.html', my_data=my_data)



@app.route("/inference_requests/upload-image-sr1", methods=["GET", "POST"])
def upload_1():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file  = received.files['imageFile']
        nparr = frombuffer(file.read(), uint8)
        img = imdecode(nparr, IMREAD_COLOR)
        save_ml_img(img, 1)
        print("ESP32 CAM SR LOC 1 Image Received")        
        
        return "[SUCCESS] Image Received", 201
    else:
        return "[FAILED] Image Not Received", 204
    
@app.route("/inference_requests/upload-image-sr2", methods=["GET", "POST"])
def upload_2():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file  = received.files['imageFile']
        nparr = frombuffer(file.read(), uint8)
        img = imdecode(nparr, IMREAD_COLOR)
        save_ml_img(img, 2)
        print("ESP32 CAM SR LOC 2 Image Received")        
        
        return "[SUCCESS] Image Received", 201
    else:
        return "[FAILED] Image Not Received", 204

@app.route("/inference_requests/upload-image-sr3", methods=["GET", "POST"])
def upload_3():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file  = received.files['imageFile']
        nparr = frombuffer(file.read(), uint8)
        img = imdecode(nparr, IMREAD_COLOR)
        save_ml_img(img, 3)
        print("ESP32 CAM SR LOC 3 Image Received")        
        
        return "[SUCCESS] Image Received", 201
    else:
        return "[FAILED] Image Not Received", 204 

@app.route("/inference_requests/upload-image-sr4", methods=["GET", "POST"])
def upload_4():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file  = received.files['imageFile']
        nparr = frombuffer(file.read(), uint8)
        img = imdecode(nparr, IMREAD_COLOR)
        save_ml_img(img, 4)
        print("ESP32 CAM SR LOC 4 Image Received")        
        
        return "[SUCCESS] Image Received", 201
    else:
        return "[FAILED] Image Not Received", 204                   

@app.route("/inference_requests/upload-image-sr5", methods=["GET", "POST"])
def upload_5():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file  = received.files['imageFile']
        nparr = frombuffer(file.read(), uint8)
        img = imdecode(nparr, IMREAD_COLOR)
        save_ml_img(img, 5)
        print("ESP32 CAM SR LOC 5 Image Received")        
        
        return "[SUCCESS] Image Received", 201
    else:
        return "[FAILED] Image Not Received", 204    
    
@app.route('/auxiliary-requests/post', methods=['POST', 'GET'])
def ar_tasks():
    if request.method == 'POST':
        if request.form.get('fetch-latest-status') == 'Fetch Latest Status':
            #fetch_data_esp_ground_sensor()
            my_data_aux = auxiliaryReadings.query.order_by(auxiliaryReadings.datetime)
            return render_template('auxiliary-sensors.html', my_data_aux=my_data_aux)
            
        
    elif request.method == 'GET':
        pass

@app.route('/live-inference')
def live_inference():
    my_data = inferenceReadings.query.order_by(inferenceReadings.datetime)
    return render_template('live-inference.html', my_data=my_data)

@app.route('/live-inference-espcam-sr1', methods=['GET', 'POST'])
def esp_show():
    if request.method == 'POST':
        if request.form.get('sr1') == 'View Location 1':
            full = 'static/image-sr1.jpg'
            return render_template('esp_image_show.html', user_image=full)
        if request.form.get('sr2') == 'View Location 2':
            full = 'static/image-sr2.jpg'
            return render_template('esp_image_show.html', user_image=full)
        if request.form.get('sr3') == 'View Location 3':
            full = 'static/image-sr3.jpg'
            return render_template('esp_image_show.html', user_image=full)
    my_data = inferenceReadings.query.order_by(inferenceReadings.datetime)
    return render_template('live-inference.html', my_data=my_data)    
    

@app.route('/view-database')
def view_database():
    return render_template('view-database.html')


@app.route('/auxiliary-sensors')
def auxiliary_sensors():
    my_data_aux = auxiliaryReadings.query.order_by(auxiliaryReadings.datetime)
    return render_template('auxiliary-sensors.html', my_data_aux=my_data_aux)

@app.route('/about-project')
def about_project():
    return render_template('about-project.html')

if __name__ == '__main__':
    
    app.run(debug='true', host='0.0.0.0', port=8000)
  
