from runpy import run_module
from cv2 import IMWRITE_JPEG2000_COMPRESSION_X1000
from flask import Flask, render_template, Response, request
from flask_sqlalchemy import SQLAlchemy
import inference_script
import cv2
from os import mkdir
from os import path
from datetime import datetime
import numpy as np
import RPi.GPIO as GPIO
import threading

sum_array = np.zeros(10)
#global flags
global capture, play, runmodelcntnode
runmodelcntnode = 0
capture = 0
play = 1
image_model_arr = np.zeros(10)


# making directory to save pictures
try:
    mkdir('./saved')

except OSError as error:
    pass


app = Flask(__name__, template_folder='./templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
save_path = '\saved'
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)



class sensornode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    auxiliarysens = db.relationship('auxiliarysens', backref='sensornode')

    def __repr__(self):
        return 'Sensor Node ID - %r' % (self.id)


class auxiliarysens(db.Model):
    #__tablename__ = 'auxiliary-sens'
    id = db.Column(db.Integer, primary_key=True)
    datetime = db.Column(db.DateTime, nullable=False, default=datetime.now())
    temp_reading = db.Column(db.Integer, nullable=False)
    moisture_reading = db.Column(db.Integer, nullable=False)
    sensornode_id = db.Column(db.Integer, db.ForeignKey('sensornode.id'))

    def __repr__(self):
        return 'auxiliary-sens ID is %r taken at %r' % (self.id, self.datetime)

class inferencenode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    inferencereadings = db.relationship('inferencereadings', backref='inference_node')
    
    def __repr__(self):
        return 'Inference Node ID - %r' % (self.id)

class inferencereadings(db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    datetime = db.Column(db.DateTime, nullable=False, default=datetime.now())
    fuzzy_reading = db.Column(db.Float, nullable=False)
    model_reading = db.Column(db.Float, nullable=False)
    net_reading = db.Column(db.Float, nullable=False)
    sum_reading = db.Column(db.Float, nullable=False)
    inferencenode_id = db.Column(db.Integer, db.ForeignKey('inferencenode.id'))

    def __repr__(self):
        return 'inference-readings ID is %r taken at %r and accumulation sum is %r' % (self.id, self.datetime, self.sum)


def gen():
    camera = cv2.VideoCapture(0)
    global capture

    if not play:
        return

    if play:
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                #capture code here
                if capture:
                    now = datetime.now()
                    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
                    filename = '%s.jpg' % current_time
                    path = 'saved/'
                    cv2.imwrite(f'{path}{filename}', frame)
                    #reset capture
                    capture = 0
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def gen_model():
    global runmodelcntnode
    sum_array = np.zeros(10)
    if runmodelcntnode:
        while True:
            
            if (not runmodelcntnode):
                sum_array = [0,0,0,0,0,0,0,0,0,0]
                break
            
            #Pass method as 0
            inference_arr = list(inference_script.main(method = 0))
            #inference_arr = inference_script.main()
            print(inference_arr)
            sum_array = np.roll(sum_array, 1)
            sum_array[0] = inference_arr[3]
            
            sum = np.sum(sum_array)
            
            data_in = inferencereadings(inferencenode_id = inference_arr[0],
                        fuzzy_reading=inference_arr[2], model_reading=inference_arr[1], net_reading=inference_arr[3], sum_reading=sum)
            
            print(sum_array)            
            
            db.session.add(data_in)
            db.session.commit()

            #Buzzer Activation
            if sum >= 7.0:
                GPIO.output(18, GPIO.HIGH)
            else:
                GPIO.output(18, GPIO.LOW)            
    
    #my_data = inferencereadings.query.order_by(inferencereadings.datetime)
    #return render_template('live-inference.html', my_data=my_data)


def run_inference(method):
    #Pass method as given
    inference_arr = inference_script.main(method = method)
    image_model_arr = np.zeros(10)
    print(inference_arr)
    value = round(inference_arr, 4)
    image_model_arr = np.roll(image_model_arr, 1)
    image_model_arr[0] = inference_arr
            
    sum = np.sum(image_model_arr)
            
    data_in = inferencereadings(inferencenode_id = method,
                        model_reading=value, net_reading=value,
                        fuzzy_reading=0, sum_reading=sum)
            
    print(sum_array)            
            
    db.session.add(data_in)
    db.session.commit()

    #Buzzer Activation
    if sum >= 7.0:
        GPIO.output(18, GPIO.HIGH)
    else:
        GPIO.output(18, GPIO.LOW)            


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video-stream')
def video_stream():
    return render_template('video-stream.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
        t1 = threading.Thread(target=gen_model)
        t1.start()    
    
    my_node = inferencenode.query.all()
    my_data = inferencereadings.query.order_by(inferencereadings.datetime)
    return render_template('live-inference.html', my_data=my_data, my_node=my_node)

#Update on 10/09
@app.route("/inference_requests/upload-image-sr1", methods=["GET", "POST"])
def upload_image1():

    if request.method == "POST":
        image_raw_bytes = request.get_data()  #get the whole body
        filename = 'image-sr1.jpg'
        path = 'saved/esp_cam_images/'

        f = open(f'{path}{filename}', 'wb') #  string
        f.write(image_raw_bytes)
        f.close()
        print("Image Received from Serial Location 1") 
        
        #Run ML Model on Saved Image
        run_inference(1)
        return render_template("live-inference.html")

    if request.method == "GET": #ifimage.

        return render_template("live-inference.html")
    return "if you see this, that is bad request method"
@app.route("/inference_requests/upload-image-sr2", methods=["GET", "POST"])
def upload_image2():

    if request.method == "POST":
        image_raw_bytes = request.get_data()  #get the whole body
        filename = 'image-sr2.jpg'
        path = 'saved/esp_cam_images/'

        f = open(f'{path}{filename}', 'wb') # wb for write byte data in the file instead of string
        f.write(image_raw_bytes)
        f.close()
        print("Image Received from Serial Location 2") 
        
        #Run ML Model on Saved Image
        run_inference(2)
        return render_template("live-inference.html")

    if request.method == "GET": #if 
        return render_template("live-inference.html")
    return "if you see this, that is bad request method"
@app.route("/inference_requests/upload-image-sr3", methods=["GET", "POST"])
def upload_image3():

    if request.method == "POST":
        image_raw_bytes = request.get_data()  #get the whole body
        filename = 'image-sr3.jpg'
        path = 'saved/esp_cam_images/'

        f = open(f'{path}{filename}', 'wb') # wb for
        f.write(image_raw_bytes)
        f.close()
        print("Image Received from Serial Location 2") 
        
        #Run ML Model on Saved Image
        run_inference(3)
        return render_template("live-inference.html")

    if request.method == "GET": #i

        return render_template("live-inference.html")
    return "if you see this, that is bad request method"  
@app.route("/inference_requests/upload-image-sr4", methods=["GET", "POST"])
def upload_image4():

    if request.method == "POST":
        image_raw_bytes = request.get_data()  #get the whole body
        filename = 'image-sr4.jpg'
        path = 'saved/esp_cam_images/'

        f = open(f'{path}{filename}', 'wb') # 
        f.write(image_raw_bytes)
        f.close()
        print("Image Received from Serial Location 4") 
        
        #Run ML Model on Saved Image
        run_inference(4)
        return render_template("live-inference.html")

    if request.method == "GET": #

        return render_template("live-inference.html")
    return "if you see this, that is bad request method"
@app.route("/inference_requests/upload-image-sr5", methods=["GET", "POST"])
def upload_image5():

    if request.method == "POST":
        image_raw_bytes = request.get_data()  #get the whole body
        filename = 'image-sr5.jpg'
        path = 'saved/esp_cam_images/'

        f = open(f'{path}{filename}', 'wb') # wb for write byte data in the file instead of string
        f.write(image_raw_bytes)
        f.close()
        print("Image Received from Serial Location 5") 
        
        #Run ML Model on Saved Image
        run_inference(5)
        return render_template("live-inference.html")

    if request.method == "GET": #e.
        return render_template("live-inference.html")
    return "if you see this, that is bad request method"          

@app.route('/auxiliary-requests/post', methods=['POST', 'GET'])
def ar_tasks():
    if request.method == 'POST':
        data_out_aux = request.data
        # convert bytes to string
        data_out_str = str(data_out_aux, 'UTF-8')
        data_out_str_list = data_out_str.split()
        data_out_str_list[0] = int(float(data_out_str_list[0]))
        data_out_str_list[1] = int(float(data_out_str_list[1]))
        data_out_str_list[2] = int(float(data_out_str_list[2]))

        data_in_aux = auxiliarysens(
            sensornode_id=data_out_str_list[0], moisture_reading=data_out_str_list[1], smoke_reading=data_out_str_list[2])
        db.session.add(data_in_aux)
        db.session.commit()
        # means return 200 response code
        return ''
    elif request.method == 'GET':
        pass

@app.route('/live-inference')
def live_inference():
    my_node = inferencenode.query.all()
    my_data = inferencereadings.query.order_by(inferencereadings.datetime)
    return render_template('live-inference.html', my_data=my_data, my_node=my_node)


@app.route('/view-database')
def view_database():
    return render_template('view-database.html')


@app.route('/auxiliary-sensors')
def auxiliary_sensors():
    my_data_aux = auxiliarysens.query.order_by(auxiliarysens.datetime)
    return render_template('auxiliary-sensors.html', my_data_aux=my_data_aux)


@app.route('/about-project')
def about_project():
    return render_template('about-project.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
