import time
import io
import threading
from picamera import PiCamera
import cv2
import numpy as np

from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

class Camera():
    thread = None  # background thread that reads frames from camera
    process = None
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera

    def initialize(self):
        if Camera.process is None:
            # start background frame thread
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()
        
            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def get_frame(self):
        Camera.last_access = time.time()
        self.initialize()
        return self.frame

    @classmethod
    def _thread(cls):
        with PiCamera() as camera:
            # camera setup
            camera.resolution = (320, 240)
            camera.hflip = True
            camera.vflip = True

            # let camera warm up
            #camera.start_preview()
            time.sleep(10)

            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream, 'jpeg',
                                                 use_video_port=True):
                # store frame
                stream.seek(0)
                cls.frame = stream.read()

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()

                # if there hasn't been any clients asking for frames in
                # the last 10 seconds stop the thread
                if time.time() - cls.last_access > 10:
                    break
        cls.thread = None
   
camera = Camera()
camera.initialize()
frame = camera.get_frame()
npstring = np.frombuffer(frame, dtype= np.uint8)
img = cv2.imdecode(npstring, 1)
cv2.imshow("Frame", img)
cv2.waitKey(0)
print(npstring)
print("Frame End")

