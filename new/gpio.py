import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
#GPIO.setmode(GPIO.BOARD)

GPIO.setmode(18, GPIO.OUT)
GPIO.output(18, GPIO.HIGH)