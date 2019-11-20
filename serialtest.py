import serial
import time

ser = serial.Serial("/dev/cu.usbserial-A105A98D", 9600)
time.sleep(5)
ser.write(b'hello')
