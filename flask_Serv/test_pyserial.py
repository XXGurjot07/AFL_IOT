m=1
path=f'ahfdoiwahodwa'
path=f'ahfdoiwahodwa'
path=f'ahfdoiwahodwa'
path=f'ahfdoiwah{m}odwa'
print(path)

import serial

ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
ser.flush()
                    
while True:
    if ser.in_waiting > 0:
        row_data = ser.readline().decode('ascii').rstrip()
        split = row_data.split(" ")
        data_list = list(map(int, split))
            
        flame_val = max(data_list[0:5])
        sd_val = data_list[5]
        print(data_list)
        

