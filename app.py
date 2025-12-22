import eventlet
eventlet.monkey_patch() 

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from tf_keras.models import load_model
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia'

socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", ping_timeout=10, ping_interval=5)

print("Memuat Model...")

detector = HandDetector(maxHands=1)
model = load_model('keras_model.h5', compile=False)
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[-1] if ' ' in line.strip() else line.strip() for line in f.readlines()]

print("Model siap!")


cap = cv2.VideoCapture(0) 
offset = 20
imgSize = 300
modelInputSize = 224


previous_label = ""
start_time = 0
confirmation_time = 3.0 
is_waiting = False 

def generate_frames():
    global previous_label, start_time, is_waiting, cap
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img, draw=False)

        current_label = "Netral"
        confidence_score = 0

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            try:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                y1, y2 = max(0, y-offset), min(img.shape[0], y+h+offset)
                x1, x2 = max(0, x-offset), min(img.shape[1], x+w+offset)
                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size != 0:
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap] = imgResize

                    imgInput = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
                    imgInput = cv2.resize(imgInput, (modelInputSize, modelInputSize))
                    imgInput = np.asarray(imgInput, dtype=np.float32).reshape(1, modelInputSize, modelInputSize, 3)
                    imgInput = (imgInput / 127.5) - 1

                    prediction = model.predict(imgInput, verbose=0)
                    index = np.argmax(prediction)
                    confidence = prediction[0][index]
                    current_label = class_names[index]
                    confidence_score = confidence

                    if confidence > 0.7:
                        cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+w+offset, y-offset), (0, 255, 0), cv2.FILLED)
                        cv2.putText(imgOutput, f"{current_label} {int(confidence*100)}%", (x-offset, y-offset-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 4)

            except Exception as e:
                pass

        if hands and confidence_score > 0.8:
            if current_label == previous_label:
                if not is_waiting:
                    start_time = time.time()
                    is_waiting = True
                
                duration = time.time() - start_time
                
                bar_width = int((duration / confirmation_time) * 100)
                cv2.rectangle(imgOutput, (50, 50), (50 + bar_width * 3, 80), (0, 255, 255), cv2.FILLED)
                cv2.putText(imgOutput, f"Tahan: {duration:.1f}s", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if duration > confirmation_time:
                    print(f"KIRIM KE WEB: {current_label}")
                    
                    socketio.emit('input_huruf', {'char': current_label})
                    
                    start_time = time.time() + 1.5 
                    is_waiting = False
                    previous_label = "" 
            else:
                previous_label = current_label
                is_waiting = False
                start_time = time.time()
        else:
            previous_label = ""
            is_waiting = False

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        socketio.sleep(0.01) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("Server berjalan di: http://127.0.0.1:5500")
        socketio.run(app, host='0.0.0.0', port=5500, debug=True)
    finally:
        cap.release()