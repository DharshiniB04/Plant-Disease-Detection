import threading
from flask import Flask, render_template, Response
import cv2
import time
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
class_names = ['Cercospora_leaf_spot_Gray_leaf_spot', 'Corn;Common_rust', 'Corn;healthy', 'Corn;Northern_Leaf_Blight']
def prediction_handler(p_prediction):
    # Split the prediction string using a semicolon as a delimiter
    prediction = p_prediction.split(';')
    
    if "Healthy".lower() in prediction[-1].lower():
        # If the prediction indicates the plant is healthy
        plant_name = prediction[0]
        plant_status = "Healthy"
        plant_disease = "NIL"
        plant_solution = "NIL"
    else:
        plant_name = prediction[0]
        if "Unknown" in plant_name:
            # If the plant name is "Unknown," indicating an unknown plant species
            plant_status = "Unknown"
            plant_disease = "NIL"
            plant_solution = "Take Another Picture"
            plant_disease = prediction[-1]  # Assuming there is a disease name in the prediction
        else:
            # If the prediction indicates the plant is unhealthy
            plant_status = "Unhealthy"
            plant_disease = p_prediction.split(';')[-1]
            plant_solution = "Increase pH level of water"  # A hardcoded solution for an unhealthy plant
    
    # Return a dictionary with information about the plant prediction
    return {
        "species": plant_name,
        "status": plant_status,
        "disease": plant_disease,
        "solution": plant_solution
    }



class VideoStreamer:
    def __init__(self):
        self.camera = cv2.VideoCapture(1)
        self.lock = threading.Lock()

    def gen_frames(self):
        while True:
            with self.lock:
                success, frame = self.camera.read()

            if not success:
                break

            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            img_ = np.expand_dims(frame, axis=0)
            pred = model.predict(img_)
            prediction_result = class_names[np.argmax(pred)]
            if pred[0][np.argmax(pred[0])] < 0.85:
                prediction_result = "Unknown"
            print(pred[0][np.argmax(pred[0])])
            handler = prediction_handler(prediction_result)

            frame = cv2.copyMakeBorder(
                frame, 90, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            frame = cv2.putText(frame, str("species : ") + str(handler['species']), (0, 15), font,
                                fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, str("status : ") + str(handler['status']), (0, 30), font,
                                fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, str("disease : ") + str(handler['disease']), (0, 45), font,
                                fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, str("solution : ") + str(handler['solution']), (0, 60), font,
                                fontScale, color, thickness, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)
streamer = VideoStreamer()
model = tf.keras.models.load_model('mnet_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.35
color = (255, 0, 0)
thickness = 1


@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/video_feed')
def video_feed():
    return Response(streamer.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
