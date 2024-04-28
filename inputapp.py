from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import UnidentifiedImageError
import time


app = Flask(__name__)

# Load the pre-trained model for plant disease prediction
model = tf.keras.models.load_model('mnet_model.h5')

def predict_disease(image):
    # Resize the image to match the input size expected by the model (224x224)
    image = image.resize((224, 224), Image.BILINEAR)

    # Convert the image to a NumPy array
    image = np.array(image)

    # Expand the dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = model.predict(image)

    # Get the predicted disease label
    disease_label = np.argmax(prediction)

    # Get the prediction score (probability) for the predicted class
    prediction_score = prediction[0][disease_label]

    return disease_label, prediction_score

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        timestamp = int(time.time())  # Generate a timestamp to break the cache
        try:
            image_file = request.files['image']  # Get the uploaded image file

            # Check if the image file is empty
            if image_file.filename == '':
                return render_template('index.html', disease_name='Error: No file selected', timestamp=timestamp)

            # Check if the file has an allowed extension (e.g., .jpg, .jpeg, .png)
            allowed_extensions = {'jpg', 'jpeg', 'png'}
            if not '.' in image_file.filename or image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return render_template('index.html', disease_name='Error: Invalid file format', timestamp=timestamp)

            image = Image.open(image_file)  # Load the image
            image.save('static/uploaded_image.jpg')  # Save the image to the static folder
            disease_label, prediction_score = predict_disease(image)  # Predict the disease and get the prediction score

            # Round the prediction_score to two decimal places
            prediction_score = round(prediction_score, 2)

            disease_classes = ['Cercospora_leaf_spot_Gray_leaf_spot', 'Corn;Common_rust', 'Corn;healthy', 'Corn;Northern_Leaf_Blight', 'Unpredictable']
            disease_name = disease_classes[disease_label]  # Get the name of the predicted disease

            return render_template('index.html', disease_name=disease_name, prediction_score=prediction_score*100, timestamp=timestamp)
        except UnidentifiedImageError:
            return render_template('index.html', disease_name='Error: Invalid image format', timestamp=timestamp)
        except Exception as e:
            return render_template('index.html', disease_name=f'Error: {str(e)}', timestamp=timestamp)
    else:
        return render_template('index.html')

app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
