# Modules
import time
from flask import Flask, render_template, Response, redirect, url_for

import os
from numpy import array as np_array
from video_camera import VideoCamera

# Global Variables

# Stores actual training image pixels for faces (images are clipped from training image
# and resized to the input size of the embedding we want to use)
TRAINING_FACE_IMAGES_OUTPUT_FILE = 'training_data/training-data-faces-dataset.npz'

# Stores embeddings for faces we want to identify
LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE = 'training_data/trained-faces-embeddings.npz'

camera = VideoCamera()



app = Flask(__name__)

@app.route('/')
def index():
    ''' function index ...

    Args: None

    Returns:
        rendered index.html template
    '''
    return render_template('index.html')




@app.route('/video_feed')
def video_feed(): 
    
    return Response(camera.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognise', methods=['POST'])
def recognise():
    # Add logic to stop the facial recognition process.
    # For now, redirect to the home page or display a confirmation.
    return redirect(url_for('index'))  # Replace 'index' with your main route.



@app.route('/capture_image', methods=['POST'])
def capture_image():
   
   return camera.capture()
    



if __name__ == '__main__':

    # start local web server
    app.run(host='0.0.0.0', port='4000', debug=True)
