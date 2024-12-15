from cv2 import VideoCapture, \
                CAP_PROP_FRAME_WIDTH, \
                CAP_PROP_FRAME_HEIGHT, \
                CAP_PROP_FPS, \
                WINDOW_NORMAL, \
                COLOR_BGR2RGB, \
                COLOR_RGB2BGR, \
                namedWindow, \
                cvtColor, \
                imencode, \
                imwrite, \
                waitKey
import time
import os

# Reduce image size to increase performance
CAMERA_RESOLUTION = (1280, 720)

# Set this to true to print out performance times for detecting/recognizing
PRINT_PERFORMANCE_INFO = True

class VideoCamera():

    def __init__(self):
        print("Camera images being processed at resolution: {}".format(CAMERA_RESOLUTION))
        self.video = VideoCapture(0)
        self.video.set(CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        self.video.set(CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        print("Camera FPS set at {:4.1f}".format(self.video.get(CAP_PROP_FPS)))
        
    def generate_frames(self):
        # load camera
        
        
        while True:  
            success, frame = self.video.read()  
            if not success:
                break
            else:
                # Encode frame as JPEG
                _, buffer = imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



    def capture(self):
        success, frame = self.video.read() 
        # Ensure training_data directory exists
        os.makedirs('training_data', exist_ok=True)
        # Create a unique filename with timestamp
        timestamp = int(time.time())
        filename = f"training_data/captured_image_{timestamp}.jpg"
        # Save the image
        imwrite(filename, frame)
        print(f"Image captured and saved to {filename}")

        return "Image captured successfully!", 200
        