from edgetpu.detection.engine import DetectionEngine
from numpy import asarray
from PIL import Image
# load the MobileNet V2 SSD Face model
print("Using SSD MobileNet V2 for face detection")

SSD_MOBILENET_V2_FACE_MODEL = 'models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite' # trained with Celebrity imageset
# Keras model converted to quantized tflite model (via convert_h5_to_tflite.py)
FACE_EMBEDDING_CELEBRITY_TFLITE_MODEL_PATH = 'models/facenet_keras_edgetpu.tflite'

face_detection_engine = DetectionEngine(SSD_MOBILENET_V2_FACE_MODEL)



def detect_faces(self, rgb_array):
    results = [] # assume no faces are detected

    frame_as_image = Image.fromarray(rgb_array)
    detected_faces = face_detection_engine.detect_with_image(
                frame_as_image,
                threshold=0.5,
                keep_aspect_ratio=True,
                relative_coord=False,
                top_k=5,
                resample=Image.BOX)

    if len(detected_faces) == 0:
        return results

    # extract the bounding box from the first face
    for detected_face in detected_faces:
        # convert the bounding box to the format we want
        x_1, y_1, x_2, y_2 = detected_face.bounding_box.flatten().astype("int")
        width = abs(x_2 - x_1)
        height = abs(y_2 - y_1)
        result = (x_1, y_1, width, height)
        results.append(result)

    return results


def extract_face(self, rgb_array, embedding_model):
        

        detected_faces = detect_faces(rgb_array)
        if len(detected_faces) == 0:
            return []

        if detected_faces[0][2] == 0:
            return []

        x_1, y_1, width, height = tuple(detected_faces[0])
        x_1, y_1 = abs(x_1), abs(y_1)
        x_2, y_2 = x_1 + width, y_1 + height

        # extract a cropped image of the detected face
        face = rgb_array[y_1:y_2, x_1:x_2]

        # resize pixels to the dimension required for the specified embedding model
        image = Image.fromarray(face)
        image = image.resize((160, 160))

        # convert image to numpy array
        face_rgb_array = asarray(image)
        return face_rgb_array
