''' learn_faces.py

    Purpose: Generate embeddings for training faces
'''
from os import listdir
from os.path import isdir
import platform
import argparse

from numpy import asarray, savez_compressed, copy as copyarray
from numpy import asarray, array as np_array

from enum import Enum
from numpy import expand_dims, uint8 as np_uint8

from numpy import asarray
from PIL import Image


SSD_MOBILENET_V2_FACE_MODEL = 'models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite' # trained with Celebrity imageset

def load_faces(directory, face_detection_engine, embedding_model):
    ''' function load_faces

    Load images and extract a single face for all files in a directory

    Args:
        directory (string): The directory to load the image files from
        face_detection_engine (FaceDetectionEngine): The engine to use for detecting faces
        embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                        embeddings for face images

    Returns:
        A list of numpy arrays for faces detected in the image files in the given directory
    '''

    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # load image from file
        image = Image.open(path)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to numpy array
        rgb_array = np_array(image)

        # get face
        face = face_detection_engine.extract_face(rgb_array, embedding_model)
        # store
        if len(face) > 1:
            faces.append(face)
        else:
            print("Face not detected in file: {}".format(path))
    return faces
def get_processed_training_data(directory, face_detection_engine, embedding_model):
    ''' function get_processed_training_data

    Load a dataset that contains one subdir for each class that in turn contains images

    Args:
        directory (string): The directory to load the image files from
        face_detection_engine (FaceDetectionEngine): Detector being used to detect faces
        embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                        embeddings for face images

    Returns:
        A Tuple of two numpy arrays:
            - An array of arrays:
                for each subdirectory
                    an array of faces detected in the image files
            - An array of arrays:
                for each subdirectory
                    an array of labels for the faces detected in the image files
    '''

    face_images, face_labels = list(), list()
    # enumerate folders, one per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that also might be in the directory
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path, face_detection_engine, embedding_model)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        face_images.extend(faces)
        face_labels.extend(labels)
    return asarray(face_images), asarray(face_labels)




# Face Detection Engine
class FaceDetectionMethodEnum(Enum):
    ''' enum DetectionMethod

        Enumerates all methods supported for detecting faces
    '''
    MTCNN = 1 # currently only supported on Ubuntu
    SSD_MOBILENET_V2 = 2 # currently only supported on Coral dev board

class FaceDetectionEngine:
    ''' class FaceDetectionEngine

        Purpose: detect faces in an image
    '''

    def __init__(self, detection_method):
        ''' function constructor

        Constructor for FaceDetectionEngine

        Args:
            detection_method (DetectionMethod): Method to use for detection

        Returns:
            None
        '''

        # We only want to import these modules at run-time since
        # they will only be installed on certain platforms.
        # pylint: disable=import-outside-toplevel, import-error

        self.detection_method = detection_method
        if self.detection_method == FaceDetectionMethodEnum.SSD_MOBILENET_V2:
            # load the MobileNet V2 SSD Face model
            print("Using SSD MobileNet V2 for face detection")
            from edgetpu.detection.engine import DetectionEngine
            self.face_detection_engine = DetectionEngine(SSD_MOBILENET_V2_FACE_MODEL)
        else:
            raise Exception("Invalid detection method: {}".format(detection_method))

    # detect faces
    def detect_faces(self, rgb_array):
        ''' function detect_faces

        Detect any faces that are present in the given image.

        Args:
            rgb_array (numpy.ndarray): An image that may or may not contain faces

        Returns:
            An array of bounding boxes (top_left_x, top_left_y, width, height)
            for each face detected in the given image
        '''

        results = [] # assume no faces are detected

        # detect faces in the image
        if self.detection_method == FaceDetectionMethodEnum.SSD_MOBILENET_V2:
            frame_as_image = Image.fromarray(rgb_array)
            detected_faces = self.face_detection_engine.detect_with_image(
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
        ''' function extract_face

        Extract a single face from the given frame

        Args:
            rgb_array (numpy.ndarray): The image that may or may not contain
                            one or more faces
            embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                            embeddings for face images

        Returns:
            If a face is detected, returns an RGB numpy.ndarray of the face extracted from
            the given frame of the dimensions required for the given embedding model.
            Otherwise it returns an empty array.
        '''

        detected_faces = self.detect_faces(rgb_array)
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








#

# Keras model converted to quantized tflite model (via convert_h5_to_tflite.py)
FACE_EMBEDDING_CELEBRITY_TFLITE_MODEL_PATH = 'models/facenet_keras_edgetpu.tflite'

class FaceEmbeddingModelEnum(Enum):
    ''' enum FaceEmbeddingModelEnum

        Enumerates all models supported for identifying faces
    '''
    CELEBRITY_KERAS = 1  # Keras model pre-trained using  MS-Celeb-1M
    CELEBRITY_TFLITE = 2 # tflite version of CELEBRITY_KERAS

class FaceEmbeddingEngine:
    ''' class FaceEmbeddingEngine

        Purpose: generate embeddings for images of faces
    '''

    def __init__(self, embedding_model):
        ''' function constructor

        Constructor for FaceEmbeddingEngine

        Args:
        embedding_model (FaceEmbeddingModelEnum): The model to use for generating
                        embeddings for face images

        Returns:
            None
        '''

        # We only want to import these modules at run-time since
        # they will only be installed on certain platforms.
        # pylint: disable=import-outside-toplevel, import-error

        self.embedding_model = embedding_model
        self.required_image_shape = get_image_dimensions_for_embedding_model(embedding_model) + (3,) # need 3 arrays for RGB

        if self.embedding_model == FaceEmbeddingModelEnum.CELEBRITY_KERAS:
            print("Using Celebrity trained Keras model for face embeddings")
            from keras.models import load_model
            self.face_embedding_engine = load_model(FACE_EMBEDDING_CELEBRITY_KERAS_MODEL_PATH, compile=False)
        elif self.embedding_model == FaceEmbeddingModelEnum.CELEBRITY_TFLITE:
            print("Using Celebrity trained tflite model for face embeddings")
            from edgetpu.basic.basic_engine import BasicEngine
            self.face_embedding_engine = BasicEngine(FACE_EMBEDDING_CELEBRITY_TFLITE_MODEL_PATH)
            print("Embedding model input tensor shape: {}".format(self.face_embedding_engine.get_input_tensor_shape()))
            print("Embedding model input size: {}".format(self.face_embedding_engine.required_input_array_size()))
        else:
            raise Exception("Invalid embedding mode method: {}".format(embedding_model))

    def get_embedding_model(self):
        ''' function get_embedding_model

        Get the embedding model being used by this instance of the FaceEmbeddingEngine

        Args:
            None

        Returns:
            The FaceEmbeddingModelEnum being used by this instance of FaceEmbeddingEngine
        '''
        return self.embedding_model

    # get the face embedding for one face
    def get_embedding(self, face_pixels):
        ''' function get_embedding

        Generate an embedding for the given face

        Args:
            face_pixels (cv2 image): The image of the face to generate the
                            embedding for. The dimensions of the image must
                            match the dimensions required by the selected
                            embedding model.

        Returns:
            A numpy array with the embedding that was generated
        '''

        # Confirm we're using a proper sized image to generate the embedding with
        if face_pixels.shape != self.required_image_shape:
            raise Exception("Invalid shape: {} for embedding mode method: {}".format(face_pixels.shape, self.embedding_model))

        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        sample = expand_dims(face_pixels, axis=0)

        # get embedding
        if self.embedding_model == FaceEmbeddingModelEnum.CELEBRITY_KERAS:
            embeddings = self.face_embedding_engine.predict(sample)
            result = embeddings[0]
        else:
            sample = sample.flatten()
            # normalize values to between 0 and 255 (UINT)
            sample *= 255.0/sample.max()
            # convert to UNIT8
            sample = sample.astype(np_uint8)
            embeddings = self.face_embedding_engine.run_inference(sample)
            result = embeddings[1]

        return result

def get_image_dimensions_for_embedding_model(embedding_model):
    ''' function get_image_dimensions_for_embedding_model

    Get the required dimensions for images to use with the given embedding model

    Args:
        embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                        embeddings for face images

    Returns:
        A tuple of the required width,height dimensions for images used by the given embedding model
    '''
    result = None
    if embedding_model in (FaceEmbeddingModelEnum.CELEBRITY_KERAS, FaceEmbeddingModelEnum.CELEBRITY_TFLITE):
        result = (160, 160)
    else:
        raise Exception("Invalid embedding model: {}".format(embedding_model))

    return result












# Stores actual training image pixels for faces (images are clipped from training image
# and resized to the input size of the embedding we want to use)
TRAINING_FACE_IMAGES_OUTPUT_FILE = 'training_data/training-data-faces-dataset.npz'

# Stores embeddings for faces we want to identify
LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE = 'training_data/trained-faces-embeddings.npz'

# Set this to true to print out performance times for detecting/recognizing
PRINT_PERFORMANCE_INFO = False



TRAINING_DATA_DIRECTORY = 'training_data/train/'
VALIDATION_DATA_DIRECTORY = 'training_data/val/'
SAVE_TRAINING_DATA = True
FACENET_EMBEDDING_MODEL_PATH = 'models/facenet_keras.h5' # trained with Celebrity imageset
FACENET_TFLITE_EMBEDDING_MODEL_PATH = 'models/facenet_keras_edgetpu.tflite'

# Ignore long lines as I'm using verbose variable names for easier understanding
# pylint: disable=line-too-long

def main(face_detector, face_embedder, face_embedder_model, skip_embeddings):
    ''' function main

    Main processing function - create face embeddings for all training images

    Args:
        face_detector (FaceDetectionEngine): Engine used to detect faces
        face_embedder (FaceEmbeddingEngine): Engine used to identify faces

    Returns:
        None
    '''

    # load training dataset
    training_images, training_labels = get_processed_training_data(TRAINING_DATA_DIRECTORY, face_detector, face_embedder_model)
    print("Total training data - images: {}, labels: {}".format(training_images.shape, training_labels.shape))

    # load validation dataset
    validation_images, validation_labels = get_processed_training_data(VALIDATION_DATA_DIRECTORY, face_detector, face_embedder_model)
    print("Total test data - images: {}, labels: {}".format(validation_images.shape, validation_labels.shape))

    if SAVE_TRAINING_DATA:
        # save arrays to one file in compressed format
        print("Saving image training data to: {}".format(TRAINING_FACE_IMAGES_OUTPUT_FILE))
        savez_compressed(TRAINING_FACE_IMAGES_OUTPUT_FILE, training_images, training_labels, validation_images, validation_labels)

    if not skip_embeddings:

        # convert each face in the training set to an embedding
        training_embeddings = list()
        for face_pixels in training_images:
            embedding = face_embedder.get_embedding(face_pixels)
            training_embeddings.append(copyarray(embedding))
        training_embeddings = asarray(training_embeddings)

        # convert each face in the validation set to an embedding
        validation_embeddings = list()
        for face_pixels in validation_images:
            embedding = face_embedder.get_embedding(face_pixels)
            validation_embeddings.append(copyarray(embedding))
        validation_embeddings = asarray(validation_embeddings)

        print("training embeddings shape: ", training_embeddings.shape)
        print("validation embeddings shape: ", validation_embeddings.shape)

        # save arrays to one file in compressed format
        print("Saving image training embeddings to: {}".format(LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE))
        savez_compressed(LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE, training_embeddings, training_labels, validation_embeddings, validation_labels)

parser = argparse.ArgumentParser()
parser.add_argument('--skip-embeddings', dest='skip_embeddings', action='store_true')
parser.set_defaults(skip_embeddings=False)
args = parser.parse_args()

if __name__ == '__main__':
    face_embedding_engine = None
    face_embedding_model = None
    if is_coral_dev_board:
        face_detection_engine = FaceDetectionEngine(FaceDetectionMethodEnum.SSD_MOBILENET_V2)
        face_embedding_model = FaceEmbeddingModelEnum.CELEBRITY_TFLITE
        if not args.skip_embeddings:
            face_embedding_engine = FaceEmbeddingEngine(face_embedding_model)
    else:
        raise Exception("Unsupported platform")

    if args.skip_embeddings:
        print("Skipping embeddings, only generating face images")

    main(face_detection_engine, face_embedding_engine, face_embedding_model, args.skip_embeddings)