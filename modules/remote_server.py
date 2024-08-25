import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request
from flask_cors import CORS
import cv2
import numpy as np
from insightface.app.common import Face
from insightface.app import FaceAnalysis
from modules.processors.frame.face_swapper import process_frame  # Ensure you adapt this for your server environment


app = Flask(__name__)

# Allow CORS requests
CORS(app)

# Initialize the face analysis model
face_analyzer = FaceAnalysis()
face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))

@app.route('/process', methods=['POST'])
def process():
    frame_data = request.files['frame'].read()
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

    source_image_file = request.files.get('source_image')
    if source_image_file:
        source_image_data = source_image_file.read()
        source_image_np = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR)

        # Detect and create Face object
        faces = face_analyzer.get(source_image_np)
        source_face = faces[0] if faces else None
    else:
        source_face = None
    # source_image = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR) if source_image_data else None

    # Process the frame (use your existing face swap logic)
    processed_frame = process_frame(source_face, frame)

    # Encode the processed frame to send back
    _, img_encoded = cv2.imencode('.jpg', processed_frame)
    return img_encoded.tobytes()


def create_face_object(image):
    # Assuming you have a model or method to detect and create a Face object
    face_detector = Face()
    detected_faces = face_detector.detect(image)  # Replace with actual detection method
    if detected_faces:
        # Assuming the first detected face is what we need
        face = detected_faces[0]
        # Assuming `normed_embedding` is calculated as part of the detection pipeline
        face.normed_embedding = face_detector.get_embedding(face)
        return face
    else:
        return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
