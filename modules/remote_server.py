from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from modules.processors.frame.face_swapper import process_frame  # Ensure you adapt this for your server environment

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():
    frame_data = request.files['frame'].read()
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

    source_image_data = request.files.get('source_image')
    source_image = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR) if source_image_data else None

    # Process the frame (use your existing face swap logic)
    processed_frame = process_frame(source_image, frame)

    # Encode the processed frame to send back
    _, img_encoded = cv2.imencode('.jpg', processed_frame)
    return img_encoded.tobytes()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
