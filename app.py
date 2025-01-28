from flask import Flask
from flask_socketio import SocketIO
from PIL import Image
import base64
from io import BytesIO
from ultralytics import YOLO

# Inisialisasi Flask dan Socket.IO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model YOLO
model = YOLO('./model/yolo.tflite', task="detect")

@socketio.on("process_frame")
def handle_frame(image_data):
    try:
        # Decode frame dari Base64
        img_bytes = base64.b64decode(image_data.split(",")[1])
        img = Image.open(BytesIO(img_bytes))

        results = model.predict(img)
        boxes = results[0].boxes.data.cpu().numpy()

        data = [
            {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1]),
                "class": int(box[5]),
            }
            for box in boxes
        ]
        
        socketio.emit("bbox", data)
    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
