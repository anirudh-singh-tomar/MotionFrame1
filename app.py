from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

# Temporary folder for saving files
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    video_path = os.path.join(OUTPUT_FOLDER, video_file.filename)
    video_file.save(video_path)
    
    # Process the video and save landmarks
    output_files = process_video(video_path)
    
    return jsonify(output_files), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    posList, RHList, LHList = [], [], []
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image for pose and hand landmarks
        pose_results = pose.process(img_rgb)
        hand_results = hands.process(img_rgb)

        if pose_results.pose_landmarks:
            lmString = ''
            for lm in pose_results.pose_landmarks.landmark:
                x, y, z = int(lm.x * img.shape[1]), img.shape[0] - int(lm.y * img.shape[0]), lm.z
                lmString += f'{x},{y},{z},'
            posList.append(lmString)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_type = handedness.classification[0].label
                lmString = ''
                for lm in hand_landmarks.landmark:
                    x, y, z = int(lm.x * img.shape[1]), img.shape[0] - int(lm.y * img.shape[0]), lm.z
                    lmString += f'{x},{y},{z},'
                if hand_type == "Right":
                    RHList.append(lmString)
                else:
                    LHList.append(lmString)

    cap.release()
    
    # Save the lists to files
    body_file = os.path.join(OUTPUT_FOLDER, "BodyLandmarks.txt")
    right_hand_file = os.path.join(OUTPUT_FOLDER, "RightHandLandmarks.txt")
    left_hand_file = os.path.join(OUTPUT_FOLDER, "LeftHandLandmarks.txt")
    
    with open(body_file, 'w') as f:
        f.writelines([f"{item}\n" for item in posList])
    with open(right_hand_file, 'w') as f:
        f.writelines([f"{item}\n" for item in RHList])
    with open(left_hand_file, 'w') as f:
        f.writelines([f"{item}\n" for item in LHList])
    
    return {
        "body_landmarks": f"/download/BodyLandmarks.txt",
        "right_hand_landmarks": f"/download/RightHandLandmarks.txt",
        "left_hand_landmarks": f"/download/LeftHandLandmarks.txt"
    }

if __name__ == '__main__':
    app.run(port=5000)