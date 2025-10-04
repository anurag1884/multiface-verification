# OpenCV library
import cv2
# NumPy library
import numpy as np
# ONNXRuntime library
import onnxruntime as ort

# OS library
import os
# Shell Utilities (shutil) library
import shutil

# Numba library
from numba import cuda
# InsightFace library
from insightface.app import FaceAnalysis

SRC_IMG_PATHS = [
    os.path.join('./assets', file) for file in os.listdir('./assets')
]

OUTPUT_PROFILES_DIR = './profiles'

MODEL_NAME = 'buffalo_l'
MODEL_DIR = './'
MODEL_DET_SIZE = (960, 960)

FACE_SIZE = (112, 112)
FACE_THRESH_SIZE = (80, 80)

LAPLACIAN_THRESH = 70

STANDARD_POINT_INDICES = [
    38,  # Left eye center (approx)
    88,  # Right eye center (approx)
    86,  # Nose tip
    52,  # Left mouth corner
    61,  # Right mouth corner
]

REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041]    # Right mouth corner
], dtype=np.float32)

class DetectionModel:
    # Creates a 'DetectionModel' instance
    def __init__(self, model_name, model_root, det_size=(640, 640)):
        # Initialize app (it will load defaults first)
        if cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
            print('Running on the GPU...')
            self.app = FaceAnalysis(name=model_name, root=model_root, providers=['CUDAExecutionProvider'])
            self.ctx_id = 0
        else:
            print('Running on the CPU...')
            self.app = FaceAnalysis(name=model_name, root=model_root, providers=['CPUExecutionProvider'])
            self.ctx_id = -1
        # Prepare the app for usage
        self.app.prepare(ctx_id=self.ctx_id, det_size=det_size)

    # Generates a list of faces detected in a group image data
    def get_group_faces(self, img_data):
        # Return the fetched faces from the image object
        return self.app.get(img_data)

    # Returns the largest face (if any) detected in an image data
    def get_largest_face(self, img_data):
        # Find faces in the specified image
        faces = self.get_group_faces(img_data)
        # If no faces were found, return None
        if not faces:
            return None
        # Return the largest detected face in the image
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

if os.path.exists(OUTPUT_PROFILES_DIR):
    shutil.rmtree(OUTPUT_PROFILES_DIR)
os.makedirs(OUTPUT_PROFILES_DIR, exist_ok=True)

model = DetectionModel(
    model_name=MODEL_NAME,
    model_root=MODEL_DIR,
    det_size=MODEL_DET_SIZE
)

landmark_model = model.app.models['landmark_2d_106']

count = 1
for path in SRC_IMG_PATHS:
    img = cv2.imread(path)
    if img is None:
        continue
    faces = model.get_group_faces(img)
    for face in faces:
        bbox = np.round(face.bbox).astype(np.int32)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = np.clip(bbox, [0, 0, 0, 0], [w, h, w, h])
    
        # Skip small faces
        if x2 - x1 < FACE_THRESH_SIZE[0] or y2 - y1 < FACE_THRESH_SIZE[1]:
            continue
    
        os.makedirs(os.path.join(OUTPUT_PROFILES_DIR, f'profile{count}'), exist_ok=True)

        # Run 106-point landmark detector
        lmks = landmark_model.get(img, face)
    
        # Pick standard points corresponding to each reference point
        std_points = np.array([lmks[i] for i in STANDARD_POINT_INDICES], dtype=np.float32)

        # Align face
        M, _ = cv2.estimateAffinePartial2D(std_points, REFERENCE_POINTS, method=cv2.LMEDS)
        if M is None:
            continue
        aligned_face = cv2.warpAffine(img, M, FACE_SIZE, borderValue=0)
        
        # Skip blurry faces
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < LAPLACIAN_THRESH:
            continue

        cv2.imwrite(os.path.join(OUTPUT_PROFILES_DIR, f'profile{count}/profile{count}.jpg'), aligned_face)
        count += 1

shutil.make_archive(OUTPUT_PROFILES_DIR, 'zip', OUTPUT_PROFILES_DIR)
