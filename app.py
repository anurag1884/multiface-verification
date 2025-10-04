# OpenCV library
import cv2
# NumPy library
import numpy as np
# ONNXRuntime library
import onnxruntime as ort

# Time library (for benchmarking)
import time
# OS library
import os

# Numba library
from numba import cuda
# InsightFace library
from insightface.app import FaceAnalysis
# scikit-learn library
from sklearn.metrics.pairwise import cosine_distances
# scikit-image library
from skimage import restoration, img_as_float
# Realtime DeepSORT library
from deep_sort_realtime.deepsort_tracker import DeepSort

# AlbumentationsX library
import albumentations as A

PROFILES_DIR = './profiles'

VIDEO_PATH = './vid.mp4'

OUTPUT_DIR = './'

MODEL_NAME = 'buffalo_l'
MODEL_DIR = './'
MODEL_DET_SIZE = (640, 640)

FACE_SIZE = (112, 112)

DET_RESOLUTION_WIDTH = 640

LAPLACIAN_THRESH = 70

CONFIDENCE_THRESH = 0.85

N_AUGMENTS = 10

MIN_CONFIRMATION_FRAMES = 2  # Frames to confirm a track
MAX_COS_DIST_THRESH = 0.5    # DeepSort appearance tolerance (higher = more tolerant)
MAX_FACE_FEATURES = 150      # How many appearance features to keep per track
NMS_MAX_OVERLAP = 0.6        # Allow detection jitter before NMS suppression

DEFAULT_COS_THRESH = 0.41
COS_THRESH_ADJUSTMENT_SCALE = -0.04
COS_THRESH_MOTION_FREQ = 20

DEFAULT_MAX_AGE = 10
MAX_AGE_ADJUSTMENT_SCALE = -9
MAX_AGE_MOTION_FREQ = 0.8

DEBLUR_MAG_LOW_THRESH = 3.6   # Threshold for triggering deblur
GUIDER_MAG_HIGH_THRESH = 1.8  # Threshold for triggering 'Best Shot' guide

GUIDER_RATIO = 0.038
TRACKER_GAP = 4

DET_BOX_THICKNESS = 2
TRACKER_THICKNESS = 1

FONT_THICKNESS = 1
FONT_SCALE = 0.8

TEXT_OFFSET = 5

SMOOTH_STEP = 0.2  # Interpolation smoothness step for the guider
SNAP_THRESH = 0.7  # Intersection threshold between the guider and the center point

# Laplacian variance to determine quality of an image/crop
def laplacian_variance(gray_img):
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

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

class ProfileIdentifier:
    # Creates a 'ProfileIdentifier' instance
    def __init__(self, model):
        # Detection model
        self.model = model
        # Stores individual profiles
        self.profiles = []

    # Creates a profile/candidate to match against
    def create_profile(self, info, profile_img_data=None, profile_embeddings=None):
        # Stores all embeddings attached to this profile
        embs = []
        # Add embeddings from image paths if available
        if profile_img_data is not None:
            for img_data in profile_img_data:
                # Gets the largest face detected in the image
                face = self.model.get_largest_face(img_data)
                # If no faces were found, raise an error
                if face is None:
                    raise ValueError(f'No face detected in frame.')
                # Otherwise, append the new face's embedding to the profile
                embs.append(face.normed_embedding)
        # Add embeddings explicitly if available
        if profile_embeddings is not None:
            embs.extend(profile_embeddings)
        # Picks the embedding closest to the rest
        def medoid_embedding(embs):
            D = cosine_distances(embs)
            medoid_idx = np.argmin(D.sum(axis=0))
            return embs[medoid_idx]
        # Create a new profile for the person with their newly generated facial and embedding data
        self.profiles.append({
            # Compute the overall mean embedding
            'embedding': medoid_embedding(embs),
            'info': info
        })

    # Finds profile matches within a specified group image using a given cosine threshold and appends them to a dictionary
    def find_matches(self, group_faces, img, laplacian_thresh, threshold, unique_per_profile=True):
        matched, unmatched = {}, {}
        # If no faces were found, return an empty list
        if not group_faces:
            return matched, unmatched
        # Detect and store any existing profile matches in the image
        for face in group_faces:
            # Round the fields of the bounding box to the nearest integer
            bbox = np.round(face.bbox).astype(np.int32)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = np.clip(bbox, [0, 0, 0, 0], [w, h, w, h])
            gray_face_crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            # Skip blurry faces
            if laplacian_variance(gray_face_crop) < laplacian_thresh:
                continue
            face_emb = np.ravel(face.normed_embedding)  # Shape: (512,)
            max_sim_idx, max_sim = None, -1
            for idx, profile in enumerate(self.profiles):
                # Compute the cosine similarity
                sim = np.dot(profile['embedding'], face_emb)
                # Record the profile with the highest similarity
                if max_sim_idx is None or sim > max_sim:
                    max_sim_idx, max_sim = idx, sim
            # Compare the profile with the highest similarity against a threshold and classify it as matched or unmatched
            if max_sim_idx is not None:
                if max_sim > threshold:
                    if unique_per_profile:
                        # Keep only the *best* match for each profile
                        if max_sim_idx not in matched or max_sim > max(matched[max_sim_idx]['sims']):
                            matched[max_sim_idx] = {
                                'confidence_scores': [face.det_score],
                                'bboxes': [face.bbox],
                                'sims': [max_sim],
                                'profile': self.profiles[max_sim_idx]
                            }
                    else:
                        # Allow multiple matches
                        matched.setdefault(max_sim_idx, {
                            'confidence_scores': [],
                            'bboxes': [],
                            'sims': [],
                            'profile': self.profiles[max_sim_idx]
                        })
                        matched[max_sim_idx]['confidence_scores'].append(face.det_score)
                        matched[max_sim_idx]['bboxes'].append(face.bbox)
                        matched[max_sim_idx]['sims'].append(max_sim)
                else:
                    unmatched.setdefault(max_sim_idx, {
                        'confidence_scores': [],
                        'bboxes': []
                    })
                    unmatched[max_sim_idx]['confidence_scores'].append(face.det_score)
                    unmatched[max_sim_idx]['bboxes'].append(face.bbox)
        return matched, unmatched

    # Resets the current set of profiles
    def reset_current_profiles(self):
        self.profiles.clear()

# Intersection Over Union (IOU)
def iou(boxA, boxB):
    # Extract positions from each box
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    # Compute intersection
    interX1 = max(xA1, xB1)
    interY1 = max(yA1, yB1)
    interX2 = min(xA2, xB2)
    interY2 = min(yA2, yB2)
    interW = max(0, interX2 - interX1)
    interH = max(0, interY2 - interY1)
    interArea = interW * interH
    # Compute union
    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    boxBArea = (xB2 - xB1) * (yB2 - yB1)
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

# Estimate average scene motion using optical flow, now including average angle.
def estimate_scene_motion(frame, prev_gray, sample_step=16):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0, 0, gray
    # Calculate dense optical flow (Farnebäck)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    # Sample flow vectors to reduce noise
    h, w = flow.shape[:2]
    flow_sampled = flow[::sample_step, ::sample_step]
    # Compute magnitude and angle
    flow_x = flow_sampled[..., 0]
    flow_y = flow_sampled[..., 1]
    mag, ang = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
    avg_mag = np.mean(mag)
    avg_ang = np.mean(ang)
    return avg_mag, avg_ang, gray

augment = A.Compose([
    # Horizontal flip is standard for faces
    A.HorizontalFlip(p=0.5),

    # Gentle geometric transforms
    A.Affine(
        scale=(0.9, 1.1),                # Zoom in/out
        translate_percent=(0.05, 0.05),  # Random shifts
        rotate=(-15, 15),                # Rotation range
        shear=(-10, 10),                 # Shearing
        p=0.7
    ),

    # Slight perspective jitter mimics camera angle variation
    A.Perspective(scale=(0.02, 0.05), p=0.3),

    # Color & illumination variation
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02,
        p=0.5
    ),

    # Random blur/sharpness for robustness to focus changes
    A.OneOf([
        A.MotionBlur(blur_limit=3, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Sharpen(p=0.3)
    ], p=0.3),

    # Small JPEG compression noise (simulates bad uploads)
    A.ImageCompression(quality_range=(60, 100), p=0.3),

    # Final resize/crop to target face size
    A.Resize(FACE_SIZE[0], FACE_SIZE[1], p=1),
])

profiles = {}

model = DetectionModel(
    model_name=MODEL_NAME,
    model_root=MODEL_DIR,
    det_size=MODEL_DET_SIZE
)

rec_model = model.app.models['recognition']
for name in os.listdir(PROFILES_DIR):
    embs = []
    profile_path = os.path.join(PROFILES_DIR, name)
    for img_path in os.listdir(profile_path):
        img = cv2.imread(os.path.join(profile_path, img_path))
        if img is None:
            continue
        emb = rec_model.get_feat(img)    # Give raw BGR image, ArcFaceONNX handles preprocessing
        emb = emb[0]                     # Strip batch dimension
        emb = emb / np.linalg.norm(emb)  # Normalize embedding
        embs.append(emb)
        rgb_face_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Apply augmentations
        for i in range(N_AUGMENTS):
            aug_face_img = augment(image=rgb_face_img)['image']
            aug_face_img = cv2.cvtColor(aug_face_img, cv2.COLOR_RGB2BGR)
            aug_emb = rec_model.get_feat(aug_face_img)[0]
            aug_emb = aug_emb / np.linalg.norm(aug_emb)
            embs.append(aug_emb)
    # Only save if at least one embedding was assigned to the profile
    if embs:
        profiles[name] = embs

identifier = ProfileIdentifier(model)

# Create profiles for each individual from the generated profile embedding list
for name, embs in profiles.items():
    identifier.create_profile(
        profile_embeddings=embs,
        info={
            'name': name
        }
    )

print(f'{len(profiles)} profile(s) created.')

if VIDEO_PATH:
    # Using an existing video
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    # Fallback to using webcam
    cap = cv2.VideoCapture(0)

orig_size = (round(cap.get(3)), round(cap.get(4)))
scale = DET_RESOLUTION_WIDTH / orig_size[0]  # Width scaling factor
new_size = (DET_RESOLUTION_WIDTH, round(orig_size[1] * scale))

filename_ext = os.path.splitext(os.path.basename(VIDEO_PATH))
output_path = os.path.join(OUTPUT_DIR, filename_ext[0]) + '_output' + filename_ext[1]
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, orig_size)

# Initialize tracker
sort = DeepSort(
    max_cosine_distance=MAX_COS_DIST_THRESH,
    nn_budget=MAX_FACE_FEATURES,
    nms_max_overlap=NMS_MAX_OVERLAP
)

t = time.time()

laplacian_thresh = LAPLACIAN_THRESH * scale

guider_radius = round(orig_size[1] * GUIDER_RATIO)
tracker_radius = round(guider_radius + TRACKER_GAP / scale)

font_thickness = round(FONT_THICKNESS / scale)
det_box_thickness = round(DET_BOX_THICKNESS / scale)
tracker_thickness = round(TRACKER_THICKNESS / scale)
font_scale = round(FONT_SCALE / scale)
text_offset = round(TEXT_OFFSET / scale)

track_best_conf = {}
old_pos = new_pos = prev_gray = None
bound_box = [None, None, None, None]
det_list, info_list = [], []
while True:
    # Read frame from input
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, new_size)

    # Optical flow magnitude, angle, or gyro if available
    scene_motion_mag, scene_motion_ang, prev_gray = estimate_scene_motion(resized_frame, prev_gray)

    # Deblur if significant camera shake/motion detected
    if scene_motion_mag > DEBLUR_MAG_LOW_THRESH:
        # Convert to gray image for scikit-image
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        gray = img_as_float(gray)
        # Define a PSF (here, a small motion blur kernel)
        psf = np.ones((5, 5)) / 25
        # Apply Richardson-Lucy deconvolution for deblurring the image
        deconvolved = restoration.richardson_lucy(gray, psf, num_iter=30)
        # Convert the deconvolved grayscale image back to BGR for OpenCV
        resized_frame = (deconvolved * 255).astype(np.uint8)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
        track_best_conf.clear()
        sort.delete_all_tracks()
        sort.n_init = 1
    else:
        sort.n_init=MIN_CONFIRMATION_FRAMES

    # Generate faces from the group image
    group_faces = model.get_group_faces(resized_frame)

    # Crowd density (Number of faces in current frame)
    crowd_density = len(group_faces)

    cos_threshold = DEFAULT_COS_THRESH + COS_THRESH_ADJUSTMENT_SCALE * min(1, crowd_density / COS_THRESH_MOTION_FREQ)
    sort.max_age = round(DEFAULT_MAX_AGE + MAX_AGE_ADJUSTMENT_SCALE * min(1, scene_motion_mag / MAX_AGE_MOTION_FREQ))

    matched, unmatched = identifier.find_matches(group_faces, resized_frame, laplacian_thresh, cos_threshold)
    
    # Build detections from matched profiles
    for m in matched.values():
        for confidence, bbox in zip(m['confidence_scores'], m['bboxes']):
            det_list.append(([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], confidence))
            info_list.append(m['profile']['info'])
            bbox = np.round(bbox / scale).astype(np.int32)
            if bound_box[0] is None or bbox[0] < bound_box[0]:
                bound_box[0] = bbox[0]
            if bound_box[1] is None or bbox[1] < bound_box[1]:
                bound_box[1] = bbox[1]
            if bound_box[2] is None or bbox[2] > bound_box[2]:
                bound_box[2] = bbox[2]
            if bound_box[3] is None or bbox[3] > bound_box[3]:
                bound_box[3] = bbox[3]
    
    # Build detections from unmatched profiles
    for m in unmatched.values():
        for confidence, bbox in zip(m['confidence_scores'], m['bboxes']):
            det_list.append(([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], confidence))
            info_list.append(None)
            bbox = np.round(bbox / scale).astype(np.int32)
            if bound_box[0] is None or bbox[0] < bound_box[0]:
                bound_box[0] = bbox[0]
            if bound_box[1] is None or bbox[1] < bound_box[1]:
                bound_box[1] = bbox[1]
            if bound_box[2] is None or bbox[2] > bound_box[2]:
                bound_box[2] = bbox[2]
            if bound_box[3] is None or bbox[3] > bound_box[3]:
                bound_box[3] = bbox[3]

    # Update tracker with matched detections
    tracks = sort.update_tracks(det_list, others=info_list, frame=resized_frame)

    det_list.clear()
    info_list.clear()

    box_overlay = frame.copy()
    text_overlay = np.zeros_like(frame)
    for track in tracks:
        bbox = np.round(track.to_tlbr() / scale).astype(np.int32)
        if track.is_confirmed():
            current_conf = track.get_det_conf()
            if current_conf and current_conf > track_best_conf.get(track.track_id, 0):
                track_best_conf[track.track_id] = current_conf
            best_conf = track_best_conf.get(track.track_id, 0)
            info = track.get_det_supplementary()
            is_matched = info != None
            color = (0, 255, 0) if is_matched else (0, 255, 255) if best_conf > CONFIDENCE_THRESH else (0, 0, 255)
            cv2.rectangle(box_overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, det_box_thickness)
            if is_matched:
                cv2.putText(text_overlay, info['name'], (bbox[2] + text_offset, bbox[3]), cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(0, 255, 0), thickness=font_thickness, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(box_overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), det_box_thickness)
    cv2.addWeighted(box_overlay, 0.5, frame, 0.5, 0, frame)
    text_overlay_gray = cv2.cvtColor(text_overlay, cv2.COLOR_BGR2GRAY)
    _, text_overlay_mask = cv2.threshold(text_overlay_gray, 1, 255, cv2.THRESH_BINARY)  # Everything > 0 becomes white
    text_overlay_mask_inv = cv2.bitwise_not(text_overlay_mask)
    bg_part = cv2.bitwise_and(frame, frame, mask=text_overlay_mask_inv)
    fg_part = cv2.bitwise_and(text_overlay, text_overlay, mask=text_overlay_mask)
    frame = cv2.add(bg_part, fg_part)

    if scene_motion_mag < GUIDER_MAG_HIGH_THRESH:
        if None not in bound_box and 0 < iou(bound_box, [0, 0, *orig_size]) <= 1:
            old_pos = new_pos
            new_pos = np.array([bound_box[0] + bound_box[2], bound_box[1] + bound_box[3]]) // 2
        if old_pos is not None:
            overlay = frame.copy()
            interpolated_pos = np.round((1 - SMOOTH_STEP) * old_pos + SMOOTH_STEP * new_pos).astype(np.int32)
            center = np.array(orig_size) // 2
            if np.linalg.norm(center - interpolated_pos) < SNAP_THRESH * guider_radius:
                cv2.circle(overlay, tuple(interpolated_pos), guider_radius, (0, 215, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, tuple(interpolated_pos), tracker_radius, (0, 215, 255), tracker_thickness, lineType=cv2.LINE_AA)
            else:
                cv2.circle(overlay, tuple(interpolated_pos), guider_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, tuple(center), tracker_radius, (255, 255, 255), tracker_thickness, lineType=cv2.LINE_AA)
                cv2.putText(overlay, 'Best Shot', tuple(interpolated_pos + guider_radius + text_offset), cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    bound_box[0] = bound_box[1] = bound_box[2] = bound_box[3] = None
    
    # Write frame to output
    out.write(frame) 

print(f'Time spent: {(time.time() - t):.2f} seconds')

cap.release()
out.release()
