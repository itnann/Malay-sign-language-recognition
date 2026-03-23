import torch
import numpy as np
import cv2
import mediapipe as mp

GESTURES = np.array(['abang', 'ada', 'ambil', 'anak_lelaki', 'anak_perempuan', 'apa', 'apa_khabar',
            'arah', 'assalamualaikum', 'ayah', 'baca', 'bagaimana', 'bahasa_isyarat',
            'baik', 'baik_2', 'bapa', 'bapa_saudara', 'bas', 'bawa', 'beli', 'beli_2',
            'berapa', 'berjalan', 'berlari', 'bila', 'bola', 'boleh', 'bomba', 'buang',
            'buat', 'curi', 'dapat', 'dari', 'emak', 'emak_saudara', 'hari', 'hi', 'hujan',
            'jahat', 'jam', 'jangan', 'jumpa', 'kacau', 'kakak', 'keluarga', 'kereta',
            'kesakitan', 'lelaki', 'lemak', 'lupa', 'main', 'makan', 'mana', 'marah', 'mari',
            'masa', 'masalah', 'minum', 'mohon', 'nasi', 'nasi_lemak', 'panas', 'panas_2',
            'pandai', 'pandai_2', 'payung', 'pen', 'pensil', 'perempuan', 'pergi', 'pergi_2',
            'perlahan', 'perlahan_2', 'pinjam', 'polis', 'pukul', 'ribut', 'sampai',
            'saudara', 'sejuk', 'sekolah', 'siapa', 'sudah', 'suka', 'tandas', 'tanya',
            'teh_tarik', 'teksi', 'tidur', 'tolong'])
INPUT_SIZE = 258
HIDDEN_SIZE = 64
NUM_CLASSES = len(GESTURES)

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion from BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is no longer writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)   # Color conversion RGB to BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # Draw right connections

def draw_styled_landmarks(image,results):

    # Draw pose connection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )
    # Draw left hand connection
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )
    # Draw right hand connection
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])


def process_video_file(video_path, sequence_length=30):
    """
    智能处理策略：
    1. 遍历视频，记录每一帧的关键点 + 是否检测到手。
    2. 找到动作的'有效区间' (从第一次出现手到最后一次出现手)。
    3. 对有效区间进行'均匀采样'或'补零'，统一为30帧。
    """
    cap = cv2.VideoCapture(video_path)
    raw_frames = []  # 存关键点数据
    hand_indices = []  # 存检测到手的帧索引 (0, 1, 5, 6...)

    frame_idx = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe 处理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # 提取特征 (确保你的 extract_keypoints 返回的是 numpy array)
            keypoints = extract_keypoints(results)
            raw_frames.append(keypoints)

            # 判断当前帧是否有手 (左手 OR 右手)
            # 注意：results.left_hand_landmarks 为 None 表示没检测到
            has_hand = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

            if has_hand:
                hand_indices.append(frame_idx)

            frame_idx += 1

    cap.release()

    # 转换为 numpy 方便切片
    raw_data = np.array(raw_frames)  # Shape: (Total_Frames, 258)
    total_frames = raw_data.shape[0]

    if total_frames == 0:
        return None

    # --- 策略核心：确定有效区间 (Region of Interest) ---
    if len(hand_indices) > 0:
        # 找到第一帧有手和最后一帧有手的位置
        start_idx = hand_indices[0]
        end_idx = hand_indices[-1]

        # 可选：增加一点缓冲 (Buffer)，比如前后各多取 2 帧，防止动作切太死
        start_idx = max(0, start_idx - 2)
        end_idx = min(total_frames - 1, end_idx + 2)

        # 提取有效片段
        valid_data = raw_data[start_idx: end_idx + 1]
    else:
        # 如果全程都没检测到手，无奈只能用全视频（或者是空视频）
        # 这里选择用全视频作为兜底
        valid_data = raw_data

    valid_frames_count = valid_data.shape[0]

    # --- 数据对齐：统一到 sequence_length (30) ---
    if valid_frames_count == sequence_length:
        # 刚好30帧，完美
        final_data = valid_data

    elif valid_frames_count > sequence_length:
        # 策略：均匀采样 (Uniform Sampling)
        # 比“取中间”更好，因为它保留了动作的开始、中间和结束
        indices = np.linspace(0, valid_frames_count - 1, sequence_length, dtype=int)
        final_data = valid_data[indices]

    else:  # valid_frames_count < sequence_length
        # 策略：末尾补零 (Zero Padding)
        padding_len = sequence_length - valid_frames_count
        padding = np.zeros((padding_len, 258))  # 假设 input_size 是 258
        final_data = np.vstack((valid_data, padding))

    # 返回 Tensor: (1, 30, 258)
    return torch.tensor(final_data, dtype=torch.float32).unsqueeze(0)