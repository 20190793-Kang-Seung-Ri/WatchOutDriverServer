from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import io
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import math
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import nest_asyncio

# nest_asyncio 적용
nest_asyncio.apply()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# GPU 사용 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU 사용 가능")
    except RuntimeError as e:
        print(e)

# 고개 각도 임계값
PITCH_DIFFERENCE_THRESHOLD = 15

# MediaPipe 및 모델 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
model = load_model("eye_model_fold_1.h5")
input_size = (224, 224)

# 눈 랜드마크 인덱스
left_eye_indices = [33, 160, 158, 157, 154, 153, 145, 144, 163, 7, 173, 246, 130, 133, 159]
right_eye_indices = [362, 385, 387, 388, 390, 373, 374, 380, 382, 263, 466, 359, 386, 384, 362]

# 서버 시작 시 변수 생성
global average_ear, eye_closed_count, initial_pitch, sleep_state  # 눈 벌림 정도, 눈 감은 횟수, 고개 초기값, 졸음 레벨
global eye_small_start, eye_small_time                            # 눈 작아진 시작, 눈 작아진 시간
global closed_start_time, eye_closed_time                         # 눈 감은 시작, 눈 감은 시간
global awake_start_time, awake_time                               # 눈 뜬 시작, 눈 뜬 시간
global head_down_start, head_down_time                            # 고개 내린 시작, 고개 내린 시간

# 초기화
average_ear = 1
eye_closed_count = 0
initial_pitch = None
sleep_state = 0

eye_small_start = None
eye_small_time = 0

closed_start_time = None
eye_closed_time = 0

awake_start_time = None
awake_time = 0

head_down_start = None
head_down_time = 0

def get_eye_roi(frame, face_landmarks, eye_indices, w, h, eye_margin=5):
    coords = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in eye_indices])
    eye_rect = cv2.boundingRect(coords)
    x, y, w, h = eye_rect
    return frame[y-eye_margin:y+h+eye_margin, x-eye_margin:x+w+eye_margin]

def calculate_head_angle(face_landmarks):
    nose_tip = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]
    pitch = math.degrees(math.atan2(nose_tip.y - chin.y, nose_tip.z - chin.z))
    return pitch

# EAR(Eye Aspect Ratio) 계산 함수
def calculate_ear(eye_landmarks, frame_width, frame_height):
    # 수직 거리
    A = np.linalg.norm(np.array([eye_landmarks[1].x * frame_width, eye_landmarks[1].y * frame_height]) -
                       np.array([eye_landmarks[5].x * frame_width, eye_landmarks[5].y * frame_height]))
    B = np.linalg.norm(np.array([eye_landmarks[2].x * frame_width, eye_landmarks[2].y * frame_height]) -
                       np.array([eye_landmarks[4].x * frame_width, eye_landmarks[4].y * frame_height]))
    # 수평 거리
    C = np.linalg.norm(np.array([eye_landmarks[0].x * frame_width, eye_landmarks[0].y * frame_height]) -
                       np.array([eye_landmarks[3].x * frame_width, eye_landmarks[3].y * frame_height]))
    ear = (A + B) / (2.0 * C)
    return ear

# 눈 벌림 정도 계산을 위한 함수
def calculate_eye_opening(left_eye_landmarks, right_eye_landmarks, frame_width, frame_height):
    left_ear = calculate_ear(left_eye_landmarks, frame_width, frame_height)
    right_ear = calculate_ear(right_eye_landmarks, frame_width, frame_height)

    # 눈 벌림 정도 (EAR의 평균을 사용)
    average_ear = (left_ear + right_ear) / 2.0
    return round(average_ear, 2)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/initialize/")
async def initialize_endpoint():
    global average_ear, eye_closed_count, initial_pitch, sleep_state  # 눈 벌림 정도, 눈 감은 횟수, 고개 초기값, 졸음 레벨
    global eye_small_start, eye_small_time                            # 눈 작아진 시작, 눈 작아진 시간
    global closed_start_time, eye_closed_time                         # 눈 감은 시작, 눈 감은 시간
    global awake_start_time, awake_time                               # 눈 뜬 시작, 눈 뜬 시간
    global head_down_start, head_down_time                            # 고개 내린 시작, 고개 내린 시간
    
    # 초기화
    average_ear = 1
    eye_closed_count = 0
    initial_pitch = None
    sleep_state = 0

    eye_small_start = None
    eye_small_time = 0

    closed_start_time = None
    eye_closed_time = 0

    awake_start_time = None
    awake_time = 0
    
    head_down_start = None
    head_down_time = 0

    return {"message": "서버가 초기화되었습니다."}

@app.post("/process_video/")
async def upload_image(file: UploadFile = File(...)):
    global average_ear, eye_closed_count, initial_pitch, sleep_state  # 눈 벌림 정도, 눈 감은 횟수, 고개 초기값, 졸음 레벨
    global eye_small_start, eye_small_time                            # 눈 작아진 시작, 눈 작아진 시간
    global closed_start_time, eye_closed_time                         # 눈 감은 시작, 눈 감은 시간
    global awake_start_time, awake_time                               # 눈 뜬 시작, 눈 뜬 시간
    global head_down_start, head_down_time                            # 고개 내린 시작, 고개 내린 시간
    try:
        # 요청 수신 로그
        logging.info(f"이미지 파일 {file.filename} 수신 중...")

        # 파일 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logging.info(f"이미지 크기: {image.size}")

        # 이미지 크기 확인 (224x224)
        if image.size != (224, 224):
            logging.warning(f"이미지 크기 불일치: {image.size}, 224x224 크기 요구됨.")
            return {"error": "Image must be 224x224 in size."}

        # 이미지 처리 (예: numpy 배열로 변환)
        img_array = np.array(image)
        logging.info(f"이미지 처리 완료, 배열 크기: {img_array.shape}")

        # 여기에서 필요한 처리를 수행
        # 예: 모델 예측 등

        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 얼굴 랜드마크 및 처리
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                pitch = calculate_head_angle(face_landmarks)

                if initial_pitch is None:
                    initial_pitch = pitch

                head_down_detected = initial_pitch - pitch > PITCH_DIFFERENCE_THRESHOLD

                # 눈 ROI 추출
                left_eye_roi = get_eye_roi(frame, face_landmarks, left_eye_indices, w, h)
                right_eye_roi = get_eye_roi(frame, face_landmarks, right_eye_indices, w, h)

                left_eye_resized = cv2.resize(left_eye_roi, input_size)
                right_eye_resized = cv2.resize(right_eye_roi, input_size)

                # EAR 계산
                average_ear = calculate_eye_opening([face_landmarks.landmark[i] for i in left_eye_indices], 
                                                    [face_landmarks.landmark[i] for i in right_eye_indices], 
                                                    w, h)

                # 눈 예측
                left_eye_input = np.expand_dims(left_eye_resized / 255.0, axis=0)
                right_eye_input = np.expand_dims(right_eye_resized / 255.0, axis=0)

                if head_down_detected:
                    if head_down_start is None:
                        head_down_start = time.time()

                    head_down_time = time.time() - head_down_start

                    if head_down_time > 1 and sleep_state >= 1:
                        sleep_state = 3
                    elif head_down_time > 5 and sleep_state == 0:
                        sleep_state = 3
                else:
                    left_eye_pred = model.predict(left_eye_input, verbose=0)[0][0]
                    right_eye_pred = model.predict(right_eye_input, verbose=0)[0][0]

                    # 수면 상태 판별 로직
                    if left_eye_pred < 0.5 or right_eye_pred < 0.5:
                        if closed_start_time is None:
                            closed_start_time = time.time()  # 눈 감기 시작 시간 기록

                        eye_closed_count += 1
                        eye_closed_time = time.time() - closed_start_time  # 눈 감고 있는 시간 계산

                        if 0.5 < eye_closed_time <= 1:    
                            sleep_state = 1
                            awake_start_time = None
                            awake_time = 0

                        if 1 < eye_closed_time <= 2:
                            sleep_state = 2

                        if eye_closed_time > 2:
                            sleep_state = 3
                    else:
                        head_down_start = None
                        head_down_time = 0
                        closed_start_time = None
                        eye_closed_time = 0
                        eye_small_start = None
                        eye_small_time = 0

                        if awake_start_time is None:
                            awake_start_time = time.time()
                        
                        awake_time = time.time() - awake_start_time

                        if average_ear < 0.45:
                            if eye_small_start is None:
                                eye_small_start = time.time()

                            eye_small_time = time.time() - eye_small_start

                            if eye_small_time >= 10 and sleep_state == 0:
                                sleep_state = 1

                        # 깨어 있는 상태가 10초 이상 지속되면 수면 상태 초기화
                        if awake_time >= 10 and sleep_state > 0:
                            sleep_state -= 1
                            awake_start_time = None
                            awake_time = 0

        logging.info("이미지 처리 및 응답 전송 완료")
        return {"filename": file.filename, "message": "Image received successfully", "sleep_state": sleep_state, "close_count": average_ear}
    
    except Exception as e:
        logging.error(f"이미지 처리 중 오류 발생: {e}")
        return {"error": "Failed to process the image"}

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
        <form action="/process_video/" enctype="multipart/form-data" method="post">
        <label for="file">Upload your image (JPG only):</label>
        <input name="file" type="file" accept="image/jpeg" required>
        <input type="submit" value="Upload">
        </form>
    """

    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn

    # SSL 인증서 및 키 경로 설정
    ssl_certfile = "C:/OpenSSL/bin/private.crt"  # 인증서 파일 경로
    ssl_keyfile = "C:/OpenSSL/bin/private.key"   # 개인 키 파일 경로

    logging.info("서버 시작 중...")
    # uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    uvicorn.run(app, host="0.0.0.0", port=8000)

