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

# 상태 변수 초기화
eye_closed_time = 0
closed_start_time = None
eye_closed_count = 0
sleep_state = 0
initial_pitch = None
last_sleep_state_1_time = None

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

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_video/")
async def upload_image(file: UploadFile = File(...)):
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

                global initial_pitch, eye_closed_time, closed_start_time, eye_closed_count, sleep_state, last_sleep_state_1_time

                if initial_pitch is None:
                    initial_pitch = pitch

                head_down_detected = initial_pitch - pitch > PITCH_DIFFERENCE_THRESHOLD

                # 눈 ROI 추출
                left_eye_roi = get_eye_roi(frame, face_landmarks, left_eye_indices, w, h)
                right_eye_roi = get_eye_roi(frame, face_landmarks, right_eye_indices, w, h)

                left_eye_resized = cv2.resize(left_eye_roi, input_size)
                right_eye_resized = cv2.resize(right_eye_roi, input_size)

                # 눈 예측
                left_eye_input = np.expand_dims(left_eye_resized / 255.0, axis=0)
                right_eye_input = np.expand_dims(right_eye_resized / 255.0, axis=0)

                if not head_down_detected:
                    left_eye_pred = model.predict(left_eye_input, verbose=0)[0][0]
                    right_eye_pred = model.predict(right_eye_input, verbose=0)[0][0]

                    if left_eye_pred < 0.5 and right_eye_pred < 0.5:
                        logging.info("눈이 닫혀 있다고 판단했습니다.")
                        if closed_start_time is None:
                            closed_start_time = time.time()
                            eye_closed_time = time.time() - closed_start_time
                        if 0.6 < eye_closed_time < 1.5:
                            eye_closed_count += 1
                            sleep_state = 1
                            last_sleep_state_1_time = time.time()
                        elif sleep_state == 1 and time.time() - last_sleep_state_1_time < 30:
                            sleep_state = 2
                        elif eye_closed_time >= 1.5:
                            sleep_state = 2
                    else:
                        logging.info("눈이 열려 있다고 판단했습니다.")
                        closed_start_time = None
                        eye_closed_time = 0

        logging.info("이미지 처리 및 응답 전송 완료")
        return {"filename": file.filename, "message": "Image received successfully", "sleep_state": sleep_state, "close_count": eye_closed_count}
    
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

