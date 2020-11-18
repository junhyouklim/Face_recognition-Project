import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_classifier = load_model('emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

# 캠으로 비디오 캡쳐
camera = cv2.VideoCapture(0)

while True:
    # 캠화면 캡쳐
    ret, frame = camera.read()
    # 캡쳐 프레임 그레이스케일
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 캡쳐 프레임에서 얼굴 인식
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))

    # 빈 이미지 생성
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    # 얼굴이 인식 된 경우 얼굴 표현 표시
    if len(faces) > 0:
        # 사이즈 큰 이미지 정렬
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        for face in faces:
            (fX, fY, fW, fH) = face
            # 이미지 사이즈 48x48 변환
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 얼굴 표현 예측
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # 정답 얼굴 화면에 표시
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 255, 255), 2)

            # 정답 및 그래프 표시
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 123, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow("Probabilities", canvas)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()