import cv2

# 加载 OpenCV 自带的 Haar Cascade 分类器，用于人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头（默认摄像头）
cap = cv2.VideoCapture(0)

while True:
    # 捕获视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图，因为 Haar Cascade 分类器在灰度图上效果更好
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸，scaleFactor 和 minNeighbors 是调整检测灵敏度的参数
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 遍历检测到的每个人脸并在其周围画矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示带有标记人脸的视频帧
    cv2.imshow('Face Detection', frame)

    # 按键 'q' 退出视频窗口
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
