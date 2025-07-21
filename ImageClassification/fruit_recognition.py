import cv2
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from fruit_recognition_train import train_dataset

# 加载训练好的模型
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model.load_state_dict(torch.load('fruit_recognition_model.pth'))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 RGB 格式并预处理
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = transform(pil_img).unsqueeze(0)

    # 如果有 GPU 可用，使用 GPU
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
        model.to('cuda')

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()

    # 显示识别结果
    label = train_dataset.classes[predicted_class]
    cv2.putText(frame, f"Prediction: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Fruit Recognition", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
