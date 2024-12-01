import torch
import os
import shutil
from torchvision import models, transforms
from PIL import Image

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载保存的模型
def load_model(model_path, device):
    model = models.vgg19(pretrained=False)  # 不加载预训练权重
    model.classifier[6] = torch.nn.Linear(4096, 2)  # 重新定义最后一层为二分类
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # 设置模型为推理模式
    return model


# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加一个批次维度
    return image


# 推理函数：使用保存的模型进行推理
def predict(model, image_path, device):
    image = preprocess_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
    return predicted.item()  # 返回预测类别的标签


# 分类文件并移动
def classify_and_move_files(model, image_folder, class_folders, device):
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        if os.path.isfile(image_path):
            label = predict(model, image_path, device)

            # 根据预测结果将文件分类到不同的文件夹
            destination_folder = class_folders[label]
            destination_path = os.path.join(destination_folder, filename)

            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            shutil.move(image_path, destination_path)  # 将文件移动到目标文件夹
            print(f'Moved {filename} to {destination_folder}')


# 使用保存的模型进行推理并分类
def run_inference_and_classify(model_path, image_folder, class_folders, device):
    model = load_model(model_path, device)
    classify_and_move_files(model, image_folder, class_folders, device)


model_path = 'vgg19_model.pth'  # 已保存的模型路径
image_folder = 'images_to_classify'  # 需要分类的图像所在文件夹
class_folders = {0: 'outImg/left', 1: 'outImg/right'}  # 类别文件夹，0类和1类的图像分别存放在不同的文件夹中

run_inference_and_classify(model_path, image_folder, class_folders, device)
