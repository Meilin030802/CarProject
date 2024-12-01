import torch
import numpy as np
import cv2
import os
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# 加载 SAM 模型，并设置为使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").to(device)
predictor = SamPredictor(sam)

input_folder = r'C:\Users\Administrator\car-project\sam\img\img'
output_folder = r'C:\Users\Administrator\car-project\sam\out_file' 

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 遍历文件夹中的所有图像文件
for image_file in image_files:
    # 加载图像
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # 获取图像中心坐标
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    input_point = np.array([[center_x, center_y]])  # 使用图像中心作为分割点
    input_label = np.array([1])  # 默认标签为1（表示目标区域）

    # 设置预测器的图像，并将图像传送到GPU
    predictor.set_image(image)

    # 执行预测，使用较高的阈值（例如0.95）以分割更多区域
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,  # 启用多掩码输出
        return_logits=False
    )

    # 获取第一个掩码（可以根据分数选择最好的掩码）
    best_mask = masks[0]

    # 将掩码从256x256大小映射回原始图像大小
    best_mask_resized = cv2.resize(best_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    # 将分割掩码转为RGB格式，背景为白色
    mask_rgb = np.repeat(best_mask_resized[:, :, np.newaxis], 3, axis=2)  # 转为RGB格式掩码
    segmented_image = np.where(mask_rgb == 1, image, 255)

    # 保存结果图像
    output_image_path = os.path.join(output_folder, f"{image_file}")
    cv2.imwrite(output_image_path, segmented_image)

    # 可选：显示当前处理的图像
    print(f"Processing {output_image_path}")

print("批量处理完成。")
