import os
import random
import shutil

def move_files_by_parity(source_dir, odd_dir, even_dir, num_files=2000):
    # 打印当前工作目录
    print(f"当前工作目录: {os.getcwd()}")

    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"源文件夹 {source_dir} 不存在！")
        return

    # 获取目录中的所有文件名（只考虑数字命名的文件）
    all_files = [f for f in os.listdir(source_dir) if f.split('.')[0].isdigit()]  # 忽略扩展名

    # 分割文件为奇数和偶数文件
    odd_files = [f for f in all_files if int(f.split('.')[0]) % 2 != 0]
    even_files = [f for f in all_files if int(f.split('.')[0]) % 2 == 0]

    # 如果文件不足2000，调整数量为实际文件数
    odd_files_to_move = odd_files[:min(len(odd_files), num_files)]
    even_files_to_move = even_files[:min(len(even_files), num_files)]

    # 打印已选择的文件数量
    print(f"选取了 {len(odd_files_to_move)} 个单数文件。")
    print(f"选取了 {len(even_files_to_move)} 个双数文件。")

    # 创建目标文件夹（如果不存在）
    os.makedirs(odd_dir, exist_ok=True)
    os.makedirs(even_dir, exist_ok=True)

    # 将文件移动到对应的文件夹
    for file in odd_files_to_move:
        shutil.move(os.path.join(source_dir, file), os.path.join(odd_dir, file))

    for file in even_files_to_move:
        shutil.move(os.path.join(source_dir, file), os.path.join(even_dir, file))

    print(f"Moved {len(odd_files_to_move)} odd files to {odd_dir}")
    print(f"Moved {len(even_files_to_move)} even files to {even_dir}")

# 使用示例
source_directory = r'.\img'
odd_directory = r'.\setData\left'
even_directory = r'.\setData\right'

move_files_by_parity(source_directory, odd_directory, even_directory)
