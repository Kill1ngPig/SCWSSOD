import os
import shutil

# 原始路径
base_dir = './data/testing'
img_dir = os.path.join(base_dir, 'img')
gt_dir = os.path.join(base_dir, 'gt')

# 输出路径
output_base = './data'

# 要处理的数据集
datasets = os.listdir(img_dir)  # 自动读取所有子目录

for dataset in datasets:
    print(f'处理数据集: {dataset}')

    # 创建目标路径
    out_dataset_dir = os.path.join(output_base, dataset)
    image_out = os.path.join(out_dataset_dir, 'image')
    scribble_out = os.path.join(out_dataset_dir, 'scribble')
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(scribble_out, exist_ok=True)

    # 源路径
    img_src = os.path.join(img_dir, dataset)
    gt_src = os.path.join(gt_dir, dataset)

    # 复制图片文件
    for file in os.listdir(img_src):
        if file.endswith('.jpg') or file.endswith('.png'):
            shutil.copy(os.path.join(img_src, file), os.path.join(image_out, file))

    # 复制GT标注文件
    for file in os.listdir(gt_src):
        if file.endswith('.png') or file.endswith('.jpg'):
            shutil.copy(os.path.join(gt_src, file), os.path.join(scribble_out, file))

    # 生成 test.txt
    names = [os.path.splitext(f)[0] for f in os.listdir(image_out) if f.endswith('.jpg') or f.endswith('.png')]
    txt_path = os.path.join(out_dataset_dir, 'test.txt')
    with open(txt_path, 'w') as f:
        for name in sorted(names):
            f.write(name + '\n')

    print(f'完成: {dataset}, 共 {len(names)} 张图像')
