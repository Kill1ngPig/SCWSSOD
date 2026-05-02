import os

# 数据路径
image_dir = './data/DUTS/image'
txt_path = './data/DUTS/train.txt'

# 获取所有文件名（去掉扩展名）
names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 写入 train.txt
with open(txt_path, 'w') as f:
    for name in names:
        f.write(name + '\n')

print(f"train.txt 已生成，共 {len(names)} 张图像")
