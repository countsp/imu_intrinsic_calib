import numpy as np
import cv2
import os
from PIL import Image

# 遍历照片
def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path)
                images.append(image_path)
            except IOError:
                print("Cannot open image: ", filename)
    return images

# 标定板格点数量和大小
pattern_size = (8, 11)  # 内部角点数量
square_size = 60  # 棋盘格方块大小（毫米）

# 存储棋盘格角点的3D坐标
obj_points = []
# 存储棋盘格对应的图像点坐标
img_points = []

# 准备棋盘格的3D坐标
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 指定文件夹路径
folder_path = "/home/office2004/cam_calib/241224front"
# 调用函数读取图片
images = read_images_from_folder(folder_path)

# 遍历所有标定图像
for image_path in images:
    # 读取图像并将其转换为灰度图
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # 如果找到棋盘格角点则存储对应的3D和2D坐标
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        # 在图像上绘制棋盘格角点
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        cv2.imshow('Chessboard Corners', image)
        cv2.waitKey(500)

# 进行相机内参标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# 打印相机内参和畸变系数
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)
np.savetxt('camera_matrix.txt', camera_matrix)
np.savetxt('dist_coeffs.txt', dist_coeffs)
