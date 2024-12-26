import cv2
import numpy as np
import glob

# 设置棋盘格的大小：棋盘格的内角点数 (列数-1, 行数-1)
chessboard_size = (8, 11) 

# 设置棋盘格的单个方格的大小（单位：mm）
square_size = 60  # 假设每个方格的大小是25mm，实际情况根据你的棋盘格修改

# 准备棋盘格的三维世界坐标 (0,0,0), (1,0,0), (2,0,0)... 与实际物理尺寸相对应
object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
object_points *= square_size

# 存储每张图像的棋盘格角点
object_points_list = []  # 3D点在世界坐标系中的位置
image_points_list = []  # 2D点在图像平面上的位置

# 获取路径下所有的棋盘格图像
images = glob.glob('/home/office2004/cam_calib/241224front/*.png')

# 如果没有图像，报错退出
if len(images) == 0:
    print("未找到任何棋盘格图像，检查路径是否正确。")
    exit()

# 遍历每张图像，进行角点检测
for image_path in images:
    # 读取图像
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"无法读取图像: {image_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 如果找到了角点，将3D世界坐标和2D图像坐标分别存储
        object_points_list.append(object_points)
        image_points_list.append(corners)

        # 在图像上显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)  # 显示500毫秒

cv2.destroyAllWindows()

# 获取第一张图像的尺寸，用于标定
if len(image_points_list) > 0:
    height, width = gray.shape[:2]
else:
    print("没有有效的角点被检测到，无法进行标定。")
    exit()

# 使用所有角点进行相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points_list, image_points_list, (width, height), None, None
)

# 输出标定结果
print("标定结果：")
print("相机矩阵 (camera_matrix):\n", camera_matrix)
print("畸变系数 (dist_coeffs):\n", dist_coeffs)

# 保存标定结果到文件
np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# 如果需要，还可以计算重投影误差
mean_error = 0
for i in range(len(object_points_list)):
    image_points2, _ = cv2.projectPoints(object_points_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(image_points_list[i], image_points2, cv2.NORM_L2)
    mean_error += error**2

mean_error = np.sqrt(mean_error / len(object_points_list))
print(f"重投影误差: {mean_error}")
