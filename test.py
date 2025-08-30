import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_image(image_path):
    """读取图片并转换为RGB格式"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def svd_image_simplification(image, k=50):
    """使用SVD对图像进行简化"""
    simplified = np.zeros_like(image, dtype=np.float32)

    for channel in range(3):
        img_channel = image[:, :, channel].astype(np.float32)
        U, S, Vt = np.linalg.svd(img_channel, full_matrices=False)

        k = min(k, len(S))
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]

        simplified_channel = U_k @ S_k @ Vt_k
        simplified_channel = (simplified_channel - simplified_channel.min()) / (
                    simplified_channel.max() - simplified_channel.min()) * 255
        simplified[:, :, channel] = simplified_channel

    return simplified.astype(np.uint8)


def generate_xz_plane_cloud(image, point_density, y_range):
    """生成XZ平面上的点云, Y轴随机分配"""
    height, width = image.shape[:2]
    points = []
    colors = []

    for x in range(width):
        for z in range(height):

            if random.random() < point_density:
                color = image[z, x] / 255.0
                y = random.uniform(-y_range, y_range)
                points.append([x, y, -z])
                colors.append(color)

    return np.array(points), np.array(colors)

def plot_cloud(points, colors):
    """给定点云数据, 绘制3D图像"""

    # 建立3D图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=10.0, alpha=0.8)
    ax.axis('off')
    ax.axis('equal')
    plt.show()

    return

# 测试代码
if __name__ == "__main__":
    # 读取原始图像
    image_path = "input_image.jpg"
    original_img = load_image(image_path)

    # SVD图像简化
    simplified_img = svd_image_simplification(original_img, k=30)

    # 生成云图
    points, colors = generate_xz_plane_cloud(
        simplified_img,
        point_density=0.05,
        y_range=50
    )
    plot_cloud(points,colors)
