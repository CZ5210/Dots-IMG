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


def generate_cloud_method1(image, point_density, y_range):
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

def generate_cloud_method2(image, point_density, y_range, rev = False):
    """生成XZ平面上的点云, Y轴按照颜色信息分配"""
    height, width = image.shape[:2]
    points = []
    colors = []
    # 转换为HSV颜色空间，便于根据饱和度等特征采样
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for x in range(width):
        for z in range(height):
            # 提取HSV值
            h, s, v = hsv_image[z, x]
            
            # 基于饱和度调整采样概率，饱和度高的像素更可能被采样
            adjusted_density = point_density * (s / 255.0)
            if random.random() < adjusted_density:
                color = image[z, x] / 255.0
                # 可以尝试更复杂的Y轴映射，比如结合颜色的多通道信息
                # 这里简单示例，也可根据实际需求调整
                y = (color[0] + color[1] + color[2]) / 3 * y_range
                if rev:
                    y = -y
                points.append([x, y, -z])
                colors.append(color)

    return np.array(points), np.array(colors)


def plot_cloud(points, colors, size=10, alpha=0.8):
    """给定点云数据, 绘制3D图像"""

    # 建立3D图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=size, alpha=alpha)
    ax.axis('off')
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


    return

# 测试代码
if __name__ == "__main__":
    # 读取原始图像
    image_path = "0.png" 
    original_img = load_image(image_path)

    # SVD图像简化
    simplified_img = svd_image_simplification(original_img, k=50)

    # 生成云图
    points, colors = generate_cloud_method2(
        simplified_img,
        point_density=0.005,
        y_range=500,
        # rev=True
    )
    plot_cloud(points,colors,size=100)
