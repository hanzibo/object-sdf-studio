"""RGB conversion and per-frame depth scaling for Orbbec SDK v2."""
import cv2
import numpy as np
from pyorbbecsdk import OBFormat

def color_frame_to_bgr(frame):
    """将 Orbbec 彩色帧转换为 OpenCV 使用的 BGR ``uint8`` 图像。

    相机可能根据 profile 输出 MJPG、RGB、YUYV 等不同格式。OpenCV 的
    ``imshow``/``imwrite`` 默认使用 BGR，因此应先按帧格式转换。
    """
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    # 无压缩 RGB/BGR：每个像素占 3 字节。
    if color_format == OBFormat.RGB:
        image = data.reshape(height, width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return data.reshape(height, width, 3).copy()
    # Gemini 336L 默认彩色 profile 是 MJPG，需要先进行 JPEG 解码。
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    # YUYV/UYVY 是打包的 YUV 4:2:2，每个像素平均占 2 字节。
    if color_format == OBFormat.YUYV:
        image = data.reshape(height, width, 2)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        image = data.reshape(height, width, 2)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    # I420/NV12/NV21 是 YUV 4:2:0，高度按 1.5 倍组织在缓冲区内。
    if color_format == OBFormat.I420:
        image = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    if color_format == OBFormat.NV12:
        image = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        image = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)

    raise ValueError(f"Unsupported color format: {color_format}")

def depth_frame_to_mm(frame) -> np.ndarray:
    """将 Y16 深度帧转换成以毫米为单位的二维 ``float32`` 数组。

    原始 ``uint16`` 数值必须乘以当前帧的 ``depth_scale``。数值 0 表示
    该像素没有有效深度，不能假设比例永远等于 1。
    """
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
        frame.get_height(), frame.get_width()
    )
    return raw.astype(np.float32) * frame.get_depth_scale()
