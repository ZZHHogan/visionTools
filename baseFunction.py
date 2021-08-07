import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import math


# ===============================*****====================================
def cropImg():
    """
    截取图片
    顺序：高、宽、通道
    """
    img = cv2.imread("01.jpg")
    region = img[50:500, 50:100, :]  # h w c
    cv2.imshow("img", region)
    cv2.waitKey()
cropImg()
# ===============================*****====================================
def cv_draw_rect_put_text(img_path, x1y1_x2y3, text):
    """
    对opencv读取的图片进行画框、写文字，
    rectangle输入点的坐标如：((50, 50), (200, 200))，即左上角与右下角位置
    putText输入点的坐标如：(50, 50)，即左上角位置，但输入中文会乱码
    注意颜色顺序为BGR
    :return:
    """
    img = cv2.imread(img_path)
    cv2.rectangle(img,
                  (x1y1_x2y3[0][0], x1y1_x2y3[0][1]),
                  (x1y1_x2y3[1][0], x1y1_x2y3[1][1]),
                  (0, 255, 0),
                  4)
    cv2.putText(img, text, (x1y1_x2y3[0][0], x1y1_x2y3[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
    cv2.putText(img, "你好！！！", (x1y1_x2y3[0][0] + 50, x1y1_x2y3[0][1] + 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv_draw_rect_put_text("01.jpg", ((50, 50), (200, 200)),"hello-world!!!")
# ===============================*****====================================
def pil_draw_text(img_path, points, text):
    """
    对PIL读取的图片进行写文字，
    需要设置字体的大小、字体、坐标（左上角起始点）
    输入中文不会乱码
    注意颜色顺序为RGB
    :return:
    """
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
    draw.text((points[0], points[1]), text, (255, 0, 255), font=font)
    draw.text((points[0] + 100, points[1] + 100), "你好！！！", (255, 0, 255), font=font)
    img.show()
pil_draw_text("01.jpg", (50, 50),"hello-world!!!")
# ===============================*****====================================
def createBackground():
    """
    与numpy配合生成背景
    :return:
    """
    imgzero = np.zeros(shape=(300, 500, 3), dtype=np.uint8)  # h w c
    imgzero[:, :] = (0, 0, 255)  # 更改颜色
    cv2.imshow("imgzero", imgzero)
    cv2.waitKey()
createBackground()
# ===============================*****====================================
def cv_stack_img(img_path):
    """
    拼接图片
    """
    img = cv2.imread(img_path)
    I = np.zeros(img.shape, dtype=np.uint8)
    img1 = cv2.hconcat([img, I])  # 水平拼接
    img2 = cv2.vconcat([img, I])  # 垂直拼接
    cv2.imshow('img_h', img1)
    cv2.imshow('img_v', img2)
    cv2.waitKey(0)
cv_stack_img("111.jpg")
# ===============================*****====================================
def pil_stack_img(img_path):
    """
    拼接图片
    """
    img = Image.open(img_path)
    I = Image.new('RGB', img.size, (255, 255, 255))
    size1, size2 = img.size, I.size
    joint = Image.new('RGB', (size1[0] + size2[0], size1[1]), (255, 255, 255))
    loc1, loc2 = (0, 0), (size1[0], 0)
    joint.paste(img, loc1)
    joint.paste(I, loc2)
    plt.imshow(joint)
    plt.axis('on')
    plt.title('image')
    plt.show()
pil_stack_img("111.jpg")
# ===============================*****====================================
def capture_video(video_path):
    """
    打开摄像头，video_path可以是视频路径，也可以是摄像头端口号，默认为0
    """
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 镜像操作
        cv2.imshow("video", frame)
        key = cv2.waitKey(50)
        if key == ord('q'):  # 判断是哪一个键按下，退出
            break
    cv2.destroyAllWindows()
capture_video(0)
# ===============================*****====================================
def img2video(imgs_path, fps):
    """
    :param imgs_path: 图片文件夹路径
    :param fps: 帧数
    :return:
    """
    imgs_list = os.listdir(imgs_path)
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #  用于avi格式的生成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv2.VideoWriter("a.mp4", fourcc, fps, (512, 512))     # resize img shape
    for img_path in imgs_list:
        # print(img_path)
        img = cv2.imdecode(np.fromfile(imgs_path + img_path, dtype=np.uint8), 1)  # 中文路径
        img = cv2.resize(img, (512, 512))
        # size = img.shape[:2]
        videowriter.write(img)
    videowriter.release()
img2video("tmp/", 25)




