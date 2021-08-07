import cv2
from PIL import Image
import numpy as np

# ===============================*****====================================
def BGR2GRAY():
    """
    opencv默认以BGR通道读取
    :return:
    """
    img_path = "02.jpg"
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('BGR', img)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
BGR2GRAY()
# ===============================*****====================================
def BGR2RGB():
    """
    opencv默认以BGR通道读取
    :return:
    """
    img_path = "02.jpg"
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('BGR', img)
    cv2.imshow('rgb', rgb)
    cv2.waitKey(0)
BGR2RGB()
# ===============================*****====================================
def BGR2HSV():
    """
    opencv默认以BGR通道读取
    :return:
    """
    img_path = "02.jpg"
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('BGR', img)
    cv2.imshow('hsv', hsv)
    cv2.waitKey(0)
BGR2HSV()
# ===============================*****====================================
def BGR2YUV():
    """
    opencv默认以BGR通道读取
    :return:
    """
    img_path = "02.jpg"
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    cv2.imshow('BGR', img)
    cv2.imshow('yuv', hsv)
    cv2.waitKey(0)
BGR2YUV()
# ===============================*****====================================
def CV2PIL():
    """
    opencv默认以BGR通道读取
    PIL则为RGB通道顺序
    :return:
    """
    img_path = "02.jpg"
    cv_img = cv2.imread(img_path)
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    cv2.imshow('BGR', cv_img)
    cv2.waitKey(0)
    pil_img.show()
CV2PIL()
# ===============================*****====================================
def PIL2CV():
    """
    opencv默认以BGR通道读取
    PIL则为RGB通道顺序
    :return:
    """
    img_path = "02.jpg"
    pil_img = Image.open(img_path)
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('BGR', cv_img)
    cv2.waitKey(0)
    pil_img.show()
PIL2CV()


