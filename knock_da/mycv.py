from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2


BLACK   = np.array([  0,   0,   0])
GREY    = np.array([127, 127, 127])
WHITE   = np.array([255, 255, 255])

RED     = np.array([  0,   0, 255])
GREEN   = np.array([  0, 255,   0])
BLUE    = np.array([255,   0,   0])

CYAN    = np.array([255, 255,   0])
MAGENTA = np.array([255,   0, 255])
YELLOW  = np.array([  0, 255, 255])


def imread(path: Path, verbose: bool = False):
    img = cv2.imread(str(path))
    if verbose:
        height, width = img.shape[:2]
        print(f'height: {height}, width: {width}')
    return img


def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis("off")
    plt.show()


def cap_open(mov, verbose=False):
    cap = cv2.VideoCapture(str(mov))

    if not cap.isOpened():
        frm = -1
        wid = -1
        hei = -1
        fps = -1
    else:
        frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if verbose:
            print(wid, 'x', hei, 'pixels,', round(fps, 3), 'fps,', frm, 'frames,', round(frm / fps, 3), 's,', time_str(frm / fps)[0])

    return cap, wid, hei, fps, frm


def time_str(value,
               units = (
                   (' d ', 60 * 60 * 24),
                   (' h ', 60 * 60),
                   (' m ', 60),
                   (' s ', 1),
                   (' ms', 1/1000),
               )):
    result = [0] * len(units)
    result_str = ''
    for i, u in enumerate(units):
        if u[1] < value:
            result[i] = int(value / u[1])
            value %= u[1]
            result_str += str(result[i]) + u[0]

    return result_str, result


def wri_open(path, wid, hei, fps=30, codec='XVID'):
    f = open(path, 'w') ; f.close() # 試しに新規作成してみる (ディレクトリが書き込み禁止なら例外が発生する)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(path, fourcc, fps, (int(wid), int(hei)))


def cv2_line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8):
    pt1 = np.round(pt1).astype('i')
    pt2 = np.round(pt2).astype('i')
    cv2.line(img, tuple(pt1), tuple(pt2), color, thickness, lineType)

def cv2_piutText(img, text, org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=BLACK, thickness=1, lineType=cv2.LINE_AA):
    org = np.round(org).astype('i')
    thickness = np.round(thickness)
    cv2.putText(img, str(text), tuple(org), fontFace, fontScale, color, int(thickness), lineType)

def cv2_getPerspectiveTransform(src, dst):
    src = np.array(src).astype('f4')
    dst = np.array(dst).astype('f4')
    return cv2.getPerspectiveTransform(src, dst)

def cv2_floodFill(img, seedPoint, newVal, loDiff=None, upDiff=None, flags=4):
    h, w = img.shape[: 2]
    mask = np.zeros((h + 2, w + 2), dtype='u1')
    return cv2.floodFill(img, mask, seedPoint=seedPoint, newVal=newVal, loDiff=loDiff, upDiff=upDiff, flags=flags)


def trim_and_resize(img, size):
    h0, w0 = img.shape[: 2]
    w1, h1 = size
    r0 = h0 / w0
    r1 = h1 / w1

    if r0 < r1:
        new_w0 = h0 / r1
        left_trim = round((w0 - new_w0) / 2)
        img = img[:, left_trim : round(new_w0) + left_trim]
    elif r1 < r0:
        new_h0 = w0 * r1
        upper_trim = round((h0 - new_h0) / 2)
        img = img[upper_trim : round(new_h0) + upper_trim, :]
        print(img.shape)

    return cv2.resize(img, (w1, h1))

def rotate_img(img, angle, centre=None, out_size=None, scale=1., borderMode=None):
    hei, wid = img.shape[: 2]
    if centre is None:
        centre = np.round(np.array([wid, hei]) / 2).astype('f')
    if out_size is None:
        out_size = (wid, hei)
    mat = cv2.getRotationMatrix2D(tuple(centre), angle , scale)
    return cv2.warpAffine(img, mat, out_size, borderMode=borderMode)


def bgr2gry(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def gry2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def rgb2gry(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def gry2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
def hsv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def bgr2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
def yuv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)