import datetime
import shutil

#numpy, ceras, tensorflow,tensorflow, patplotlip, opencv, numpy
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
def h_show(title, image):
    resize = True
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))
    if resize:
        image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))
    #cv2.imshow(title, image)

#return a quantified differnece between two frames

def plotdiffs(vid):
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    diffs = np.array([])
    for i in range(1, int(frame_count)-1):
        dif = diffinframes(readframe(i), readframe(i+1))
        print(i)
        diffs = np.append(diffs,dif)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.arange(1, int(frame_count)-1),diffs, label='diffs between frames')
    plt.show()


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def diffinframes(frame1, frame2):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2= cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    quanti = mse(frame1,frame2)
    #dif = frame1 - frame2
    #sumdif = dif.sum()
    #res = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF)
    #I wanted to use templatemaching but atm I use simple dif
    return quanti



def readVideo(vid):
    while vid.isOpened():
        now = datetime.now()
        ret, frame = vid.read()
        print(datetime.now() -now)
        if not ret:
            break

def readframe(frame_number):
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number)-1)  # optional
    success, frame = vid.read()
    if success:
        return frame
    return False

from datetime import datetime


if __name__ == "__main__":
    vidarr = []
    now = datetime.now()
    vid = cv2.VideoCapture('Videos/CV20_video_1.mp4')
    if not vid.isOpened():
        print("Error opening video stream or file")

    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print("videodata: fps: ", fps, "framecount: ", frame_count)

    #showVideo(vid)
    plotdiffs(vid)
    #vid.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_number = 1)
    #ret, frame1 = vid.read()
    #vid.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_number = 2)
    #ret, frame2 = vid.read()
    #diffinframes(frame1, frame2)
    print("total time", datetime.now()-now)
    cv2.waitKey(0)
    vid.release()
    cv2.destroyAllWindows()

