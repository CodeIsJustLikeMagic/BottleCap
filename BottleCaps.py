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

    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prevframe = vid.read()
    while vid.isOpened():
        ret, currentframe = vid.read()
        if not ret:
            break
        dif = diffinframes(prevframe, currentframe)
        diffs = np.append(diffs, dif)
        prevframe = currentframe
    h_plot(diffs, "mean square diffs", 0)
    findszene(diffs)
def diffofdiffs(diffarr):
    #similarity to previous error value
    diffofdiffs = np.array([])

    for i in range(len(diffarr) - 1):
        diffofdiffs = np.append(diffofdiffs, np.abs(diffarr[i]-diffarr[i+1]))

    print(diffofdiffs.min())

    h_plot(diffofdiffs, "diffofdiffs")

def findszene(diffarr):
    minaountofframes = 5
    maxdiff = 20

    # I want seqzenzes of this were the diff is small, how many frames? how small is it?
    lines= []
    linestarts = []
    currentline = np.array([])
    currentlinestart = 0
    for index, d in enumerate(diffarr, start=1):
        if d < maxdiff:
            currentline = np.append(currentline, d)
        else:
            #break line
            if(len(currentline) > minaountofframes):#cut the line here
                lines.append(currentline)
                #lines.append(lines, currentline)
                currentline = np.array([])
                linestarts.append(currentlinestart)
            currentlinestart = index +1
    if(len(currentline) > minaountofframes):
        lines.append(currentline)
        linestarts.append(currentlinestart)
    print(len(lines), len(linestarts))
    for line,linest in zip(lines, linestarts):
        h_plot(line, "line "+str(linest), linest)
        #current diff,  currentline, start index, mean diff in line, length
    #there should be one szene right at the start, one in the middle and one at the end
    print("found", len(lines), "sequenzes")


def h_plot(arr, desciption):
    h_plot(arr, desciption, 0)

def h_plot(arr, description, start):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(start, len(arr) + start)
    ax.scatter(x, arr, label='diffs between frames')
    ax.set_ylabel(description)
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
        ret, frame = vid.read()
        if not ret:
            break

from datetime import datetime


if __name__ == "__main__":

    now = datetime.now()
    vid = cv2.VideoCapture('Videos/CV20_video_50.mp4') #50 is black
    if not vid.isOpened():
        print("Error opening video stream or file")

    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print("videodata: fps: ", fps, "framecount: ", frame_count)

    #readVideo(vid)
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

