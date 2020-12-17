import datetime
import shutil

#numpy, ceras, tensorflow,tensorflow, patplotlip, opencv, numpy
import cv2
import numpy as np
from matplotlib import pyplot as plt
import statistics
#region help methods
def h_show(title, image):
    resize = True
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))
    if resize:
        image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))
    cv2.imshow(title, image)

def h_plot(arr, desciption):
    h_plot(arr, desciption, 0)

def h_plot(arr, description, start):
    return
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(start, len(arr) + start)
    ax.scatter(x, arr, marker=',', label='diffs between frames', s=4)
    ax.set_ylabel(description)
    global maxdiff
    ax.axhline(y=maxdiff, color = "black");
    plt.show()
#return a quantified differnece between two frames
#endregion
#region find frames
def diffsbetweenallframes(vid):

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
    return diffs

def diffofdiffs(diffarr):
    #similarity to previous error value
    diffofdiffs = np.array([])

    for i in range(len(diffarr) - 1):
        diffofdiffs = np.append(diffofdiffs, np.abs(diffarr[i]-diffarr[i+1]))

    print(diffofdiffs.min())

    h_plot(diffofdiffs, "diffofdiffs")

def findsequences(diffarr):
    minaountofframes = 10
    global maxdiff

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
    #print(len(lines), len(linestarts))
    for line,linest in zip(lines, linestarts):
        h_plot(line, "line "+str(linest), linest)
        #current diff,  currentline, start index, mean diff in line, length
    #print("found", len(lines), "sequenzes")
    return linestarts, lines

    
def getframeindeces(linestarts, lines):
    #there should be one szene right at the start, one in the middle and one at the end
    if 0 in linestarts:
        linestarts.pop(0)
        lines.pop(0)
        # the sequzence at the start is worse than the main sequenze.
        # now there should be one or two good seqzences.
        return 0, findbestlines(linestarts, lines)
    else:
        print("AHHHHH there is no seqzence at the start, pls help")
        return 0, findbestlines(linestarts, lines)


    # remove bas sequenzs until we only have two left.
    # one should be in the middle or at the end if there are only two good sequences

def findbestlines(linestarts, lines):
    nlines = []
    for line, linest in zip(lines, linestarts):
        #print(linest, np.std(line), np.mean(line))
        frameindeces = np.array(range(linest, linest+len(line)))
        line = np.vstack([line, frameindeces])
        nlines.append(line)
    #print(nlines)
    nlines.sort(key=sortbymean)
    #so now the first or the second one is our sequence
    besttwo = nlines[:2]
    #if the second line mean is muuch worse than the first one there is probably one one static szene
    besttwo.sort(key=smallerstart)
    #print(besttwo)
    #choose the frame with the smallest score from the first one
    bestone = besttwo[0]
    #print('startframe of bestone', bestone[1][0])
    bestone = zip(bestone[0], bestone[1])
    m = min(bestone, key=lambda t: t[0])
    return m[1]

def smallerstart(elem):
    return elem[1][0]

def sortbymean(elem):
    return np.mean(elem[0])

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err  # return the MSE, the lower the error, the more "similar" the two images are

def diffinframes(frame1, frame2):# is really slow
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
import os



def readframe(vid, frame_number):
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))  # optional
    success, frame = vid.read()
    if success:
        return frame
    return False
#endregion

#region regions of interest
def diffbetweenFrames(emptyframe, fullframe,filename):
    difference = cv2.subtract(emptyframe, fullframe)
    difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference = cv2.GaussianBlur(difference, (5,5), 0)
    #h_show('dif gray_'+filename, difference)
    ret, thresh = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # black roi on white
    kernel = np.ones((3, 3), np.uint8)
    #h_show('thresh_' + filename, thresh)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    #erosion followed by dilation
    #h_show('opening' + filename, opening)
    dilated = cv2.dilate(opening, kernel, iterations=1) # grow background. get rid of artifacts
    h_show('dilated_' + filename, dilated)
    #ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #difference[mask != 255] = [255, 255, 255]

    #h_show('emptyframe_'+filename, emptyframe)
    #h_show('fullframe_'+filename, fullframe)

#endregion

def testRegionOfInterest(low, high):
    for i in range(low, high+1):
        print(i)
        fullframe = cv2.imread('Results/CV20_video_'+str(i)+'.png')
        emptyframe = cv2.imread('Results/CV20_video_'+str(i)+'_empty.png')
        diffbetweenFrames(emptyframe, fullframe, 'Video_'+str(i))


def testAllVideos():
    #estimate every image in the folder
    folder = 'Videos'
    files = os.listdir(folder)
    paths = [folder+'/' + file for file in files]
    for path, file in zip(paths, files):
        if file.endswith('.mp4'):
            testVideo(path, file)

def testVideo(path,file):
    print('start', path, '...')
    now = datetime.now()
    vid = cv2.VideoCapture(path)  # 50 is black
    if not vid.isOpened():
        print("Error opening video stream or file")

    fps = vid.get(cv2.CAP_PROP_FPS)
    #frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    #print("videodata: fps: ", fps, "framecount: ", frame_count)
    diffs = diffsbetweenallframes(vid)
    linestarts, lines = findsequences(diffs)
    if len(lines) < 1:
        print('to much action, threshhold excluded  everything')
        return
    emptyscene, fullscene = getframeindeces(linestarts, lines)
    emptyframe = readframe(vid, emptyscene)
    fullframe = readframe(vid, fullscene)
    print('choose frame: ', fullscene, ' min ', fullscene/fps)
    cv2.imwrite('Results/'+file[:-4]+'.png', fullframe)
    cv2.imwrite('Results/' + file[:-4] + '_empty.png', emptyframe)
    vid.release()
    #difbetweenFrames(emptyframe, fullframe)
    print("end. time for", file, datetime.now()-now)


maxdiff = 50


if __name__ == "__main__":

    now = datetime.now()


    #readVideo(vid)
    #showVideo(vid)
    #testAllVideos()
    #testVideo("Videos/CV20_video_3.mp4", "CV20_video_3.mp4")
    testRegionOfInterest(1, 10)
    print("total time", datetime.now()-now)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

