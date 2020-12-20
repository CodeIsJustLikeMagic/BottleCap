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

def h_plot(arr, description, start = 0):
    plot = False
    if plot:
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
def initialRegionsOfInterest(emptyframe, fullframe, filename):
    kernel = np.ones((3, 3), np.uint8)
    h_show('fullframe'+filename, fullframe)
    difference = cv2.subtract(emptyframe, fullframe)
    difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference = cv2.GaussianBlur(difference, (5,5), 2)
    difference = (255-difference)
    #difference = difference * 5
    ret, thresh = cv2.threshold(difference, 0, 255,  cv2.THRESH_OTSU)
    # black roi on white
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  #erosion followed by dilation
    differentregions = cv2.erode(opening, kernel, iterations=2) # grow bottle cap regions
    differentregions = 255 - differentregions
    #h_show('differenceMask', differentregions)
    imshow_components(differentregions, filename,fullframe)
    #watershedthings(fullframe,differentregions, filename)
    differenceMask = np.ones(fullframe.shape[:2], dtype="uint8")
    differenceMask[:, :] = (differentregions != 0)  # 0 or 1 depending on wehter it is ==0




    #regularasstemplatematching(cleanededges, fullframe, filename)
    #regularasstemplatematching(fullframe, "img" + str(i))

    cleanedfullframe = cv2.cvtColor(fullframe, cv2.COLOR_BGR2GRAY)
    #h_show('startcleanedframe' + filename, cleanedfullframe)
    cleanedfullframe = (1 - differenceMask) * 255 + differenceMask * cleanedfullframe
    #h_show('cleanedframe'+filename, cleanedfullframe)

def imshow_components(img,filename,originalimage, saveCrops = False):
    num_labels, labels = cv2.connectedComponents(img)
    # Map component labels to hue val
    occurenceOfLabels = np.bincount(labels.flatten())
    occurenceOfLabels = occurenceOfLabels[1::]

    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    h_show('labeled'+filename, labeled_img)
    #labels = labels.flatten()
    labelsbackwards = np.flip(labels)
    w, h = img.shape[::-1]
    print(w,h)
    #label 0 is background (black)
    for occ, lblind in zip(occurenceOfLabels,range(1,num_labels)):
        if occ > 300:
            # get min x,y and max x,y position of l
            x1 = np.argmax(labels == lblind, axis = 1)# along x axis
            x1 = minwithout0(x1)

            x2 = np.argmax(labelsbackwards == lblind, axis = 1)
            x2 = minwithout0(x2)
            x2 = w-x2

            y1 = np.argmax(labels == lblind, axis = 0)# along x axis
            y1 = minwithout0(y1)

            y2 = np.argmax(labelsbackwards == lblind, axis = 0)# along x axis
            y2 = minwithout0(y2)
            y2 = h - y2

            #print(lblind, 'x', x1, x2, 'y', y1, y2)
            crop_img = originalimage[y1:y2, x1:x2].copy()
            if saveCrops:
                cv2.imwrite("Crops/"+str(lblind)+"_"+filename[:-4]+'.png', crop_img)
            objclass, prob= predictionModel.prediction(crop_img)
            cv2.rectangle(originalimage, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (x1, y1)
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (0, 0, 255)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            originalimage = cv2.putText(originalimage, objclass+str(prob), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            #h_show("testcropimg"+ str(lblind)+ filename, crop_img)
        # crop out rectangular region (of orig image) around label.
    h_show("detected"+filename, originalimage)
    #0,0 is upper right corner


def minwithout0(ar):
    ar = ar[ar != 0]
    if len(ar) == 0:
        return 0
    else:
        return np.amin(ar)


def watershedthings(coloredimage, roiImage, filename):
    coloredimage = cv2.cvtColor(roiImage, cv2.COLOR_GRAY2RGB)
    #roiImage = cv2.cvtColor(roiImage, cv2.COLOR_BGR2GRAY)
    h_show("roiImage"+filename, roiImage)
    ret, markers = cv2.connectedComponents(roiImage)
    # print('markers',np.unique(markers))
    print(np.unique(markers))
    markers = markers + 1
    labels = cv2.watershed(coloredimage, markers)
    print(labels)
    for label in np.unique(labels):
        mask = np.zeros(roiImage.shape, dtype="uint8")
        mask[labels == label] = 1
        image = roiImage.copy()
        image = mask * 255 + (1-mask) * 0
        h_show("single Region "+str(label)+" "+filename, image)
#endregion

def fuckaroundwithedges(fullframe, filename, differenceMask,differentregions,kernel):
    #what is, just hear me out here, what if we mask the edges?
    edges = cv2.Canny(fullframe, 30, 200, kernel)
    cleanededges = (1-differenceMask) * 0 + differenceMask * edges
    h_show('cleanededges' + filename, cleanededges)
    #cleanededges  = cv2.dilate(cleanededges, kernel, iterations=2)
    #h_show('dilated edges'+filename, cleanededges)

    fullmask = differentregions + cleanededges
    h_show('differenceMask'+filename, differentregions)
    h_show('fullmask'+filename, fullmask)

def testRegionOfInterest(low, high = None):
    if high is None:
        high = low
    for i in range(low, high+1):
        print(i)
        fullframe = cv2.imread('Results/CV20_video_'+str(i)+'.png')
        emptyframe = cv2.imread('Results/CV20_video_'+str(i)+'_empty.png')
        if fullframe is not None and emptyframe is not None:
            initialRegionsOfInterest(emptyframe, fullframe, 'Video_' + str(i)+'.png')
            #regularasstemplatematching(fullframe, filename= 'Video'+str(i)+'.png')



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
    print("end. time for", file, datetime.now()-now)


maxdiff = 50


def regularasstemplatematching(matchImage, projektionimage = np.array([]), filename ="", templatefilepath = 'faceDown.png'):
    if len(projektionimage) == 0:
        projektionimage = matchImage
    img_gray = cv2.cvtColor(matchImage, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(templatefilepath, 0)
    h_show("matchImage", matchImage)
    h_show("template",template)
    w, h = template.shape[::-1]
    res= cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    h_show('template matching '+filename, res)
    threshold = 0.5
    print(np.unique(res))
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(projektionimage, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    h_show('res ' + filename, projektionimage)

import tensorflow as tf

class PredictionModel:
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    def __init__(self):
        print('setting up TensorFlowModel')

        img_data, class_name = self.create_dataset(r'TensorImages')
        target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
        print(target_dict)
        target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
        print(len(target_val))
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4)
            ])
        print(model.summary())
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x=tf.cast(np.array(img_data), tf.float64), y=tf.cast(list(map(int, target_val)), tf.int32),
                            epochs=5)
        print(history)
        print('finished TensorFlow Model')
        self.model = model
        self.class_name = class_name

    def create_dataset(self, img_folder):
        img_data_array = []
        class_name = []

        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
        return img_data_array, class_name

    def prediction(self, image):
        image = cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image)
        print(predictions)
        m = np.argmax(predictions[0])
        return np.unique(self.class_name)[m], predictions[0][m]
        #use target dict instead {'FaceDowns': 0, 'FaceUps': 1}

predictionModel = PredictionModel()

if __name__ == "__main__":

    now = datetime.now()

    #network()
    #readVideo(vid)
    #showVideo(vid)
    #testAllVideos()
    #testVideo("Videos/CV20_video_3.mp4", "CV20_video_3.mp4")
    testRegionOfInterest(1, 3)
    print("total time", datetime.now()-now)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

