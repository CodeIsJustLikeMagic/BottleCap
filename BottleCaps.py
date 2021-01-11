#numpy, ceras, tensorflow,tensorflow, patplotlip, opencv, numpy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os

#region help methods
def h_show(title, image,show = False):
    resize = True
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))
    if resize:
        image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))
    if show:
        cv2.imshow(title, image)

def h_plot(arr, description, filename = '', start = 0, save = False):
    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(start, len(arr) + start)
        m = arr.mean()
        ax.scatter(x, arr, marker=',', label='diffs between frames', s=4)
        ax.set_ylabel(description)
        global maxdiff
        ax.axhline(y=maxdiff, color="black")
        ax.axhline(y=m, color="red")
        if save:
            fig.savefig('Results/'+filename[:-4]+'_'+description+'.png')
        plt.show()

def readVideo(vid): # show video
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

#endregion
#region find frames
def diffsbetweenallframes(vid, filename, savePlot =False):

    diffs = np.array([])

    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prevframe = vid.read()
    while vid.isOpened():
        ret, currentframe = vid.read()
        if not ret:
            break
        dif = diffbetweenframes(prevframe, currentframe)
        diffs = np.append(diffs, dif)
        prevframe = currentframe
    h_plot(diffs, "mean square diffs", start = 0, filename = filename )
    return diffs

def diffofdiffs(diffarr):
    #similarity to previous error value
    diffofdiffs = np.array([])

    for i in range(len(diffarr) - 1):
        diffofdiffs = np.append(diffofdiffs, np.abs(diffarr[i]-diffarr[i+1]))

    print(diffofdiffs.min())

    h_plot(diffofdiffs, "diffofdiffs")

def findsequences(diffarr):
    maxdiff = diffarr.mean()
    linestarts, lines, _ = h_findsequences(diffarr, 30, maxdiff)
    while (len(lines)<1 or (len(lines) == 1 and linestarts[0] == 0)) and maxdiff > 0: # lower the threshhold
        maxdiff = maxdiff - diffarr.mean()/2
        linestarts, lines, _ = h_findsequences(diffarr, 30, maxdiff)
    while (len(lines)<1 or (len(lines) == 1 and linestarts[0] == 0)) and maxdiff < 10000: # raise the threshhold
        maxdiff = maxdiff + diffarr.mean()/2
        linestarts, lines, _ = h_findsequences(diffarr, 30, maxdiff)
    return linestarts,lines,maxdiff

def h_findsequences(diffarr, minaountofframes,  maxdiff):
    # I want seqzenzes of this were the diff is small, how many frames? how small is it?
    lines = []
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
            else:#clear what we already have
                currentline = np.array([])
            currentlinestart = index
    if(len(currentline) > minaountofframes):
        lines.append(currentline)
        linestarts.append(currentlinestart)
    #print(len(lines), len(linestarts))
    for line,linest in zip(lines, linestarts):
        h_plot(line, "line "+str(linest), start = linest)
        #current diff,  currentline, start index, mean diff in line, length
    #print("found", len(lines), "sequenzes")
    return linestarts, lines, maxdiff

    
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
    bestone = besttwo[0] # list index out of range
    #print('startframe of bestone', bestone[1][0])
    l = len(bestone[0])
    bestone = list(map(list, zip(bestone[0], bestone[1]))) #0 is all difference scores, 1 is the frame
    m = min(bestone, key=lambda t: t[0])
    ret = bestone[int(l/2)]
    return ret[1]

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

def diffbetweenframes(frame1, frame2):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2= cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    quanti = mse(frame1,frame2)
    #dif = frame1 - frame2
    #sumdif = dif.sum()
    #res = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF)
    #I wanted to use templatemaching but atm I use simple dif
    return quanti

def readframe(vid, frame_number):
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))  # optional
    success, frame = vid.read()
    if success:
        return frame
    return False
#endregion
#region regions of interest
# important function
def initialRegionsOfInterest(emptyframe, fullframe, filename):
    kernel = np.ones((3, 3), np.uint8)
    difference = cv2.subtract(emptyframe, fullframe)
    difference = cv2.add(difference, cv2.subtract( fullframe, emptyframe))
    difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference = cv2.GaussianBlur(difference, (5,5), 2)
    difference = (255-difference)
    ret, thresh = cv2.threshold(difference, 0, 255,  cv2.THRESH_OTSU) # black roi on white
    differentregions = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  #erosion followed by dilation
    differentregions = 255 - differentregions
    guessCropsWithTensor(differentregions, filename, fullframe)

# important function
def guessCropsWithTensor(img, filename, originalimage, saveCrops = False):
    num_labels, labels = cv2.connectedComponents(img)
    # Map component labels to hue val
    occurenceOfLabels = np.bincount(labels.flatten())
    occurenceOfLabels = occurenceOfLabels[1::]
    labelsbackwards = np.flip(labels)
    w, h = img.shape[::-1]
    #label 0 is background (black)
    shapes = []
    if predictionModel is not None:

        for occ, lblind in zip(occurenceOfLabels,range(1,num_labels)):
            if occ > 300:
                #calculate boundinBox for connectedcomponent
                # get min x,y and max x,y position of l
                x1 = np.argmax(labels == lblind, axis = 1)# along x axis
                x1 = minwithout0(x1)

                x2 = np.argmax(labelsbackwards == lblind, axis = 1)
                x2 = minwithout0(x2)
                x2 = w - x2

                y1 = np.argmax(labels == lblind, axis = 0)# along x axis
                y1 = minwithout0(y1)

                y2 = np.argmax(labelsbackwards == lblind, axis = 0)# along x axis
                y2 = minwithout0(y2)
                y2 = h - y2

                #print(lblind, 'x', x1, x2, 'y', y1, y2)
                crop_img = originalimage[y1:y2, x1:x2].copy()
                if saveCrops:
                    cv2.imwrite("Crops/"+str(lblind)+"_"+filename[:-4]+'.png', crop_img)

                # guess if boundingbox contains bottlecap
                objclass, prob= predictionModel.prediction(crop_img)
                cv2.rectangle(originalimage, (x1, y1), (x2, y2), (0, 0, 255), 2)

                #save resulting boundingBox in json file
                resultdict = {}
                resultdict["label"]=str(objclass)
                resultdict["points"]= [[str(x1),str(y1)],[str(x2),str(y2)]]
                shapes.append(resultdict)

                #save Image with Boxes
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # org
                org = (x1, y1-10)
                # fontScale
                fontScale = 0.7
                # Blue color in BGR
                color = (0, 0, 255)
                # Line thickness of 2 px
                thickness = 1
                # Using cv2.putText() method
                originalimage = cv2.putText(originalimage, objclass+" "+str(prob), org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                h_show("testcropimg"+ str(lblind)+ filename, crop_img, show= False)
                if True:
                    cv2.imwrite('Results/CV20_' + filename[:-4] + '_caps.png', originalimage)
            # crop out rectangular region (of orig image) around label.

    jsonDict = {}
    jsonDict["shapes"] = shapes
    st = 'Results/CV20_'+filename[:-4]+'.json'
    with open ('Results/CV20_'+filename[:-4]+'.json','w') as outfile:
        json.dump(jsonDict, outfile)
    showcoloredConnectedComponents(labels, filename)
    h_show("detected"+filename, originalimage)

def showcoloredConnectedComponents(labels,filename, save = True):
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    coloredROIS = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    coloredROIS = cv2.cvtColor(coloredROIS, cv2.COLOR_HSV2BGR)

    # set bg label to black
    coloredROIS[label_hue==0] = 0
    h_show('labeled'+filename, coloredROIS)
    if save:
        cv2.imwrite('Results/CV20_'+ filename[:-4] + '_components.png', coloredROIS)


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

def regularasstemplatematching(matchImage, projektionimage = np.array([]), filename ="", templatefilepath = 'faceDown.png'):
    if len(projektionimage) == 0:
        projektionimage = matchImage
    img_gray = cv2.cvtColor(matchImage, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(templatefilepath, 0)
    h_show("matchImage", matchImage)
    h_show("template", template)
    w, h = template.shape[::-1]
    res= cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    h_show('template matching '+filename, res)
    threshold = 0.5
    print(np.unique(res))
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(projektionimage, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    h_show('res ' + filename, projektionimage)
#endregion

def startFindCapsFromImages(low, high = None):
    if high is None:
        high = low
    for i in range(low, high+1):
        print('start working on Video: ',i)
        fullframe = cv2.imread('Results/CV20_video_'+str(i)+'.png')
        emptyframe = cv2.imread('Results/CV20_video_'+str(i)+'_empty.png')
        if fullframe is not None and emptyframe is not None:
            findCaps(emptyframe,fullframe, i)

def startFindCapsFromVideo(low, high = None, saveImages = True):
    if high is None:
        high = low
    for i in range(low,high+1):
        print('start working on Video: ', i)
        emptyframe, fullframe = findStaticFramesSingle(i, filename= 'CV20_video_'+str(i)+'.mp4')
        if saveImages:
            cv2.imwrite('Results/CV20_video_'+str(i)+'.png', fullframe)
            cv2.imwrite('Results/CV20_video_'+str(i)+'_empty.png', emptyframe)
        findCaps(emptyframe, fullframe, i)

def findCaps(emptyframe,fullframe, i):
    print('start Find Caps for Video', i, '...')
    now = datetime.now()
    if fullframe is not None and emptyframe is not None:
        initialRegionsOfInterest(emptyframe, fullframe, 'Video_' + str(i) + '.png')
        # regularasstemplatematching(fullframe, filename= 'Video'+str(i)+'.png')
    print('end Find Caps. Time for Video', i, datetime.now()-now)

def startFindStaticFrames(low, high = None, savePlot = False):
    if high is None:
        high = low
    for i in range(low, high +1):
        try:
            emptyframe, fullframe = findStaticFramesSingle(i, savePlot= savePlot)
            cv2.imwrite('Results/CV20_video_'+str(i) + '.png', fullframe)
            cv2.imwrite('Results/CV20_video_'+str(i)+ '_empty.png', emptyframe)
        except:
            print("caught an error with video file")

# important function
def findStaticFramesSingle(filenum, path = 'Videos/', filename = None, savePlot = False):
    if filename is None:
        filename = "CV20_video_"+str(filenum)+".mp4"
    path = path + filename
    print('start frameselection', path, '...')
    now = datetime.now()
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print("Error opening video stream or file")
        return

    fps = vid.get(cv2.CAP_PROP_FPS)
    #frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    #print("videodata: fps: ", fps, "framecount: ", frame_count)
    diffs = diffsbetweenallframes(vid, filename, savePlot = savePlot)
    print("mean", diffs.mean())
    linestarts, lines, maxdiff = findsequences(diffs)

    if len(lines) < 1:
        print('to much action, threshhold excluded  everything')
        return
    emptysceneindex, fullsceneindex = getframeindeces(linestarts, lines)
    illustratelines(diffs, linestarts, lines, filename, fullsceneindex, maxdiff)
    emptyframe = readframe(vid, emptysceneindex)
    fullframe = readframe(vid, fullsceneindex)
    #if diffbetweenframes(emptyframe, fullframe) > diffs.mean:
     #   print("")
    print('choosing frame:', fullsceneindex, '; min', fullsceneindex/fps)
    vid.release()
    print("end frameselection. time for", filename, datetime.now() - now)
    return emptyframe, fullframe
    # sometimes real sequence is to short and gets cut out

def illustratelines(diffs, linestats, lines, filename, choosen, maxdiff, save = True):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(0, len(diffs))
    m = diffs.mean()
    ax.scatter(x, diffs, marker=',', label='diffs between frames', s=4)
    ax.set_ylabel("cut into sequences")
    #ax.axhline(y=maxdiff, color="black")
    ax.axhline(y=maxdiff, color="red")
    ax.axvline(x = choosen)

    for line, start in zip(lines,linestats):
        x = np.arange(start, len(line)+start)
        ax.scatter(x, line, marker=',', s = 4)
    if save:
        fig.savefig('Results/' + filename[:-4] + '_diffsplot.png')
    plt.show()


# important function
import json

def evaluateResults(low, high= None):
    if high is None:
        high = low
    for i in range(low, high +1):
        evaluateResultsSinge(i)
def evaluateResultsSinge(videonum):
    success = True
    # visually compare the still frame of the video and the chosen still frame
    # evaluate how well the video frame was choosen
    intendedFrame = cv2.imread('Videos/CV20_image_'+str(videonum)+'.png')
    fullFrame = cv2.imread('Results/CV20_video_'+str(videonum)+'.png')
    h_show('intended frame' + str(videonum), intendedFrame, show = True)
    h_show('chosenFrame'+ str(videonum) ,fullFrame, show = True)
    dif = diffbetweenframes(intendedFrame, fullFrame)
    difference = cv2.subtract(intendedFrame, fullFrame)
    h_show('how well the frame was choosen '+str(videonum), difference, show= True)
    if dif > 50:
        success = False


    # read the json file of the video to see how well caps were detected
    with open ('Videos/CV20_label_renamed_'+str(videonum)+'.json') as f:
        with open('Results/CV20_Video_' + str(videonum) + '.json') as f2:
            realCaps = json.load(f)
            intendedShapes =  realCaps.get("shapes")
            foundcaps = json.load(f2)
            foundShapes = foundcaps.get("shapes")
            for rshape in intendedShapes:
                rlabel = rshape.get("label")
                rmidpoint = middleofpoints(rshape.get("points"))
                #compare to points of my estimation
                for fshape in foundShapes:

                    foundlabel = fshape.get("label")
                    fmidpoint = middleofpoints(fshape.get("points"))



    print('Evaluating Video', videonum, 'frame eval: ', dif)

def middleofpoints(points):
    if points is None:
        return [0,0]
    sumX = 0
    sumY = 0
    length = len(points)
    for point in points:
        sumX = sumX + int(point[0])
        sumY = sumY + int(point[1])
    midpointX = sumX/length
    midpointY = sumY/length
    return [midpointX,midpointY]


import tensorflow as tf
class PredictionModel:
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    def __init__(self):
        print('setting up TensorFlowModel')
        now = datetime.now()

        img_data, class_name = self.create_dataset(r'TensorImages')
        target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
        self.inv_target_dict = {value: key for (key, value) in target_dict.items()}
        print('target dict', target_dict)
        target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
        print(len(target_val))
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(len(target_dict), activation = 'softmax')
                #https://datascience.stackexchange.com/questions/69664/how-to-interpret-keras-predict-output
            ])
        print(model.summary())
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x=tf.cast(np.array(img_data), tf.float64), y=tf.cast(list(map(int, target_val)), tf.int32),
                            epochs=4)
        print(history)
        print('finished TensorFlow Model. Time', datetime.now()-now)
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
        #print(predictions)
        m = np.argmax(predictions[0])
        return self.inv_target_dict[m], predictions[0][m]

if True:
    predictionModel = PredictionModel()
else:
    predictionModel = None
maxdiff = 50# max diff for sequence of frames
plot = True
if __name__ == "__main__":

    now = datetime.now()
    #startFindCapsFromImages(1,187)
    #startFindStaticFrames(1,100, savePlot = True)
    #startFindStaticFramesAllVideos()
    #startFindStaticFrames(1,100, savePlot = True)
    startFindCapsFromVideo(51)
    #startFindCapsFromImages(1,100)

    #evaluateResults(25,30)
    print("total time", datetime.now()-now, 'without setting up tensorflow model')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # put hough voting on the black-white before connected components
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

