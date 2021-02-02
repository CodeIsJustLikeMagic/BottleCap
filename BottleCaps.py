# numpy, ceras, tensorflow,tensorflow, patplotlip, opencv, numpy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import json


# region help methods
def h_show(title, image, show=False, resize=True):
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))
    if resize:
        image = cv2.resize(image, (int(width * 0.5), int(height * 0.5)))
    if show or all_show: # allshow overwrites whatever specifics I added to each plot call
        cv2.imshow(title, image)


def h_plot(arr, description, filename='', start=0, maxdiff = None, save=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(start, len(arr) + start)
    #m = arr.mean()
    ax.scatter(x, arr, marker=',', label='diffs between frames', s=4)
    ax.set_ylabel(description)
    #ax.axhline(y=maxdiff, color="black")
    #ax.axhline(y=m, color="red")
    if save:
        fig.savefig('misc/' + filename[:-4] + '_' + description + '.png')
    if plot:
        plt.show()


def read_video(vid):  # show video
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break


# endregion
# region find frames
def diffs_between_all_frames(vid, filename, savePlot=False):
    diffs = np.array([])

    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prevframe = vid.read()
    while vid.isOpened():
        ret, currentframe = vid.read()
        if not ret:
            break
        dif = diff_between_frames(prevframe, currentframe)
        diffs = np.append(diffs, dif)
        prevframe = currentframe
    h_plot(diffs, "mean square diffs", start=0, save=savePlot, filename=filename)
    return diffs


def find_sequences(diffarr):
    maxdiff = diffarr.mean()
    linestarts, lines, _ = h_find_sequences(diffarr, 30, maxdiff)
    while (len(lines) < 1 or (len(lines) == 1 and linestarts[0] == 0)) and maxdiff > 0:  # lower the threshhold
        print('threhhold has been lowered')
        # dont accept lines if
            # there is exactly one line and it starts with 0
            # there are 0 lines
        # do accept lines if
            # at least one of the sequences does not start at 0
        maxdiff = maxdiff - diffarr.mean() / 2
        linestarts, lines, _ = h_find_sequences(diffarr, 30, maxdiff)
    while (len(lines) < 1 or (len(lines) == 1 and linestarts[0] == 0)) and maxdiff < 10000:  # raise the threshhold
        print('threhhold has been raised')
        maxdiff = maxdiff + diffarr.mean() / 2
        linestarts, lines, _ = h_find_sequences(diffarr, 30, maxdiff)
    return linestarts, lines, maxdiff


def h_find_sequences(diffarr, minaountofframes, maxdiff):
    # cut video into sequences of at least n>minamounfofframes frames and the frames have an mse<maxdiff
    lines = []
    linestarts = []
    currentline = np.array([])
    currentlinestart = 0
    for index, d in enumerate(diffarr, start=1):
        if d < maxdiff:
            currentline = np.append(currentline, d)
        else:
            # break line
            if (len(currentline) > minaountofframes):  # cut the line here
                lines.append(currentline)
                currentline = np.array([])
                linestarts.append(currentlinestart)
            else:  # clear what we already have
                currentline = np.array([])
            currentlinestart = index
    if (len(currentline) > minaountofframes):
        lines.append(currentline)
        linestarts.append(currentlinestart)
    # print(len(lines), len(linestarts))
    for line, linest in zip(lines, linestarts):
        h_plot(line, "line " + str(linest), maxdiff = maxdiff,  start=linest)
        # current diff,  currentline, start index, mean diff in line, length
    # print("found", len(lines), "sequenzes")
    return linestarts, lines, maxdiff


def get_frame_indeces_of_best_line(linestarts, lines):
    # there should be one szene right at the start, one in the middle and one at the end
    if 0 in linestarts:# if there is a scequence that starts at 0
        linestarts.pop(0) # discard it
        lines.pop(0)
        # the sequzence at the start is worse than the main sequenze.
        # now there should be one or two good seqzences.
        return 0, find_bestline(linestarts, lines)
    else:
        # print("AHHHHH there is no seqzence at the start, pls help")
        return 0, find_bestline(linestarts, lines)


def find_bestline(linestarts, lines):
    nlines = []
    for line, linest in zip(lines, linestarts):
        # print(linest, np.std(line), np.mean(line))
        frameindeces = np.array(range(linest, linest + len(line)))
        line = np.vstack([line, frameindeces])
        nlines.append(line)
    nlines.sort(key=sort_by_mean)
    # so now the first or the second one is our sequence
    besttwo = nlines[:2]
    # if the second line mean is muuch worse than the first one there is probably one one static szene
    besttwo.sort(key=smaller_start)
    # choose the frame with the smallest score from the first one
    bestone = besttwo[0]  # list index out of range
    l = len(bestone[0])
    bestone = list(map(list, zip(bestone[0], bestone[1])))  # 0 is all difference scores, 1 is the frame
    m = min(bestone, key=lambda t: t[0])
    ret = bestone[int(l / 2)]
    return ret[1]


def sort_by_mean(elem):
    return np.mean(elem[0])


def smaller_start(elem):
    return elem[1][0]


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err  # return the MSE, the lower the error, the more "similar" the two images are


def diff_between_frames(frame1, frame2):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    quanti = mse(frame1, frame2)
    # dif = frame1 - frame2
    # sumdif = dif.sum()
    # res = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF)
    # I wanted to use templatemaching but atm I use simple dif
    return quanti


def read_frame(vid, frame_number):
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))  # optional
    success, frame = vid.read()
    if success:
        return frame
    return False


# endregion
# region regions of interest
# important function
def initial_regions_of_interest(emptyframe, fullframe, filename):
    kernel = np.ones((3, 3), np.uint8)
    difference = cv2.subtract(emptyframe, fullframe)
    difference = cv2.add(difference, cv2.subtract(fullframe, emptyframe))
    difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference = cv2.GaussianBlur(difference, (5, 5), 2)
    difference = (255 - difference)
    ret, thresh = cv2.threshold(difference, 0, 255, cv2.THRESH_OTSU)  # black roi on white
    differentregions = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # erosion followed by dilation
    differentregions = 255 - differentregions
    lines = findLineswithHough(differentregions, filename, threshhold=180, minLineLength=20, maxLineGap=150, show=True)
    h_show('regions' + filename, differentregions)
    if lines is not None:  # remove lines if there are any :)
        lines = cv2.dilate(lines, kernel, iterations=2)
        differentregions = cv2.subtract(differentregions, lines)

    # findContours(fullframe, filename)
    lines = findContainer(emptyframe,filename)
    # houghTransformSkimage(cv2.Canny(fullframe, 50, 150, apertureSize=3))
    # findCirclewithHough(fullframe,differentregions,filename)

    guess_crops_with_tensor(differentregions, filename, fullframe)


def houghTransformSkimage(edges):
    # https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
    from skimage import io
    from skimage.transform import probabilistic_hough_line
    from skimage.feature import canny
    from matplotlib import cm
    from skimage.color import rgb2gray

    # Line finding using the Probabilistic Hough Transform
    image = io.imread('Results/CV20_video_6_empty.png')
    image = rgb2gray(image)
    # edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                     line_gap=3)

    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def findContainer(emptyframe, filename):
    gray = cv2.cvtColor(emptyframe, cv2.COLOR_BGR2GRAY)
    lineView = findLineswithHough(gray, filename, threshhold=180, minLineLength=20, maxLineGap=150)
    lineView = findLineswithHough(lineView, filename, threshhold=200, minLineLength=100, maxLineGap=500, save=True)


def findLineswithHough(gray, filename, rho=1, theta=np.pi / 180, threshhold=180, minLineLength=20, maxLineGap=150,
                       save=False, show=False):
    edges = cv2.Canny(gray, 10, 150)
    lineView = np.zeros(gray.shape, dtype=np.uint8)
    h_show('edges' + filename, edges, show=show)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=180, minLineLength=20, maxLineGap=150)
    if lines is None:
        print("no lines")
        return
    # h_show('houghlines5' + filename, lines, show=True)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lineView, (x1, y1), (x2, y2), (255), 2)

    h_show('houghlines' + filename, lineView, show=show)
    if save:
        cv2.imwrite('Results/CV20_' + filename[:-4] + '_lines.png', lineView)
    return lineView


def findContours(img, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # --- First obtain the threshold using the greyscale image ---
    ret, th = cv2.threshold(gray, 127, 255, 0)

    # --- Find all the contours in the binary image ---
    contours, hierarchy = cv2.findContours(edges, 2, 1)
    cnt = contours
    big_contour = []
    max = 0
    for i in cnt:
        area = cv2.contourArea(i)  # --- find the contour having biggest area ---
        if (area > max):
            max = area
            big_contour = i

    final = cv2.drawContours(img, big_contour, -1, (0, 255, 0), 3)
    h_show('contours' + filename, final)


def findCirclewithHough(img, gray, filename):
    img = cv2.medianBlur(img, 5)
    # gray = 255-gray
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cimg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    h_show("edges" + filename, edges, show=True)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=3, maxRadius=50)
    if circles is None:
        print("No cirlces")
        return
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    h_show('detected circles' + filename, cimg, show=True)


# important function
def guess_crops_with_tensor(img, filename, originalimage, saveCrops=False):
    global predictionModel
    if predictionModel is None:
        predictionModel = PredictionModel()
    num_labels, labels = cv2.connectedComponents(img)
    # Map component labels to hue val
    occurenceOfLabels = np.bincount(labels.flatten())
    occurenceOfLabels = occurenceOfLabels[1::]
    labelsbackwards = np.flip(labels)
    w, h = img.shape[::-1]
    # label 0 is background (black)
    shapes = []
    if predictionModel is not None:

        for occ, lblind in zip(occurenceOfLabels, range(1, num_labels)):
            if occ > 300:
                # calculate boundinBox for connectedcomponent
                # get min x,y and max x,y position of l
                x1 = np.argmax(labels == lblind, axis=1)  # along x axis
                x1 = minwithout0(x1)

                x2 = np.argmax(labelsbackwards == lblind, axis=1)
                x2 = minwithout0(x2)
                x2 = w - x2

                y1 = np.argmax(labels == lblind, axis=0)  # along x axis
                y1 = minwithout0(y1)

                y2 = np.argmax(labelsbackwards == lblind, axis=0)  # along x axis
                y2 = minwithout0(y2)
                y2 = h - y2

                # print(lblind, 'x', x1, x2, 'y', y1, y2)
                crop_img = originalimage[y1:y2, x1:x2].copy()
                if saveCrops:
                    cv2.imwrite("Crops/" + str(lblind) + "_" + filename[:-4] + '.png', crop_img)

                # guess if boundingbox contains bottlecap
                objclass, prob = predictionModel.prediction(crop_img)
                cv2.rectangle(originalimage, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # save resulting boundingBox in json file
                resultdict = {}
                resultdict["label"] = str(objclass)
                resultdict["points"] = [[str(x1), str(y1)], [str(x2), str(y2)]]
                shapes.append(resultdict)

                # save Image with Boxes
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # org
                org = (x1, y1 - 10)
                # fontScale
                fontScale = 0.7
                # Blue color in BGR
                color = (0, 0, 255)
                # Line thickness of 2 px
                thickness = 1
                # Using cv2.putText() method
                originalimage = cv2.putText(originalimage, objclass + " " + str(prob), org, font,
                                            fontScale, color, thickness, cv2.LINE_AA)
                h_show("testcropimg" + str(lblind) + filename, crop_img, show=False)
                if True:
                    cv2.imwrite('Results/CV20_' + filename[:-4] + '_caps.png', originalimage)
            # crop out rectangular region (of orig image) around label.

    jsonDict = {}
    jsonDict["shapes"] = shapes
    st = 'Results/CV20_' + filename[:-4] + '.json'
    with open('Results/CV20_' + filename[:-4] + '.json', 'w') as outfile:
        json.dump(jsonDict, outfile)
    show_colored_connected_components(labels, filename)
    # h_show("detected"+filename, originalimage)


def show_colored_connected_components(labels, filename, save=True):
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    coloredROIS = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    coloredROIS = cv2.cvtColor(coloredROIS, cv2.COLOR_HSV2BGR)

    # set bg label to black
    coloredROIS[label_hue == 0] = 0
    # h_show('labeled'+filename, coloredROIS)
    if save:
        cv2.imwrite('Results/CV20_' + filename[:-4] + '_components.png', coloredROIS)


def minwithout0(ar):
    ar = ar[ar != 0]
    if len(ar) == 0:
        return 0
    else:
        return np.amin(ar)

# endregion

def start_find_caps_from_images(low, high=None):
    if high is None:
        high = low
    for i in range(low, high + 1):
        print('start working on Video: ', i)
        fullframe = cv2.imread('Results/CV20_video_' + str(i) + '.png')
        emptyframe = cv2.imread('Results/CV20_video_' + str(i) + '_empty.png')
        find_caps(emptyframe, fullframe, i)


def start_find_caps_from_video(low, high=None, saveImages=True):
    if high is None:
        high = low
    for i in range(low, high + 1):
        print('start working on Video: ', i)
        emptyframe, fullframe = find_static_frames_single(i, filename='CV20_video_' + str(i) + '.mp4')
        if saveImages:
            cv2.imwrite('Results/CV20_video_' + str(i) + '.png', fullframe)
            cv2.imwrite('Results/CV20_video_' + str(i) + '_empty.png', emptyframe)
        find_caps(emptyframe, fullframe, i)


def find_caps(emptyframe, fullframe, i):
    print('start Find Caps for Video', i, '...')
    now = datetime.now()
    if fullframe is not None and emptyframe is not None:
        initial_regions_of_interest(emptyframe, fullframe, 'Video_' + str(i) + '.png')
        print('end Find Caps. Time for Video', i, datetime.now() - now)
    else:
        print('no images found for Video', i)


def start_find_static_frames(low, high=None, savePlot=False):
    if high is None:
        high = low
    for i in range(low, high + 1):
        try:
            emptyframe, fullframe = find_static_frames_single(i, savePlot=savePlot)
            cv2.imwrite('Results/CV20_video_' + str(i) + '.png', fullframe)
            cv2.imwrite('Results/CV20_video_' + str(i) + '_empty.png', emptyframe)
        except:
            print("caught an error with video file")


# important function
def find_static_frames_single(filenum, path='Videos/', filename=None, savePlot=False):
    if filename is None:
        filename = "CV20_video_" + str(filenum) + ".mp4"
    path = path + filename
    print('start frameselection', path, '...')
    now = datetime.now()
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print("Error opening video stream or file")
        return

    fps = vid.get(cv2.CAP_PROP_FPS)
    diffs = diffs_between_all_frames(vid, filename, savePlot=True)
    print("mean", diffs.mean())
    linestarts, lines, maxdiff = find_sequences(diffs)

    if len(lines) < 1:
        print('to much action, threshhold excluded  everything')
        return
    emptysceneindex, fullsceneindex = get_frame_indeces_of_best_line(linestarts.copy(), lines.copy())
    illustrate_lines(diffs, linestarts, lines, filename, fullsceneindex, maxdiff)
    emptyframe = read_frame(vid, emptysceneindex)
    fullframe = read_frame(vid, fullsceneindex)
    print('choosing frame:', fullsceneindex, '; min', fullsceneindex / fps)
    vid.release()
    print("end frameselection. time for", filename, datetime.now() - now)
    return emptyframe, fullframe
    # sometimes real sequence is to short and gets cut out

def illustrate_lines(diffs, linestats, lines, filename, choosen, maxdiff, save=True):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(0, len(diffs))
    m = diffs.mean()
    ax.scatter(x, diffs, marker=',', label='diffs between frames', s=4)
    ax.set_ylabel("cut into sequences")
    # ax.axhline(y=maxdiff, color="black")
    ax.axhline(y=maxdiff, color="red")
    ax.axvline(x=choosen)

    for line, start in zip(lines, linestats):
        x = np.arange(start, len(line) + start)
        ax.scatter(x, line, marker=',', s=4)
    if save:
        fig.savefig('Results/' + filename[:-4] + '_diffsplot.png')
    plt.show()

# important function
def evaluate_results(low, high=None):
    if high is None:
        high = low
    for i in range(low, high + 1):
        evaluate_results_singe(i)


def evaluate_results_singe(video_num):
    success = True
    # visually compare the still frame of the video and the chosen still frame
    # evaluate how well the video frame was choosen
    intended_frame = cv2.imread('Videos/CV20_image_' + str(video_num) + '.png')
    found_full_frame = cv2.imread('Results/CV20_video_' + str(video_num) + '.png')
    h_show('intended frame' + str(video_num), intended_frame)
    h_show('chosenFrame' + str(video_num), found_full_frame)
    dif = diff_between_frames(intended_frame, found_full_frame)
    difference = cv2.subtract(intended_frame, found_full_frame)
    h_show('how well the frame was choosen ' + str(video_num), difference)
    if dif > 50:
        success = False

    # read the json file of the video to see how well caps were detected
    with open('Videos/CV20_label_renamed_' + str(video_num) + '.json') as f:
        with open('Results/CV20_Video_' + str(video_num) + '.json') as f2:
            real_caps = json.load(f)
            intended_shapes = real_caps.get("shapes")
            found_caps = json.load(f2)
            found_shapes = found_caps.get("shapes")
            for rshape in intended_shapes:
                rlabel = rshape.get("label")
                rmidpoint = middle_of_points(rshape.get("points"))
                # compare to points of my estimation
                for fshape in found_shapes:
                    foundlabel = fshape.get("label")
                    fmidpoint = middle_of_points(fshape.get("points"))

    print('Evaluating Video', video_num, 'frame eval: ', dif)


def middle_of_points(points):
    if points is None:
        return [0, 0]
    sumX = 0
    sumY = 0
    length = len(points)
    for point in points:
        sumX = sumX + int(point[0])
        sumY = sumY + int(point[1])
    midpointX = sumX / length
    midpointY = sumY / length
    return [midpointX, midpointY]


class PredictionModel:
    IMG_WIDTH = 200
    IMG_HEIGHT = 200

    def __init__(self):
        import tensorflow as tf
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
                tf.keras.layers.Dense(len(target_dict), activation='softmax')
                # https://datascience.stackexchange.com/questions/69664/how-to-interpret-keras-predict-output
            ])
        print(model.summary())
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(x=tf.cast(np.array(img_data), tf.float64), y=tf.cast(list(map(int, target_val)), tf.int32),
                            epochs=4)
        print(history)
        print('finished TensorFlow Model. Time', datetime.now() - now)
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
        # print(predictions)
        m = np.argmax(predictions[0])
        return self.inv_target_dict[m], predictions[0][m]


predictionModel = None

plot = True
all_show = False
if __name__ == "__main__":
    now = datetime.now()
    start_find_static_frames(59)
    #start_find_caps_from_video(1,100)
    #evaluate_results(25,30)
    print("total time", datetime.now() - now)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # put hough voting on the black-white before connected components
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
