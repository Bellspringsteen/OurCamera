r"""Analyze Traffic Images

This executable is used to annotate traffic images to highlight vehicle types and to produce stats
and graphs for the amount of time bicycle lanes and bus stops are blocked by vehicles:


Example usage:
    ./analyzeimages \
        -path_images ./data/rawimages/
        -path_labels_map data/car_label_map.pbtxt
        -save_directory data/processedimages/
"""

import sys

from matplotlib.ticker import FormatStrFormatter, FuncFormatter

sys.path.append('./models/research/')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import argparse
from argparse import RawTextHelpFormatter
import time
import numpy as np
import os
import tensorflow as tf
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
import matplotlib.path as mpltPath

from PIL import Image
import scipy.misc


def processimages(path_images_dir, path_labels_map,save_directory):
    pathcpkt = 'data/output_inference_graph.pb/frozen_inference_graph.pb'
    csv_file = 'data/csvfile.csv'
    num_classes = 6

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pathcpkt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(path_labels_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    f = open(csv_file, 'w')
    #f.write(
    #    'timestamp,number cars in bike lane, number trucks in bike lane, '
    #    'number cars in bus stop, number trucks in bus stop\n')

    def load_image_into_numpy_array(imageconvert):
        (im_width, im_height) = imageconvert.size
        try:
            return np.array(imageconvert.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)
        except ValueError:
            return np.array([])

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            polygon_right_lane = [(178, 122), (188, 240), (231, 240), (187, 125)]
            polygon_left_lane = [(108, 143), (0, 215), (0, 233), (123, 142), (108, 97)]
            polygon_bus_lane = [(200, 155), (230, 240), (292, 240), (225, 157)]

            pathrightlane = mpltPath.Path(polygon_right_lane)
            pathleftlane = mpltPath.Path(polygon_left_lane)
            pathbuslane = mpltPath.Path(polygon_bus_lane)
            for testpath in os.listdir(path_images_dir):

                start_time = time.time()
                timestamp = testpath.split(".jpg")[0]

                try:
                    image = Image.open(path_images_dir + '/' + testpath)
                    image_np = load_image_into_numpy_array(image)
                except IOError:
                    print("Issue opening "+testpath)
                    continue

                if image_np.size == 0:
                    print("Skipping image "+testpath)
                    continue
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    min_score_thresh=0.4,
                    use_normalized_coordinates=True,
                    line_thickness=2)
                scores = np.squeeze(scores)
                boxes = np.squeeze(boxes)
                num_cars_in_bikelane, num_cars_in_bus_stop, num_trucks_in_bike_lane, num_trucks_in_bus_stop = 0, 0, 0, 0
                for i in range(boxes.shape[0]):
                    if scores[i] > .4:
                        box = tuple(boxes[i].tolist())

                        ymin, xmin, ymax, xmax = box

                        center_x = (((xmax * 352) - (xmin * 352)) / 2) + (xmin * 352)
                        center_y = (((ymax * 240) - (ymin * 240)) / 2) + (ymin * 240)
                        classes = np.squeeze(classes).astype(np.int32)
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'

                        if class_name == 'car':
                            points = [(center_x, center_y)]
                            if pathrightlane.contains_points(points) or pathleftlane.contains_points(points):
                                num_cars_in_bikelane += 1
                            elif pathbuslane.contains_points(points):
                                num_cars_in_bus_stop += 1

                        elif class_name == 'truck' or class_name == 'police' or class_name == 'ups':
                            points = [(center_x, center_y)]
                            if pathrightlane.contains_points(points) or pathleftlane.contains_points(points):
                                num_trucks_in_bike_lane += 1
                            elif pathbuslane.contains_points(points):
                                num_trucks_in_bus_stop += 1

                # write to a csv file whenever there is a vehicle, how many and of what type with timestamp
                f.write(timestamp + ',' + str(num_cars_in_bikelane) + ',' + str(num_trucks_in_bike_lane) + ',' + str(
                    num_cars_in_bus_stop) + ',' + str(num_trucks_in_bus_stop) + '\n')
                print("Process Time " + str(time.time() - start_time))
                scipy.misc.imsave(save_directory + testpath, image_np)

        f.close()
        return csv_file


def initialize_datastore():
    blankarray = [0] * 24
    alldata = [[list(blankarray), list(blankarray), list(blankarray)],
               [list(blankarray), list(blankarray), list(blankarray)]]
    # alldata [ [cars_blocking_bikelane[24],trucks_blocking_bikelane[24],eitherblockingbikelane[24]
    #           [cars_blocking_buslane[24],trucks_blocking_buslane[24],eitherblockingbuslane[24]]

    weekdaydata = [[list(blankarray), list(blankarray), list(blankarray)],
                   [list(blankarray), list(blankarray), list(blankarray)]]
    # same as alldata above but for weekdays, weekenddata same but for weekends
    weekenddata = [[list(blankarray), list(blankarray), list(blankarray)],
                   [list(blankarray), list(blankarray), list(blankarray)]]

    return [alldata, weekdaydata, weekenddata]


def weekday(datevalue):
    if datevalue.weekday() < 5:
        return True
    else:
        return False


def incrementarray(array, blockagearray, delta_time):
    timestamp_string = (blockagearray[0].split(".jpg"))[0]
    datetime_object = datetime.strptime(timestamp_string, '%Y-%m-%d %H:%M:%S.%f')

    hour = datetime_object.hour
    num_cars_in_bike_lane = int(blockagearray[1])
    num_trucks_in_bike_lane = int(blockagearray[2])
    num_cars_in_bus_stop = int(blockagearray[3])
    num_truck_in_bus_stop = int(blockagearray[4])

    if num_cars_in_bike_lane > 0:
        array[0][0][hour] += delta_time
    if num_trucks_in_bike_lane > 0:
        array[0][1][hour] += delta_time
    if num_cars_in_bike_lane > 0 or num_trucks_in_bike_lane > 0:
        array[0][2][hour] += delta_time
    if num_cars_in_bus_stop > 0:
        array[1][0][hour] += delta_time
    if num_truck_in_bus_stop > 0:
        array[1][1][hour] += delta_time
    if num_cars_in_bus_stop > 0 or num_truck_in_bus_stop > 0:
        array[1][2][hour] += delta_time


def incrementarrays(dataarrays, blockagearray, delta_time):
    alldata = dataarrays[0]
    weekdaydata = dataarrays[1]
    weekenddata = dataarrays[2]

    datetime_object = datetime.strptime((blockagearray[0].split(".jpg"))[0], '%Y-%m-%d %H:%M:%S.%f')

    incrementarray(alldata, blockagearray, delta_time)
    if weekday(datetime_object):
        incrementarray(weekdaydata, blockagearray, delta_time)
    else:
        incrementarray(weekenddata, blockagearray, delta_time)

    return [alldata, weekdaydata, weekenddata]


def buildsaveplot(list_to_graph, title):
    label = ['', '', '', '', '', '6 am', '',
             '', '', '', '', '12 noon', '', '', '', '', '', '6 Pm', '',
             '',
             '', '', '', 'Midnight']
    index = np.arange(len(label))
    plt.bar(index, list_to_graph)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title(title)
    plt.plot()

    plt.ylim([0, 100.0])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
    plt.savefig("output/"+title.replace(" ", "") + ".png", bbox_inches='tight')
    plt.close()


def analyzeresults(csv_file):
    total_time_secs, total_time_bike_lane_blocked_secs, total_time_bus_stop_blocked_secs = 0, 0, 0

    weekdaytotalseconds = [1] * 24  # where we are going to store how many seconds worth of images there are
    weekendtotalseconds = [1] * 24  # for each hour this is necessary beecause we may be missing images

    previous_timestamp = 0
    dataarrays = initialize_datastore()

    data = csv.reader(open(csv_file, 'r'))
    data = sorted(data, key=lambda rowparse: datetime.strptime((rowparse[0].split(".jpg"))[0], '%Y-%m-%d %H:%M:%S.%f'))

    for row in data:
        datetime_object = datetime.strptime((row[0].split(".jpg"))[0], '%Y-%m-%d %H:%M:%S.%f')
        timestamp = float(datetime_object.strftime('%s'))
        hour = datetime_object.hour

        if previous_timestamp != 0:
            delta_time = timestamp - previous_timestamp
            if delta_time > 30:
                print("DELTA TIME LARGE")
                delta_time = 30

            total_time_secs += delta_time
            if weekday(datetime_object):
                weekdaytotalseconds[hour] += delta_time  # necessary because there may be time stamps missing in images
            else:
                weekendtotalseconds[hour] += delta_time

            dataarrays = incrementarrays(dataarrays, row, delta_time)
        previous_timestamp = timestamp

    weekendpercentageblocked = [[0] * 24, [0] * 24]  # bike lane first array and bus lane second
    weekdaypercentageblocked = [[0] * 24, [0] * 24]

    for hour in range(0, 24):
        total_time_bike_lane_blocked_secs += dataarrays[0][0][2][hour]
        total_time_bus_stop_blocked_secs += dataarrays[0][1][2][hour]
        weekdaypercentageblocked[0][hour] = 100 * (dataarrays[1][0][2][hour] / weekdaytotalseconds[hour])
        weekendpercentageblocked[0][hour] = 100 * (dataarrays[2][0][2][hour] / weekendtotalseconds[hour])
        weekdaypercentageblocked[1][hour] = 100 * (dataarrays[1][1][2][hour] / weekdaytotalseconds[hour])
        weekendpercentageblocked[1][hour] = 100 * (dataarrays[2][1][2][hour] / weekendtotalseconds[hour])

    total_time_seven2seven, blockedbikelaneseven2seven, blockedbuslaneseven2seven = 0, 0, 0
    for x in range(7, 19):
        total_time_seven2seven += weekdaytotalseconds[x]
        blockedbikelaneseven2seven += dataarrays[1][0][2][x]
        blockedbuslaneseven2seven += dataarrays[1][1][2][x]

    print("RESULTS \n Total Time " + str(total_time_secs) + " blocked bike lane time " + str(
        total_time_bike_lane_blocked_secs) + "blocked truck lane time" + str(total_time_bus_stop_blocked_secs))
    print("Bike lane blocked " + str(100 * (total_time_bike_lane_blocked_secs / total_time_secs)) + "% of the time")
    print("Bus lane blocked " + str(100 * (total_time_bus_stop_blocked_secs / total_time_secs)) + "% of the time")
    print("Bike lane blocked " + str(
        100 * (blockedbikelaneseven2seven / total_time_seven2seven)) + "% of the time durring weekday from 7 am to 7pm")
    print("Bus lane blocked " + str(
        100 * (blockedbuslaneseven2seven / total_time_seven2seven)) + "% of the time durring weekday from 7 am to 7pm")

    buildsaveplot(weekdaypercentageblocked[0], 'Weekday Bike Lane Percentage Blocked by Hour')
    buildsaveplot(weekdaypercentageblocked[1], 'Weekday Bus Stop Percentage Blocked by Hour')
    buildsaveplot(weekendpercentageblocked[0], 'Weekend Bike Lane Percentage Blocked by Hour')
    buildsaveplot(weekendpercentageblocked[1], 'Weekend Bus Stop Percentage Blocked by Hour')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze traffic images to determine rate of blocking bike'
                    'and bus lanes', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-path_images', help='the folder with all the downloaded images in it')
    parser.add_argument('-path_labels_map', help='the file with the integer to label map')
    parser.add_argument('-save_directory', help='the directory you want to save the annotated images to')
    args = parser.parse_args()
    #csv_file = processimages(args.path_images,args.path_labels_map,args.save_directory)
    analyzeresults('data/analysis10days.csv')
    analyzeresults(csv_file)