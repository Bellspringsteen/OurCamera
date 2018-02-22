r"""Processing Image and Annotation Dataset

This executable is used to convert the image and label dataset to test and train TF Records:

A folder should be specified which contains folders labeled images/ and labels/
Each jpg image file should contain a corresponding label file with the same name but with .xml extension
A ratio of test to training should be specified i.e. .70

Example usage:
    ./generate_tfrecord \
        --folder=path/to/data_dir \
        --train_ratio=.70

"""


from __future__ import absolute_import
from PIL import Image

import random
import argparse
from argparse import RawTextHelpFormatter
import os
import io
import tensorflow as tf
import sys
import xml.etree.ElementTree as eT
sys.path.append('./models-master/research/')
from object_detection.utils import dataset_util


def class_text_to_int(row_label):
    if row_label == 'car':
        return 1
    elif row_label == 'truck':
        return 2
    elif row_label == 'bus':
        return 3
    elif row_label == 'ups':
        return 4
    elif row_label == 'pedestrian':
        return 5
    elif row_label == 'bicycle':
        return 6
    elif row_label == 'bicycke':
        return 6
    elif row_label == 'police':
        return 7
    else:
        raise ValueError('Unknown label found in data, label is '+row_label)


def create_tf_example_from_jpg_xml(jpg_file, xml_file):

    with tf.gfile.GFile(jpg_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    tree = eT.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        xmins.append(int(member[4][0].text) / width)
        xmaxs.append(int(member[4][2].text) / width)
        ymins.append(int(member[4][1].text) / height)
        ymaxs.append(int(member[4][3].text) / height)
        classes_text.append(member[0].text)
        classes.append(class_text_to_int(member[0].text))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(jpg_file),
        'image/source_id': dataset_util.bytes_feature(jpg_file),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    parser = argparse.ArgumentParser(
        description='Process a folder of jpg and xml training files into test and train tf records \n'
                    ' assumes that in folder there are two folders images and labels \n'
                    ' images contains jpg images and labels contain xml', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-folder', help='an integer for the accumulator')
    parser.add_argument('-train_ratio', type=float,
                        help='a float between 0 - 1.0 that sets the ratio of files that should be training vs testing')
    args = parser.parse_args()
    print("Running on " + args.folder + " with ratio " + str(args.train_ratio))

    images_path = args.folder+'images/'
    labels_path = args.folder+'labels/'

    arr = os.listdir(images_path)

    random.shuffle(arr)

    numberofimages = len(arr)

    start_value = 0
    end_value = int(numberofimages*args.train_ratio)
    for i in ['test', 'train']:

        writer = tf.python_io.TFRecordWriter(args.folder+i + '.record')
        for x in range(start_value, end_value):
            if arr[x].endswith(".jpg"):
                filename = os.path.splitext(arr[x])[0]
                tf_example = create_tf_example_from_jpg_xml(images_path+filename+'.jpg', labels_path+filename+'.xml')
                writer.write(tf_example.SerializeToString())
        start_value = end_value + 1
        end_value = numberofimages
        writer.close()


if __name__ == '__main__':
    tf.app.run()

