# OurCamera

This project uses Google's TensorFlow Machine learning package to identify 
and categorize types of vehicles on NYC streets. 

The data images are downloaded from NYC DOT cameras.

The training data folder has a series of images and training annotations.

## Setup

* Install Tensorflow from https://www.tensorflow.org/install/
* Download the github repo for [tensorflow models](https://github.com/tensorflow/models) and place it in this folder
* Follow the Object Detection Installation instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Run

Create a `test.record` file and a `train.record` file:

```
./generate_tfrecordfinal --folder=path/to/data_dir --train_ratio=.70
```

Copy Model from `?`

In three seperate command line windows run:

1)

```
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/path/data/faster_rcnn_resnet101_cars.config \
    --train_dir=/path/to/models/model/train
```

2)
    
```
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/path/data/faster_rcnn_resnet101_cars.config \
    --checkpoint_dir=/path/data/models/model/train \
    --eval_dir=/path/data/models/model/eval
```
        
3)

```
tensorboard --logdir=/path/models/model/
```

After a thousand or so steps you should be getting results. Look at your tensorflow eval images to gauge when to stop. FYI it took my `4770k Intel i7` about `24 hours` to train.

Create a frozen version of your graph by selecting a checkpoint:

```
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /path/data/models/faster_rcnn_resnet101_cars.config \
    --trained_checkpoint_prefix /path/data/models/model/train/model.ckpt-18557 \
    --output_directory output_inference_graph.pb
```

Run `downloadimages.py` to create a folder of images.

Run `analyze.py` to analyze the images

![Alt text](blockedlanes.gif?raw=true "Left: Identifying Vehicles Right: Identifying Blocked Lanes")



