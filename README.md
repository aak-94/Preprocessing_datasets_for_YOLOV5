# 06_Preprocessing_datasets_for_YOLOV5
Toolchain_for_preprocessing_multiple_datasets_for_YOLO_models

## Goal
- To reduce the data-preprocessing for neural network training
- To facilitate the use of multiple datasets having different image sizes together

## Purpose
This toolchain helps to pre-process the  multiple datasets simultaneously and performs follwoing tasks:
- filtering the xmls without corrosponding img file
- filtering the xmls without the trainable object annotations (or tags)
- resizing of the images and xml files so that complete dataset will have one standard image size.
  Thus multiple datasets with different image sizes can be combined easily to train the network
 - Converts the labels from pascal VOC  format (.xml)  to the YOLO format (.txt)
- combines multiple dataset to the single dataset and then splits it in train-test datasets
- generates the bar graph to visualize the dataset distribution before as well as after the split.

# Requirements
 basic python libraries numpy, matplotlib, opencv
 
## Working 
Following image shows the working of the complete pre-processing pipeline
![image](https://github.com/aak-94/06_Preprocessing_datasets_for_YOLOV5/blob/master/flowchart.JPG)

## Testing
This pipline has been tested with the datasets for [object detection](https://data.mendeley.com/datasets/5ty2wb6gvg/1)
## Usage
- download the repo
- specify the paths of datasets in the datasets.json file
- specify the desired img size in the datasets.json file
- run the file intial_filter.py

## references:
[Resize-pascal-voc](https://github.com/italojs/resize_dataset_pascalvoc)

