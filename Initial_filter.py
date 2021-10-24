import os
from pathlib import Path   # to handle the slashes as per the operating system
import cv2 as cv2
import random
from sys import platform
from utils import *
from  dict_to_yolo_label import *
import json

#%% The number of total images and total labels in each gov
def main(data_dir,destination): 
    result_dir,obs_path,label_path,output_xmls,out_img,train_test_paths= create_result_dir(destination)

    if platform == 'win32':
        seperator ='\\'
    else: 
        seperator ='/'
    
    annotation_path ={}
    image_path ={}
    images={}
    xml_list={}
    for key in data_dir['Datasets']:
        data_directory = data_dir['Datasets'][key]
        xml_list[key], image_list, images[key], annotation_path[key], image_path[key] = get_path_lists(data_directory,seperator)
        print(key+' datasets total_annotaitons: ', len(xml_list[key]))
        print(key+' dataset total_images: ', len(image_list))

    all_class_names= get_class_names(xml_list)
    unique_class_names,all_class_names=replace_del_names(all_class_names,data_dir['replace_del'])
    print(unique_class_names)
    create_save_bar_chart(unique_class_names, all_class_names,'Class distribution of filtered dataset',result_dir)
    total_images = 0
    img_size=data_dir['img_size']
    for key in data_dir['Datasets']:
        for i,file in enumerate(images[key]):  #processing one image 
            xml_name = file[:-4]+'.xml'
            if os.path.isfile(os.path.join(annotation_path[key],xml_name)):  #checking if corresponding xml exists
                dict,reshaping = xml_to_dict(annotation_path[key], xml_name, image_path[key], file, data_dir, output_xmls,out_img)    
                if len(dict)!=0:                                        #checking if xml has an trainable object 
                    dict_to_yolo_label(file,dict,img_size,unique_class_names,label_path)
                    total_images+=1
                    if i % 100 ==0:
                        #pass
                        write_marked_images(image_path[key], file, dict, obs_path,reshaping,img_size)
                else:
                    print('empty xml_file: ', xml_name)
            else:
                print('Annotation file for',file,'not found')
    print('total_valid_images:', total_images)
    train_test_split(total_images,train_test_paths,out_img,label_path,output_xmls)
    create_data_representation(train_test_paths[0],seperator,'train dataset',result_dir,data_dir['replace_del'])
    create_data_representation(train_test_paths[4],seperator,'test dataset',result_dir,data_dir['replace_del'])

if __name__ =="__main__":
    with open('./datasets.json') as json_params:
        data_dir = json.load(json_params)
    destination = os.getcwd()   
    main(data_dir,destination)