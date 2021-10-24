import cv2 as cv2
import os
from xml.etree import ElementTree
import numpy as np
import matplotlib.pyplot as plt
import math
from random import choice
import shutil

def get_path_lists(data_dir,seperator):
    xml_list=[]
    image_list=[]
    images=[]
    for looproot, _, filenames in os.walk(data_dir):
        for filename in filenames:          #limit the the images number here during trials
            if filename.endswith('xml'):
                xml_list.append(os.path.join(looproot, filename))
            if filename.endswith('jpg'):
                image_list.append(os.path.join(looproot, filename))
                images.append(filename)
    annotation_path = xml_list[0].rsplit(seperator,1)[0]
    image_path = image_list[0].rsplit(seperator,1)[0]
    return xml_list,image_list,images,annotation_path,image_path

def create_result_dir(destination):
    result_dir= os.path.join(destination,'result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    else:
        shutil.rmtree(result_dir) # delete existing folder and subfolders
        os.mkdir(result_dir)

    obs_path = os.path.join(result_dir,'observation_set')
    os.mkdir(obs_path)
    label_path = os.path.join(result_dir,'labels')
    os.mkdir(label_path)
    output_xmls = os.path.join(result_dir,'annotations')
    os.mkdir(output_xmls)
    out_imgs = os.path.join(result_dir,'images')
    os.mkdir(out_imgs)

    #train_test_paths    
    train_test_paths=[]
    train_path = os.path.join(result_dir,'train')
    os.mkdir(train_path)
    train_imgs = os.path.join(train_path,'images')
    os.mkdir(train_imgs)    
    train_lbls = os.path.join(train_path,'labels')
    os.mkdir(train_lbls)
    train_xmls = os.path.join(train_path,'xmls')
    os.mkdir(train_xmls) 
    test_path = os.path.join(result_dir,'test')
    os.mkdir(test_path)
    test_imgs = os.path.join(test_path,'images')
    os.mkdir(test_imgs)    
    test_lbls = os.path.join(test_path,'labels')
    os.mkdir(test_lbls)
    test_xmls = os.path.join(test_path,'xmls')
    os.mkdir(test_xmls) 

    train_test_paths.append(train_path)
    train_test_paths.append(train_imgs)
    train_test_paths.append(train_lbls)
    train_test_paths.append(train_xmls)
    train_test_paths.append(test_path)
    train_test_paths.append(test_imgs)
    train_test_paths.append(test_lbls)
    train_test_paths.append(test_xmls)

    return result_dir,obs_path,label_path,output_xmls,out_imgs,train_test_paths

def write_marked_images (image_path,file, dict, obs_path,reshaping,img_size):
    '''
    img= input image
    dict = each key contains the list of class name & bounding box coordinates in order [cls_name, x_min, y_min, x_max, y_max]
    obs_path = location to save the marked images
    '''
    font = cv2.FONT_HERSHEY_PLAIN
    img = cv2.imread(os.path.join(image_path, file))
    if reshaping:
        img=cv2.resize(img,(img_size,img_size))
    for key in dict:
        values=dict[key]
        cv2.rectangle(img, (values[1], values[2]), (values[3], values[4]), (255,255,0), 2)
        cv2.putText(img, values[0], (values[1], values[2]+ 30), font, 3, (255,0,0), 3)
    cv2.imwrite(os.path.join(obs_path,file),img)
    print(file+' marked and saved to '+ obs_path)

def get_class_names(xml_dict):
    '''
    input:
    xml_list : contating the path each xml files

    output:
    class_names = list with the unique class names
    '''
    all_class_names=[]
    if isinstance(xml_dict, dict):
        for key in xml_dict:
            for xml_file in xml_dict[key]:
                infile_xml = open(xml_file)
                tree = ElementTree.parse(infile_xml)
                root = tree.getroot()
                for index,obj in enumerate(root.iter('object')):
                    cls_name = obj.find('name').text
                    all_class_names.append(cls_name)
    else:                                          # treat as a list
        for xml_file in xml_dict:
                infile_xml = open(xml_file)
                tree = ElementTree.parse(infile_xml)
                root = tree.getroot()
                for index,obj in enumerate(root.iter('object')):
                    cls_name = obj.find('name').text
                    all_class_names.append(cls_name)
    return all_class_names

def xml_to_dict(annotation_path,xml_name,image_path,file_name,dict_in,output_xmls,out_img):
    '''
    input:
    annotation_path : contating the path for annotation xml file
    xml_name: name of the xml file
    image_path: contating the path for image file
    file_name: iamge file name
    img_size: desired img_sizes
    output_xmls : path to save vlaid xmls
    out_img:  path to save vlaid images

    output:
    dict: for each key it holds a list. and each list contains the parameters for bounding box and class name.
    e.g. dict[key]: [cls_name, x_min, y_min, x_max, y_max]
    reshaping: Boolean tag to enable the reshaping of images

    Stores the valid xmls and images to the given output paths

    '''
    img_size =dict_in['img_size']
    infile_xml = open(os.path.join(annotation_path,xml_name))
    tree = ElementTree.parse(infile_xml)
    root = tree.getroot()
    size_node = root.find('size')
    img_W = int(size_node.find('width').text)
    img_H = int(size_node.find('height').text)

    if img_W !=img_size or img_H != img_size:
        scale_x = img_size / img_W
        scale_y = img_size / img_H
        size_node.find('width').text = str(img_size)
        size_node.find('height').text = str(img_size)
        reshaping =True
    else:
        reshaping = False

    dict={}
    temp_list=[]
    if root.find('object')!=None:           #if xml does not contain the trainable object
        for index,obj in enumerate(root.iter('object')):
            cls_name = obj.find('name').text
            if cls_name in dict_in['replace_del']['tag_del']:
                continue
            elif cls_name in dict_in['replace_del']['tag_replacement'].keys():
                cls_name=dict_in['replace_del']['tag_replacement'][cls_name]
            
            for index_2,box in enumerate(obj.iter('bndbox')):
                if not reshaping:
                    x_min = int(box.find('xmin').text)
                    y_min = int(box.find('ymin').text)
                    x_max = int(box.find('xmax').text)
                    y_max = int(box.find('ymax').text)
                    temp_list.append(cls_name)
                    temp_list.append(x_min)
                    temp_list.append(y_min)
                    temp_list.append(x_max)
                    temp_list.append(y_max)
                    dict[(index,index_2)]=temp_list
                    temp_list=[]
                else:
                    #read and scale the bnd box dimensions
                    x_min = round(int(box.find('xmin').text)*scale_x)
                    y_min = round(int(box.find('ymin').text)*scale_y)
                    y_min = round(int(box.find('ymin').text)*scale_y)
                    x_max = round(int(box.find('xmax').text)*scale_x)
                    y_max = round(int(box.find('ymax').text)*scale_y)

                    #store new values
                    box.find('xmin').text = str(x_min)
                    box.find('ymin').text = str(y_min)
                    box.find('xmax').text = str(x_max)
                    box.find('ymax').text = str(y_max)

                    temp_list.append(cls_name)
                    temp_list.append(x_min)
                    temp_list.append(y_min)
                    temp_list.append(x_max)
                    temp_list.append(y_max)

                    dict[(index,index_2)]=temp_list
                    temp_list=[]
        tree.write(os.path.join(output_xmls,xml_name))
        img = cv2.imread(os.path.join(image_path, file_name))
        if reshaping:
            img=cv2.resize(img,(img_size,img_size))
        cv2.imwrite(os.path.join(out_img,file_name),img)
    return dict,reshaping

def addlabels(x,y):
    space =max(y)/55
    for i in range(len(x)):
        plt.text(i-0.25,y[i]+space,y[i])

def create_save_bar_chart(unique_class_names, all_class_names,plot_title,result_dir):
    unique_class_names.append('total')
    cls_count = []
    for cls_name in unique_class_names:
        if cls_name != 'total':
            cls_count.append(all_class_names.count(cls_name))
        else:
            cls_count.append(sum(cls_count))    

    #%% Number of each class labels
    f, ax =plt.subplots()
    ax.bar(unique_class_names,cls_count)
    ax.set_xlabel("Class Names")
    ax.set_ylabel("Total occurances")
    ax.set_title('{}'.format(plot_title))
    addlabels(unique_class_names,cls_count)
    #plt.show()
    f.savefig(os.path.join(result_dir,(plot_title +'.tiff')))
    print(plot_title +' saved to '+ result_dir)


def train_test_split(total_images,train_test_paths,out_img,label_path,output_xmls):
    limit=math.ceil(total_images*0.8)
    x=0
    print('spliting the dataset into train and test datasets...')
    while x <=total_images:
        label_list = os.listdir(label_path)
        if len(label_list)!=0:
            label_name = choice(label_list)
        else:
            break
        img_name =label_name[:-4]+'.jpg'
        xml_name=label_name[:-4]+'.xml'
        original_imgpath=os.path.join(out_img,img_name)
        original_labelpath=os.path.join(label_path,label_name)
        original_xmlpath=os.path.join(output_xmls,xml_name)  
        if os.path.isfile(original_imgpath) and os.path.isfile(original_labelpath) and os.path.isfile(original_xmlpath) :
            if x <=limit:   # for train dataset
                i=1
                j=2
                k=3
            else:           #for test dataset
                i=5
                j=6
                k=7
            shutil.move(original_imgpath,os.path.join(train_test_paths[i],img_name))
            shutil.move(original_labelpath,os.path.join(train_test_paths[j],label_name))
            shutil.move(original_xmlpath,os.path.join(train_test_paths[k],xml_name))
            x=x+1
        else:
            x=x
            total_images-=1
            print('image for ', label_name, ' is not found: selecting next pair')
    print('Train and test datest are ready')

def create_data_representation(data_dir,seperator,plot_title,result_dir,data_dir_replace):
    xml_list, image_list, _,_,_= get_path_lists(data_dir,seperator)
    print('total_annotaitons: ', len(xml_list))
    print('total_images: ', len(image_list))
    all_class_names= get_class_names(xml_list)
    unique_class_names,all_class_names=replace_del_names(all_class_names,data_dir_replace)
    create_save_bar_chart(unique_class_names, all_class_names,plot_title,result_dir)

def create_data_representation(train_test_paths,seperator,plot_title,result_dir,data_dir_replace):
    xml_list, image_list, _,_,_= get_path_lists(train_test_paths,seperator)
    print('total_annotaitons of ',plot_title,' : ', len(xml_list))
    print('total_images in',plot_title,' : ', len(image_list))
    all_class_names= get_class_names(xml_list)
    unique_class_names,all_class_names=replace_del_names(all_class_names,data_dir_replace)
    create_save_bar_chart(unique_class_names, all_class_names,plot_title,result_dir)

def replace_del_names(all_class_names,dict):
    '''
    This function replaces and removes the tags specified in json file
    input:
    unique_class_names
    all_class_names
    dict

    output:
    filtered unique_class_names and all_class_names
    '''
    if len(dict['tag_replacement'])==0 and len(dict['tag_del'])==0:
        pass
    else:
        r=0
        for n,i in enumerate(list(all_class_names)):
            if i in dict['tag_replacement'].keys():
                all_class_names[n-r]=dict['tag_replacement'][i]
            if i in dict['tag_del']:
                all_class_names.remove(i)
                r+=1
    unique_class_names=np.unique(np.array(all_class_names))
    unique_class_names=unique_class_names.tolist()
    return unique_class_names,all_class_names