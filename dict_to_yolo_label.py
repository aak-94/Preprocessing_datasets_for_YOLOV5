import os

def dict_to_yolo_label(file,dict,img_size,unique_class_names,label_path):
    '''
    converts the dict in to a yolo lable

    input:
    file: input image name e.g. xyz.jpg
    dict: for each key it holds a list. and each list contains the parameters for bounding box and class name.
    e.g. dict[key]: [cls_name, x_min, y_min, x_max, y_max]
    img_size: desired shape of the images
    label_path: path to store the yolo label

    output:
    sotres the yolo txt label to the result directory
    '''

    # Theoretical Notes
    '''
    POC_label_format = [xmin, ymin,xmax,ymax]
    YOLO format = [x_center, Y_center,width,hight]
    image_width =600
    image_height =600
    '''

    df=[]
    for key in dict:
        values=dict[key]
        decimal_limit=3
        x_center=round(((values[1]+values[3])/2)/img_size,decimal_limit)
        y_center=round(((values[1]+values[3])/2)/img_size,decimal_limit)
        width =round((values[3]-values[1])/img_size,decimal_limit)
        height=round((values[4]-values[2])/img_size,decimal_limit)
        index_id = unique_class_names.index(values[0])  #get the index of the label
        row = [str(index_id),' ', str(x_center),' ', str(y_center),' ', str(width),' ', str(height)]
        df.append(row)
    name = file[:-4]+'.txt'
    txt_file = open(os.path.join(label_path,name),'w')  # mode W: create file if does not exist else overwrite new data
    for i in range(len(df)):
        txt_file.writelines(df[i])
        txt_file.write('\n')
    txt_file.close()