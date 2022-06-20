import os
import random

import config_global
from config_global import TEETH_DATASET_PATH
from config_global import teeth_class,landmark_class
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


model_name = "model223"

models = [os.path.join(model.path,'input')  for model in os.scandir(TEETH_DATASET_PATH) if model.name == model_name]
X =  []

for model in models:

    images = os.scandir(model)
    for image in images:
        #if "x0y0z0.png" in image.name :
        X.append(image.path)

number_of_plot = 20
size = (512,512)
for plot in range(len(X)):
    #selected_input_path = X[random.randint(0, len(X))]
    masks_paths = []
    masks_landmark_paths = []
    for t_class in teeth_class:
        mask_path = X[plot].replace('input', t_class+os.sep+'binary_mask')
        if os.path.exists(mask_path):
            masks_paths.append(mask_path)

    for t_class in landmark_class:
        mask_path = X[plot].replace('input', t_class+os.sep+'binary_mask')
        mask_path = mask_path.replace('teeth-mask', 'landmark-mask')
        if os.path.exists(mask_path):
            masks_landmark_paths.append(mask_path)
    print(masks_landmark_paths)
    fig, ax = plt.subplots(1,3)
    #ax[0].imshow(Image.open(X[plot]).resize(size))
    mask_multiple_class = np.zeros(size)
    mask_singe_class = np.zeros(size)
    Image_teeth = np.array(Image.open(X[plot]).convert('RGB').resize(size))
    Image_masked = np.array(Image.open(X[plot]).convert('RGB').resize(size))
    Image_masked_landmark = np.array(Image.open(X[plot]).convert('RGB').resize(size))

    color_index = 0
    class_labels = config_global.teeth_class
    cmap = plt.get_cmap("tab20b")

    colors = [cmap(j) for j in np.linspace(0, 1, len(class_labels))]
    print(colors[0][0:3])
    for i in masks_paths:
        class_of_teeth =  i.split('\\')[5]
        print(teeth_class.index(class_of_teeth) + 1 )
        mask_of_teeth = np.array(Image.open(i).resize(size))
        mask_singe_class[mask_of_teeth != 0 ] = 1 #
        mask_multiple_class[mask_of_teeth != 0] = teeth_class.index(class_of_teeth) + 1
        select_color = colors[color_index][0:3]
        list_select_color = []
        for index in range(len(select_color)):
             list_select_color.append( select_color[index] *180)
        Image_masked[mask_of_teeth != 0]  =  np.array(list_select_color,dtype='uint8')
        color_index += 1

    color_index = 0
    class_labels = config_global.landmark_class
    cmap = plt.get_cmap("tab20b")

    colors = [cmap(j) for j in np.linspace(0, 1, len(class_labels))]
    print(colors[0][0:3])
    for i in masks_landmark_paths:
        print(i)
        class_of_teeth =  i.split('\\')[5]

        print(landmark_class.index(class_of_teeth) + 1 )
        mask_of_teeth = np.array(Image.open(i).resize(size))
        mask_singe_class[mask_of_teeth != 0 ] = 1 #
        mask_multiple_class[mask_of_teeth != 0] = landmark_class.index(class_of_teeth) + 1
        select_color = colors[color_index][0:3]
        list_select_color = []
        for index in range(len(select_color)):
             list_select_color.append( select_color[index] *180)
        Image_masked_landmark[mask_of_teeth != 0] = np.array(list_select_color,dtype='uint8')
        color_index += 1

    Image_masked = Image.fromarray(Image_masked)
    #ax[1].imshow(mask_singe_class)
    #ax[2].imshow(mask_multiple_class)
    ax[0].imshow(Image_teeth)
    ax[1].imshow(Image_masked)
    ax[2].imshow(Image_masked_landmark)

    #ax.set_axis_off()
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()


    plt.show()
print(X)
