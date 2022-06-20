import os
import random

from config_global import TEETH_DATASET_PATH
from config_global import teeth_class



import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model_name = "model223"

models = [os.path.join(model.path,'input') for model in os.scandir(TEETH_DATASET_PATH) if model.name == model_name ]
X = []

for model in models:
    images = os.scandir(model)
    for image in images:
        #if image.name == "model223x0y0z0.png":
            X.append(image.path)

number_of_plot = 6
for plot in range(number_of_plot):
    selected_input_path = X[random.randint(0, len(X))]
    masks_path = []
    for t_class in teeth_class:
        masks_path.append(selected_input_path.replace('input', t_class + os.sep + 'binary_mask'))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(plt.imread(selected_input_path))
    mask_multiple_class = np.zeros((500, 500))
    mask_single_class = np.zeros((500, 500))

    for i in masks_path:
        if os.path.exists(i):
            class_of_teeth = i.split('\\')[5]
            print(teeth_class.index(class_of_teeth) + 1)
            mask_of_teeth = np.array(Image.open(i))
            mask_multiple_class[mask_of_teeth != 0] = teeth_class.index(class_of_teeth) + 1
            mask_single_class[mask_of_teeth != 0] = 1
    ax[1].imshow(mask_multiple_class)
    #ax[2].imshow(mask_single_class)
    ax[0].set_axis_off()
    #ax[2].set_axis_off()
    ax[1].set_axis_off()
    plt.show()
#print(X)