# author: xosval03
# email: xosval03@fit.vutbr.cz
"""
This script generate binary mask from blender renderer

"""
import os
import cv2
import numpy as np

from config_global import TEETH_DATASET_PATH

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # 'M'

for teeth in os.scandir(TEETH_DATASET_PATH):
    for type_of_teeth in os.scandir(teeth):
        if type_of_teeth.name in types:
            abs_path_of_folder_type_of_teeth = os.path.join(type_of_teeth.path, 'output-mask')
            for image in os.scandir(abs_path_of_folder_type_of_teeth):

                mask_image = cv2.imread(image.path)
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)


                #red
                lower_red = np.array([0, 50, 50])
                upper_red = np.array([10, 255, 255])
                mask0 = cv2.inRange(mask_image, lower_red, upper_red)

                # upper mask (170-180)
                lower_red = np.array([170, 50, 50])
                upper_red = np.array([180, 255, 255])
                mask1 = cv2.inRange(mask_image, lower_red, upper_red)
                mask_red = mask0 + mask1

                path_red = os.path.join(type_of_teeth.path+'_left', 'binary_mask')

                try:
                    os.makedirs(path_red)
                except OSError as error:
                    pass
                    #print(error)

                cv2.imwrite(os.path.join(path_red,image.name), mask_red)



                #blue
                lower_blue = np.array([111,189,195])
                upper_blue = np.array([149,255,226])
                mask_blue = cv2.inRange(mask_image, lower_blue, upper_blue)
                path_blue = os.path.join(type_of_teeth.path+'_right', 'binary_mask')
                try:
                    os.makedirs(path_blue)
                except OSError as error:
                    pass
                    #print(error)

                cv2.imwrite(os.path.join(path_blue,image.name), mask_blue)
