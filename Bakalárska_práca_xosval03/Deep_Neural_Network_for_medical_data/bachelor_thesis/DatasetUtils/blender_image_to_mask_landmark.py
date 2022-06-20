# author: xosval03
# email: xosval03@fit.vutbr.cz
"""
This script generate binary mask from blender renderer

"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config_global import LANDMARK_DATASETH_PATH

def return_gaussian(pts):
    sigma = 3
    peaks_img = np.zeros((500, 500))
    peaks_img[np.int_(pts[:, 1]), np.int_(pts[:, 0])] = 1
    density_img = cv2.GaussianBlur(peaks_img, (0,0), sigma)
    return density_img

types = ['A_left', 'B_left', 'C_left', 'D_left', 'E_left', 'F_left', 'G_left',
         'A_right', 'B_right', 'C_right', 'D_right', 'E_right', 'F_right', 'G_right']
bad_teeth = []
for teeth in os.scandir(LANDMARK_DATASETH_PATH):
    for type_of_teeth in os.scandir(teeth):
        if type_of_teeth.name in types:
            print(type_of_teeth.name)
            abs_path_of_folder_type_of_teeth = os.path.join(type_of_teeth.path, 'output-mask')
            for image in os.scandir(abs_path_of_folder_type_of_teeth):

                index_error_flag = False


                mask_image = cv2.imread(image.path)
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)


                #red
                lower_red = np.array([0, 50, 50])
                upper_red = np.array([10, 255, 255])
                mask0 = cv2.inRange(mask_image, lower_red, upper_red)

                lower_red = np.array([170, 50, 50])
                upper_red = np.array([180, 255, 255])
                mask1 = cv2.inRange(mask_image, lower_red, upper_red)
                mask_red = mask0 + mask1

                #cv2.imshow('helloWorld', mask_red)




                path_red = os.path.join(type_of_teeth.path+'_1','binary_mask')
                try:
                    os.makedirs(path_red)
                except OSError as error:
                    pass
                path_red = os.path.join(path_red,image.name)

                """
                try:
                    contours, hierarchy = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    M = cv2.moments(contours[0])
                except IndexError:
                    index_error_flag = True

                if( index_error_flag == False  ):
                    try :
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        location = np.array([[cX,cY]])
                        image_mask = return_gaussian(location)
                        plt.imsave(path_red,image_mask, cmap='Greys_r')
                    except ZeroDivisionError  :
                        print(path_red)
                        bad_teeth.append(path_red)
                        cv2.imwrite(path_red, cv2.blur(mask_red, (10, 10)))
                else:
                
                    print(path_red)
                    bad_teeth.append(path_red)
                """
                cv2.imwrite(path_red, cv2.blur(mask_red, (10, 10)))
                   # cv2.imwrite(path_red, mask_red)



                index_error_flag = False




                #blue
                lower_blue = np.array([111, 189, 195])
                upper_blue = np.array([149, 255, 226])
                mask_blue = cv2.inRange(mask_image, lower_blue, upper_blue)


                path_blue = os.path.join(type_of_teeth.path + '_2', 'binary_mask')

                try:
                    os.makedirs(path_blue)
                except OSError as error:
                    pass
                path_blue = os.path.join(path_blue, image.name)


                """
                try:
                    contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    M = cv2.moments(contours[0])
                except IndexError:
                    index_error_flag = True


                if index_error_flag == False :
                    try:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        location = np.array([[cX, cY]])
                        image_mask = return_gaussian(location)
                        plt.imsave(path_blue,image_mask, cmap='Greys_r')
                    except ZeroDivisionError :
                        bad_teeth.append(path_blue)
                        print(path_blue)
                        cv2.imwrite(path_blue, cv2.blur(mask_red, (10, 10)))
                else:

                    bad_teeth.append(path_blue)
                    print(path_blue)
                    """
                cv2.imwrite(path_blue, cv2.blur(mask_blue, (10, 10)))
                #    cv2.imwrite(path_blue, mask_blue)


print(bad_teeth)
