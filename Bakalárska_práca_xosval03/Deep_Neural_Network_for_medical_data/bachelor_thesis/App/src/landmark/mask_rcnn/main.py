# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uiOfApp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import time

from PyQt5 import QtCore, QtGui, QtWidgets
import qdarkstyle
from matplotlib import pyplot as plt

from stl.mesh import Mesh
from PIL.ImageQt import ImageQt

from torchvision import transforms
from PIL import ImageDraw
import vtkplotlib as vpl

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL  import  Image
import numpy as np


import torch

import config_global


class FigureAndButton(QtWidgets.QWidget):
    def __init__(self,parent = None):

        QtWidgets.QWidget.__init__(self, parent)

        # Go for a vertical stack layout.
        vbox = QtWidgets.QVBoxLayout(parent)
        self.setLayout(vbox)

        # Create the figure
        self.figure = vpl.QtFigure()

        # Create a button and attach a callback.
        self.button = QtWidgets.QPushButton("Select a teeth")
        self.button.released.connect(self.button_pressed_cb)

        # QtFigures are QWidgets and are added to layouts with `addWidget`
        vbox.addWidget(self.figure)
        vbox.addWidget(self.button)


    def button_pressed_cb(self):
        """Plot commands can be called in callbacks. The current working
        figure is still self.figure and will remain so until a new
        figure is created explicitly. So the ``fig=self.figure``
        arguments below aren't necessary but are recommended for
        larger, more complex scenarios.
        """

        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Teeth files (*.stl)")[0].replace('/','\\')
        print(fname)


        mesh = Mesh.from_file(fname)

        # Plot the mesh
        vpl.mesh_plot(mesh,color='silver')
        fig = vpl.gcf()
        fig.background_color = "black"
        

        # Randomly place a ball.


        # Reposition the camera to better fit to the balls.
        vpl.reset_camera(self.figure)

        # Without this the figure will not redraw unless you click on it.
        self.figure.update()


    def show(self):
        # The order of these two are interchangeable.
        super().show()
        self.figure.show()


    def closeEvent(self, event):
        """This isn't essential. VTK, OpenGL, Qt and Python's garbage
        collect all get in the way of each other so that VTK can't
        clean up properly which causes an annoying VTK error window to
        pop up. Explicitly calling QtFigure's `closeEvent()` ensures
        everything gets deleted in the right order.
        """
        self.figure.closeEvent(event)




class Ui_Dialog(object):


    def analyze(self):
        mask_rcnn_path = "C:\\Users\\mosva\\Documents\\Code\\bachelor_trainning\\pytorch_google_colab\\bachelor_thesis\\selected_checkpoints\\landmark\\mask-rcnn\\mask-rcnn-model-all0.pt"
        # --------------- INPUT --------------------

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        teeth_screenshot = Image.fromarray(np.array(vpl.screenshot_fig()).copy())
        loader = transforms.Compose([transforms.ToTensor()])

        # --------------- INPUT -------------------- 

        # --------------- Resizing for Neural Network -------------------- 

        teeth_screenshot_mask_rcnn_input = teeth_screenshot.resize((256,256))

        # --------------- Resizing for Neural Network --------------------

        # -----------------MASK-RCNN-----------------------------------------------

        class_labels = config_global.landmark_class
        cmap = plt.get_cmap("tab20b")

        colors = [cmap(j) for j in np.linspace(0, 1, len(class_labels))]
        modelMaskRcnn = torch.load(mask_rcnn_path)
        modelMaskRcnn.eval()


        teeth_screenshot_mask_rcnn_input_tensor = loader(teeth_screenshot_mask_rcnn_input)
        prediction_mask_rcnn = modelMaskRcnn([teeth_screenshot_mask_rcnn_input_tensor.to(device)])
        teeth_screenshot_mask_rcnn_input_numpy = np.array(teeth_screenshot_mask_rcnn_input)

        scores = prediction_mask_rcnn[0]['scores'].cpu()
        boxes = prediction_mask_rcnn[0]['boxes'].byte().cpu()
        print(prediction_mask_rcnn[0])
        labels = prediction_mask_rcnn[0]['labels'].cpu()

        for i in range(len(prediction_mask_rcnn[0]['masks'])):
        # iterate over masks
            #
            if float(scores[i]) > 0.5:
                selected_color = colors[int(labels[i])][0:3]
                selected_color = [ color*255 for color in selected_color]
                mask = prediction_mask_rcnn[0]['masks'][i, 0]
                mask = mask.mul(255).byte().cpu().numpy()
                teeth_screenshot_mask_rcnn_input_numpy[ mask > 1] = np.array(selected_color,dtype='uint8')

        mask_rcnn_out = Image.fromarray(teeth_screenshot_mask_rcnn_input_numpy)
        image_draw_rect = ImageDraw.Draw(mask_rcnn_out)

        for box_index in range(len(boxes)):
            if float(scores[box_index]) > 0.5:
                selected_color = colors[int(labels[i])][0:3]
                selected_color = [color * 255 for color in selected_color]
                x1,y1,x2,y2 = boxes[box_index]
                image_draw_rect.rectangle([x1,y1,x2,y2],fill=None,width=1,outline="red")

        mask_rcnn_rezised_out = Image.fromarray(np.array(mask_rcnn_out).copy()).resize((450,450))
        
        self.widget_2.setPixmap(
            QtGui.QPixmap.fromImage(
                ImageQt(mask_rcnn_rezised_out)
            )
        )
       
        time.sleep(4)
        del  image_draw_rect, mask_rcnn_out, teeth_screenshot_mask_rcnn_input_numpy, prediction_mask_rcnn, teeth_screenshot_mask_rcnn_input_tensor , modelMaskRcnn

        # -----------------MASK-RCNN-----------------------------------------------




    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1448, 728)

        self.widget = FigureAndButton(Dialog)
        self.widget.setGeometry(QtCore.QRect(40, 60, 512, 512))
        self.widget_2 = QtWidgets.QLabel(Dialog)
        self.widget_2.setGeometry(QtCore.QRect(620, 46, 512, 512))
        self.widget_2.setObjectName("widget_2")

        self.widget_3 = QtWidgets.QLabel(Dialog)
        self.widget_3.setGeometry(QtCore.QRect(1120, 76, 256, 256))
        self.widget_3.setObjectName("widget_3")
        self.widget_3.setPixmap(
            QtGui.QPixmap.fromImage(
                ImageQt(Image.open("C:\\Users\\mosva\\Documents\\Code\\bachelor_trainning\\pytorch_google_colab\\bachelor_thesis\\App\\src\\app.PNG").resize((286,256)))
            )
        )

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(50, 30, 47, 13))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(630, 30, 131, 16))
        self.label_2.setObjectName("label_2")

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(255, 580, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.released.connect(self.analyze)

    

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)



    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Analyze"))
        self.label.setText(_translate("Dialog", "Teeth"))
        self.label_2.setText(_translate("Dialog", "Teeth analyzed"))
       # self.pushButton_2.setText(_translate("Dialog", "Select  a teeth"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())

    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())


