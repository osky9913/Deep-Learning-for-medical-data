# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uiOfApp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import time

from PyQt5 import QtCore, QtGui, QtWidgets
import qdarkstyle

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

from bachelor_thesis.Training.landmark.unet.inference import inference_one
from bachelor_thesis.Training.landmark.unet.train import cfg
from bachelor_thesis.Training.landmark.unet.utils.colors import get_colors
from bachelor_thesis.Training.landmark.unet.unet import UNet# dont delete

net = eval(cfg.model)(cfg)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = eval(cfg.model)(cfg)
net.to(device=device)
net.load_state_dict(torch.load("C:\\Users\\mosva\\Documents\\Code\\bachelor_trainning\\pytorch_google_colab\\bachelor_thesis\\selected_checkpoints\\landmark\\unet\\epoch_6.pth", map_location=device))
print("loaded")








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
        # --------------- INPUT --------------------

        teeth_screenshot = Image.fromarray(np.array(vpl.screenshot_fig()).copy())
        img = teeth_screenshot.convert('RGB').resize((256, 256))
        mask = inference_one(net=net,
                             image=img,
                             device=device)

        colors = get_colors(n_classes=cfg.n_classes)
        w, h = img.size
        img_mask = np.zeros([h, w, 3], np.uint8)
        for idx in range(0, len(mask)):
            image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
            array_img = np.asarray(image_idx)
            img_mask[np.where(array_img == 255)] = colors[idx]
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
        output = cv2.addWeighted(img, 0.7, img_mask, 0.3, 0)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(output).resize((480,480))

        self.widget_2.setPixmap(
            QtGui.QPixmap.fromImage(
                ImageQt(im_pil)
            )
        )
       


        # -----------------MASK-RCNN-----------------------------------------------

        # ----------------YoloV3 -------------------
        """
        img = np.array(teeth_screenshot_yolo_input)
        augmentations = test_transforms(image=img)
        img = augmentations["image"]
        show_predict_yolo( img,teeth_screenshot_yolo_input_draw, 0.6, 0.5)
        teeth_screenshot_yolo_input = teeth_screenshot_yolo_input.resize((250,250))
        self.widget_2.setPixmap(
            QtGui.QPixmap.fromImage(
                ImageQt(teeth_screenshot_yolo_input)
            )
        )
        """



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

        """
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(1230, 240, 91, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        """
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(255, 580, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.released.connect(self.analyze)

    

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)



    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        """
        self.groupBox.setTitle(_translate("Dialog", "Option"))
        self.checkBox.setText(_translate("Dialog", "Mask-Rcnn Masks"))
        self.checkBox_2.setText(_translate("Dialog", "Mask-RCNN  Boxes ( red )"))
        self.checkBox_5.setText(_translate("Dialog", " Yolo Boxes ( green )"))
        self.checkBox_4.setText(_translate("Dialog", "Unet Landmarks per teeth"))
        self.checkBox_3.setText(_translate("Dialog", "Unet Landmarks"))
        """
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


