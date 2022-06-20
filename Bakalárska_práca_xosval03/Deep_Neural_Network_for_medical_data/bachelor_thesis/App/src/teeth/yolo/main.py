# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uiOfApp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import time

from PyQt5 import QtCore, QtGui, QtWidgets
import qdarkstyle
from matplotlib import pyplot as plt, patches

from stl.mesh import Mesh
from PIL.ImageQt import ImageQt
from torch import optim

from torchvision import transforms
from PIL import ImageDraw
import vtkplotlib as vpl

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL  import  Image
import numpy as np


import torch

from bachelor_thesis.Training.teeth.yolo import config
from bachelor_thesis.Training.teeth.yolo.model import YOLOv3
from bachelor_thesis.Training.teeth.yolo.utils import cells_to_bboxes, non_max_suppression

IMAGE_SIZE = 512
NUM_CLASSES = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
DEVICE = config.DEVICE
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

yolo_v3_path = "C:\\Users\\mosva\Documents\\Code\\bachelor_trainning\\pytorch_google_colab\\bachelor_thesis\\models_checkpoint\\teeth\\yolo\\checkpoint_67pth.tr"
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)

def box_convertor(image, boxes):
    print(type(image),print(image))
    im = image
    print(im.shape)
    height, width, _ = im.shape
    pil_boxes = []
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        pil_box = [upper_left_x * width,
                   upper_left_y * height,
                   upper_left_x * width + box[2] * width,
                   upper_left_y * height + box[3] * height]
        pil_boxes.append((pil_box, class_pred))
    return pil_boxes
    """
    print("hello", pil_boxes)
    img_pil = Image.fromarray(image)
    print("world")
    img_pil_draw = ImageDraw.Draw(img_pil)
    print("Let's")

    for box in pil_box:
        print(box)
        img_pil_draw.rectangle(box,fill=None,width=1,outline="yellow")
    return img_pil
    """


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

load_checkpoint(yolo_v3_path, model, optimizer, config.LEARNING_RATE)
model.eval()
with torch.no_grad():
    scaled_anchors = (
            torch.tensor(ANCHORS)
            * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(DEVICE)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.TEETH_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        print(class_pred)
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    ax.axis('off')
    plt.show()
    #fig.tight_layout(pad=0)

    # To remove the huge white borders
    #ax.margins(0)

    #fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot


def show_predict_yolo(img, img_draw, thresh, iou_thresh ):

    print("preditcting")
    img_tensor = img.unsqueeze(0).to(DEVICE)
    print("still dont exist ")

    with torch.no_grad():
        print("still exist 1")
        try:
            out = model(img_tensor)
        except:
            e = sys.exc_info()
            print(e)
        print("still predict exist 1")

        bboxes = [[] for _ in range(img_tensor.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box


    #display image
    print("still exist")
    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )

    image = img_tensor[0].permute(1,2,0).detach().cpu()
    #numpy_image = image.numpy()
    image = plot_image(image, nms_boxes)
    return image
    #boxes = box_convertor(numpy_image, nms_boxes)


    #print("Let's")

    #for box in boxes:
    #    print(box)
    #    img_draw.rectangle(box[0],fill=None,width=1,outline="")






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

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        teeth_screenshot = Image.fromarray(np.array(vpl.screenshot_fig()).copy())
        loader = transforms.Compose([transforms.ToTensor()])

        # --------------- INPUT -------------------- 

        # --------------- Resizing for Neural Network -------------------- 

        teeth_screenshot_yolo_input = teeth_screenshot.resize((512,512))
        teeth_screenshot_yolo_input_draw = ImageDraw.Draw(teeth_screenshot_yolo_input)



        # ----------------YoloV3 -------------------

        img = np.array(teeth_screenshot_yolo_input)
        augmentations = test_transforms(image=img)
        img = augmentations["image"]
        try:
            show_predict_yolo( img,teeth_screenshot_yolo_input_draw, 0.6, 0.5)
        except:
            print(sys.exc_info())
        teeth_screenshot_yolo_input = teeth_screenshot_yolo_input.resize((250,250))
        self.widget_2.setPixmap(
            QtGui.QPixmap.fromImage(
                ImageQt(teeth_screenshot_yolo_input)
            )
        )




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


