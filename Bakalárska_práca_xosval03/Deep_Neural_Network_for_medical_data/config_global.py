import torch,os

#----------------------------------------------------------------------------------------------
RENDERS = "D:\\Renders\\RenderAnnotationVer30"
ABS_PATH_OF_PROJECT = "C:\\Users\\mosva\\Documents\\Code\\bachelor_trainning\\pytorch_google_colab\\bachelor_thesis"

#----------------------------------------------------------------------------------------------
TEETH_DATASET_PATH = os.path.join(RENDERS,"teeth-mask")
LANDMARK_DATASETH_PATH = os.path.join(RENDERS,"landmark-mask")

#----------------------------------------------------------------------------------------------
teeth_class = [
'A_left','B_left','C_left','D_left', 'E_left', 'F_left', 'G_left','H_left',
'A_right' ,'B_right' ,'C_right', 'D_right', 'E_right', 'F_right', 'G_right','H_right'
]

landmark_class = ['A_left_1', 'B_left_1', 'C_left_1', 'D_left_1', 'E_left_1', 'F_left_1', 'G_left_1', 'H_left_1',
                  'A_left_2', 'B_left_2', 'C_left_2', 'D_left_2', 'E_left_2', 'F_left_2', 'G_left_2', 'H_left_2',
                  'A_right_1', 'B_right_1', 'C_right_1', 'D_right_1', 'E_right_1', 'F_right_1', 'G_right_1', 'H_right_1',
                  'A_right_2', 'B_right_2', 'C_right_2', 'D_right_2', 'E_right_2', 'F_right_2', 'G_right_2', 'H_right_2']

#----------------------------------------------------------------------------------------------
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



SIZE = (256)
TUPLE_SIZE = (256,256)

MASK_RCNN_TEETH_IMAGE_SIZE = (416,416)
YOLO_TEETH_IMAGE_SIZE = SIZE
UNET_TEETH_IMAGE_SIZE = TUPLE_SIZE
MASK_RCNN_LANDMARK_IMAGE_SIZE = TUPLE_SIZE
YOLO_LANDMARK_IMAGE_SIZE = SIZE
UNET_LANDMARK_IMAGE_SIZE = (300,300)


#----------------------------------------------------------------------------------------------
MASK_RCNN_TEETH_EPOCH = 4
MASK_RCNN_LANDMARK_EPOCH = 20
YOLO_TEETH_EPOCH = 2
YOLO_LANDMARK_EPOCH = 2
UNET_TEETH_EPOCH = 100
UNET_LANDMARK_EPOCH = 80
#----------------------------------------------------------------------------------------------

MASK_RCNN_TEETH_BATCH = 1
MASK_RCNN_LANDMARK_BATCH = 2
YOLO_TEETH_BATCH =  4
YOLO_LANDMARK_BATCH = 4
# at least 2
UNET_TEETH_BATCH = 2
UNET_LANDMARK_BATCH = 2
#----------------------------------------------------------------------------------------------

MASK_RCNN_TEETH_WORKER = 2
MASK_RCNN_LANDMARK_WORKER = 2
YOLO_TEETH_WORKER =  4
YOLO_LANDMARK_WORKER = 4

UNET_TEETH_WORKER = 4
UNET_LANDMARK_WORKER = 2

#----------------------------------------------------------------------------------------------
MODELS_CHECKPOINT = os.path.join(ABS_PATH_OF_PROJECT,"models_checkpoint")
MODELS_CHECKPOINT_TEETH = os.path.join(MODELS_CHECKPOINT,"teeth")
MODELS_CHECKPOINT_LANDMARK = os.path.join(MODELS_CHECKPOINT,"landmark")

MODEL_PATH_MASK_RCNN_TEETH = os.path.join(MODELS_CHECKPOINT_TEETH,'mask-rcnn')
MODEL_PATH_MASK_RCNN_LANDMARK = os.path.join(MODELS_CHECKPOINT_LANDMARK,'mask-rcnn')
MODEL_PATH_YOLO_TEETH = os.path.join(MODELS_CHECKPOINT_TEETH,'yolo')
MODEL_PATH_YOLO_LANDMARK = os.path.join(MODELS_CHECKPOINT_LANDMARK,'yolo')

MODEL_PATH_UNET_TEETH = os.path.join(MODELS_CHECKPOINT_TEETH,'unet')
MODEL_PATH_UNET_LANDMARK = os.path.join(MODELS_CHECKPOINT_LANDMARK,'unet')

#----------------------------------------------------------------------------------------------
TENSORBOARD_PATH = os.path.join(ABS_PATH_OF_PROJECT,'TensorBoard')

TENSORBOARD_LANDMARK_PATH = os.path.join(TENSORBOARD_PATH,'landmark')
TENSORBOARD_TEETH_PATH = os.path.join(TENSORBOARD_PATH,'teeth')


TENSORBOARD_LANDMARK_UNET_PATH = os.path.join(TENSORBOARD_LANDMARK_PATH,'unet')
TENSORBOARD_LANDMARK_YOLO_PATH = os.path.join(TENSORBOARD_LANDMARK_PATH,'yolo')
TENSORBOARD_LANDMARK_MASKRCNN_PATH = os.path.join(TENSORBOARD_LANDMARK_PATH,'mask-rcnn')

TENSORBOARD_TEETH_UNET_PATH = os.path.join(TENSORBOARD_TEETH_PATH,'unet')
TENSORBOARD_TEETH_YOLO_PATH = os.path.join(TENSORBOARD_TEETH_PATH,'yolo')
TENSORBOARD_TEETH_MASKRCNN_PATH = os.path.join(TENSORBOARD_TEETH_PATH,'mask-rcnn')



#----------------------------------------------------------------------------------------------
IMAGE_FOLDER_PATH = os.path.join(ABS_PATH_OF_PROJECT, 'images')
IMAGE_FOLDER_PATH_TEETH = os.path.join(IMAGE_FOLDER_PATH,'teeth')
IMAGE_FOLDER_PATH_LANDMARK = os.path.join(IMAGE_FOLDER_PATH,'landmark')
IMAGE_FOLDER_PATH_MASK_RCNN_TEETH = os.path.join(IMAGE_FOLDER_PATH_TEETH,'mask-rcnn')
IMAGE_FOLDER_PATH_MASK_RCNN_LANDMARK = os.path.join(IMAGE_FOLDER_PATH_LANDMARK,'mask-rcnn')
IMAGE_FOLDER_PATH_MASK_RCNN_TEETH = os.path.join(IMAGE_FOLDER_PATH_TEETH,'mask-rcnn')
IMAGE_FOLDER_PATH_MASK_RCNN_LANDMARK = os.path.join(IMAGE_FOLDER_PATH_LANDMARK,'mask-rcnn')
IMAGE_FOLDER_PATH_YOLO_TEETH = os.path.join(IMAGE_FOLDER_PATH_TEETH,'yolo')
IMAGE_FOLDER_PATH_YOLO_LANDMARK = os.path.join(IMAGE_FOLDER_PATH_LANDMARK,'yolo')
IMAGE_FOLDER_PATH_UNET_TEETH = os.path.join(IMAGE_FOLDER_PATH_TEETH,'unet')
IMAGE_FOLDER_PATH_UNET_LANDMARK = os.path.join(IMAGE_FOLDER_PATH_LANDMARK,'unet')
#----------------------------------------------------------------------------------------------
YOLO_PLOT_DURRING_TRAINNING_FLAG = False

