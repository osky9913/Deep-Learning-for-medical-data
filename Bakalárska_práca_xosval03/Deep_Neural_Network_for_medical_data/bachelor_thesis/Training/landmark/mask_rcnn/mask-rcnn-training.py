# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import torchvision
from torch.utils.data import random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import matplotlib

matplotlib.use( 'tkagg' )

import os
import numpy as np
import torch
import torch.utils.data
from config_global import LANDMARK_DATASETH_PATH,\
    landmark_class,\
    IMAGE_FOLDER_PATH_MASK_RCNN_LANDMARK,\
    MODEL_PATH_MASK_RCNN_LANDMARK,\
    TENSORBOARD_LANDMARK_MASKRCNN_PATH, MASK_RCNN_LANDMARK_EPOCH
from torch.utils.tensorboard import SummaryWriter

MODEL_PATH = MODEL_PATH_MASK_RCNN_LANDMARK



class TeethDataset(object):
    def __init__(self, imgs, transforms ):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs
        self.transforms = transforms
        

    def __getitem__(self, idx):
        # load images and masks
        size = (256,256)
        img_path = self.imgs[idx]
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB").resize(size)
        #img = img.thumbnail(size,Image.ANTIALIAS)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        img_path = self.imgs[idx]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        image = Image.open(img_path).convert("RGB").resize(size) #.resize(size)
        mask = np.zeros(size)


        #obj_ids = []
        for index in range(len(landmark_class)):
            mask_path = img_path.replace('input', landmark_class[index] + os.sep + 'binary_mask')
            if os.path.exists(mask_path):
                mask_bin = np.array(Image.open(mask_path).convert('L').resize(size))
                mask_bin_fixed = np.where(mask_bin != 0, 1, 0)
                mask[mask_bin_fixed == 1] = index+1
                #obj_ids.append(index+1)

        mask = mask
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin == xmax:
                xmax += 0.5
            if ymin == ymax:
                ymax += 0.5

            boxes.append([xmin, ymin,xmax,ymax ])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        #labels = torch.ones((num_objs,), dtype=torch.int64)

        #there is a multiple classes
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)


        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target['iscrowd'] = iscrowd


        img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,progress=True,num_classes=2,pretrained_backbone=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform():
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)



def visualize(model,device,dataset):
    import matplotlib.pyplot as plt

    for  item  in range(len(dataset)):
        print(item)
        model.eval()
        image = dataset[item][0].mul(255).permute(1, 2, 0).byte().numpy()

        with torch.no_grad():
            
            prediction = model([dataset[item][0].to(device)])

        image1 = Image.fromarray(image)
        image2 = Image.fromarray(image)
        image3 = np.array(Image.fromarray(image))
        image4 = np.array(Image.fromarray(image))
        image5 = np.array(Image.fromarray(image))

    
        info = dataset[item][1]
        boxes = info['boxes'].numpy()
        masks = info['masks'].mul(255).byte().cpu().numpy()

        fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, figsize=(8,6))
        plt.tight_layout()

        for mask in masks:    
            image3[ mask > 1] =  np.array([0,0,255],dtype='uint8')
            image4[ mask > 1] +=  np.array([0,0,60],dtype='uint8')

        for i in range(len(prediction[0]['masks'])):
        # iterate over masks
            mask = prediction[0]['masks'][i, 0]
            mask = mask.mul(255).byte().cpu().numpy()
            image5[ mask > 1] +=  np.array([60,0,0],dtype='uint8')


        image3 = Image.fromarray(image3)
        image4 = Image.fromarray(image4)
        image5 = Image.fromarray(image5)


        image_draw2 = ImageDraw.Draw(image2)
        image_draw4 = ImageDraw.Draw(image4)
        image_draw5 = ImageDraw.Draw(image5)


        for box in boxes:
            x1,y1,x2,y2 =box
            image_draw2.rectangle([x1,y1,x2,y2],fill=None,width=1,outline="blue")
            image_draw4.rectangle([x1,y1,x2,y2],fill=None,width=1,outline="blue")

        for box in prediction[0]['boxes'].byte().cpu().numpy():
            x1,y1,x2,y2 = box
            image_draw5.rectangle([x1,y1,x2,y2],fill=None,width=1,outline="red")

        ax1.imshow(image)
        ax2.imshow(image2)
        ax3.imshow(image3)
        ax4.imshow(image4)
        ax5.imshow(image5)
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH_MASK_RCNN_LANDMARK, 'scatter_%s.png' %  str(item)))
        plt.close('all') 
        del image1, image2, image3, image4, image5,info,boxes,masks

def main(imgs):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has teeth_class plus background
    num_classes = len(landmark_class)+1
    # use our dataset and defined transformations






    dataset = TeethDataset(imgs ,get_transform())
    val_percent = 0.3
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    print(n_val,n_train)

    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-350])
    #dataset_test = torch.utils.data.Subset(dataset_test, indices[-350:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 3 epochs
    num_epochs = MASK_RCNN_LANDMARK_EPOCH
    step = 0
    writer = SummaryWriter(log_dir=TENSORBOARD_LANDMARK_MASKRCNN_PATH)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        _, step_temp = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,writer=writer,step=step)
        step = step_temp
        torch.save(model.state_dict(), os.path.join(MODEL_PATH,'mask-rcnn-model' + str(epoch) +'.pt'))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model,os.path.join(MODEL_PATH,'mask-rcnn-model-all' + str(epoch) +'.pt'))
    writer.close()

    return model,device,dataset



if __name__ == '__main__':
    from numba import cuda 
    from PIL import Image, ImageDraw

    cudagpu = cuda.get_current_device()
    cudagpu.reset()



    imgs = []
    root = LANDMARK_DATASETH_PATH
    scandirs = [ os.scandir(teeth_id_input) for teeth_id_input in [ os.path.join(teeth,'input') for teeth in [teeth_id.path for teeth_id in os.scandir(root)]]]
    for scandir in scandirs:
        for rotation in scandir:
            imgs.append(rotation.path)
    imgs.sort()

    model,device,dataset = main(imgs)
    visualize(model,device,dataset)