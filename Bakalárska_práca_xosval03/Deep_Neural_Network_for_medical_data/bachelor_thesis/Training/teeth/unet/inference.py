
import argparse
import logging
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from bachelor_thesis.Training.teeth.unet.dataset  import TeethDataset
from bachelor_thesis.Training.teeth.unet.config  import UNetConfig

cfg = UNetConfig()

def our_preprocess(pil_img, scale):

    print("here is a problem")
    w, h = pil_img.size
    print("here is a problem 2")
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))
    print("here is a problem 3")

    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        # mask target image
        img_nd = np.expand_dims(img_nd, axis=2)
    else:
        # grayscale input image
        # scale between 0 and 1
        img_nd = img_nd / 255
    # HWC to CHW
    print('I am here')
    img_trans = img_nd.transpose((2, 0, 1))
    return img_trans.astype(float)

def inference_one(net, image, device):
    net.eval()
    print(cfg.scale)
    a = our_preprocess(image, cfg.scale)
    print(a)
    img = torch.from_numpy(a)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if cfg.deepsupervision:
            output = output[-1]

        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image.size[1], image.size[0])),
                transforms.ToTensor()
            ]
        )
        
        if cfg.n_classes == 1:
            probs = tf(probs.cpu())
            mask = probs.squeeze().cpu().numpy()
            return mask > cfg.out_threshold
        else:
            masks = []
            for prob in probs:
                prob = tf(prob.cpu())
                mask = prob.squeeze().cpu().numpy()
                mask = mask > cfg.out_threshold
                masks.append(mask)
            return masks
  
  
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', dest='input', type=str, default='',
                        help='Directory of input images')
    parser.add_argument('--output', '-o', dest='output', type=str, default='',
                        help='Directory of ouput images')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    input_imgs = os.listdir(args.input)

    net = eval(cfg.model)(cfg)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, img_name in tqdm(enumerate(input_imgs)):
        logging.info("\nPredicting image {} ...".format(img_name))

        img_path = osp.join(args.input, img_name)
        img = Image.open(img_path).convert('RGB').resize((256,256))

        mask = inference_one(net=net,
                             image=img,
                             device=device)
        img_name_no_ext = osp.splitext(img_name)[0]
        output_img_dir = osp.join(args.output, img_name_no_ext)
        os.makedirs(output_img_dir, exist_ok=True)

        if cfg.n_classes == 1:
            image_idx = Image.fromarray((mask * 255).astype(np.uint8))
            image_idx.save(osp.join(output_img_dir, img_name))
        else:
            for idx in range(0, len(mask)):
                img_name_idx = img_name_no_ext + "_" + str(idx) + ".png"
                image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
                image_idx.save(osp.join(output_img_dir, img_name_idx))
