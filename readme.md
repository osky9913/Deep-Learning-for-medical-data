There are 6 experimental NN. Mask-RCNN, YOLOv3. Unet each for object and for point detections. In this project is also a app
for comparation between this NN.      




How to run this project

1. You need to be on os windows 10 because of  https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. After downloading the c++ build tools choose the first option for installing ( C++ build tools)
3. Install it 
4. Restart PC ( because of yml in next step )
5. Download anaconda program from https://www.anaconda.com/products/individual and install it
6.a Run on terminal in root directory where is enviroment.yml

   "conda env create -f  enviroment.yml"

6.b if there will be a problem to install pycocotools 
 run this pip install pycocotools

conda env update --file pytorch_google_colab.yml --prune

7. also pip install tqdm
pip install -U albumentations
   
8. In config_global.py change the abs path for RENDERS and ABS_PATH_OF_PROJECT



![Unet][imgs/unet_2.PNG]
