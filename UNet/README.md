# UNet 

![UNet](/unet.png)

## Description
This subproject focuses on implementing the UNet model from scratch. UNet is a convolutional neural network designed for biomedical image segmentation. Its architecture is built upon the fully convolutional network, and it was developed with the purpose of performing precise segmentation of medical images.

## Key Features
- **From Scratch Implementation**: Understanding and building the UNet model from the ground up.
- **Customizable Architecture**: Easily modify the number of layers and filters to suit your specific needs.
- **Preprocessing and Augmentation**: Includes comprehensive scripts for preprocessing and augmenting medical image datasets.
- **Performance Metrics**: Integrated metrics for evaluating model performance, including Dice coefficient and Intersection over Union (IoU).

## Model Summary
```
Model: "UNet"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 128, 128, 1) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 128, 128, 64) 640         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 128, 128, 64) 36928       conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 64, 64, 64)   0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, 64, 128)  73856       max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 64, 128)  147584      conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 32, 32, 128)  0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 256)  295168      max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 256)  590080      conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 16, 16, 256)  0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 16, 16, 512)  1180160     max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 16, 16, 512)  2359808     conv2d_6[0][0]                   
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 32, 32, 512)  0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 256)  1179904     up_sampling2d[0][0]              
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 256)  590080      conv2d_8[0][0]                   
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 64, 64, 256)  0           conv2d_9[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 64, 64, 128)  295040      up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 64, 64, 128)  147584      conv2d_10[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 128, 128, 128 0           conv2d_11[0][0]                  
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 128, 128, 64) 73792       up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 128, 128, 64) 36928       conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 128, 128, 1)  65          conv2d_13[0][0]                  
==================================================================================================
Total params: 7,759,297
Trainable params: 7,759,297
Non-trainable params: 0
__________________________________________________________________________________________________
```

## Installation
 Clone the repository:
    ```bash
    git clone https://github.com/silvaxxx1/PAPER2CODE.git
    cd PAPER2CODE/UNet
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The UNet architecture is based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
- Thanks to all contributors and the open-source community for their valuable input and support.
