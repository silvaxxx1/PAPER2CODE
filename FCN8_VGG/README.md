
# FC8_VGG

![FC8_VGG](C:/Users/USER/SILVA/PAPER2CODE/FC8_VGG/images/FCN_8.png)

## Description
This subproject implements the FC8 VGG model from scratch, based on the original research paper. The model is designed for image segmentation and classification tasks, utilizing the pre_trained VGG16 architecture with additional layers for fine-tuning.

## Dataset 
the dataset [custom dataset](https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing) is provided by [divamgupta](https://github.com/divamgupta/image-segmentation-keras) . This contains video frames from a moving vehicle and is a subsample of the [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset. 

![Sample of the dataset](C:/Users/USER/SILVA/PAPER2CODE/FC8_VGG/images/img1.png)

## Model summary 


Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                )]                                                                
                                                                                                  
 block_1_conv1 (Conv2D)         (None, 224, 224, 64  1792        ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 block_1_conv2 (Conv2D)         (None, 224, 224, 64  36928       ['block_1_conv1[0][0]']          
                                )                                                                 
                                                                                                  
 block_1_pool2 (MaxPooling2D)   (None, 112, 112, 64  0           ['block_1_conv2[0][0]']          
                                )                                                                 
                                                                                                  
 block_2_conv1 (Conv2D)         (None, 112, 112, 12  73856       ['block_1_pool2[0][0]']          
                                8)                                                                
                                                                                                  
 block_2_conv2 (Conv2D)         (None, 112, 112, 12  147584      ['block_2_conv1[0][0]']          
                                8)                                                                
                                                                                                  
 block_2_pool2 (MaxPooling2D)   (None, 56, 56, 128)  0           ['block_2_conv2[0][0]']          
                                                                                                  
 block_3_conv1 (Conv2D)         (None, 56, 56, 256)  295168      ['block_2_pool2[0][0]']          
                                                                                                  
 block_3_conv2 (Conv2D)         (None, 56, 56, 256)  590080      ['block_3_conv1[0][0]']          
                                                                                                  
 block_3_conv3 (Conv2D)         (None, 56, 56, 256)  590080      ['block_3_conv2[0][0]']          
                                                                                                  
 block_3_pool3 (MaxPooling2D)   (None, 28, 28, 256)  0           ['block_3_conv3[0][0]']          
                                                                                                  
 block_4_conv1 (Conv2D)         (None, 28, 28, 512)  1180160     ['block_3_pool3[0][0]']          
                                                                                                  
 block_4_conv2 (Conv2D)         (None, 28, 28, 512)  2359808     ['block_4_conv1[0][0]']          
                                                                                                  
 block_4_conv3 (Conv2D)         (None, 28, 28, 512)  2359808     ['block_4_conv2[0][0]']          
                                                                                                  
 block_4_pool3 (MaxPooling2D)   (None, 14, 14, 512)  0           ['block_4_conv3[0][0]']          
                                                                                                  
 block_5_conv1 (Conv2D)         (None, 14, 14, 512)  2359808     ['block_4_pool3[0][0]']          
                                                                                                  
 block_5_conv2 (Conv2D)         (None, 14, 14, 512)  2359808     ['block_5_conv1[0][0]']          
                                                                                                  
 block_5_conv3 (Conv2D)         (None, 14, 14, 512)  2359808     ['block_5_conv2[0][0]']          
                                                                                                  
 block_5_pool3 (MaxPooling2D)   (None, 7, 7, 512)    0           ['block_5_conv3[0][0]']          
                                                                                                  
 conv6 (Conv2D)                 (None, 7, 7, 4096)   102764544   ['block_5_pool3[0][0]']          
                                                                                                  
 conv7 (Conv2D)                 (None, 7, 7, 4096)   16781312    ['conv6[0][0]']                  
                                                                                                  
 conv2d_transpose (Conv2DTransp  (None, 16, 16, 12)  786432      ['conv7[0][0]']                  
 ose)                                                                                             
                                                                                                  
 cropping2d (Cropping2D)        (None, 14, 14, 12)   0           ['conv2d_transpose[0][0]']       
                                                                                                  
 conv2d (Conv2D)                (None, 14, 14, 12)   6156        ['block_4_pool3[0][0]']          
                                                                                                  
 add (Add)                      (None, 14, 14, 12)   0           ['cropping2d[0][0]',             
                                                                  'conv2d[0][0]']                 
                                                                                                  
 conv2d_transpose_1 (Conv2DTran  (None, 30, 30, 12)  2304        ['add[0][0]']                    
 spose)                                                                                           
                                                                                                  
 cropping2d_1 (Cropping2D)      (None, 28, 28, 12)   0           ['conv2d_transpose_1[0][0]']     
                                                                                                  
 conv2d_1 (Conv2D)              (None, 28, 28, 12)   3084        ['block_3_pool3[0][0]']          
                                                                                                  
 add_1 (Add)                    (None, 28, 28, 12)   0           ['cropping2d_1[0][0]',           
                                                                  'conv2d_1[0][0]']               
                                                                                                  
 conv2d_transpose_2 (Conv2DTran  (None, 224, 224, 12  9216       ['add_1[0][0]']                  
 spose)                         )                                                                 
                                                                                                  
 activation (Activation)        (None, 224, 224, 12  0           ['conv2d_transpose_2[0][0]']     
                                )                                                                 
                                                                                                  
==================================================================================================
Total params: 135,067,736
Trainable params: 135,067,736
Non-trainable params: 0
__________________________________________________________________________________________________


### Loading Pretrained Weights
To load pretrained VGG16 weights, use:
```bash
python MODEL/load_weights.py
```
### Training the Model
To train the model, run the following command:
```bash
python MODEL/train.py
```

### Running Inference
To run inference on test images, use:
```bash
python main.py
```

## Structure
- `MODEL/`: Contains the model architecture, training script, and pretrained weights.
- `UTILS/`: Utility scripts for metrics and visualization.
- `data/`: Scripts and data for training and testing.
- `images/`: Contains images of the output.




