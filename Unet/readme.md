# Implementing unet in pytorch
# Database
Epithelium database and code from: http://www.andrewjanowczyk.com/deep-learning/

# Training
* trained using google VM on Kesla T80 took ~30 minutes for 100 epochs
* Best test loss 0.1131
* Training parameters can be seen in train-2.py

# Results
My trained model can be found in the .pth file. 
Attached are some screenshots of the output, showing the input image, output, mask, and also a series of images representing different layer activations 

Top row is from the training dataset, the mask, and mask weights. The bottom row shows the output, output mask, actual mask, and the input image from the validation set. 
![Screenshot](https://github.com/awoloshuk/IUSM/blob/master/Unet/output.png)

This image shows a validation set image and the activation of unet layers
![Screenshot](https://github.com/awoloshuk/IUSM/blob/master/Unet/activations.png)
