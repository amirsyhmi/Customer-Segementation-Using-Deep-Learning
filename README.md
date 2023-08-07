# Customer-Segementation-Using-Deep-Learning
Develop deep learning model using TensorFlow keras

## Objective
To develop a classifier model by using TensorFlow keras to identity the group of customer.
- Model Training: `Deep Learning`
- Method: `Squential`
- Module: `Tensorflow`

The dataset that used in this analysis can be obtain from [[Customer Segmentation
Multiclass Classification]](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## About the Dataset:
There datasets that used in this analysis
- customer_segmentation.csv(8067 entries data with 11 columns)

Our target is on the 11th columns, which name as `Segmentation`. The dataset will be used to develop the deeplearnng model.

## Results
- Model Summary

<p align="center">
  <img src="Images/model summary.jpg" alt="Model Summary">
</p>

The model consist of 3 dense layer and 4 output one for each classes.

- Model Architecture
- 
<p align="center">
  <img src="Images/model.png" alt="Model">
</p>

The model was called to train with an epochs of 100 and stop with the used of `early stopping` at 34 epochs. The accuracy and the loss are as follows:

<p align="center">
  <img src="Images/tensorboard.png" alt="TensorBoard">
</p>

## Model Evaluation Results
<p align="center">
  <img src="Images/model evaluation.jpg" alt="TensorBoard">
</p>

## Conclusion
For this project, I only manage to get Accuracy of 52%. Some improvement neeed to be made to get better results in the future. :(



