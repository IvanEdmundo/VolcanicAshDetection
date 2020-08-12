# VolcanicAshDetection
## Master Thesis Project

This repository shows the code that I used in my Master Thesis Project. In this work, neural networks have been used for the identification of volcanic ash clouds through multispectral measurements of the Moderate Resolution Imaging Spectroradiometer (MODIS). Different neural networks were constructed for each parameter to be recovered, experimenting with different topologies and evaluating their performance.

In this case Segnet model and MLP model were used to make the classification.

In the repository you can see 2 folders, one for MLP and the other one for Segnet. Each folder has all the code that I used to implement each model. I will now explain the contents of the files

### MLP Folder
 
 * loadMLP-- Load the data, makes normalization and one hot encoding
 * Undersampling-- Use undersampling method in order to balanced the data
 * OverSampling-- Use overampling method in order to balanced the data
 * SMOTE--- Use SMOTE method in order to balanced the data
 * traingen -- Make the model and train it
 * trainover -- Make the model and train it using the data with oversampling method
 * trainsmot -- Make the model and train it using the data with SMOTE method
 * trainunder -- Make the model and train it using the data with undersampling method
 * MLPvisualizacion-- Testing the trained model
 
 ### Segnet
 * load_v4-- Load the data, makes normalization and one hot encoding
 * build_modelHT-- Define the model architecture 
 * SegNet-Basic -- Training the model and save it
 * Segnet-Evaluation-Visualization-- Testing the model
