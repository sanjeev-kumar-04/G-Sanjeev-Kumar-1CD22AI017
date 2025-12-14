## Platform
The codes are meant to be run **only in Google Colab**.

## Topic 1: Binary Image Classification

In this topic, I modified the  image classification code to work with a different dataset while keeping the same model structure and logic.

## Dataset Used
I used the **Malaria dataset from TensorFlow Datasets (TFDS)**, which contains two classes:
- Parasitized
- Uninfected

This dataset is directly available in Colab and does not require manual downloading or extraction.

## What I Changed
- Removed ZIP file download and extraction used in the original code  
- Removed manual directory-based dataset loading  
- Replaced it with direct dataset loading from TFDS  
- Kept the CNN architecture and training logic unchanged  
- Did not use advanced techniques or extra libraries 

## Output
- Training and validation accuracy
- Training and validation loss
- Prediction on a sample image
- Feature map visualization of CNN layers
