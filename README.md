## Platform
The codes are meant to be run **only in Google Colab**.

## Topic 1: Binary Image Classification Cats vs Dogs

## Modifications Made

- Replaced manually uploaded dataset with TensorFlow inbuilt Cats vs Dogs dataset (TFDS)
- Added Batch Normalization layers to stabilize training
- Added Dropout layer to reduce overfitting
- Optimized data pipeline using caching and prefetching for faster execution
- Included single-image testing after training
- Generated a clean CNN architecture diagram for report visualization

## Topic 2: Graph-Based Q-Learning for Autonomous Robot Navigation

## Modifications Made

- Changed the application scenario from crime investigation to warehouse robot navigation  
- Reinterpreted graph nodes and edges to represent warehouse locations and paths  
- Replaced police locations with hazard zones  
- Replaced drug trace locations with charging stations  
- Corrected optimal path output formatting for clarity  
- Retained all visualizations and training convergence plots

## Topic 3: Time Series Forecasting Using LSTM (Cherry Blossom Bloom Prediction)

### Modifications Made

- Replaced external CSV dataset with TensorFlow Datasets inbuilt *Cherry Blossoms* dataset  
- Changed use case from airline passenger forecasting to environmental time-series prediction  
- Handled missing values in real-world dataset by removing invalid records  
- Adapted data preprocessing pipeline for TFDS-based time-series input  
- Applied Min-Max normalization suitable for seasonal numerical data

# Topic 4: Character-Level Text Generation using SimpleRNN (Optimized Version)

##Modifications Made

### 1. Improved Training Stability
- Input preprocessing is streamlined using NumPy arrays and TensorFlow one-hot encoding.
- Ensured consistent sequence lengths during both training and inference to avoid runtime shape errors.

### 2. Faster Convergence
- Optimized the number of RNN units to balance learning capacity and speed.
- Reduced unnecessary computations during text generation by limiting prediction overhead.

### 3. Better Output Consistency
- Used a different but structurally richer example sentence to improve character transition learning.
- Ensured generated text strictly follows the learned character vocabulary.

### 4. Robust Text Generation
- Added safe character-to-index mapping during inference to prevent invalid indexing.
- Maintained deterministic prediction using greedy decoding for reproducibility.




