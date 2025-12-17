## Platform
The codes are meant to be run **only in Google Colab**.

## Topic 1: Binary Image Classification Cats vs Dogs

### Modifications Made

- Replaced manually uploaded dataset with TensorFlow inbuilt Cats vs Dogs dataset (TFDS)
- Added Batch Normalization layers to stabilize training
- Added Dropout layer to reduce overfitting
- Optimized data pipeline using caching and prefetching for faster execution
- Included single-image testing after training
- Generated a clean CNN architecture diagram for report visualization

## Topic 2: Graph-Based Q-Learning for Autonomous Robot Navigation

### Modifications Made

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

## Topic 4: Character-Level Text Generation Using Simple RNN

### Modifications Made

- Used a richer and more descriptive example sentence to improve character learning patterns  
- Optimized sequence length and RNN units for faster convergence  
- Improved input sequence handling to prevent shape mismatch errors  
- Enhanced training stability without altering the original RNN logic  
- Maintained character-level modeling while improving output coherence  




