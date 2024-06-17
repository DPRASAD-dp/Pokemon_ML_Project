
# Pokémon Legendary Status Prediction

## Project Overview

This repository contains two main components:
1. Exploratory Data Analysis (EDA) of the Pokémon dataset
2. Predictive modeling to determine a Pokémon's legendary status

The goal of this project is to understand the characteristics that define legendary Pokémon and build a high-performance model to predict whether a given Pokémon is legendary or not.

## Dataset

The dataset used in this project contains information about various Pokémon, including their attributes such as HP, Attack, Defense, type, and whether they are legendary or not. The original dataset suffers from class imbalance, with significantly fewer legendary Pokémon compared to non-legendary ones.

## Part 1: Exploratory Data Analysis (EDA)

### Files
- `Pokemon_EDA_Code.ipynb`: Jupyter notebook containing the EDA code and visualizations
- `Pokemon.csv`: The original Pokémon dataset

### Key Features
- Comprehensive analysis of Pokémon attributes
- Visualization of distributions and relationships between features
- Investigation of legendary vs. non-legendary Pokémon characteristics
- Correlation analysis and identification of key relationships (e.g., inverse relationship between base total and capture rate)
- Handling of missing values and anomalies

### Findings
- Legendary Pokémon generally have higher base stats (HP, Attack, Defense, etc.)
- There's a strong correlation between a Pokémon's base egg steps and its legendary status
- The dataset is highly imbalanced, with legendary Pokémon being the minority class

## Part 2: Predictive Modeling

### Files
- `Pokemon_Predictor.ipynb`: Python script for data preprocessing, model training, and evaluation
- `Pokemon_resampled.csv`: The resampled dataset after addressing class imbalance

### Key Features
- Data resampling using ADASYN (oversampling) and NearMiss (undersampling) to address class imbalance
- Ensemble of Logistic Regression and Long Short-Term Memory (LSTM) neural network
- Attention mechanism for dynamic weighting of model predictions
- Comprehensive evaluation using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC

### Model Performance
- Accuracy: 98.25%
- AUC-ROC: 0.999
- Precision: 99.19%
- Recall: 98.80%
- F1-score: 98.99%

## Requirements
- Python 3.7+
- Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, TensorFlow, Keras, imbalanced-learn

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pokemon-legendary-prediction.git
   cd pokemon-legendary-prediction
   ```

2. Run the EDA:
   ```
   jupyter notebook Pokemon_EDA_Code.ipynb
   ```

4. Run the predictive model:
   ```
   python Pokemon_Predictor_Code.ipynb
   ```

## Future Work
- Investigate model interpretability to understand feature importance
- Test the model on new, unseen data for generalization
- Experiment with additional feature engineering
- Deploy the model as a web application or API

## Conclusion

This project demonstrates a comprehensive approach to binary classification on an imbalanced dataset. By combining thorough exploratory data analysis with advanced machine learning techniques, we achieved high-performance predictions of a Pokémon's legendary status. The ensemble model, with its near-perfect AUC-ROC score, proves to be highly effective in discriminating between legendary and non-legendary Pokémon.

## Author
DURGA PRASAD KAVALI

## License
This project is open-sourced under the [MIT license](LICENSE).
```
