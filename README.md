# Diabetes Prediction using PCA and Random Forest

A machine learning notebook that predicts diabetes from patient health data using a custom **Principal Component Analysis (PCA)** implementation for dimensionality reduction and a **Random Forest Classifier** for classification. Built in Python and designed to run on Google Colab.

## Dataset

- **Source:** [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (Kaggle)
- **Features:** gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level
- **Target:** `diabetes` (binary — 0 or 1)

## Pipeline

### 1. Preprocessing
- Rows with missing values dropped
- Samples with gender `Other` removed
- Categorical features (`gender`, `smoking_history`) label-encoded to integers
- Features normalised using `MinMaxScaler`

### 2. Class Balancing
The dataset is heavily imbalanced (far more non-diabetic than diabetic samples). **Random Under-Sampling** is applied to balance the classes before training.

### 3. Dimensionality Reduction (Custom PCA)
A PCA implementation built from scratch using NumPy:
- Mean-centres the data
- Computes the covariance matrix
- Extracts and sorts eigenvectors by eigenvalue magnitude
- Projects the data onto the top **6 principal components**

### 4. Classification
A **Random Forest Classifier** (100 estimators) is trained on the PCA-reduced data with an 80/20 train-test split. Model performance is evaluated using accuracy score, classification report, and a normalised confusion matrix.

## Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
mlxtend
seaborn
matplotlib
opendatasets
```

## Usage

1. Open the notebook in Google Colab
2. Run the install cell (`!pip install opendatasets`)
3. Provide your Kaggle API credentials when prompted by `opendatasets`
4. Run cells sequentially
