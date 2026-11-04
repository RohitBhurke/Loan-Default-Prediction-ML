
# Loan Default Prediction Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop a machine learning-based loan default prediction system. Five classification algorithms are implemented and evaluated to identify their performance in predicting loan default risk.

## 2. Dataset Description

- Dataset source: Kaggle
- Total instances: 45,000
- Total columns: 14
- Input features: 13
- Target variable: loan_status
- Problem type: Binary Classification

The dataset contains applicant demographic information, income, employment experience, loan details, credit history and previous loan defaults.

Target classes:
- 0: No Default
- 1: Default

## 3. GitHub Repository Link

https://github.com/RohitBhurke/Loan-Default-Prediction-ML

## 4. Models and Evaluation Metrics

The following five classification algorithms were implemented:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8993 | 0.9562 | 0.7888 | 0.7470 | 0.7673 | 0.7036 |
| Decision Tree | 0.8999 | 0.8558 | 0.7738 | 0.7765 | 0.7751 | 0.7108 |
| KNN | 0.8956 | 0.9246 | 0.7909 | 0.7205 | 0.7541 | 0.6891 |
| Naive Bayes | 0.7356 | 0.9398 | 0.4566 | 0.9990 | 0.6267 | 0.5484 |
| SVM | 0.8994 | 0.9562 | 0.7889 | 0.7475 | 0.7677 | 0.7040 |

## 5. Model Performance Observations

### Logistic Regression
Achieved 89.93% accuracy and an AUC score of 0.9562, demonstrating strong classification and discrimination performance.

### Decision Tree
Achieved the highest accuracy of 89.99%, F1 score of 0.7751 and MCC of 0.7108 among the five models.

### K-Nearest Neighbors
Achieved 89.56% accuracy and the highest precision of 0.7909, but its recall was comparatively lower.

### Naive Bayes
Achieved very high recall of 0.9990 but relatively low precision of 0.4566, indicating a high number of false-positive predictions.

### Support Vector Machine
Achieved 89.94% accuracy and an AUC score of 0.9562, matching Logistic Regression on AUC.

## 6. Overall Best Model

Decision Tree is the overall best-performing model based on accuracy, F1 score and Matthews Correlation Coefficient.

Logistic Regression and SVM achieved the highest AUC score of 0.9562.

## 7. Project Files

- app.py: Streamlit application
- requirements.txt: Required Python libraries
- test_data.csv: Labelled test dataset
- model/: Saved machine learning pipelines
- README.md: Project documentation

## 8. Streamlit Application

Live application link: https://loan-default-prediction-ml2.streamlit.app
