# ML Classification Models - Breast Cancer Dataset

## Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting breast cancer diagnosis (malignant vs. benign) using diagnostic measurements. The goal is to identify which classification algorithm provides the best performance in distinguishing between benign and malignant tumors, thereby assisting in medical diagnosis and treatment planning.

---

## Dataset Description

### Dataset Information
- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / scikit-learn datasets
- **Instances:** 569 samples
- **Features:** 30 numerical features
- **Target:** Binary Classification (0 = Benign, 1 = Malignant)
- **Class Distribution:** 
  - Benign (0): 357 samples (62.7%)
  - Malignant (1): 212 samples (37.3%)

### Features Explanation

Each sample contains 10 real-valued features computed for each cell nucleus:

1. **Radius** - Mean distance from center to points on the perimeter
2. **Texture** - Standard deviation of gray-scale values
3. **Perimeter** - Outer boundary length
4. **Area** - Cell nucleus area
5. **Smoothness** - Local variation in radius lengths
6. **Compactness** - Perimeter² / area - 1.0
7. **Concavity** - Severity of concave portions of the contour
8. **Concave Points** - Number of concave portions of the contour
9. **Symmetry** - Symmetry of the nucleus
10. **Fractal Dimension** - "Coastline approximation" - 1

For each of these 10 features, three statistical measures are computed:
- **Mean** values (10 features)
- **Standard Error (SE)** values (10 features)
- **Worst (Largest)** values (10 features)

This results in a total of **30 features** per sample.

### Data Characteristics
- No missing values
- All features are numerical (continuous)
- Features are in different scales (scaling is required)
- Class imbalance is present (62.7% vs 37.3%)
- Well-established binary classification problem

---

## Models Used

### 1. Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | 0.9825 |
| AUC | 0.9954 |
| Precision | 0.9861 |
| Recall | 0.9861 |
| F1 Score | 0.9861 |
| MCC | 0.9623 |

**Observation about model performance:**
Logistic Regression provides the best overall performance on this dataset with excellent results across all metrics. Being a linear classifier, it learns a clear decision boundary between benign and malignant cases. Its high AUC score (0.9954) indicates excellent discrimination ability across all classification thresholds. The perfect balance of precision and recall (both 0.9861) makes it highly reliable for clinical applications. The model benefits from feature scaling and demonstrates excellent generalization capability.

### 2. Decision Tree Classifier

| Metric | Score |
|--------|-------|
| Accuracy | 0.9123 |
| AUC | 0.9157 |
| Precision | 0.9559 |
| Recall | 0.9028 |
| F1 Score | 0.9286 |
| MCC | 0.8174 |

**Observation about model performance:**
The Decision Tree (with max_depth=10) shows the lowest performance among all models. While it provides interpretable decision rules, it fails to capture the complex relationships in the data as effectively as other algorithms. The lower AUC (0.9157) indicates moderate discrimination ability. The gap between precision and recall suggests potential bias in predictions. The controlled tree depth helps prevent severe overfitting, but it also limits the model's ability to learn complex patterns from the 30 features.

### 3. K-Nearest Neighbors Classifier

| Metric | Score |
|--------|-------|
| Accuracy | 0.9561 |
| AUC | 0.9788 |
| Precision | 0.9589 |
| Recall | 0.9722 |
| F1 Score | 0.9655 |
| MCC | 0.9054 |

**Observation about model performance:**
KNN (k=5) is an instance-based lazy learner that makes predictions based on the nearest neighbors in the feature space. It achieves solid performance with balanced precision and recall. The high AUC (0.9788) demonstrates good discrimination ability across thresholds. The higher recall (0.9722) means it effectively captures most malignant cases, which is critical for medical applications. However, performance is sensitive to feature scaling and the choice of k. For large-scale deployments, KNN becomes computationally expensive due to its need to compute distances to all training samples.

### 4. Naive Bayes Classifier (Gaussian)

| Metric | Score |
|--------|-------|
| Accuracy | 0.9298 |
| AUC | 0.9868 |
| Precision | 0.9444 |
| Recall | 0.9444 |
| F1 Score | 0.9444 |
| MCC | 0.8492 |

**Observation about model performance:**
Gaussian Naive Bayes assumes feature independence and normal distribution, which are simplifying assumptions given the correlated features in this dataset. Despite these assumptions, it achieves excellent AUC score (0.9868), suggesting the class-conditional probability distributions are highly separable. The perfect balance of precision and recall (both 0.9444) indicates well-calibrated predictions. The model trains very quickly and requires minimal hyperparameter tuning, making it efficient for practical deployment. However, the independence assumption limits its ability to fully exploit feature correlations.

### 5. Random Forest Classifier (Ensemble)

| Metric | Score |
|--------|-------|
| Accuracy | 0.9561 |
| AUC | 0.9939 |
| Precision | 0.9589 |
| Recall | 0.9722 |
| F1 Score | 0.9655 |
| MCC | 0.9054 |

**Observation about model performance:**
Random Forest (100 trees) provides excellent performance through ensemble voting. The outstanding AUC score (0.9939) demonstrates superior discrimination ability. The high recall (0.9722) ensures most malignant cases are identified, critical for medical applications. The model effectively reduces overfitting through bagging and feature randomness. It provides feature importance rankings, enabling identification of the most discriminative features for clinical focus. The ensemble approach captures various non-linear patterns from different subsets of data and features, resulting in robust and stable predictions.

### 6. XGBoost Classifier (Ensemble)

| Metric | Score |
|--------|-------|
| Accuracy | 0.9561 |
| AUC | 0.9901 |
| Precision | 0.9467 |
| Recall | 0.9861 |
| F1 Score | 0.9660 |
| MCC | 0.9058 |

**Observation about model performance:**
XGBoost (100 estimators) achieves excellent performance with outstanding AUC score (0.9901) indicating near-perfect class separation capability. The model uses gradient boosting with sequential error correction, allowing each tree to focus on previously misclassified samples. The highest recall (0.9861) is especially important for medical applications as it minimizes false negatives (missing malignant cases). This makes XGBoost highly reliable for clinical diagnosis where missing malignant cases is more costly than false positives. While computationally more expensive during training, it provides highly reliable predictions.

---

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9561 | 0.9901 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

---

## Model Performance Analysis

### Key Findings

1. **Top Performers:** Logistic Regression and Naive Bayes achieve the highest AUC scores (0.9954 and 0.9868 respectively), showing excellent discrimination ability.

2. **Best Accuracy:** Logistic Regression leads with 98.25% accuracy, providing the most reliable overall predictions for binary classification.

3. **High Recall Leaders:** XGBoost has the highest recall (0.9861), ensuring minimal false negatives—critical for medical diagnosis where missing malignant cases is dangerous.

4. **Ensemble Methods:** Random Forest and XGBoost demonstrate the power of ensemble learning with high AUC scores (0.9939 and 0.9901), though they tie with KNN on accuracy.

5. **Balanced Predictions:** Naive Bayes and Logistic Regression show perfect balance between precision and recall, indicating well-calibrated classifiers.

6. **Decision Tree Limitation:** Decision Tree significantly underperforms with lowest AUC (0.9157), highlighting the limitations of single trees for this complex dataset.

7. **Consistency:** Most ensemble and sophisticated models cluster around 95.61% accuracy, showing robust performance when properly implemented.

### Critical Recall Metric:** XGBoost's recall of 0.9861 means it catches 98.61% of malignant cases, minimizing dangerous false negatives—the most critical metric in cancer diagnosis.
- **High Precision Advantage:** Logistic Regression's precision of 0.9861 reduces unnecessary interventions due to false positives.
- **Safe Baseline:** Naive Bayes offers a computationally efficient alternative with a strong AUC (0.9868), suitable for resource-constrained settings.
- **Practical Recommendation:** **Logistic Regression** is recommended for deployment due to its highest accuracy (0.9825), perfect precision-recall balance, and interpretability for clinical decisions. **XGBoost** is recommended as an alternative if maximizing sensitivity (recall) for identifying malignant cases is the prioritys.
- **Low False Positive Rate:** Logistic Regression's precision of 0.9811 indicates fewer unnecessary biopsies due to false positives.
- **Practical Recommendation:** XGBoost is recommended for deployment due to its superior recall and overall metrics in a medical context.

---

## Project Structure

```
MLAssignment/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── notebooks/
│   └── ML_Classification_Models.ipynb  # Complete training notebook
├── models/                             # Saved model files
│   ├── logistic_regression.pkl        # Logistic Regression model
│   ├── decision_tree.pkl              # Decision Tree model
│   ├── knn.pkl                        # KNN model
│   ├── naive_bayes.pkl                # Naive Bayes model
│   ├── random_forest.pkl              # Random Forest model
│   ├── xgboost.pkl                    # XGBoost model
│   ├── scaler.pkl                     # StandardScaler for feature scaling
│   └── results.pkl                    # Evaluation metrics results
├── data/                              # Dataset directory
└── model_comparison.png               # Performance comparison visualization
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MLAssignment.git
   cd MLAssignment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook (for model training/analysis):**
   ```bash
   jupyter notebook notebooks/ML_Classification_Models.ipynb
   ```

4. **Run the Streamlit app (locally):**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501`

---

## Streamlit Application Features

The deployed Streamlit app includes:

### 1. **Model Metrics Dashboard**
   - Display of all evaluation metrics for each model
   - Metric cards showing Accuracy, Precision, Recall, F1, AUC, and MCC
   - Detailed comparison table

### 2. **Performance Comparison**
   - Side-by-side accuracy and AUC comparison charts
   - Heatmap visualization of all metrics across all models
   - Easy identification of top-performing models

### 3. **Data Upload & Prediction**
   - CSV file upload capability for test data
   - Real-time predictions using selected model
   - Probability estimates for both classes
   - Download predictions as CSV

### 4. **Dataset Information**
   - Detailed description of the Breast Cancer dataset
   - Feature explanations
   - Model descriptions
   - Evaluation metrics definitions

### 5. **Interactive Model Selection**
   - Dropdown menu to select any of the 6 models
   - Dynamically updates all displays based on selection

---

## Deployment on Streamlit Cloud

### Steps to Deploy:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Visit Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create New App:**
   - Click "New app"
   - Select your repository
   - Choose branch: `main`
   - Set path to: `app.py`
   - Click "Deploy"

4. **Access Your App:**
   - Your app will be available at: `https://[username]-mlassignment.streamlit.app`

---

## Technologies Used

- **Machine Learning:** scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Data Source:** UCI Machine Learning Repository

---

## Performance Summary

| Metric | Best Model | Score |
|--------|-----------|-------|
| **Accuracy** | Logistic Regression | 0.9825 |
| **AUC** | Logistic Regression | 0.9954 |
| **Precision** | Logistic Regression | 0.9861 |
| **Recall** | XGBoost | 0.9861 |
| **F1 Score** | Logistic Regression | 0.9861 |
| **MCC** | Logistic Regression | 0.9623 |

---

## Conclusion

This project demonstrates that multiple machine learning approaches can achieve excellent performance on the Breast Cancer classification task. Logistic Regression emerges as the top overall performer with the best accuracy (0.9825), AUC (0.9954), and balanced metrics across all evaluation criteria. It combines high performance with model simplicity and interpretability, making it ideal for clinical deployment.

The comprehensive comparison of six different algorithms reveals distinct strengths:
- **Logistic Regression**: Best all-around performer with perfect precision-recall balance
- **XGBoost**: Highest recall (0.9861) for minimizing false negatives
- **Naive Bayes**: Strong AUC (0.9868) with minimal computational overhead
- **Random Forest**: Excellent AUC (0.9939) with feature importance insights
- **KNN**: Solid performance (0.9561) but computationally expensive
- **Decision Tree**: Interpretable but lower performance (0.9123)

The Streamlit web application provides an interactive platform for exploring model performance and making predictions, making these sophisticated ML models accessible to non-technical stakeholders and medical professionals. The deployment on Streamlit Community Cloud enables real-time access to these predictive tools from anywhere.

---

## Author

ML Assignment - BITS Pilani
February 2026

---

## References

1. UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) Dataset
2. Scikit-learn Documentation: https://scikit-learn.org/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. Streamlit Documentation: https://docs.streamlit.io/
