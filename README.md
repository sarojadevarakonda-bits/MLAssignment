# ML Classification Models

## Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting breast cancer diagnosis (malignant vs. benign) using diagnostic measurements. The goal is to identify which classification algorithm provides the best performance in distinguishing between benign and malignant tumors, thereby assisting in medical diagnosis and treatment planning.

---

## Dataset Description

**Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** UCI Machine Learning Repository  
**Instances:** 569 samples  
**Features:** 30 numerical features  
**Target:** Binary Classification (0 = Benign, 1 = Malignant)

Each sample contains 10 real-valued features computed for each cell nucleus:
1. Radius - Mean distance from center to points on the perimeter
2. Texture - Standard deviation of gray-scale values
3. Perimeter - Outer boundary length
4. Area - Cell nucleus area
5. Smoothness - Local variation in radius lengths
6. Compactness - Perimeter² / area - 1.0
7. Concavity - Severity of concave portions of the contour
8. Concave Points - Number of concave portions of the contour
9. Symmetry - Symmetry of the nucleus
10. Fractal Dimension - "Coastline approximation" - 1

For each of these 10 features, three statistical measures are computed: Mean, Standard Error, and Worst (Largest) values, resulting in a total of 30 features per sample.

**Class Distribution:**
- Benign (0): 357 samples (62.7%)
- Malignant (1): 212 samples (37.3%)

---

## Models Used

### Comparison Table - Evaluation Metrics for All 6 Models

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9561 | 0.9901 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression provides the best overall performance with excellent results across all metrics. Being a linear classifier, it learns a clear decision boundary between benign and malignant cases. Its high AUC score (0.9954) indicates excellent discrimination ability. The perfect balance of precision and recall (both 0.9861) makes it highly reliable for clinical applications. |
| Decision Tree | The Decision Tree (max_depth=10) shows the lowest performance among all models. While it provides interpretable decision rules, it fails to capture complex relationships in the data as effectively as other algorithms. The lower AUC (0.9157) indicates moderate discrimination ability. The controlled tree depth helps prevent overfitting but limits its ability to learn complex patterns. |
| kNN | KNN (k=5) is an instance-based lazy learner that achieves solid performance with balanced precision and recall. The high AUC (0.9788) demonstrates good discrimination ability across thresholds. The higher recall (0.9722) effectively captures most malignant cases, which is critical for medical applications. However, performance is sensitive to feature scaling and the choice of k. |
| Naive Bayes | Gaussian Naive Bayes assumes feature independence and normal distribution. Despite these simplifying assumptions, it achieves excellent AUC score (0.9868), suggesting highly separable class-conditional probability distributions. The perfect balance of precision and recall (both 0.9444) indicates well-calibrated predictions. The model trains very quickly with minimal hyperparameter tuning. |
| Random Forest (Ensemble) | Random Forest (100 trees) provides excellent performance through ensemble voting. The outstanding AUC score (0.9939) demonstrates superior discrimination ability. The high recall (0.9722) ensures most malignant cases are identified. The model effectively reduces overfitting through bagging and feature randomness, capturing various non-linear patterns and resulting in robust predictions. |
| XGBoost (Ensemble) | XGBoost (100 estimators) achieves excellent performance with outstanding AUC score (0.9901). The model uses gradient boosting with sequential error correction, allowing each tree to focus on previously misclassified samples. The highest recall (0.9861) is especially important for medical applications as it minimizes false negatives. This makes XGBoost highly reliable for clinical diagnosis. |
