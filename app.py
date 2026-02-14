import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request

# Set page config
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("🤖 ML Classification Models - Breast Cancer Dataset")
st.markdown("""
This application demonstrates the performance of 6 different machine learning classification models
trained on the Breast Cancer dataset.
""")

# Load models and scaler
@st.cache_resource
def load_models():
    models_dir = os.path.dirname(__file__)
    models = {}
    
    model_files = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'knn': 'KNN',
        'naive_bayes': 'Naive Bayes',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }
    
    for file_name, display_name in model_files.items():
        try:
            with open(os.path.join(models_dir, 'models', f'{file_name}.pkl'), 'rb') as f:
                models[display_name] = pickle.load(f)
        except:
            st.warning(f"Could not load {display_name} model")
    
    try:
        with open(os.path.join(models_dir, 'models', 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = StandardScaler()
    
    try:
        with open(os.path.join(models_dir, 'models', 'results.pkl'), 'rb') as f:
            results = pickle.load(f)
    except:
        results = {}
    
    return models, scaler, results

# Load test data from UCI
@st.cache_resource
def load_test_data():
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        data = pd.read_csv(url, header=None)
        
        # Extract features and target
        X = data.iloc[:, 2:32].values
        y = data.iloc[:, 1].map({'M': 1, 'B': 0}).values
        
        # Convert to DataFrame for easier handling
        X = pd.DataFrame(X)
        
        # Split data (80-20 split)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_test, y_test
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return None, None

models, scaler, results = load_models()
X_test, y_test = load_test_data()

# Sidebar
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox(
    "Select a Model:",
    list(models.keys()),
    index=0
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Metrics", "📈 Performance Comparison", "📁 Data Upload & Prediction"])

# Tab 1: Model Metrics
with tab1:
    st.header(f"Evaluation Metrics - {selected_model}")
    
    if results and selected_model in results:
        col1, col2, col3 = st.columns(3)
        
        metrics = results[selected_model]
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        
        with col2:
            st.metric("AUC Score", f"{metrics['AUC']:.4f}")
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{metrics['F1']:.4f}")
            st.metric("MCC Score", f"{metrics['MCC']:.4f}")
        
        # Display as table
        st.subheader("Detailed Metrics Table")
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ["Score"]
        st.dataframe(metrics_df.round(4))
    else:
        st.warning("Results not available for this model")

# Tab 2: Performance Comparison
with tab2:
    st.header("Model Comparison")
    
    if results:
        # Create comparison dataframe
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        
        st.subheader("All Models Metrics Comparison")
        st.dataframe(results_df)
        
        # Confusion Matrices Visualization
        st.subheader("Confusion Matrices for All Models")
        st.markdown("""
        **Confusion Matrix Components:**
        - **True Negatives (TN):** Correctly predicted Benign cases
        - **False Positives (FP):** Benign cases incorrectly predicted as Malignant
        - **False Negatives (FN):** Malignant cases missed (most critical in medical diagnosis)
        - **True Positives (TP):** Correctly predicted Malignant cases
        """)
        
        if X_test is not None and y_test is not None:
            # Generate confusion matrices for all models
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            model_names = list(models.keys())
            for idx, model_name in enumerate(model_names):
                model = models[model_name]
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                cm = confusion_matrix(y_test, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                           xticklabels=['Benign', 'Malignant'], 
                           yticklabels=['Benign', 'Malignant'])
                axes[idx].set_title(f'{model_name}')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Could not load test data for confusion matrices")

# Tab 3: Data Upload & Prediction
with tab3:
    st.header("Predictions on Custom Data")
    
    uploaded_file = st.file_uploader("Upload test data (CSV file)", type=['csv'], key='upload_data')
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # Check if data has correct number of features
            if data.shape[1] == 30:  # Breast cancer dataset has 30 features
                st.success(f"✓ Dataset has {data.shape[0]} samples and {data.shape[1]} features")
                
                # Make predictions
                if selected_model in models:
                    X_scaled = scaler.transform(data)
                    predictions = models[selected_model].predict(X_scaled)
                    probabilities = models[selected_model].predict_proba(X_scaled)
                    
                    # Display results
                    result_df = pd.DataFrame({
                        'Prediction': predictions,
                        'Probability Class 0': probabilities[:, 0],
                        'Probability Class 1': probabilities[:, 1]
                    })
                    
                    st.subheader(f"Predictions using {selected_model}")
                    st.dataframe(result_df)
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"❌ Dataset has {data.shape[1]} features. Expected 30 features for Breast Cancer dataset.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("📁 Upload a CSV file with 30 features to make predictions")
        
        # Provide sample data info
        st.subheader("Sample Test Data Format")
        st.markdown("""
        The test data should have 30 features corresponding to the Breast Cancer dataset:
        - mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
        - SE (Standard Error) versions of the above
        - worst (largest values) versions of the above
        
        **CSV Format:** Each row is a sample, each column is a feature (no header recommended)
        """)
