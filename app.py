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
import requests

# Get base features from UCI names file
@st.cache_resource
def get_base_features():
    try:
        names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names"
        response = requests.get(names_url, verify=False)
        
        base_features = []
        for line in response.text.split('\n'):
            if ':' in line and line[0].isdigit():
                # Extract feature description (e.g., "1) radius_mean")
                parts = line.split(')')
                if len(parts) > 1:
                    feature_desc = parts[1].strip().lower()
                    # Remove prefixes (mean, se, worst) to get base name
                    for prefix in ['_mean', '_se', '_worst']:
                        feature_desc = feature_desc.replace(prefix, '')
                    
                    if feature_desc and feature_desc not in base_features:
                        base_features.append(feature_desc.replace(' ', '_'))
        
        # Return unique base features (first 10)
        return sorted(list(set(base_features)))[:10] if base_features else None
    except:
        return None

BASE_FEATURES = get_base_features()

# Set page config
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("🤖 ML Classification Models")
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

# Get feature names from UCI names file
@st.cache_resource
def get_feature_names():
    try:
        # Get base features from UCI (dynamically parsed)
        base_features = get_base_features()
        
        # Fallback if parsing fails
        if not base_features or len(base_features) < 10:
            base_features = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                           'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
        
        names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names"
        response = requests.get(names_url, verify=False)
        
        # Parse feature names from .names file
        lines = response.text.split('\n')
        feature_names = []
        
        for line in lines:
            if ':' in line and any(x in line.lower() for x in base_features):
                # Extract feature name from description line
                parts = line.split(':')
                if len(parts) > 0:
                    feature_name = parts[0].strip().lower().replace(' ', '_')
                    if feature_name and feature_name not in feature_names:
                        feature_names.append(feature_name)
        
        # If parsing fails, generate default names
        if len(feature_names) < 30:
            feature_names = (
                [f'mean_{f}' for f in base_features] +
                [f'SE_{f}' for f in base_features] +
                [f'worst_{f}' for f in base_features]
            )
        
        return feature_names[:30]  # Return exactly 30 features
    except Exception as e:
        # Fallback: generate default names
        base_features = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                       'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
        return (
            [f'mean_{f}' for f in base_features] +
            [f'SE_{f}' for f in base_features] +
            [f'worst_{f}' for f in base_features]
        )

models, scaler, results = load_models()
X_test, y_test = load_test_data()
feature_names = get_feature_names()

# Sidebar
st.sidebar.header("Configuration")

# Model selector in sidebar
st.sidebar.markdown("**🤖 Select Model**")
selected_model = st.sidebar.selectbox(
    "Choose a model to view details:",
    list(models.keys()),
    key="model_select",
    label_visibility="collapsed"
)

# Create tabs
tab1 = st.tabs(["📊 Complete Analysis"])[0]

# Main Content - All in one place
with tab1:
    # ===== SECTION 1: Model Analysis =====
    st.subheader("1️⃣ Model Performance Metrics")
    
    if results and selected_model in results:
        st.markdown(f"**📈 Metrics for {selected_model}**")
        metrics = results[selected_model]
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col_b:
            st.metric("AUC", f"{metrics['AUC']:.4f}")
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col_c:
            st.metric("F1", f"{metrics['F1']:.4f}")
            st.metric("MCC", f"{metrics['MCC']:.4f}")
    
    st.markdown("**All Models Comparison**")
    if results:
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        st.dataframe(results_df, use_container_width=True)
    
    st.divider()
    
    # ===== SECTION 2: Confusion Matrices =====
    st.subheader("2️⃣ Confusion Matrices for All Models")
    st.markdown("""
    **Components:**
    - **TN:** Correctly predicted Benign | **FP:** Benign → Malignant (False Alarm)
    - **FN:** Malignant → Benign (Missed) | **TP:** Correctly predicted Malignant
    """)
    
    if X_test is not None and y_test is not None:
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
    
    st.divider()
    
    # ===== SECTION 3: Download Sample Data =====
    st.subheader("3️⃣ Download Sample Test Data")
    st.markdown("**Sample Data Format:**")
    st.markdown("""
    - 30 features: mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
    - Plus SE (Standard Error) and worst (largest) versions of each
    - CSV Format: Each row is a sample, each column is a feature
    """)
    
    if X_test is not None:
        sample_data = X_test.sample(n=min(20, len(X_test)), random_state=42)
        sample_data.columns = feature_names
        sample_csv = sample_data.to_csv(index=False, header=True)
        
        st.download_button(
            label="📥 Download Sample Test Data",
            data=sample_csv,
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download a sample CSV file"
        )
        st.caption("✓ Sample data ready for testing predictions")
    else:
        st.warning("Could not load sample data")
    
    st.divider()
    
    # ===== SECTION 4: Make Predictions =====
    st.subheader("4️⃣ Make Predictions on Custom Data")
    
    uploaded_file = st.file_uploader("Upload test data (CSV file)", type=['csv'], key='upload_data')
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # Check if data has correct number of features
            if data.shape[1] == 30:
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
            else:
                st.error(f"❌ Dataset has {data.shape[1]} features. Expected 30 features for Breast Cancer dataset.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("📁 Upload a CSV file with 30 features to make predictions")
