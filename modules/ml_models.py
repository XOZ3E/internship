import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import shap
import matplotlib.pyplot as plt

def show():
    st.title("🤖 ML Models Module")
    
    if 'processed_data' not in st.session_state:
        st.warning("Please process data in the Data Processing module first.")
        return
    
    df = st.session_state.processed_data.copy()  # Work on a copy
    
    # Fix Arrow serialization warnings for timestamp columns
    timestamp_cols = ['Order Date', 'Ship Date']  # Add any other timestamp columns from your dataset
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)  # Convert to string for compatibility
    
    # Select target and features
    st.subheader("🎯 Select Target for Modeling")
    target_col = st.selectbox("Target Column", df.columns)
    features = [col for col in df.columns if col != target_col]
    
    # Split data
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing: Select numeric columns, fill NaNs, replace inf
    orig_feature_cols = X_train.columns.tolist()
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_test_num = X_test[X_train_num.columns]
    
    dropped = [c for c in orig_feature_cols if c not in X_train_num.columns]
    if dropped:
        st.warning(f"Dropped non-numeric columns for modeling: {dropped}")
    
    X_train_num = X_train_num.fillna(0).replace([np.inf, -np.inf], 0)
    X_test_num = X_test_num.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Handle categorical target
    if y.dtype == 'O' or y.dtype.name == 'category':
        unique_labels = np.unique(y)
        le = LabelEncoder()
        le.fit(unique_labels)
        
        y_train_mod = le.transform(y_train)
        try:
            y_test_mod = le.transform(y_test)
        except ValueError as e:
            st.warning(f"Unseen labels in test set: {e}. Mapping to -1.")
            y_test_mod = np.array([le.transform([label])[0] if label in le.classes_ else -1 for label in y_test])
    else:
        y_train_mod = y_train.values  # Ensure array
        y_test_mod = y_test.values
    
    # Reset indices to align everything
    X_train_num = X_train_num.reset_index(drop=True)
    y_train_mod = pd.Series(y_train_mod).reset_index(drop=True)  # Convert to Series for indexing
    
    # Hyperparameter Tuning Widgets
    st.subheader("⚙️ Hyperparameter Tuning")
    n_estimators = st.slider("Number of Trees", 10, 100, 20)
    sample_size = st.slider("Data Sample Size (%) for Training", 10, 100, 20)
    max_depth = st.slider("Max Tree Depth (to reduce memory)", 5, 20, 8)
    
    # AutoML Interface: One-click training with spinner
    if st.button("Train Random Forest Model"):
        if len(X_train_num.columns) == 0:
            st.error("No numeric features available for training. Please add/process numeric columns in Data Processing.")
            return
        
        with st.spinner("Training model... This may take a moment. Use lower sample size if it's slow."):
            sample_frac = sample_size / 100.0
            if sample_frac <= 0 or len(X_train_num) == 0:
                st.error("Sample size too small or no data available. Increase sample size.")
                return
            
            # Sample using reset indices
            sample_df = X_train_num.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            sample_indices = sample_df.index.tolist()  # Now 0-based positional indices
            
            if len(sample_indices) == 0:
                st.error("Sampled data is empty. Increase sample size.")
                return
            
            X_train_sample = sample_df
            y_train_sample = y_train_mod.iloc[sample_indices]  # Use iloc for positional access
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train_sample, y_train_sample)
            y_pred = model.predict(X_test_num)
            accuracy = accuracy_score(y_test_mod, y_pred)
            st.success(f"Model Accuracy: {accuracy:.2f}")
            
            # Model Performance Dashboard
            st.subheader("📊 Model Performance Dashboard")
            st.text(classification_report(y_test_mod, y_pred))
            
            # Model Interpretability: Feature Importance
            st.subheader("🔍 Feature Importance")
            trained_features = X_train_num.columns.tolist()
            importance = pd.DataFrame({
                'feature': trained_features,
                'importance': model.feature_importances_
            })
            st.dataframe(importance)
            
            fig = px.bar(importance, x='importance', y='feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Visualizations (FIXED for multi-class and indexing errors)
            st.subheader("🔍 Model Interpretability (SHAP)")
            try:
                explainer = shap.TreeExplainer(model)
                
                # Sample X_test_num to avoid memory issues (e.g., first 100 rows)
                X_test_sample = X_test_num.iloc[:100] if len(X_test_num) > 100 else X_test_num
                
                if len(X_test_sample) == 0:
                    st.warning("No test data available for SHAP. Skipping.")
                else:
                    shap_values = explainer.shap_values(X_test_sample)
                    
                    # Handle multi-class: If shap_values is a list (multi-class), aggregate to 2D
                    num_classes = len(np.unique(y_train_mod))
                    if isinstance(shap_values, list) and num_classes > 2:
                        # Aggregate absolute mean across classes for summary
                        shap_values = np.abs(np.array(shap_values)).mean(axis=0)  # Shape: (n_samples, n_features)
                        st.info("Multi-class detected. Showing aggregated SHAP values (mean absolute across classes).")
                    elif isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary, take positive class
                    
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test_sample, show=False)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP plot failed: {str(e)}. This may be due to multi-class targets or data size. Try a smaller sample.")
    
            # Store model
            st.session_state.trained_model = model
