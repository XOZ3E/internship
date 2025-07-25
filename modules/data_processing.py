import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def show():
    st.title("🛠️ Data Processing Module")
    
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload data from the Home page first.")
        return
    
    df = st.session_state.uploaded_data.copy()
    # After df = st.session_state.uploaded_data.copy()
    timestamp_cols = ['Order Date', 'Ship Date']  # Add others if needed
    for col in timestamp_cols:
     if col in df.columns:
        df[col] = df[col].astype(str)

    
    # Interactive Data Cleaning
    st.subheader("🧹 Interactive Data Cleaning")
    missing_strategy = st.selectbox(
        "Handle Missing Values",
        ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
    )
    if missing_strategy == "Drop rows":
        df = df.dropna()
    elif missing_strategy == "Fill with mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == "Fill with median":
        df = df.fillna(df.median(numeric_only=True))
    elif missing_strategy == "Fill with mode":
        df = df.fillna(df.mode().iloc[0])
    
    # Feature Engineering Widgets
    st.subheader("🔧 Feature Engineering")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        col_to_scale = st.selectbox("Select column to scale (normalize)", numeric_cols)
        if st.button("Normalize Selected Column"):
            df[col_to_scale] = (df[col_to_scale] - df[col_to_scale].min()) / (df[col_to_scale].max() - df[col_to_scale].min())
            st.success(f"Normalized {col_to_scale}")
    
    # Pipeline Visualization: Real-time preview
    st.subheader("📋 Pipeline Visualization")
    st.dataframe(df.head(10))  # Preview first 10 rows
    
    if len(numeric_cols) > 0:
        fig = px.histogram(df, x=numeric_cols[0], title="Distribution After Processing")
        st.plotly_chart(fig, use_container_width=True)
    
    # Export Functionality
    st.subheader("💾 Export Cleaned Data")
    if st.button("Download Processed Data as CSV"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
    
    # Store processed data in session state for other modules
    st.session_state.processed_data = df
