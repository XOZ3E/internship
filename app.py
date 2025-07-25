import streamlit as st
import pandas as pd
import numpy as np
from modules import data_discovery, data_processing, ml_models, deployment

def main():
    st.set_page_config(
        page_title="Smart Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.sidebar.title("Analytics Modules")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Module",
        ["Home", "Data Discovery", "Data Processing", "ML Models", "Deployment"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Data Discovery":
        data_discovery.show()
    elif page == "Data Processing":
        data_processing.show()
    elif page == "ML Models":
        ml_models.show()
    elif page == "Deployment":
        deployment.show()

def show_home():
    st.title("🎯 Smart Analytics Dashboard")
    st.markdown("""
    **Transform your data into actionable insights with our integrated analytics platform**
    
    ### 🚀 Features:
    - **Data Discovery**: Automated profiling and visualization
    - **Data Processing**: Intelligent cleaning and transformation
    - **ML Models**: Automated model development and comparison
    - **Deployment**: One-click model deployment and monitoring
    """)
    
    # Upload area
    uploaded_file = st.file_uploader(
        "Upload your dataset to get started",
        type=['csv', 'xlsx', 'json']
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'csv':
            st.session_state.uploaded_data = pd.read_csv(uploaded_file)
        elif file_type == 'xlsx':
            st.session_state.uploaded_data = pd.read_excel(uploaded_file)
        elif file_type == 'json':
            st.session_state.uploaded_data = pd.read_json(uploaded_file)
        st.success("Data uploaded successfully! Navigate to modules to begin analysis.")

if __name__ == "__main__":
    main()
