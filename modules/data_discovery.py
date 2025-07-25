import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np  # Add this if not present

def show():
    st.title("🔍 Data Discovery Engine")
    
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload data from the Home page first.")
        return
    
    df = st.session_state.uploaded_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Overview")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Memory Usage:** {df.memory_usage().sum() / 1024:.2f} KB")
        
        # Data types (fixed for Plotly)
        st.subheader("🏷️ Data Types")
        type_counts = df.dtypes.value_counts()
        fig = px.pie(
            values=type_counts.values.tolist(),
            names=[str(x) for x in type_counts.index.tolist()]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe())
        
        # Missing values (fixed for Plotly)
        missing = df.isnull().sum()
        if missing.any():
            st.subheader("⚠️ Missing Values")
            fig = px.bar(
                x=[str(x) for x in missing.index.tolist()],
                y=missing.values.tolist()
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Interactive visualization
    st.subheader("🎨 Interactive Visualization")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        col_x = st.selectbox("X-axis", numeric_cols)
        col_y = st.selectbox("Y-axis", numeric_cols)
        
        fig = px.scatter(df, x=col_x, y=col_y)
        st.plotly_chart(fig, use_container_width=True)
