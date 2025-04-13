import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model, save_model, load_model
from src.visualization import (
    st_plot_data_overview, 
    st_plot_model_performance, 
    st_admission_prediction_app
)
from src.utils import set_plot_style, create_directory_structure

def main():
    # Set page configuration
    st.set_page_config(
        page_title="UCLA Admission Prediction",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set the style for all plots
    set_plot_style()
    
    # Create directory structure if needed
    create_directory_structure()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Data Exploration", "Model Training", "Make Predictions"]
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Home page
    if page == "Home":
        st.title("UCLA Admission Prediction")
        st.write("""
        ## Welcome to the UCLA Admission Prediction App
        
        This application helps you predict whether a student will be admitted to UCLA based on various factors such as GRE score, TOEFL score, university rating, and more.
        
        ### Project Scope
        
        The world is developing rapidly, and continuously looking for the best knowledge and experience among people. This motivates people all around the world to stand out in their jobs and look for higher degrees that can help them in improving their skills and knowledge. As a result, the number of students applying for Master's programs has increased substantially.
        
        This app uses a neural network model to predict a student's chance of admission into UCLA.
        
        ### How to use this app
        
        1. **Data Exploration**: View and analyze the admission dataset
        2. **Model Training**: Train a neural network model on the data
        3. **Make Predictions**: Use the trained model to predict admission chances for new students
        
        """)
        
        st.info("Navigate through the app using the sidebar on the left.")
    
    # Data Exploration page
    elif page == "Data Exploration":
        st.title("Data Exploration")
        
        # Load data
        data_path = st.text_input("Enter path to the data file", "data/Admission.csv")
        load_button = st.button("Load Data")
        
        if load_button:
            data = load_data(data_path)
            if data is not None:
                st.session_state.data = data
                st.success(f"Data loaded successfully! Shape: {data.shape}")
        
        # Display data exploration if data is loaded
        if st.session_state.data is not None:
            st_plot_data_overview(st.session_state.data)
    
    # Model Training page
    elif page == "Model Training":
        st.title("Model Training")
        
        # Check if data is loaded
        if st.session_state.data is None:
            st.warning("Please load the data first in the Data Exploration page.")
            return
        
        # Model hyperparameters
        st.subheader("Model Hyperparameters")
        
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.text_input("Hidden Layer Sizes (comma-separated, e.g., 3,3)", "3,3")
            batch_size = st.number_input("Batch Size", min_value=1, value=50)
        
        with col2:
            max_iter = st.number_input("Max Iterations", min_value=1, value=200)
            random_state = st.number_input("Random State", min_value=0, value=123)
        
        # Train model button
        train_button = st.button("Train Model")
        
        if train_button:
            # Convert hidden layers string to tuple
            hidden_layer_sizes = tuple(int(x) for x in hidden_layers.split(','))
            
            # Show progress
            with st.spinner("Preprocessing data..."):
                xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, feature_columns = preprocess_data(
                    st.session_state.data, test_size=0.2, random_state=random_state
                )
            
            with st.spinner("Training model..."):
                model = train_model(
                    xtrain_scaled, ytrain,
                    hidden_layer_sizes=hidden_layer_sizes,
                    batch_size=batch_size,
                    max_iter=max_iter,
                    random_state=random_state
                )
            
            with st.spinner("Evaluating model..."):
                results = evaluate_model(model, xtest_scaled, ytest, xtrain_scaled, ytrain)
            
            # Save model, scaler, and results to session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_columns = feature_columns
            st.session_state.results = results
            
            # Save model to disk
            save_model(model, scaler)
            
            st.success("Model trained successfully!")
        
        # Display model results if available
        if st.session_state.results is not None:
            st_plot_model_performance(st.session_state.results)
    
    # Make Predictions page
    elif page == "Make Predictions":
        st.title("Make Predictions")
        
        # Check if model is trained or load from disk
        if st.session_state.model is None:
            st.info("No model in session. Attempting to load from disk...")
            model, scaler = load_model()
            
            if model is not None and scaler is not None:
                st.session_state.model = model
                st.session_state.scaler = scaler
                
                # Load data to get feature columns if needed
                if st.session_state.feature_columns is None:
                    if st.session_state.data is not None:
                        # Process data to get feature columns
                        _, _, _, _, _, feature_columns = preprocess_data(st.session_state.data)
                        st.session_state.feature_columns = feature_columns
                    else:
                        st.warning("Please load data in the Data Exploration page to get feature information.")
                        return
                
                st.success("Model loaded successfully!")
            else:
                st.error("No trained model found. Please train a model first.")
                return
        
        # Make predictions
        if st.session_state.model is not None and st.session_state.scaler is not None and st.session_state.feature_columns is not None:
            st_admission_prediction_app(
                st.session_state.model,
                st.session_state.scaler,
                st.session_state.feature_columns
            )
        else:
            st.error("Model, scaler or feature columns not available. Please train or load a model first.")

if __name__ == "__main__":
    main()