import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_correlation_matrix(data):
    """
    Create a correlation matrix plot
    
    Args:
        data (pandas.DataFrame): Data to plot
    
    Returns:
        matplotlib.figure.Figure: Correlation matrix figure
    """
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    
    return plt.gcf()

def plot_feature_distributions(data):
    """
    Create histograms of numerical features
    
    Args:
        data (pandas.DataFrame): Data to plot
    
    Returns:
        matplotlib.figure.Figure: Feature distributions figure
    """
    numeric_cols = data.select_dtypes(include=['float', 'int']).columns
    n_cols = 2
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*4))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(data[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    return fig

def plot_loss_curve(loss_curve):
    """
    Plot the loss curve for model training
    
    Args:
        loss_curve (list): Loss values during training
    
    Returns:
        matplotlib.figure.Figure: Loss curve figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def plot_confusion_matrix(cm, class_names=['Not Admitted', 'Admitted']):
    """
    Plot confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): Names of classes
    
    Returns:
        matplotlib.figure.Figure: Confusion matrix figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    return plt.gcf()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance based on model coefficients
    
    Args:
        model (MLPClassifier): Trained neural network model
        feature_names (list): Names of features
    
    Returns:
        matplotlib.figure.Figure: Feature importance figure
    """
    # For neural networks, we'll use the input layer weights as a proxy for feature importance
    # This is a simple heuristic and has limitations
    coefs = model.coefs_[0]
    importances = np.abs(coefs).mean(axis=1)
    
    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance (based on input layer weights)')
    plt.tight_layout()
    
    return plt.gcf()

# Streamlit visualization functions
def st_plot_data_overview(data):
    """
    Display data overview in Streamlit
    
    Args:
        data (pandas.DataFrame): Data to display
    """
    st.subheader("Data Overview")
    
    # Show basic info
    st.write("Dataset Shape:", data.shape)
    
    # Show data sample
    st.write("Data Sample:")
    st.dataframe(data.head())
    
    # Show summary statistics
    st.write("Summary Statistics:")
    st.dataframe(data.describe())
    
    # Show distributions
    st.write("Feature Distributions:")
    fig = plot_feature_distributions(data)
    st.pyplot(fig)
    
    # Show correlation matrix
    st.write("Correlation Matrix:")
    fig = plot_correlation_matrix(data)
    st.pyplot(fig)

def st_plot_model_performance(model_results):
    """
    Display model performance in Streamlit
    
    Args:
        model_results (dict): Model evaluation results
    """
    st.subheader("Model Performance")
    
    # Show accuracy
    train_acc = model_results.get('train_accuracy')
    test_acc = model_results.get('test_accuracy')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{train_acc:.2%}" if train_acc else "N/A")
    with col2:
        st.metric("Test Accuracy", f"{test_acc:.2%}" if test_acc else "N/A")
    
    # Show confusion matrix
    st.write("Confusion Matrix:")
    cm = model_results.get('test_confusion_matrix')
    if cm is not None:
        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)
    
    # Show loss curve
    st.write("Loss Curve:")
    loss_curve = model_results.get('loss_curve')
    if loss_curve is not None:
        fig = plot_loss_curve(loss_curve)
        st.pyplot(fig)

def st_admission_prediction_app(model, scaler, feature_columns):
    """
    Interactive admission prediction form in Streamlit
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_columns: Original feature columns
    """
    st.subheader("Admission Prediction")
    
    # Get basic feature names (before one-hot encoding)
    basic_features = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    
    # Create form for user input
    with st.form("prediction_form"):
        st.write("Enter student information:")
        
        # Create input fields
        gre = st.slider("GRE Score", 260, 340, 316)
        toefl = st.slider("TOEFL Score", 80, 120, 107)
        university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
        sop = st.slider("SOP Strength", 1.0, 5.0, 3.5, 0.5)
        lor = st.slider("LOR Strength", 1.0, 5.0, 3.5, 0.5)
        cgpa = st.slider("CGPA", 6.0, 10.0, 8.6, 0.1)
        research = st.radio("Research Experience", ["No", "Yes"], horizontal=True)
        
        # Convert research to binary
        research = 1 if research == "Yes" else 0
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'GRE_Score': [gre],
                'TOEFL_Score': [toefl],
                'University_Rating': [university_rating],
                'SOP': [sop],
                'LOR': [lor],
                'CGPA': [cgpa],
                'Research': [research]
            })
            
            # Process the input data (one-hot encoding, scaling)
            input_data['University_Rating'] = input_data['University_Rating'].astype('object')
            input_data['Research'] = input_data['Research'].astype('object')
            
            # Create dummy variables (ensuring all columns match the training data)
            input_encoded = pd.get_dummies(input_data, columns=['University_Rating', 'Research'], dtype='int')
            
            # Fix any missing columns compared to training data
            for col in feature_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Ensure columns are in the same order as during training
            input_encoded = input_encoded[feature_columns]
            
            # Scale the input data
            input_scaled = scaler.transform(input_encoded)
            
            # Make prediction
            probability = model.predict_proba(input_scaled)[0, 1]
            prediction = 1 if probability >= 0.5 else 0
            
            # Display result
            st.subheader("Prediction Result")
            
            # Create a gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Probability of Admission"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            st.plotly_chart(fig)
            
            # Display prediction
            if prediction == 1:
                st.success("The student is likely to be admitted (80% or higher chance)!")
            else:
                st.warning("The student is unlikely to be admitted (less than 80% chance).")