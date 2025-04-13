import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    """
    Set consistent plot style for all visualizations
    """
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14

def create_directory_structure():
    """
    Create project directory structure if it doesn't exist
    """
    directories = [
        'data',
        'models',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create __init__.py in src directory
    init_file = os.path.join('src', '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            pass

def get_data_summary(data):
    """
    Generate a summary of the dataset
    
    Args:
        data (pandas.DataFrame): Dataset to summarize
        
    Returns:
        dict: Summary statistics and information
    """
    summary = {}
    
    # Basic info
    summary['shape'] = data.shape
    summary['columns'] = list(data.columns)
    summary['dtypes'] = dict(data.dtypes)
    
    # Missing values
    summary['missing_values'] = dict(data.isnull().sum())
    
    # Class distribution (if Admit_Chance exists)
    if 'Admit_Chance' in data.columns:
        summary['class_distribution'] = dict(data['Admit_Chance'].value_counts())
    
    # Numeric column statistics
    numeric_cols = data.select_dtypes(include=['float', 'int']).columns
    summary['numeric_stats'] = {}
    
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'min': data[col].min(),
            'max': data[col].max(),
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std()
        }
    
    return summary

def normalize_feature_names(feature_array, original_df):
    """
    Get feature names from original dataframe that match the array
    
    Args:
        feature_array (numpy.ndarray): Feature array after one-hot encoding and scaling
        original_df (pandas.DataFrame): Original dataframe before processing
        
    Returns:
        list: Feature names matching the array
    """
    # Get numeric columns
    numeric_cols = original_df.select_dtypes(include=['float', 'int']).columns.tolist()
    
    # Get categorical columns
    categorical_cols = original_df.select_dtypes(include=['object']).columns.tolist()
    
    # Create dummy column names for categorical features
    dummy_cols = []
    for col in categorical_cols:
        unique_values = original_df[col].unique()
        for val in unique_values:
            dummy_cols.append(f"{col}_{val}")
    
    # Combine numeric and dummy columns
    all_feature_names = numeric_cols + dummy_cols
    
    # If array length doesn't match, return default indices
    if len(all_feature_names) != feature_array.shape[1]:
        return [f"feature_{i}" for i in range(feature_array.shape[1])]
    
    return all_feature_names