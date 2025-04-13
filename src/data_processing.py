import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load admission data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, test_size=0.2, random_state=123):
    """
    Preprocess admission data for modeling
    
    Args:
        data (pandas.DataFrame): Raw admission data
        test_size (float): Proportion of data to use for test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (xtrain_scaled, xtest_scaled, ytrain, ytest, scaler)
    """
    # Drop unnecessary columns
    if 'Serial_No' in data.columns:
        data = data.drop(['Serial_No'], axis=1)
    
    # Convert the target variable into a categorical variable
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    
    # Convert categorical features
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')
    
    # Create dummy variables
    clean_data = pd.get_dummies(data, columns=['University_Rating', 'Research'], dtype='int')
    
    # Split data into features and target
    x = clean_data.drop(['Admit_Chance'], axis=1)
    y = clean_data['Admit_Chance']
    
    # Split data into train and test sets
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    
    return xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, x.columns

def process_new_data(data, scaler):
    """
    Process new data for prediction
    
    Args:
        data (pandas.DataFrame): New data for prediction
        scaler (MinMaxScaler): Fitted scaler for feature normalization
        
    Returns:
        numpy.ndarray: Scaled features ready for prediction
    """
    # Convert categorical features
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')
    
    # Create dummy variables
    clean_data = pd.get_dummies(data, columns=['University_Rating', 'Research'], dtype='int')
    
    # Scale features
    scaled_data = scaler.transform(clean_data)
    
    return scaled_data