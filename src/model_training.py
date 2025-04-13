from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
import os

def train_model(X_train, y_train, hidden_layer_sizes=(3, 3), batch_size=50, max_iter=200, random_state=123):
    """
    Train a neural network model for admission prediction
    
    Args:
        X_train (numpy.ndarray): Scaled training features
        y_train (numpy.ndarray): Training target variable
        hidden_layer_sizes (tuple): Number of neurons in hidden layers
        batch_size (int): Size of mini-batches for stochastic optimizers
        max_iter (int): Maximum number of iterations
        random_state (int): Random seed for reproducibility
        
    Returns:
        MLPClassifier: Trained model
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluate the model performance
    
    Args:
        model (MLPClassifier): Trained model
        X_test (numpy.ndarray): Scaled test features
        y_test (numpy.ndarray): Test target variable
        X_train (numpy.ndarray, optional): Scaled training features
        y_train (numpy.ndarray, optional): Training target variable
        
    Returns:
        dict: Model evaluation metrics
    """
    results = {}
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_cm = confusion_matrix(y_test, y_pred)
    test_report = classification_report(y_test, y_pred, output_dict=True)
    
    results['test_accuracy'] = test_acc
    results['test_confusion_matrix'] = test_cm
    results['test_classification_report'] = test_report
    
    # Training set evaluation (if provided)
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_cm = confusion_matrix(y_train, y_pred_train)
        
        results['train_accuracy'] = train_acc
        results['train_confusion_matrix'] = train_cm
    
    # Model loss history
    results['loss_curve'] = model.loss_curve_
    
    return results

def save_model(model, scaler, model_path='models/admission_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Save the trained model and scaler
    
    Args:
        model (MLPClassifier): Trained model
        scaler (MinMaxScaler): Fitted scaler
        model_path (str): Path to save the model
        scaler_path (str): Path to save the scaler
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(model_path='models/admission_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Load the trained model and scaler
    
    Args:
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        
    Returns:
        tuple: (model, scaler) if successful, (None, None) otherwise
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None