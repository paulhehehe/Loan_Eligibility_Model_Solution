
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging
import os

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(data_path):
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            logging.error(f"File not found: {data_path}")
            return None
        
        # Import the data from 'credit.csv'
        df = pd.read_csv(data_path)

        # Impute all missing values in all the features
        df['Gender'].fillna('Male', inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        
        # Log success message with data shape
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        # Drop 'Loan_ID' variable from the data
        df = df.drop('Loan_ID', axis=1)
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None

def split_data(X, y):
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y)

    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    
    return xtrain, xtest, ytrain, ytest


def scale_data(xtrain, xtest):

    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return xtrain_scaled, xtest_scaled
