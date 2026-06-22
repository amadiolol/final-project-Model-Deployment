import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Class Preprocessing 
class CreditDataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.label_encoders = {}
        
    def _extract_months(self, text):
        if pd.isna(text): return np.nan
        years = re.search(r'(\d+)\s*Years', str(text))
        months = re.search(r'(\d+)\s*Months', str(text))
        y = int(years.group(1)) if years else 0
        m = int(months.group(1)) if months else 0
        return (y * 12) + m

    def process_data(self):
        df = pd.read_csv(self.filepath)
        
        # Data Cleaning
        cols_to_drop = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Month']
        df = df.drop(columns=cols_to_drop)
        df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        
        numeric_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                        'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly']
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['Credit_History_Age_Months'] = df['Credit_History_Age'].apply(self._extract_months)
        df = df.drop(columns=['Credit_History_Age'])
        
        
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # Encoding
        target_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
        df['Credit_Score'] = df['Credit_Score'].map(target_mapping)
        
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        # Simpan encoder untuk tahap deployment nanti
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        X = df.drop(columns=['Credit_Score'])
        y = df['Credit_Score']
        return train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Class Evaluation 
class CreditModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }
        return metrics

# 3. Class Training 
class CreditModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_model = None
        self.best_accuracy = 0
        
    def train_and_log(self, model_name, model):
        with mlflow.start_run(run_name=model_name):
            # Training
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            # Evaluasi
            metrics = CreditModelEvaluator.evaluate(self.y_test, y_pred)
            
            # Log ke MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"[{model_name}] Accuracy: {metrics['accuracy']:.4f}")
            
            # Cek model terbaik
            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']
                self.best_model = model

    def save_best_model(self):
        # Simpan model terbaik dalam format .pkl untuk deployment
        with open('best_rf_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        print("\nModel terbaik berhasil disimpan sebagai 'best_rf_model.pkl'")

#  Main Execution 
if __name__ == "__main__":
    print("Memulai Pipeline Data...")
    
    # 1. Preprocessing
    preprocessor = CreditDataPreprocessor('data_C.csv')
    X_train, X_test, y_train, y_test = preprocessor.process_data()
    
    # Set experiment MLflow
    mlflow.set_experiment("Credit_Score_Prediction")
    
    # 2. Training & Evaluation Pipeline
    print("Memulai Training dan Logging ke MLflow...")
    trainer = CreditModelTrainer(X_train, X_test, y_train, y_test)
    
    # Eksekusi Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    
    trainer.train_and_log("Random Forest", rf_model)
    trainer.train_and_log("Decision Tree", dt_model)
    
    # 3. Simpan model terbaik
    trainer.save_best_model()