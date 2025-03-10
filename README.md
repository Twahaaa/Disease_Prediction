# Medical Prediction Project

## Overview
This project implements a machine learning model to predict medical outcomes based on patient data. It uses XGBoost for classification tasks, with proper preprocessing steps to handle categorical variables and ensure accurate predictions.

## Project Structure
```
diag_pred/
├── data/                  # Raw and processed data files
├── model/             # Jupyter notebooks for data exploration
├── diag_pred_env/          # Virtual environment
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/med_pred.git
cd med_pred
```

2. Create and activate a virtual environment:
```bash
python -m venv med_pred_env
source med_pred_env/bin/activate  # On Windows: med_pred_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
The project uses scikit-learn's preprocessing tools to handle categorical variables and normalize features:

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Load data
df = pd.read_csv('data/patient_data.csv')

# Encode categorical features
le = LabelEncoder()
for cat_col in categorical_columns:
    df[cat_col] = le.fit_transform(df[cat_col])

# Split features and target
X = df.drop('outcome', axis=1)
y = df['outcome']

# Encode target variable
y = le.fit_transform(y)
```

### Model Training
Use XGBoost for classification with proper encoding:

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)
```

### Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(report)
```

## Important Notes

### Label Encoding
Ensure target variables are properly encoded before model training:
- Apply LabelEncoder to the target variable before splitting into train/test sets
- Verify class labels are zero-based (starting from 0)
- Check for class consistency across all data splits

```python
# Check encoded class distribution
print("Unique classes:", np.unique(y))
```

### Common Issues
- **Class Mismatch Error**: If you encounter `ValueError: Invalid classes inferred from unique values of y`, ensure your class labels are consistent and zero-based.
- **Solution**: Apply LabelEncoder before splitting data and verify class consistency.

## License
[MIT License](LICENSE)

## Contact
Aboobakkar Twaha - aboobakkartwahal@gmail.com
