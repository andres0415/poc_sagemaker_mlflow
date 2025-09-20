# ML Project - Iris & Wine Classification

Simple machine learning project for classification tasks using scikit-learn. Designed to work with AWS SageMaker Studio and MLflow for experiment tracking.

## 📁 Project Structure

```
poc_sagemaker_mlflow/
├── src/
│   ├── data_loader.py      # Load Iris and Wine datasets
│   ├── preprocessing.py    # Basic data preprocessing
│   ├── models.py          # Train ML models (RF, SVM, LogReg)
│   └── evaluation.py      # Model evaluation functions
├── notebooks/
│   └── iris_classification_example.ipynb
├── data/                  # For storing datasets (empty)
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Example

```python
from src.data_loader import load_iris
from src.preprocessing import preprocess_data
from src.models import compare_models

# Load data
X, y = load_iris()

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Train and compare models
results = compare_models(X_train, y_train, X_test, y_test)
print(f"Best model: {results['best_model']}")
```

### 3. Try the Jupyter Notebook

Open `notebooks/iris_classification_example.ipynb` for a complete walkthrough.

## 📊 Available Datasets

- **Iris**: 3 classes, 4 features, 150 samples
- **Wine**: 3 classes, 13 features, 178 samples

```python
from src.data_loader import load_iris, load_wine

# Load datasets
X_iris, y_iris = load_iris()
X_wine, y_wine = load_wine()
```

## 🤖 Models Included

- **Random Forest** - Ensemble method, good baseline
- **SVM** - Support Vector Machine with RBF kernel  
- **Logistic Regression** - Linear model, fast training

## 🔧 Key Functions

### Data Loading
```python
from src.data_loader import load_iris, load_wine
```

### Preprocessing
```python
from src.preprocessing import preprocess_data

# Split and scale data
X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.3, scale=True)
```

### Model Training
```python
from src.models import compare_models

# Train all models and compare
results = compare_models(X_train, y_train, X_test, y_test)
```

### Evaluation
```python
from src.evaluation import evaluate_model, plot_confusion_matrix

# Evaluate a single model
evaluate_model(model, X_test, y_test, "Model Name")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, "Model Name")
```

## ☁️ AWS SageMaker Studio Setup

### 1. Upload Project
- Clone or upload this repository to SageMaker Studio
- Open a new terminal in SageMaker Studio

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. MLflow Integration
When ready to use MLflow in SageMaker:

```python
import mlflow
import mlflow.sklearn

# Set tracking URI (SageMaker managed MLflow)
mlflow.set_tracking_uri("your-mlflow-server-uri")

# Start experiment
with mlflow.start_run():
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Log metrics
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 4. Model Cards
Use SageMaker Model Cards to document your models:
- Model purpose and use cases
- Training data and methodology  
- Performance metrics
- Limitations and biases

## 🎯 Perfect For

- **Learning ML workflows**
- **Testing AWS SageMaker Studio**
- **MLflow experiment tracking**
- **Model comparison studies**
- **ML pipeline prototyping**

## 📝 Example Output

```
Model Comparison Results:
========================================
Random Forest: 0.9778
SVM: 0.9556
Logistic Regression: 0.9556

Best Model: Random Forest
```

## 📋 Next Steps

1. **In SageMaker Studio**: Set up MLflow tracking server
2. **Experiment**: Try different hyperparameters
3. **Compare**: Log experiments with MLflow
4. **Register**: Best model in MLflow Model Registry
5. **Document**: Create model cards for governance

---

**Ready to use with AWS SageMaker Studio and MLflow! 🚀**