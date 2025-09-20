# 🌸 Iris Classification with MLflow & AWS SageMaker Studio

A complete machine learning project demonstrating MLflow experiment tracking and model registry with AWS SageMaker Studio integration.

## 🎯 Project Overview

This repository provides a complete end-to-end machine learning pipeline for iris flower classification, designed to work seamlessly with AWS SageMaker Studio's MLflow tracking servers. The project demonstrates:

- **Multiple ML Algorithms**: Logistic Regression, Random Forest, SVM, Naive Bayes, Decision Tree
- **MLflow Integration**: Complete experiment tracking, model logging, and model registry
- **SageMaker Studio Ready**: Configured for AWS SageMaker Studio MLflow tracking servers
- **Model Comparison**: Automated model evaluation and champion model selection
- **Interactive Analysis**: Jupyter notebook for exploration and visualization

## 🏗️ Project Structure

```
poc_sagemaker_mlflow/
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── train.py                # Model training with MLflow
│   │   └── evaluate.py             # Model evaluation and comparison
│   └── utils/
│       └── mlflow_utils.py         # MLflow utilities and tracking
├── config/
│   └── mlflow_config.py            # MLflow and SageMaker configuration
├── notebooks/
│   └── iris_mlflow_demo.ipynb      # Interactive demo notebook
├── requirements.txt                # Python dependencies
├── run_demo.sh                     # Quick start script
└── README.md                       # This file
```

## 🚀 Quick Start

### Option 1: Automated Demo Script
```bash
# Clone the repository
git clone https://github.com/andres0415/poc_sagemaker_mlflow.git
cd poc_sagemaker_mlflow

# Run the complete demo
./run_demo.sh
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models
python src/models/train.py --model all --experiment iris_classification

# 4. Analyze results and register best model
python src/models/evaluate.py --experiment iris_classification --register-best
```

### Option 3: Interactive Notebook
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/iris_mlflow_demo.ipynb
```

## 🔧 AWS SageMaker Studio Integration

### Prerequisites
1. AWS SageMaker Studio environment
2. MLflow Tracking Server created in SageMaker Studio
3. S3 bucket for artifact storage
4. Appropriate IAM roles and permissions

### Setup Steps

1. **Create MLflow Tracking Server**
   ```bash
   # In SageMaker Console > Studio > MLflow
   # Create a new tracking server and note the endpoint URL
   ```

2. **Configure Environment Variables**
   ```bash
   export MLFLOW_TRACKING_URI=<your-mlflow-tracking-server-url>
   export MLFLOW_ARTIFACT_LOCATION=s3://<your-s3-bucket>/mlflow-artifacts/
   export SAGEMAKER_ROLE=<your-sagemaker-execution-role-arn>
   export SAGEMAKER_DEFAULT_BUCKET=<your-s3-bucket-name>
   export AWS_DEFAULT_REGION=<your-aws-region>
   ```

3. **Upload and Run**
   ```bash
   # Upload this repository to SageMaker Studio
   # Install requirements
   pip install -r requirements.txt
   
   # Run training with SageMaker MLflow integration
   python src/models/train.py --model all --experiment iris_sagemaker_studio
   ```

## 📊 Available Models

The project includes training and comparison of multiple algorithms:

| Algorithm | Description | Key Parameters |
|-----------|-------------|----------------|
| **Logistic Regression** | Linear classifier with regularization | C, solver, max_iter |
| **Random Forest** | Ensemble of decision trees | n_estimators, max_depth, min_samples_split |
| **Support Vector Machine** | SVM with RBF kernel | C, gamma, kernel |
| **Naive Bayes** | Gaussian Naive Bayes classifier | (no hyperparameters) |
| **Decision Tree** | Single decision tree | max_depth, min_samples_split |

## 🎯 MLflow Features Demonstrated

### Experiment Tracking
- Automatic parameter logging
- Comprehensive metrics tracking (accuracy, precision, recall, F1)
- Artifact logging (confusion matrices, feature importance plots)
- Model versioning and lineage

### Model Registry
- Automatic registration of best performing models
- Model versioning and stage management
- Model metadata and descriptions
- Model loading and inference

### Visualizations
- Confusion matrices
- Feature importance plots
- Model comparison charts
- Classification reports

## 📈 Example Results

After running the demo, you'll see results like:

```
MODEL COMPARISON
════════════════════════════════════════════════════════════
model_name          test_accuracy  test_f1_weighted  test_precision_weighted
random_forest            1.0000          1.0000               1.0000
svm                      1.0000          1.0000               1.0000  
logistic_regression      1.0000          1.0000               1.0000
decision_tree            0.9667          0.9667               0.9722
naive_bayes              0.9667          0.9667               0.9722

BEST MODEL
════════════════════════════════════════════════════════════
Model: random_forest
Run ID: abc123...
Accuracy: 1.0000
```

## 🔍 MLflow UI Access

### Local Development
```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

### SageMaker Studio
- Access through SageMaker Studio's MLflow interface
- Navigate to MLflow tab in SageMaker Studio
- View experiments, runs, and model registry

## 🛠️ Command Line Interface

### Training Individual Models
```bash
# Train specific model
python src/models/train.py --model random_forest --experiment my_experiment

# Train with custom tracking URI
python src/models/train.py --model svm --tracking-uri http://my-mlflow-server
```

### Model Evaluation
```bash
# Analyze experiment results
python src/models/evaluate.py --experiment my_experiment

# Register best model
python src/models/evaluate.py --experiment my_experiment --register-best --model-name my_champion_model
```

## 📦 Model Deployment

The registered models can be deployed using various methods:

### SageMaker Endpoints
```python
import mlflow.sagemaker as mfs

# Deploy to SageMaker endpoint
mfs.deploy(
    model_uri="models:/iris_champion_model/1",
    region_name="us-east-1",
    mode="create",
    execution_role_arn="arn:aws:iam::123456789:role/SageMakerRole",
    instance_type="ml.t2.medium"
)
```

### Local Serving
```bash
# Serve model locally
mlflow models serve -m models:/iris_champion_model/1 -p 5001
```

## 🔄 Model Cards and Documentation

Each registered model includes comprehensive documentation:
- Model algorithm and hyperparameters
- Performance metrics and evaluation results
- Training dataset information
- Usage examples and deployment notes
- Model lineage and experiment tracking

## 🧪 Testing the Pipeline

```bash
# Test data loading
python src/data/data_loader.py

# Test MLflow utilities
python src/utils/mlflow_utils.py

# Run complete pipeline test
python -m pytest tests/  # (if tests are added)
```

## 📋 Requirements

- Python 3.8+
- MLflow 2.8+
- Scikit-learn 1.3+
- AWS SDK (boto3) for SageMaker integration
- Jupyter for interactive analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **MLflow UI not accessible**
   - Check if port 5000 is available
   - Verify MLflow installation
   - Check firewall settings

2. **SageMaker permissions**
   - Ensure proper IAM roles
   - Verify S3 bucket permissions
   - Check MLflow tracking server access

3. **Model registration fails**
   - Verify MLflow tracking URI
   - Check experiment exists
   - Ensure successful training runs

### Support

For issues and questions:
1. Check the troubleshooting section above
2. Review MLflow documentation
3. Check AWS SageMaker Studio documentation
4. Create an issue in this repository

---

**Happy Machine Learning! 🚀**