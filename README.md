# Water Quality Prediction - AI Model Web Integration

A complete machine learning project for predicting water quality parameters using regression models, with a responsive web interface and Flask API backend.

## 📋 Project Overview

This project implements an end-to-end regression model for water quality prediction:
- **Dataset**: Water quality measurements (pH, temperature, turbidity, etc.)
- **Model**: Best performing regression model selected from multiple algorithms
- **Deployment**: Flask API with responsive frontend interface
- **Data Split**: Train (80%), Validate (10%), Test (10%) with fixed random_state=42

## 🏗️ Project Structure

```
First_project/
├── train_and_deploy_model.ipynb  # Complete ML pipeline notebook
├── model/
│   ├── best_model.pkl             # Trained model + preprocessing objects
│   └── metrics.json               # Model performance metrics
├── front_end/                     # Responsive web interface
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── assets/
├── data/
│   ├── raw/                       # Original dataset
│   ├── train/                     # Training data (80%)
│   ├── validate/                  # Validation data (10%)
│   └── test/                      # Test data (10%)
├── PROJECT_REQUIREMENT.MD         # Detailed project requirements
├── README.md                      # This file
├── QUICK_START.md                 # Quick start guide
└── .gitignore
```

## 🚀 Quick Start

See [QUICK_START.md](QUICK_START.md) for detailed setup and running instructions.

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Modern web browser

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd First_project
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Run the Jupyter Notebook**:
```bash
jupyter notebook train_and_deploy_model.ipynb
```

2. **Execute all cells** to:
   - Load and preprocess data
   - Train and validate models
   - Select best model
   - Export model files

3. **Start Flask server** (run the last cells in notebook):
   - The server will start on `http://localhost:5000`

4. **Open web interface**:
   - Navigate to `http://localhost:5000` in your browser
   - Enter water quality parameters
   - Get predictions!

## 📊 Model Performance

The model is trained on 6,430 samples and validated on separate validation (804) and test (804) sets.

View detailed metrics in:
- `model/metrics.json` - JSON format
- Last cells of `train_and_deploy_model.ipynb` - Visual analysis

## 🎯 Features

- **Complete ML Pipeline**: Data preprocessing, training, validation, testing
- **Multiple Models Compared**: Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: Automatic optimization using GridSearchCV
- **Responsive Web UI**: Works on desktop, tablet, and mobile
- **RESTful API**: Flask-based prediction API
- **Reproducible Results**: Fixed random_state=42 for all random operations

## 📡 API Endpoints

### Health Check
```
GET /api/health
Response: {"status": "healthy", "model_loaded": true}
```

### Model Information
```
GET /api/model-info
Response: {"model_type": "...", "features": [...], "training_date": "..."}
```

### Make Prediction
```
POST /api/predict
Body: {"features": [value1, value2, ...]}
Response: {"success": true, "data": {"prediction": value}}
```

### Model Performance
```
GET /api/performance
Response: {"test_r2": ..., "test_mae": ..., "test_rmse": ...}
```

## 🛠️ Technology Stack

**Backend:**
- Python 3.8+
- Flask (Web framework)
- Scikit-learn (ML models)
- Pandas, NumPy (Data processing)
- Joblib (Model serialization)

**Frontend:**
- HTML5
- CSS3 (Responsive design)
- JavaScript (Vanilla JS)
- Fetch API (Backend communication)

**ML Pipeline:**
- Data imputation (SimpleImputer)
- Feature scaling (StandardScaler)
- Cross-validation (5-fold)
- Hyperparameter tuning (GridSearchCV)

## 📝 Data Information

- **Source**: `data/raw/DATA_FPT.csv`
- **Features**: 6 water quality parameters
  - NhietdoN_SONG (Temperature)
  - pH_N_SONG (pH level)
  - Duc_N_SONG (Turbidity)
  - Mau_N_SONG (Color)
  - SS_SONG (Suspended Solids)
  - EC_N_SONG (Electrical Conductivity)
- **Target**: Jartest_N_SONG_KN (Coagulant dosage)
- **Total Samples**: 8,038
- **Data Split**: 80/10/10 (train/validate/test)

## 🔍 Model Development Process

1. **Data Loading**: Load pre-split train/validate/test data
2. **Preprocessing**:
   - Handle missing values (mean imputation)
   - Feature scaling (standardization)
3. **Model Training**:
   - Train 6 different regression models
   - 5-fold cross-validation on training set
4. **Model Selection**:
   - Evaluate on validation set
   - Select best performing model
5. **Hyperparameter Tuning**:
   - GridSearchCV for optimal parameters
6. **Final Evaluation**:
   - Test on held-out test set
   - Generate comprehensive metrics and visualizations
7. **Export**:
   - Save model, scaler, imputer in single file
   - Save metrics as JSON

## 📖 Documentation

- **PROJECT_REQUIREMENT.MD**: Detailed project requirements and specifications
- **QUICK_START.md**: Quick start guide for new users
- **train_and_deploy_model.ipynb**: Complete pipeline with inline documentation

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the project maintainer.

## 📄 License

This project is created for educational purposes.

## 🔗 Related Files

- Detailed requirements: [PROJECT_REQUIREMENT.MD](PROJECT_REQUIREMENT.MD)
- Quick start guide: [QUICK_START.md](QUICK_START.md)
- Main notebook: [train_and_deploy_model.ipynb](train_and_deploy_model.ipynb)

---

**Last Updated**: 2026-01-31
**Version**: 2.0 (Restructured)
