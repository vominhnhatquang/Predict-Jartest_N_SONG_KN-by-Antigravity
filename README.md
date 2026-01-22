# AI Model Web Integration Platform

## 📋 Project Overview

This project integrates a regression AI model into a responsive web application with Python backend (Flask) and deployment to a public server.

## 🎯 Features

- Regression model for continuous value prediction
- RESTful API backend with Flask
- Responsive web interface (mobile-first design)
- Data preprocessing pipeline (missing value handling, scaling)
- Model validation and comparison

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **Scikit-learn** - ML library
- **Pandas & NumPy** - Data manipulation
- **Joblib** - Model serialization

### Frontend
- **HTML5** - Structure
- **CSS3** - Responsive styling
- **Vanilla JavaScript** - Interactivity

### AI Model
- Regression models: Linear Regression, Ridge, Lasso, ElasticNet
- Data preprocessing: Imputation, Scaling
- Validation: Cross-validation, Hyperparameter tuning

## 📦 Project Structure

```
First_project/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── model/
│   │   ├── model.py          # Model loading & inference
│   │   ├── trained_model.pkl # Saved model
│   │   └── scaler.pkl        # Saved scaler
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── imputer.py        # Missing value handling
│   │   ├── scaler.py         # Feature scaling
│   │   └── validator.py      # Input validation
│   ├── config.py             # Configuration
│   ├── utils.py              # Helper functions
│   └── requirements.txt      # Python dependencies
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Preprocessed data
│   └── train_test/           # Split data
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning
│   └── 03_model_training.ipynb # Model development
├── frontend/
│   ├── index.html            # Main page
│   ├── css/
│   │   └── style.css        # Responsive styles
│   ├── js/
│   │   └── main.js          # Frontend logic
│   └── assets/              # Static assets
├── tests/
│   ├── test_api.py          # API tests
│   ├── test_model.py        # Model tests
│   └── test_preprocessing.py # Preprocessing tests
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd First_project
```

2. Create virtual environment
```bash
python -m venv venv
```

3. Activate virtual environment
- Windows:
```bash
venv\Scripts\activate
```
- Mac/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### Running the Application

1. Start the Flask backend
```bash
cd backend
python app.py
```

2. Open the frontend
- Open `frontend/index.html` in your browser
- Or serve with a local server

## 📊 Model Information

- **Model Type:** Regression (Ridge/Lasso/ElasticNet)
- **Input:** Numerical features (float64)
- **Output:** Continuous value prediction
- **Preprocessing:** Missing value imputation, feature scaling
- **Validation:** k-fold cross-validation, random_state=42

## 🌐 API Endpoints

### POST /api/predict
Predict values based on input features
- **Input:** JSON with feature values
- **Output:** Prediction result

### GET /api/health
Check API health status

### GET /api/model-info
Get model metadata and capabilities

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

## 📱 Responsive Design

- **Mobile:** 320px - 767px
- **Tablet:** 768px - 1023px
- **Desktop:** 1024px+

## 🔐 Security

- Input validation
- CORS configuration
- Environment variables for sensitive data
- HTTPS in production

## 📈 Performance Goals

- API response time < 3 seconds
- Page load time < 2 seconds
- Lighthouse score > 80

## 📝 Documentation

- [Project Requirements](PROJECT_REQUIREMENTS.md)
- [Activity Log](activity.md)
- [Error Log](error_log.md)

## 🤝 Contributing

1. Follow Python PEP 8 style guide
2. Write tests for new features
3. Update documentation
4. Use meaningful commit messages

## 📄 License

This project is for educational purposes.

## 👤 Author

Võ Minh Nhật Quang - FPT AIL303 Student

## 🙏 Acknowledgments

- Scikit-learn documentation
- Flask documentation
- Project requirements and guidelines

---

*Last Updated: 2026-01-21*
