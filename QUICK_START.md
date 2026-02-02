# Quick Start Guide

Get started with the Water Quality Prediction project in 5 minutes!

## 📋 Prerequisites

Before you begin, ensure you have:
- **Python 3.8 or higher** installed ([Download Python](https://www.python.org/downloads/))
- **Jupyter Notebook** or **JupyterLab**
- **Git** (for cloning the repository)
- A modern web browser (Chrome, Firefox, Edge, or Safari)

## 🚀 Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd First_project
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\\Scripts\\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- Flask & Flask-CORS (Web framework)
- Scikit-learn (Machine learning)
- Pandas & NumPy (Data processing)
- Matplotlib & Seaborn (Visualization)
- Jupyter (Notebook interface)
- Other dependencies

## 🎯 Running the Project

### Option 1: Run Everything in One Go

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open the main notebook:**
   - Navigate to `train_and_deploy_model.ipynb`
   - Click "Cell" → "Run All"
   - Wait for all cells to execute (~2-5 minutes)

3. **Start Flask Server:**
   - Scroll to the last cell in the notebook
   - Run the cell that starts with `app.run(...)`
   - Server will start on `http://localhost:5000`

4. **Open Web Interface:**
   - Open your browser
   - Go to: `http://localhost:5000`
   - Test the prediction interface!

### Option 2: Step-by-Step Execution

1. **Launch Jupyter:**
```bash
jupyter notebook train_and_deploy_model.ipynb
```

2. **Run cells sequentially:**
   - **Section 1-2**: Setup and data loading
   - **Section 3**: Data preprocessing
   - **Section 4-5**: Model training and validation
   - **Section 6**: Model testing
   - **Section 7**: Export model
   - **Section 8**: Flask deployment

3. **After model is exported**, you can run Flask separately:

Create `run_flask.py`:
```python
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder='front_end', static_url_path='')
CORS(app)

# Load model
model_data = joblib.load('model/best_model.pkl')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    X_imputed = model_data['imputer'].transform(features)
    X_scaled = model_data['scaler'].transform(X_imputed)
    prediction = model_data['model'].predict(X_scaled)
    return jsonify({'success': True, 'data': {'prediction': float(prediction[0])}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

Then run:
```bash
python run_flask.py
```

## 🧪 Testing the Application

### Test API Endpoints

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Model Info:**
```bash
curl http://localhost:5000/api/model-info
```

**Make Prediction (example):**
```bash
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [28.0, 6.5, 25.0, 100.0, 15.0, 60.0]}'
```

### Test Web Interface

1. Open `http://localhost:5000` in your browser
2. Fill in water quality parameters:
   - Temperature: 28.0
   - pH: 6.5
   - Turbidity: 25.0
   - Color: 100.0
   - Suspended Solids: 15.0
   - Electrical Conductivity: 60.0
3. Click "Predict" button
4. View the predicted coagulant dosage

## 📊 Expected Results

After running the notebook, you should see:
- **Model trained** on 6,430 training samples
- **Validation metrics** displayed
- **Test set metrics** (R², MAE, RMSE)
- **Visualizations** saved to `front_end/assets/`
- **Model file** saved to `model/best_model.pkl`
- **Flask server** running on port 5000

##  📸 Screenshot After Deployment

After successfully deploying and testing:

1. Open the web interface at `http://localhost:5000`
2. Enter test values and get a prediction
3. Take a screenshot showing:
   - The URL bar with `localhost:5000`
   - Input fields filled with values
   - Prediction result displayed
4. Save screenshot for documentation (optional)

## 🔧 Troubleshooting

### Port Already in Use

If port 5000 is busy:
```python
# Change port in the last cell of notebook
app.run(host='0.0.0.0', port=5001, debug=False)  # Use 5001 instead
```

### Module Not Found

Reinstall dependencies:
```bash
pip install --force-reinstall -r requirements.txt
```

### Model File Not Found

Ensure you've run all cells in the notebook before starting Flask server.

### Browser Can't Connect

Check that:
- Flask server is running (check terminal output)
- No firewall blocking localhost:5000
- Using the correct URL: `http://localhost:5000` (not `https`)

## 📁 Project Files

After running, you should have:

```
First_project/
├── train_and_deploy_model.ipynb  ✅ Main notebook
├── model/
│   ├── best_model.pkl             ✅ Trained model
│   └── metrics.json               ✅ Performance metrics
├── data/
│   ├── raw/                       ✅ Original data
│   ├── train/train.csv            ✅ Training set
│   ├── validate/validate.csv      ✅ Validation set
│   └── test/test.csv              ✅ Test set
├── front_end/
│   ├── index.html                 ✅ Web interface
│   └── assets/
│       ├── test_predictions.png   ✅ Generated plots
│       └── residuals.png          ✅ Generated plots
└── README.md                      ✅ Documentation
```

## 🎓 Next Steps

1. **Explore the notebook** to understand the ML pipeline
2. **Modify hyperparameters** to see if you can improve performance
3. **Try different models** by editing the models dictionary
4. **Customize the frontend** in `front_end/` folder
5. **Read full documentation** in `PROJECT_REQUIREMENT.MD`

## 💡 Tips

- **Save your work**: The notebook auto-saves, but manually save important changes
- **Restart kernel**: If you encounter issues, try "Kernel → Restart & Run All"
- **Check logs**: Flask outputs helpful error messages in the terminal
- **Model persistence**: The model is saved to disk, no need to retrain every time

## 📞 Support

For detailed information:
- Full README: [README.md](README.md)
- Project requirements: [PROJECT_REQUIREMENT.MD](PROJECT_REQUIREMENT.MD)
- Jupyter notebook: [train_and_deploy_model.ipynb](train_and_deploy_model.ipynb)

---

**Happy Predicting! 🎉**
