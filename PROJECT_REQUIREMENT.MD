# Project Requirements: AI Model Web Integration

## 📋 Tổng Quan Dự Án (Project Overview)

**Tên dự án:** AI Model Web Integration Platform

**Mục tiêu:** Xây dựng một ứng dụng web responsive cho phép người dùng tương tác với một AI model cơ bản thông qua giao diện web, sau đó deploy lên server để công khai.

**Đối tượng sử dụng:** Người dùng cuối muốn trải nghiệm khả năng của AI model qua trình duyệt web.

---

## 🎯 Mục Tiêu Chính (Main Objectives)

1. ✅ Xây dựng regression model (nâng cấp từ Linear Regression)
2. ✅ Xử lý missing data và preprocessing pipeline
3. ✅ Tạo một trang web responsive hiện đại
4. ✅ Xây dựng backend API với Python (Flask/FastAPI)
5. ✅ Deploy thành công lên server công khai
6. ✅ Đảm bảo model prediction chính xác và stable

---

## 🛠️ Công Nghệ Sử Dụng (Technology Stack)

### Backend (Python - Required)
- **Framework:** Flask hoặc FastAPI
  - Flask: Đơn giản, phù hợp cho dự án nhỏ
  - FastAPI: Hiện đại, hỗ trợ async, tài liệu API tự động
- **AI/ML Libraries:**
  - **Scikit-learn** (Primary - cho Linear Regression models)
  - **NumPy** (Array operations)
  - **Pandas** (Data manipulation và preprocessing)
  - **Joblib** hoặc **Pickle** (Model serialization)
- **Server:** Gunicorn hoặc Uvicorn
- **Environment Management:** venv hoặc conda

### Frontend (Responsive Web)
- **HTML5:** Cấu trúc trang web
- **CSS3:** Styling với Flexbox/Grid cho responsive
  - Mobile-first approach
  - Breakpoints: 320px, 768px, 1024px, 1440px
- **JavaScript:** Xử lý tương tác và gọi API
  - Vanilla JS hoặc lightweight framework
  - Fetch API để gọi backend

### AI Model
- **Loại model:** Regression Model (Bản nâng cấp của Linear Regression)
  - Base: Linear Regression
  - Enhancements: Ridge Regression, Lasso Regression, ElasticNet, hoặc Polynomial Features
  - Purpose: Dự đoán giá trị liên tục dựa trên input features
- **Model Format:** 
  - Saved model (.pkl với pickle/joblib)
  - Scaler object (StandardScaler/MinMaxScaler)
  - Tích hợp qua Python backend
- **Input Data Type:** Numerical features only (float64)
- **Data Characteristics:**
  - ✅ All columns are float64 (numerical)
  - ⚠️ Contains missing values requiring preprocessing

### Deployment
- **Server Options:**
  - **Miễn phí:** Render, Railway, PythonAnywhere, Heroku (limited)
  - **VPS:** DigitalOcean, AWS EC2, Google Cloud
- **Domain:** Tùy chọn (có thể dùng subdomain miễn phí)
- **HTTPS:** SSL certificate (Let's Encrypt)

---

## 📱 Yêu Cầu Responsive Design

### Mobile (320px - 767px)
- ✅ Single column layout
- ✅ Touch-friendly buttons (min 44px)
- ✅ Hamburger menu nếu có navigation
- ✅ Font size tối thiểu 16px
- ✅ Optimized images

### Tablet (768px - 1023px)
- ✅ 2-column layout khi phù hợp
- ✅ Larger touch targets
- ✅ Adaptive navigation

### Desktop (1024px+)
- ✅ Multi-column layout
- ✅ Hover effects
- ✅ Full navigation bar
- ✅ Optimal content width (không quá rộng)

### Testing Requirements
- ✅ Test trên Chrome, Firefox, Safari
- ✅ Test trên iOS và Android
- ✅ Lighthouse score > 80

---

## 🎨 Tính Năng Chính (Core Features)

### 1. Trang Chủ (Home Page)
- Giới thiệu về AI model
- Demo nhanh hoặc ví dụ
- Call-to-action rõ ràng

### 2. Model Interface
- **Input Area:**
  - Form inputs với multiple numerical fields
  - Input validation (chỉ chấp nhận số)
  - Optional: CSV file upload cho batch prediction
- **Processing:**
  - Loading indicator khi đang xử lý
  - Progress feedback
- **Output Area:**
  - Predicted value (số liên tục)
  - Visualization: input features chart
  - Model confidence interval (nếu có)
  - Feature importance display

### 3. About/Documentation
- Giải thích model hoạt động như thế nào
- Limitations và use cases
- Technical details (optional)

### 4. API Endpoints (Backend)

```python
# Core API endpoints required:

POST /api/predict
- Input: User data (JSON, Form-data, hoặc File)
- Output: Prediction results (JSON)
- Status codes: 200, 400, 500

GET /api/health
- Output: Service status
- For monitoring

GET /api/model-info
- Output: Model metadata, version, capabilities
```

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

```
┌─────────────────┐
│   Web Browser   │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP/HTTPS
         ▼
┌─────────────────┐
│   Web Server    │
│  (Serve HTML/   │
│   CSS/JS)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python Backend │
│  (Flask/FastAPI)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   AI Model      │
│  (Loaded in     │
│   Memory)       │
└─────────────────┘
```

---

## 📦 Cấu Trúc Dự Án (Project Structure)

```
project-root/
│
├── backend/
│   ├── app.py                 # Main Flask/FastAPI app
│   ├── model/
│   │   ├── model.py          # Model loading & inference
│   │   ├── trained_model.pkl # Saved regression model
│   │   └── scaler.pkl        # Saved scaler object
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── imputer.py        # Missing value handling
│   │   ├── scaler.py         # Feature scaling
│   │   └── validator.py      # Input validation
│   ├── requirements.txt      # Python dependencies
│   ├── config.py             # Configuration
│   └── utils.py              # Helper functions
│
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Preprocessed data
│   └── train_test/           # Split data
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning
│   └── 03_model_training.ipynb # Model development
│
├── frontend/
│   ├── index.html            # Main page
│   ├── css/
│   │   └── style.css        # Responsive styles
│   ├── js/
│   │   └── main.js          # Frontend logic
│   └── assets/
│       └── images/          # Static assets
│
├── tests/
│   ├── test_api.py          # API tests
│   ├── test_model.py        # Model tests
│   └── test_preprocessing.py # Preprocessing tests
│
├── .gitignore
├── README.md
└── requirements.txt         # All dependencies
```

---

## 🚀 Development Workflow

### Phase 1: Setup & Planning (Day 1-2)
- [ ] Chọn dataset (California Housing, Diabetes, hoặc custom)
- [ ] Setup Python environment (Python 3.8+)
- [ ] Initialize Git repository
- [ ] Create project structure
- [ ] Install dependencies: `pip install scikit-learn pandas numpy flask`

### Phase 1.5: Data Analysis & Preprocessing (Day 3-5) ⚠️ CRITICAL
- [ ] Load và explore dataset
- [ ] Analyze missing values pattern
- [ ] Implement missing value imputation
- [ ] Feature scaling implementation
- [ ] Outlier detection và handling
- [ ] Train-test split (80-20 hoặc 70-30)
- [ ] Save preprocessing objects (imputer, scaler)
- [ ] Document preprocessing steps

### Phase 2: Model Development (Day 6-8)
- [ ] Train baseline Linear Regression
- [ ] Compare với Ridge, Lasso, ElasticNet
- [ ] Evaluate metrics: MAE, MSE, RMSE, R² Score
- [ ] Select best model variant
- [ ] Save model với joblib: `joblib.dump(model, 'model.pkl')`
- [ ] Create inference pipeline (preprocessing + prediction)
- [ ] Test model locally với sample data

### Phase 2.5: Model Validation (Day 9-10) ⚠️ CRITICAL
- [ ] Perform k-fold cross-validation (k=5 or k=10)
- [ ] Compare models using cross-validation scores
- [ ] Hyperparameter tuning với GridSearchCV/RandomizedSearchCV
- [ ] Validate on hold-out test set
- [ ] Check for overfitting/underfitting
- [ ] Generate validation report với metrics
- [ ] Ensure reproducibility (random_state=42 everywhere)

### Phase 3: Backend Development (Week 2)
- [ ] Setup Flask/FastAPI
- [ ] Create API endpoints
- [ ] Integrate model with API
- [ ] Add error handling
- [ ] Test API với Postman/curl

### Phase 4: Frontend Development (Week 2-3)
- [ ] Design UI/UX mockup
- [ ] Develop HTML structure
- [ ] Create responsive CSS
- [ ] Implement JavaScript API calls
- [ ] Add loading states & error handling
- [ ] Cross-browser testing

### Phase 5: Integration Testing (Week 3)
- [ ] Test frontend + backend integration
- [ ] Performance testing
- [ ] Responsive design testing
- [ ] Bug fixes

### Phase 6: Deployment (Week 3-4)
- [ ] Chọn hosting platform
- [ ] Setup server environment
- [ ] Configure environment variables
- [ ] Deploy backend
- [ ] Deploy frontend
- [ ] Setup domain (optional)
- [ ] Enable HTTPS
- [ ] Final testing on production

### Phase 7: Documentation (Week 4)
- [ ] User documentation
- [ ] Technical documentation
- [ ] README with setup instructions
- [ ] Demo video (optional)

---

## 🔒 Yêu Cầu Bảo Mật (Security Requirements)

- ✅ Input validation và sanitization
- ✅ Rate limiting trên API
- ✅ CORS configuration đúng
- ✅ Environment variables cho sensitive data
- ✅ HTTPS trong production
- ✅ Không expose model weights công khai

---

## 📊 Yêu Cầu Hiệu Năng (Performance Requirements)

- ✅ API response time < 3 seconds
- ✅ Page load time < 2 seconds
- ✅ Mobile-friendly (Google Mobile-Friendly Test)
- ✅ Lighthouse Performance Score > 80
- ✅ Model inference time < 2 seconds

---

## 📝 Deliverables (Sản Phẩm Bàn Giao)

1. **Source Code:**
   - Complete codebase trên GitHub
   - Clean, commented code
   - Git history rõ ràng

2. **Deployed Application:**
   - Live URL có thể truy cập
   - Working demo

3. **Documentation:**
   - README.md với setup instructions
   - API documentation
   - User guide

4. **Presentation:**
   - Demo slides/video
   - Technical explanation
   - Challenges & solutions

---

## 🎓 Tiêu Chí Đánh Giá (Evaluation Criteria)

| Tiêu chí | Trọng số | Mô tả |
|----------|----------|-------|
| **Functionality** | 30% | Model hoạt động đúng, API stable |
| **Code Quality** | 20% | Clean code, best practices, Python standards |
| **Responsive Design** | 20% | Mobile/tablet/desktop friendly |
| **Deployment** | 15% | Successfully deployed và accessible |
| **Documentation** | 10% | Clear README, comments, API docs |
| **UI/UX** | 5% | User-friendly interface |

---

## 💡 Chi Tiết Model & Data Preprocessing

### Linear Regression Enhancement Options

#### Option 1: Ridge Regression (L2 Regularization)
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
```
- **Advantages:** Giảm overfitting, stable với multicollinearity
- **Use case:** Khi features có correlation cao

#### Option 2: Lasso Regression (L1 Regularization)
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=1.0)
```
- **Advantages:** Feature selection tự động, sparse model
- **Use case:** Khi có nhiều features không quan trọng

#### Option 3: ElasticNet (L1 + L2)
```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
```
- **Advantages:** Kết hợp ưu điểm của Ridge và Lasso
- **Use case:** Balanced approach

#### Option 4: Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
model = LinearRegression()
```
- **Advantages:** Capture non-linear relationships
- **Use case:** Khi relationship không hoàn toàn linear

---

### 🔧 Data Preprocessing Requirements (CRITICAL)

> [!IMPORTANT]
> Dữ liệu có missing values - preprocessing là bắt buộc!

#### 1. Missing Data Handling

**Phương pháp Fill Missing Values:**

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Option A: Mean/Median Imputation
imputer = SimpleImputer(strategy='mean')  # hoặc 'median'
X_filled = imputer.fit_transform(X)

# Option B: Forward/Backward Fill (nếu có time series)
df.fillna(method='ffill')  # forward fill
df.fillna(method='bfill')  # backward fill

# Option C: KNN Imputer (advanced)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_filled = imputer.fit_transform(X)
```

**Bắt buộc phải làm:**
- ✅ Detect missing values: `df.isnull().sum()`
- ✅ Visualize missing pattern: `import missingno as msno`
- ✅ Choose appropriate imputation strategy
- ✅ Document imputation method trong code

#### 2. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (recommended for Linear Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# MinMaxScaler (alternative)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_filled)
```

#### 3. Outlier Detection (Optional but Recommended)

```python
from scipy import stats
import numpy as np

# Z-score method
z_scores = np.abs(stats.zscore(X))
X_no_outliers = X[(z_scores < 3).all(axis=1)]

# IQR method
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X_no_outliers = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
```

#### 4. Data Validation

```python
# Ensure all data is float64
assert X.dtypes.all() == 'float64'

# Check no missing values after imputation
assert X_filled.isnull().sum().sum() == 0

# Check data shape
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
```

---

### 🎯 Model Validation & Selection (CRITICAL)

> [!IMPORTANT]
> Model validation đảm bảo model có hiệu suất tốt nhất và không bị overfitting!

#### 1. Train-Test Split với Random State Cố Định

```python
from sklearn.model_selection import train_test_split

# CRITICAL: Always use random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2,  # 80-20 split
    random_state=42  # ⚠️ CỐ ĐỊNH = 42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

**Lý do random_state=42:**
- ✅ Reproducibility: Kết quả giống nhau mỗi lần chạy
- ✅ Debugging: Dễ dàng so sánh và debug
- ✅ Collaboration: Team members có cùng kết quả

#### 2. Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42)
}

# Perform 5-fold cross-validation
for name, model in models.items():
    # Use cv=5 or cv=10 for k-fold
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=5,  # 5-fold cross-validation
        scoring='r2',  # or 'neg_mean_squared_error'
        n_jobs=-1  # Use all CPU cores
    )
    
    print(f"{name}:")
    print(f"  Mean R² Score: {scores.mean():.4f}")
    print(f"  Std Dev: {scores.std():.4f}")
    print(f"  Scores: {scores}")
```

**Cross-validation metrics to track:**
- R² Score (coefficient of determination)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

#### 3. Hyperparameter Tuning với GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Example: Tune Ridge Regression
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

ridge = Ridge(random_state=42)

grid_search = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

#### 4. Model Evaluation on Test Set

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Test Set Performance ===")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
```

#### 5. Validation Report Template

```python
import pandas as pd

# Create validation report
validation_report = {
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet'],
    'CV_Mean_R2': [],  # From cross-validation
    'CV_Std_R2': [],
    'Test_R2': [],     # From test set
    'Test_MAE': [],
    'Test_RMSE': []
}

df_report = pd.DataFrame(validation_report)
df_report.to_csv('model_validation_report.csv', index=False)
print(df_report)
```

#### 6. Overfitting Detection

```python
# Compare training vs test performance
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

print(f"Training R² Score: {train_score:.4f}")
print(f"Test R² Score: {test_score:.4f}")
print(f"Difference: {abs(train_score - test_score):.4f}")

if abs(train_score - test_score) > 0.1:
    print("⚠️ WARNING: Possible overfitting detected!")
else:
    print("✅ Model generalizes well")
```

#### 7. Learning Curve (Optional but Recommended)

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
```

**Validation Checklist:**
- ✅ random_state=42 used in all random operations
- ✅ Cross-validation performed (k=5 or k=10)
- ✅ Multiple models compared
- ✅ Hyperparameters tuned
- ✅ Test set evaluation completed
- ✅ Overfitting checked
- ✅ Validation report generated

---

### 📊 Suggested Datasets (Float64 Numerical Only)

1. **California Housing Dataset**
   - 8 features, all numerical
   - Predict median house value
   ```python
   from sklearn.datasets import fetch_california_housing
   data = fetch_california_housing()
   ```

2. **Boston Housing Dataset** (Classic)
   - 13 features, all numerical
   - Predict house prices
   ```python
   # Note: Deprecated in sklearn, use alternative
   import pandas as pd
   url = "http://lib.stat.cmu.edu/datasets/boston"
   ```

3. **Diabetes Dataset**
   - 10 features, all numerical
   - Predict disease progression
   ```python
   from sklearn.datasets import load_diabetes
   data = load_diabetes()
   ```

4. **Custom Dataset**
   - CSV file với all float64 columns
   - Manually introduce missing values để practice preprocessing

---

## 📦 Python Requirements (requirements.txt)

```txt
# Core dependencies
Flask==3.0.0
gunicorn==21.2.0

# ML Libraries
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2

# Data preprocessing
scipy==1.11.2

# Optional: Visualization trong notebooks
matplotlib==3.7.2
seaborn==0.12.2
missingno==0.5.2

# Testing
pytest==7.4.0

# CORS support
flask-cors==4.0.0

# Environment variables
python-dotenv==1.0.0
```

> [!TIP]
> Install với: `pip install -r requirements.txt`

---

## 📚 Resources & References

### Learning Resources
- **Flask:** https://flask.palletsprojects.com/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Responsive Design:** https://web.dev/responsive-web-design-basics/
- **Deployment:** 
  - Render: https://render.com/docs
  - Railway: https://docs.railway.app/

### Python Libraries Documentation
- TensorFlow: https://www.tensorflow.org/
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/

---

## ⚠️ Common Pitfalls to Avoid

1. ❌ Model quá lớn, inference chậm
2. ❌ Không test responsive đầy đủ
3. ❌ Hardcode credentials trong code
4. ❌ Không handle errors properly
5. ❌ CORS issues khi deploy
6. ❌ Không optimize images
7. ❌ Quên setup environment variables

---

## ✅ Success Criteria

Project được coi là thành công khi:

- ✅ Website accessible qua public URL
- ✅ Model prediction hoạt động chính xác
- ✅ Responsive trên mobile, tablet, desktop
- ✅ API response time hợp lý
- ✅ Code được commit lên GitHub
- ✅ Documentation đầy đủ
- ✅ Demo thành công

---

## 📞 Next Steps

1. **Review document này** và đảm bảo hiểu rõ requirements
2. **Chọn AI model cụ thể** từ các gợi ý trên
3. **Setup development environment** (Python, IDE, Git)
4. **Bắt đầu Phase 1** theo Development Workflow
5. **Track progress** và update theo checklist

---

**Good luck with your project! 🚀**

*Document này có thể được update trong quá trình development khi có thêm requirements mới.*
