# Quick Start Guide

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt Môi Trường

```bash
# Cách 1: Tự động (Recommended)
python setup.py

# Cách 2: Thủ công
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r backend/requirements.txt
```

### 2. Train Model

```bash
# Kích hoạt virtual environment
venv\Scripts\activate

# Mở Jupyter notebook
jupyter notebook notebooks/03_model_training.ipynb

# Chạy tất cả cells để train model
# Model sẽ được lưu tự động vào backend/model/
```

### 3. Chạy Backend Server

```bash
# Di chuyển vào thư mục backend
cd backend

# Chạy Flask server
python app.py

# Server sẽ chạy tại http://localhost:5000
```

### 4. Mở Frontend

- Mở file `frontend/index.html` trong browser
- Hoặc sử dụng Live Server extension trong VS Code

### 5. Test API

Sử dụng curl hoặc Postman:

```bash
# Health check
curl http://localhost:5000/api/health

# Model info
curl http://localhost:5000/api/model-info

# Prediction (California Housing - 8 features)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]}"
```

## 📊 Feature Names (California Housing)

1. MedInc: Median income
2. HouseAge: House age
3. AveRooms: Average rooms
4. AveBedrms: Average bedrooms
5. Population: Population
6. AveOccup: Average occupancy
7. Latitude: Latitude
8. Longitude: Longitude

## 🐛 Troubleshooting

### Lỗi: Model not found
- Chưa train model → Chạy notebook 03_model_training.ipynb

### Lỗi: Module not found
- Chưa install dependencies → `pip install -r backend/requirements.txt`

### Lỗi: Port already in use
- Port 5000 đang được dùng → Đổi port trong app.py hoặc tắt process đang dùng port

### CORS Error
- Đảm bảo frontend và backend đang chạy
- Check CORS_ORIGINS trong config.py

## 📝 Next Steps

1. ✅ Train model với notebook
2. ✅ Test API endpoints
3. ✅ Test frontend predictions
4. 🔲 Deploy to production (Render/Railway)
5. 🔲 Setup custom domain (Optional)
6. 🔲 Enable HTTPS

## 📚 Documentation

- [Project Requirements](PROJECT_REQUIREMENTS.md)
- [Activity Log](activity.md)
- [Error Log](error_log.md)
- [README](README.md)
