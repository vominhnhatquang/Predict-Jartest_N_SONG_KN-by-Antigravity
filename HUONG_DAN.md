# 🚀 HƯỚNG DẪN CHẠY PROJECT - DATA_FPT

## ⚠️ YÊU CẦU

### 1. Cài Đặt Python
Trước tiên, cần cài Python 3.8 trở lên:

**Cách 1: Microsoft Store (Recommended)**
1. Mở Microsoft Store
2. Tìm "Python 3.12" hoặc "Python 3.11"
3. Click "Get" để cài đặt

**Cách 2: python.org**
1. Truy cập: https://www.python.org/downloads/
2. Download Python 3.11 hoặc 3.12
3. ✅ **QUAN TRỌNG:** Tick vào "Add Python to PATH" khi cài đặt!

### 2. Kiểm tra Python đã cài
Mở Command Prompt và chạy:
```bash
python --version
```
Hoặc:
```bash
python3 --version
```

Nếu hiển thị version (ví dụ: Python 3.11.x) → OK!

---

## 📊 BƯỚC 1: CÀI ĐẶT

### Tự động (Dễ nhất)
Double-click file: **`setup.bat`**

### Thủ công
```bash
# 1. Tạo virtual environment
python -m venv venv

# 2. Kích hoạt virtual environment
venv\Scripts\activate

# 3. Cài đặt dependencies
pip install -r backend\requirements.txt
```

---

## 🤖 BƯỚC 2: TRAIN MODEL

### Cách 1: Double-click (Dễ nhất)
Double-click file: **`train.bat`**

### Cách 2: Command line
```bash
# Kích hoạt virtual environment
venv\Scripts\activate

# Chạy training script
python train_model.py
```

**Thời gian:** Khoảng 1-2 phút

**Output:**
- `backend/model/trained_model.pkl` - Model đã train
- `backend/model/scaler.pkl` - Scaler object
- `backend/model/imputer.pkl` - Imputer object
- `data/processed/model_validation_report.csv` - Kết quả validation

---

## 🌐 BƯỚC 3: CHẠY WEBSITE

### A. Chạy Backend Server

**Cách 1: Double-click**
Double-click file: **`start_backend.bat`**

**Cách 2: Command line**
```bash
venv\Scripts\activate
cd backend
python app.py
```

Server sẽ chạy tại: **http://localhost:5000**

⚠️ **QUAN TRỌNG:** Giữ cửa sổ này mở! Đừng đóng!

### B. Mở Frontend

**Cách 1: Double-click**
Double-click file: **`open_frontend.bat`**

**Cách 2: Thủ công**
Double-click file: **`frontend/index.html`**

---

## 🎯 CÁCH SỬ DỤNG

1. ✅ Backend server đang chạy (cửa sổ CMD mở)
2. ✅ Frontend đã mở trong browser

### Nhập dữ liệu:
Nhập 6 giá trị cho các thông số nước sông:
- **Nhiệt độ Nước Sông** (VD: 28.5)
- **pH Nước Sông** (VD: 6.5)
- **Độ Đục Nước Sông** (VD: 25)
- **Màu Nước Sông** (VD: 150)
- **SS Sông** (VD: 10)
- **EC Nước Sông** (VD: 58.5)

Click **"Dự Đoán"** → Nhận kết quả Jartest!

---

## 📝 VÍ DỤ DỮ LIỆU

Từ file DATA_FPT.csv:

### Ví dụ 1:
- Nhiệt độ: 28.8
- pH: 6.6
- Độ Đục: 23
- Màu: 150
- SS: 10
- EC: 59.4
→ **Jartest dự đoán: ~16**

### Ví dụ 2:
- Nhiệt độ: 28.1
- pH: 6.5
- Độ Đục: 32
- Màu: 192
- SS: 18
- EC: 58
→ **Jartest dự đoán: ~16**

### Ví dụ 3:
- Nhiệt độ: 28.4
- pH: 6.5
- Độ Đục: 19
- Màu: 121
- SS: 6
- EC: 58.4
→ **Jartest dự đoán: ~16**

---

## 🐛 TROUBLESHOOTING

### Lỗi: "Python was not found"
→ Chưa cài Python hoặc chưa thêm vào PATH
→ **Giải pháp:** Cài lại Python, nhớ tick "Add to PATH"

### Lỗi: "Model not found"
→ Chưa train model
→ **Giải pháp:** Chạy `train.bat` trước

### Lỗi: "Port 5000 already in use"
→ Port đã được dùng
→ **Giải pháp:** Tắt ứng dụng đang dùng port 5000

### Lỗi: "Module not found"
→ Chưa cài dependencies
→ **Giải pháp:** Chạy `pip install -r backend\requirements.txt`

### Frontend không kết nối được API
→ Backend chưa chạy
→ **Giải pháp:** Chạy `start_backend.bat` trước

---

## 📂 CẤU TRÚC FILE

```
First_project/
├── train.bat              ← Chạy để train model
├── start_backend.bat      ← Chạy để start server
├── open_frontend.bat      ← Chạy để mở frontend
├── train_model.py         ← Script training model
├── setup.bat             ← Setup virtual environment
├── data/
│   └── raw/
│       └── DATA_FPT.csv  ← Your data
└── frontend/
    └── index.html        ← Web interface
```

---

## ✅ CHECKLIST

- [ ] Python đã cài (check: `python --version`)
- [ ] Virtual environment đã tạo (run: `setup.bat`)
- [ ] Model đã train (run: `train.bat`)
- [ ] Backend đang chạy (run: `start_backend.bat`)
- [ ] Frontend đã mở (run: `open_frontend.bat`)
- [ ] Test prediction thành công!

---

## 🎓 THÔNG TIN DATASET

**Dataset:** DATA_FPT.csv
**Số dòng:** ~1000 rows
**Features (6):**
1. Nhietdo_N_SONG (Nhiệt độ)
2. pH_N_SONG (pH)
3. Duc_N_SONG (Độ đục)
4. Mau_N_SONG (Màu)
5. SS_SONG (SS)
6. EC_N_SONG (EC)

**Target:** Jartest_N_SONG_KN (Giá trị cần dự đoán)

**Model:** Ridge Regression (tự động chọn best alpha)

---

**Chúc bạn thành công! 🎉**
