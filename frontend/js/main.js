/**
 * Main JavaScript for AI Model Web Integration
 * Handles API calls and UI interactions
 */

// API Configuration
// API Configuration
const API_BASE_URL = window.location.protocol === 'file:'
    ? 'http://127.0.0.1:5000'
    : window.location.origin;

const API_ENDPOINTS = {
    predict: `${API_BASE_URL}/api/predict`,
    health: `${API_BASE_URL}/api/health`,
    modelInfo: `${API_BASE_URL}/api/model-info`
};

// DOM Elements
let predictForm;
let predictBtn;
let btnText;
let loadingSpinner;
let resultContainer;
let predictionResult;
let errorContainer;
let errorMessage;
let modelInfoDiv;
let footerModelType;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Get DOM elements
    predictForm = document.getElementById('prediction-form');
    predictBtn = document.getElementById('predict-btn');
    btnText = predictBtn.querySelector('.btn-text');
    loadingSpinner = predictBtn.querySelector('.loading-spinner');
    resultContainer = document.getElementById('result-container');
    predictionResult = document.getElementById('prediction-result');
    errorContainer = document.getElementById('error-container');
    errorMessage = document.getElementById('error-message');
    modelInfoDiv = document.getElementById('model-info');
    footerModelType = document.getElementById('footer-model-type');

    // Set up event listeners
    predictForm.addEventListener('submit', handlePrediction);

    // Load initial data
    checkAPIHealth();
    loadModelInfo();
    loadPerformanceData(); // Add this line
});

/**
 * Load performance metrics
 */
async function loadPerformanceData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/performance`);
        const result = await response.json();

        if (result.success && result.data) {
            const metrics = result.data;

            const updateElement = (id, value) => {
                const el = document.getElementById(id);
                if (el) el.textContent = value;
            };

            updateElement('metric-r2', metrics.test_r2);
            updateElement('metric-mae', metrics.test_mae);
            updateElement('metric-rmse', metrics.test_rmse);
            updateElement('metric-mape', metrics.test_mape + '%');
            updateElement('metric-accuracy', metrics.test_accuracy + '%');
            updateElement('metric-model', metrics.best_model);

            // Show container
            const container = document.getElementById('performance-container');
            if (container) {
                container.style.display = 'block'; // Fix: Use style.display instead of classList
            }
        }
    } catch (error) {
        console.error('Error loading performance data:', error);
    }
}

/**
 * Check API health status
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(API_ENDPOINTS.health);
        const data = await response.json();

        if (data.success) {
            console.log('✅ API is healthy:', data.data);
        } else {
            console.warn('⚠️ API health check failed');
        }
    } catch (error) {
        console.error('❌ API health check error:', error);
        showError('Không thể kết nối với API. Vui lòng kiểm tra server.');
    }
}

/**
 * Load model information
 */
async function loadModelInfo() {
    try {
        const response = await fetch(API_ENDPOINTS.modelInfo);
        const data = await response.json();

        if (data.success && data.data) {
            displayModelInfo(data.data);
        } else {
            modelInfoDiv.innerHTML = '<p class="warning">⚠️ Model chưa được load. Vui lòng train và save model trước.</p>';
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        modelInfoDiv.innerHTML = '<p class="error">❌ Không thể tải thông tin model</p>';
    }
}

/**
 * Display model information
 */
function displayModelInfo(info) {
    let html = '';

    if (info.model_type) {
        html += `<p><strong>Model Type:</strong> <span>${info.model_type}</span></p>`;
        footerModelType.textContent = info.model_type;
    }

    if (info.n_features) {
        html += `<p><strong>Number of Features:</strong> <span>${info.n_features}</span></p>`;
    }

    html += `<p><strong>Preprocessing:</strong> <span>${info.has_scaler ? '✅ Scaler' : '❌ No Scaler'} | ${info.has_imputer ? '✅ Imputer' : '❌ No Imputer'}</span></p>`;

    modelInfoDiv.innerHTML = html;
}

/**
 * Handle prediction form submission
 */
async function handlePrediction(event) {
    event.preventDefault();

    // Hide previous results/errors
    hideResults();
    hideError();

    // Show loading state
    setLoadingState(true);

    try {
        // Get form data
        const formData = new FormData(predictForm);
        const features = [];

        for (let [key, value] of formData.entries()) {
            if (key.startsWith('feature')) {
                features.push(parseFloat(value));
            }
        }

        // Validate features
        if (features.length === 0) {
            throw new Error('Vui lòng nhập ít nhất một feature');
        }

        if (features.some(isNaN)) {
            throw new Error('Tất cả các giá trị phải là số hợp lệ');
        }

        // Make API call
        const response = await fetch(API_ENDPOINTS.predict, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features })
        });

        const data = await response.json();

        if (data.success) {
            displayResult(data.data);
        } else {
            throw new Error(data.message || 'Prediction failed');
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Có lỗi xảy ra khi thực hiện dự đoán');
    } finally {
        setLoadingState(false);
    }
}

/**
 * Display prediction result
 */
function displayResult(data) {
    let html = '';

    if (data.prediction_value !== undefined) {
        html += `<div class="prediction-value">${data.prediction_value.toFixed(4)}</div>`;
    }

    if (data.model_type) {
        html += `<p><strong>Model sử dụng:</strong> ${data.model_type}</p>`;
    }

    if (data.predictions && Array.isArray(data.predictions)) {
        html += `<p><strong>Chi tiết:</strong> ${data.predictions.join(', ')}</p>`;
    }

    predictionResult.innerHTML = html;
    resultContainer.style.display = 'block';

    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.innerHTML = `<p>${message}</p>`;
    errorContainer.style.display = 'block';

    // Scroll to error
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide results
 */
function hideResults() {
    resultContainer.style.display = 'none';
}

/**
 * Hide error
 */
function hideError() {
    errorContainer.style.display = 'none';
}

/**
 * Set loading state
 */
function setLoadingState(isLoading) {
    if (isLoading) {
        predictBtn.disabled = true;
        btnText.style.display = 'none';
        loadingSpinner.style.display = 'inline';
    } else {
        predictBtn.disabled = false;
        btnText.style.display = 'inline';
        loadingSpinner.style.display = 'none';
    }
}

/**
 * Format number for display
 */
function formatNumber(num, decimals = 2) {
    return num.toFixed(decimals);
}

/**
 * Validate numeric input
 */
function isValidNumber(value) {
    return !isNaN(parseFloat(value)) && isFinite(value);
}
