const API_URL = 'http://127.0.0.1:5000/api';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    fetchModelInfo();
    loadPerformanceData(); // NEW: Load performance data

    // Add event listener to form
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handlePrediction);
    }
});

// NEW: Function to load performance data
async function loadPerformanceData() {
    try {
        const response = await fetch(`${API_URL}/performance`);
        const result = await response.json();

        if (result.success && result.data) {
            const metrics = result.data;

            // Allow for safe element updating
            const updateElement = (id, value) => {
                const el = document.getElementById(id);
                if (el) el.textContent = value;
            };

            updateElement('metric-r2', metrics.test_r2);
            updateElement('metric-mae', metrics.test_mae);
            updateElement('metric-rmse', metrics.test_rmse);
            updateElement('metric-model', metrics.best_model);

            // Show container
            const container = document.getElementById('performance-container');
            if (container) {
                container.classList.remove('hidden');
            }
        }
    } catch (error) {
        console.error('Error loading performance data:', error);
    }
}

async function fetchModelInfo() {
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
