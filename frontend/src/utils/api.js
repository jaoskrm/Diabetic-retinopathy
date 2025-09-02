import axios from 'axios';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout for image uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth headers (if needed)
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling common errors
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.data);

      // Handle specific error codes
      switch (error.response.status) {
        case 401:
          // Unauthorized - clear token and redirect to login
          localStorage.removeItem('authToken');
          break;
        case 429:
          // Too Many Requests
          throw new Error('Too many requests. Please wait and try again.');
        case 500:
          throw new Error('Server error. Please try again later.');
        default:
          throw error;
      }
    } else if (error.request) {
      // Network error
      throw new Error('Network error. Please check your connection.');
    } else {
      // Something else happened
      throw error;
    }
  }
);

// API Functions
export const api = {
  // Health check
  healthCheck: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Get available models
  getModels: async () => {
    const response = await apiClient.get('/models');
    return response.data;
  },

  // Single model prediction
  predictSingleModel: async (file, modelName = 'efficientnet', useAdvancedOptions = {}) => {
    const formData = new FormData();
    formData.append('file', file);

    const params = {
      model_name: modelName,
      ...useAdvancedOptions
    };

    const response = await apiClient.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params,
      onUploadProgress: (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        console.log(`Upload progress: ${progress}%`);
      },
    });

    return response.data;
  },

  // Ensemble model prediction
  predictEnsemble: async (file, modelsToUse = null) => {
    const formData = new FormData();
    formData.append('file', file);

    const params = {};
    if (modelsToUse && modelsToUse.length > 0) {
      params.models_to_use = modelsToUse;
    }

    const response = await apiClient.post('/predict/ensemble', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params,
      onUploadProgress: (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        console.log(`Upload progress: ${progress}%`);
      },
    });

    return response.data;
  },

  // Batch prediction
  predictBatch: async (files, modelName = 'efficientnet') => {
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append('files', file);
    });

    const params = { model_name: modelName };

    const response = await apiClient.post('/predict/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params,
      onUploadProgress: (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        console.log(`Batch upload progress: ${progress}%`);
      },
    });

    return response.data;
  }
};

// Utility functions for file handling
export const fileUtils = {
  // Validate file type
  validateImageFile: (file) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!allowedTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload a JPEG, PNG, BMP, or TIFF image.');
    }

    if (file.size > maxSize) {
      throw new Error('File too large. Maximum size is 10MB.');
    }

    return true;
  },

  // Format file size
  formatFileSize: (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
};

export default api;