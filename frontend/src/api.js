export const getBaseUrl = () => {
  return import.meta.env.VITE_BASE_URL || 'http://localhost:8000/api';
};

export const apiCall = async (endpoint, options = {}) => {
  const baseUrl = getBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const token = localStorage.getItem('loanIntel_token');
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(url, config);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || data.message || 'Something went wrong');
    }
    
    return data;
  } catch (error) {
    throw error;
  }
};
