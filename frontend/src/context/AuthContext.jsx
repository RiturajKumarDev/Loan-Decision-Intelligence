import React, { createContext, useState, useContext, useEffect } from 'react';
import { apiCall } from '../api';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check for saved session on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('loanIntel_token');
      if (token) {
        try {
          const profile = await apiCall('/auth/profile');
          setUser(profile);
          setIsAuthenticated(true);
        } catch (error) {
          console.error("Failed to fetch profile", error);
          localStorage.removeItem('loanIntel_token');
        }
      }
      setLoading(false);
    };
    checkAuth();
  }, []);

  const login = async (userData) => {
    try {
      const response = await apiCall('/auth/login', {
        method: 'POST',
        body: JSON.stringify(userData)
      });
      
      if (response.access_token) {
        localStorage.setItem('loanIntel_token', response.access_token);
        const profile = await apiCall('/auth/profile');
        setUser(profile);
        setIsAuthenticated(true);
        return { success: true };
      }
    } catch (error) {
      throw error;
    }
  };

  const register = async (userData) => {
    try {
      const response = await apiCall('/auth/register', {
        method: 'POST',
        body: JSON.stringify({
          name: userData.fullName,
          company_name: userData.company,
          email: userData.email,
          password: userData.password
        })
      });
      return response;
    } catch (error) {
      throw error;
    }
  };

  const logout = async () => {
    try {
      await apiCall('/auth/logout');
    } catch (error) {
      console.error("Logout error", error);
    } finally {
      setUser(null);
      setIsAuthenticated(false);
      localStorage.removeItem('loanIntel_token');
    }
  };

  const changePassword = async (passwordData) => {
    try {
      return await apiCall('/auth/change_password', {
        method: 'PUT',
        body: JSON.stringify({
          old_password: passwordData.oldPassword,
          new_password: passwordData.newPassword
        })
      });
    } catch (error) {
      throw error;
    }
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, login, register, logout, changePassword, loading }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};
