import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, ArrowRight, Brain, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      await login(formData);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || "Failed to login");
    }
  };

  return (
    <div className="login-page animate-fade-in">
      <div className="container">
        <div className="form-container">
          <div className="glass-panel" style={{ padding: '40px' }}>
            <div className="auth-header">
              <div className="nav-brand-icon" style={{ margin: '0 auto 20px' }}>
                <Brain size={24} color="white" />
              </div>
              <h1>Welcome Back</h1>
              <p>Sign in to access your dashboard</p>
            </div>
            
            <form onSubmit={handleSubmit}>
              {error && <div className="alert-danger" style={{ color: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', padding: '12px', borderRadius: '8px', marginBottom: '16px', fontSize: '14px' }}>{error}</div>}
              <div className="form-group">
                <label className="form-label">Email Address</label>
                <div className="input-with-icon">
                  <Mail className="icon" size={18} />
                  <input 
                    type="email" 
                    name="email"
                    className="form-control" 
                    placeholder="name@company.com" 
                    value={formData.email}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
              
              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <label className="form-label" style={{ marginBottom: 0 }}>Password</label>
                  <Link to="#" style={{ fontSize: '12px', color: 'var(--accent-primary)' }}>Forgot Password?</Link>
                </div>
                <div className="input-with-icon">
                  <Lock className="icon" size={18} />
                  <input 
                    type={showPassword ? "text" : "password"} 
                    name="password"
                    className="form-control" 
                    placeholder="••••••••" 
                    value={formData.password}
                    onChange={handleChange}
                    required
                    style={{ paddingRight: '44px' }}
                  />
                  <button 
                    type="button" 
                    className="password-toggle" 
                    onClick={() => setShowPassword(!showPassword)}
                    tabIndex="-1"
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>
              
              <div className="form-actions" style={{ marginTop: '24px' }}>
                <button type="submit" className="btn btn-primary btn-full">
                  Sign In <ArrowRight size={16} />
                </button>
              </div>
            </form>
            
            <div className="form-footer">
              Don't have an account? <Link to="/register">Create Account</Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
