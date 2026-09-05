import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, User, ArrowRight, Brain, Briefcase, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Register = () => {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    company: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords don't match!");
      return;
    }
    try {
      await register(formData);
      navigate('/login');
    } catch (err) {
      setError(err.message || "Failed to register account");
    }
  };

  return (
    <div className="login-page animate-fade-in" style={{ paddingTop: '40px' }}>
      <div className="container">
        <div className="form-container" style={{ maxWidth: '540px' }}>
          <div className="glass-panel" style={{ padding: '40px' }}>
            <div className="auth-header">
              <div className="nav-brand-icon" style={{ margin: '0 auto 20px' }}>
                <Brain size={24} color="white" />
              </div>
              <h1>Create an Account</h1>
              <p>Join LoanIntel and start making data-driven decisions</p>
            </div>
            
            <form onSubmit={handleSubmit}>
              {error && <div className="alert-danger" style={{ color: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', padding: '12px', borderRadius: '8px', marginBottom: '16px', fontSize: '14px' }}>{error}</div>}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div className="form-group">
                  <label className="form-label">Full Name</label>
                  <div className="input-with-icon">
                    <User className="icon" size={18} />
                    <input 
                      type="text" 
                      name="fullName"
                      className="form-control" 
                      placeholder="John Doe" 
                      value={formData.fullName}
                      onChange={handleChange}
                      required
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Company Name</label>
                  <div className="input-with-icon">
                    <Briefcase className="icon" size={18} />
                    <input 
                      type="text" 
                      name="company"
                      className="form-control" 
                      placeholder="Acme Financial" 
                      value={formData.company}
                      onChange={handleChange}
                      required
                    />
                  </div>
                </div>
              </div>
              
              <div className="form-group">
                <label className="form-label">Work Email Address</label>
                <div className="input-with-icon">
                  <Mail className="icon" size={18} />
                  <input 
                    type="email" 
                    name="email"
                    className="form-control" 
                    placeholder="john@company.com" 
                    value={formData.email}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
              
              <div className="form-group">
                <label className="form-label">Password</label>
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
                    minLength={8}
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

              <div className="form-group">
                <label className="form-label">Confirm Password</label>
                <div className="input-with-icon">
                  <Lock className="icon" size={18} />
                  <input 
                    type={showConfirmPassword ? "text" : "password"} 
                    name="confirmPassword"
                    className="form-control" 
                    placeholder="••••••••" 
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    required
                    style={{ paddingRight: '44px' }}
                  />
                  <button 
                    type="button" 
                    className="password-toggle" 
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    tabIndex="-1"
                  >
                    {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>
              
              <div style={{ marginTop: '20px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                By creating an account, you agree to our <Link to="#" style={{ color: 'var(--accent-primary)' }}>Terms of Service</Link> and <Link to="#" style={{ color: 'var(--accent-primary)' }}>Privacy Policy</Link>.
              </div>

              <div className="form-actions" style={{ marginTop: '24px' }}>
                <button type="submit" className="btn btn-primary btn-full">
                  Create Account <ArrowRight size={16} />
                </button>
              </div>
            </form>
            
            <div className="form-footer">
              Already have an account? <Link to="/login">Sign In</Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Register;
