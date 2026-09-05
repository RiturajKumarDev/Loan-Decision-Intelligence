import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { Brain } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  const { isAuthenticated, logout } = useAuth();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isActive = (path) => location.pathname === path ? 'active' : '';

  return (
    <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
      <div className="container nav-container">
        <Link to="/" className="nav-brand">
          <div className="nav-brand-icon">
            <Brain size={24} color="white" />
          </div>
          LoanIntel
        </Link>
        
        <div className="nav-links">
          <Link to="/" className={`nav-link ${isActive('/')}`}>Home</Link>
          <Link to="/about" className={`nav-link ${isActive('/about')}`}>About</Link>
        </div>
        
        <div className="nav-actions">
          {isAuthenticated ? (
            <Link to="/dashboard" className="btn btn-primary" style={{ padding: '8px 20px', fontSize: '14px' }}>
              Dashboard
            </Link>
          ) : (
            <Link to="/login" className="btn btn-outline" style={{ padding: '8px 20px', fontSize: '14px' }}>
              Log In
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
};

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-col">
            <Link to="/" className="nav-brand" style={{ marginBottom: '20px', display: 'inline-flex' }}>
              <div className="nav-brand-icon" style={{ width: '32px', height: '32px' }}>
                <Brain size={18} color="white" />
              </div>
              LoanIntel
            </Link>
            <p>Empowering financial institutions with real-time, ML-driven loan decision intelligence.</p>
          </div>
          <div className="footer-col">
            <h4>Platform</h4>
            <div className="footer-links">
              <Link to="/dashboard/predict">Risk Assessment</Link>
              <Link to="#">Analytics</Link>
              <Link to="#">API Documentation</Link>
            </div>
          </div>
          <div className="footer-col">
            <h4>Company</h4>
            <div className="footer-links">
              <Link to="/about">About Us</Link>
              <Link to="#">Careers</Link>
              <Link to="#">Contact</Link>
            </div>
          </div>
          <div className="footer-col">
            <h4>Legal</h4>
            <div className="footer-links">
              <Link to="#">Privacy Policy</Link>
              <Link to="#">Terms of Service</Link>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} Loan Decision Intelligence. All rights reserved.</p>
          <p>Designed for Rituraj Kumar</p>
        </div>
      </div>
    </footer>
  );
};

const PublicLayout = () => {
  return (
    <div className="app-container">
      <Navbar />
      <main className="main-content">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
};

export default PublicLayout;
