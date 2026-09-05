import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, ShieldCheck, Zap, BarChart3, Clock } from 'lucide-react';

const Home = () => {
  return (
    <div className="home-page animate-fade-in">
      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <div className="hero-content">
            <div className="hero-text">
              <div className="hero-badge">
                <Zap size={16} /> Powered by Advanced ML
              </div>
              <h1 className="hero-title">
                Smarter <span className="text-gradient">Loan Decisions</span> in Seconds
              </h1>
              <p className="hero-desc">
                Leverage real-time risk assessment and predictive analytics to make confident, data-driven lending decisions with unprecedented accuracy.
              </p>
              <div className="hero-actions">
                <Link to="/predict" className="btn btn-primary btn-large">
                  Try Prediction Model <ArrowRight size={18} />
                </Link>
                <Link to="/about" className="btn btn-secondary btn-large">
                  Learn More
                </Link>
              </div>
            </div>

            <div className="hero-image-wrapper">
              <div className="hero-visual">
                <div className="hero-visual-inner">
                  <img src="img/Loan Decision Intelligence.png" alt="app logo" />
                </div>
              </div>

              {/* Floating elements for dynamic feel */}
              <div className="float-card top-right">
                <div className="float-icon">
                  <ShieldCheck size={20} />
                </div>
                <div className="float-text">
                  <h5>Risk Score</h5>
                  <p>98.5% Accuracy</p>
                </div>
              </div>

              <div className="float-card bottom-left">
                <div className="float-icon">
                  <Clock size={20} />
                </div>
                <div className="float-text">
                  <h5>Processing Time</h5>
                  <p>&lt; 1.2 Seconds</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Why Choose LoanIntel?</h2>
            <p className="section-desc">
              Our intelligent decision engine combines traditional financial metrics with advanced machine learning to provide a comprehensive view of borrower risk.
            </p>
          </div>

          <div className="features-grid">
            <div className="glass-card">
              <div className="feature-icon">
                <Zap size={24} />
              </div>
              <h3 className="feature-title">Real-time Assessment</h3>
              <p className="feature-desc">
                Get instant approval likelihood predictions based on real-time data analysis, eliminating days of manual review processes.
              </p>
            </div>

            <div className="glass-card delay-100">
              <div className="feature-icon">
                <ShieldCheck size={24} />
              </div>
              <h3 className="feature-title">Reduced Risk Exposure</h3>
              <p className="feature-desc">
                Our ML models identify hidden patterns in borrower data to significantly reduce default rates and improve portfolio health.
              </p>
            </div>

            <div className="glass-card delay-200">
              <div className="feature-icon">
                <BarChart3 size={24} />
              </div>
              <h3 className="feature-title">Data-Driven Insights</h3>
              <p className="feature-desc">
                Move beyond simple credit scores. We analyze employment history, income ratios, and multiple data points for holistic profiling.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
