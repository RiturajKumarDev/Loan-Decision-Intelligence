import React from 'react';
import { Shield, Brain, Users, Award } from 'lucide-react';

const About = () => {
  return (
    <div className="about-page animate-fade-in">
      <div className="container">
        <div className="page-header">
          <h1 className="hero-title" style={{ fontSize: '42px' }}>About LoanIntel</h1>
          <p className="section-desc">
            Transforming financial decision-making through advanced artificial intelligence and predictive analytics.
          </p>
        </div>
        
        <div className="about-content">
          <div className="about-text">
            <h2>Our Mission</h2>
            <p>
              Founded with the vision to democratize access to sophisticated financial technology, 
              Loan Decision Intelligence (LoanIntel) provides institutional-grade risk assessment tools 
              to lenders of all sizes.
            </p>
            <p>
              We believe that the future of lending is data-driven, equitable, and instantaneous. 
              By removing manual bottlenecks and human bias from the initial screening process, 
              we help financial institutions approve more qualified borrowers faster, while 
              significantly reducing default rates.
            </p>
            
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">98.5%</div>
                <div className="stat-label">Model Accuracy</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">1.2s</div>
                <div className="stat-label">Avg. Decision Time</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">$2.4B</div>
                <div className="stat-label">Loans Processed</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">-42%</div>
                <div className="stat-label">Default Rate</div>
              </div>
            </div>
          </div>
          
          <div className="about-visual">
            <div className="glass-panel" style={{ padding: '40px', position: 'relative' }}>
              <div style={{ position: 'absolute', top: '-20px', left: '-20px', width: '100px', height: '100px', background: 'var(--accent-primary)', filter: 'blur(50px)', opacity: '0.5', zIndex: -1 }}></div>
              <div style={{ position: 'absolute', bottom: '-20px', right: '-20px', width: '100px', height: '100px', background: 'var(--accent-tertiary)', filter: 'blur(50px)', opacity: '0.5', zIndex: -1 }}></div>
              
              <h3 style={{ marginBottom: '24px', fontSize: '24px' }}>Core Technology</h3>
              
              <div style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(99, 102, 241, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-primary)', flexShrink: 0 }}>
                  <Brain size={20} />
                </div>
                <div>
                  <h4 style={{ marginBottom: '4px', fontSize: '16px' }}>Ensemble Learning</h4>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Our proprietary ensemble models combine XGBoost, Random Forests, and deep neural networks for unparalleled predictive power.</p>
                </div>
              </div>
              
              <div style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(139, 92, 246, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-secondary)', flexShrink: 0 }}>
                  <Shield size={20} />
                </div>
                <div>
                  <h4 style={{ marginBottom: '4px', fontSize: '16px' }}>Bias Mitigation</h4>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Advanced algorithmic fairness constraints ensure our models evaluate risk without demographic prejudice.</p>
                </div>
              </div>
              
              <div style={{ display: 'flex', gap: '16px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--success)', flexShrink: 0 }}>
                  <Award size={20} />
                </div>
                <div>
                  <h4 style={{ marginBottom: '4px', fontSize: '16px' }}>Regulatory Compliance</h4>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Fully compliant with FCRA, ECOA, and built-in explainability (XAI) for adverse action reporting.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div style={{ textAlign: 'center', margin: '60px 0 100px' }}>
          <h2 style={{ marginBottom: '16px' }}>Developed By</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '18px', maxWidth: '600px', margin: '0 auto 24px' }}>
            RITURAJ KUMAR
          </p>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '12px 24px', background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', borderRadius: '100px' }}>
            <Users size={18} color="var(--accent-primary)" />
            <span>Dedicated to advancing financial inclusion through AI</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
