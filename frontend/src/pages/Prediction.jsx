import React, { useState } from 'react';
import { RefreshCw, Zap, CheckCircle, XCircle, User, DollarSign, Activity, Briefcase, GraduationCap, Home, Calendar, CreditCard, ShieldCheck } from 'lucide-react';
import { apiCall } from '../api';

const Prediction = () => {
  const initialFormState = {
    age: '30',
    annualIncome: '60000',
    loanAmount: '20000',
    creditScore: '680',
    employmentYears: '2',
    educationLevel: "Bachelor's Degree",
    housingStatus: 'Own'
  };

  const [formData, setFormData] = useState(initialFormState);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const resetFields = () => {
    setFormData(initialFormState);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const requestPayload = {
        age: parseInt(formData.age, 10),
        annual_income: parseFloat(formData.annualIncome),
        loan_amount: parseFloat(formData.loanAmount),
        credit_score: parseInt(formData.creditScore, 10),
        employment_years: parseFloat(formData.employmentYears),
        education_level: formData.educationLevel,
        housing_status: formData.housingStatus
      };

      const data = await apiCall('/ml/predict', {
        method: 'POST',
        body: JSON.stringify(requestPayload)
      });
      console.log(data);

      const isApproved = data.prediction === 1;
      const confidence = data.probability ? Math.max(...data.probability) : 0.99;

      setResult({
        approved: isApproved,
        probability: confidence,
        message: isApproved ? 'High likelihood of approval. The applicant meets the optimal risk criteria.' : 'High risk detected. Applicant falls below acceptable thresholds.',
        simulated: false
      });
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || "Failed to connect to the prediction service.");
      setResult(null);
    }

    setLoading(false);
  };

  return (
    <div className="prediction-page animate-fade-in">
      <div className="container" style={{ maxWidth: '900px' }}>
        <div className="page-header" style={{ padding: '20px 0 40px' }}>
          <h1 className="hero-title" style={{ fontSize: '42px', marginBottom: '16px' }}>
            AI Loan Assessment <ShieldCheck style={{ color: 'var(--accent-primary)', display: 'inline', verticalAlign: 'middle', marginLeft: '12px' }} size={40} />
          </h1>
          <p className="section-desc" style={{ fontSize: '18px' }}>
            Enter the applicant details below to instantly evaluate risk and generate an ML-powered loan decision.
          </p>
        </div>

        <div className="prediction-container glass-panel" style={{ padding: '0', overflow: 'hidden', margin: '0 auto 80px' }}>
          {/* Header Banner */}
          <div style={{ background: 'linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)', padding: '24px 40px', borderBottom: '1px solid var(--glass-border)' }}>
            <h3 style={{ margin: 0, fontSize: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <User size={20} color="var(--accent-primary)" /> Applicant Information
            </h3>
          </div>

          <div style={{ padding: '40px' }}>
            {error && (
              <div className="alert-danger" style={{ color: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', padding: '16px', borderRadius: '8px', marginBottom: '32px', fontSize: '15px', border: '1px solid rgba(239, 68, 68, 0.2)', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <XCircle size={20} /> {error}
              </div>
            )}
            
            <form onSubmit={handleSubmit}>
              
              {/* Personal Details Group */}
              <div style={{ marginBottom: '32px' }}>
                <h4 style={{ fontSize: '16px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <User size={16} /> Personal Details
                </h4>
                <div className="prediction-grid">
                  <div className="form-group">
                    <label className="form-label">Age</label>
                    <div className="input-with-icon">
                      <Calendar className="icon" size={18} />
                      <input type="number" name="age" min="18" max="100" className="form-control" placeholder="e.g. 30" value={formData.age} onChange={handleChange} required />
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Education Level</label>
                    <div className="input-with-icon">
                      <GraduationCap className="icon" size={18} />
                      <select name="educationLevel" className="form-control" value={formData.educationLevel} onChange={handleChange} required>
                        <option value="High School">High School</option>
                        <option value="Associate's Degree">Associate's Degree</option>
                        <option value="Bachelor's Degree">Bachelor's Degree</option>
                        <option value="Master's Degree">Master's Degree</option>
                        <option value="PhD">PhD</option>
                      </select>
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Housing Status</label>
                    <div className="input-with-icon">
                      <Home className="icon" size={18} />
                      <select name="housingStatus" className="form-control" value={formData.housingStatus} onChange={handleChange} required>
                        <option value="Own">Own</option>
                        <option value="Rent">Rent</option>
                        <option value="Mortgage">Mortgage</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ height: '1px', background: 'var(--glass-border)', margin: '40px 0' }}></div>

              {/* Financial Details Group */}
              <div style={{ marginBottom: '40px' }}>
                <h4 style={{ fontSize: '16px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <DollarSign size={16} /> Financial Profile
                </h4>
                <div className="prediction-grid">
                  <div className="form-group">
                    <label className="form-label">Annual Income (Gross)</label>
                    <div className="input-with-icon">
                      <DollarSign className="icon" size={18} />
                      <input type="number" name="annualIncome" min="0" className="form-control" placeholder="e.g. 60000" value={formData.annualIncome} onChange={handleChange} required />
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Requested Loan Amount</label>
                    <div className="input-with-icon">
                      <Briefcase className="icon" size={18} />
                      <input type="number" name="loanAmount" min="100" className="form-control" placeholder="e.g. 20000" value={formData.loanAmount} onChange={handleChange} required />
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Credit Score</label>
                    <div className="input-with-icon">
                      <Activity className="icon" size={18} />
                      <input type="number" name="creditScore" min="300" max="850" className="form-control" placeholder="300 - 850" value={formData.creditScore} onChange={handleChange} required />
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Employment (Years)</label>
                    <div className="input-with-icon">
                      <Briefcase className="icon" size={18} />
                      <input type="number" name="employmentYears" min="0" step="0.5" className="form-control" placeholder="e.g. 2" value={formData.employmentYears} onChange={handleChange} required />
                    </div>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="form-actions" style={{ marginTop: '20px', display: 'flex', gap: '16px', justifyContent: 'flex-end', borderTop: '1px solid var(--glass-border)', paddingTop: '32px' }}>
                <button type="button" className="btn btn-secondary btn-large" onClick={resetFields} style={{ padding: '14px 24px' }}>
                  Clear Form
                </button>
                <button type="submit" className="btn btn-primary btn-large" disabled={loading} style={{ padding: '14px 32px', minWidth: '220px' }}>
                  {loading ? (
                    <>Processing <RefreshCw size={18} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} /></>
                  ) : (
                    <>Run Analysis <Zap size={18} /></>
                  )}
                </button>
              </div>
            </form>

            {/* Enhanced Result Card */}
            {result && (
              <div className={`result-card show ${result.approved ? 'approved' : 'rejected'}`} style={{ marginTop: '40px', padding: '0', overflow: 'hidden', border: `1px solid ${result.approved ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`, borderRadius: '12px' }}>
                <div style={{ background: result.approved ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', padding: '30px', textAlign: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <div className="result-icon" style={{ background: 'transparent', margin: '0 auto 10px' }}>
                    {result.approved ? <CheckCircle size={56} /> : <XCircle size={56} />}
                  </div>
                  <h3 className="result-title" style={{ fontSize: '32px', marginBottom: '8px' }}>
                    {result.approved ? 'Application Approved' : 'Application Declined'}
                  </h3>
                  <p className="result-desc" style={{ fontSize: '16px', margin: '0 auto', maxWidth: '500px' }}>
                    {result.message}
                  </p>
                </div>

                <div style={{ padding: '30px 40px', background: 'rgba(0,0,0,0.2)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '15px', color: 'var(--text-secondary)', fontWeight: '500' }}>Model Confidence Score</span>
                    <span style={{ fontSize: '24px', fontWeight: 'bold', fontFamily: 'Outfit', color: result.approved ? 'var(--success)' : 'var(--danger)' }}>
                      {(result.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  {/* Progress Bar */}
                  <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ 
                      width: `${result.probability * 100}%`, 
                      height: '100%', 
                      background: result.approved ? 'var(--success)' : 'var(--danger)', 
                      transition: 'width 1.5s cubic-bezier(0.1, 0.8, 0.3, 1)' 
                    }}></div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>
                    <span>0% (Low Confidence)</span>
                    <span>100% (High Confidence)</span>
                  </div>
                </div>
              </div>
            )}

          </div>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}} />
    </div>
  );
};

export default Prediction;
