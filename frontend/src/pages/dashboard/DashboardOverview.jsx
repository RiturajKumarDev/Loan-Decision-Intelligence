import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, CheckCircle, XCircle, Clock, Zap, Activity, RefreshCw } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { apiCall } from '../../api';

const DashboardOverview = () => {
  const { user } = useAuth();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        setLoading(true);
        const data = await apiCall('/history/dashboard');
        setDashboardData(data);
      } catch (err) {
        setError(err.message || 'Failed to fetch dashboard data');
      } finally {
        setLoading(false);
      }
    };
    fetchDashboard();
  }, []);

  const approvalRate = dashboardData?.total > 0 
    ? Math.round((dashboardData.approved / dashboardData.total) * 100) 
    : 0;

  const stats = [
    { label: 'Total Predictions', value: dashboardData?.total || 0, icon: <Activity size={20} />, color: 'var(--accent-primary)' },
    { label: 'Approval Rate', value: `${approvalRate}%`, icon: <CheckCircle size={20} />, color: 'var(--success)' },
    { label: 'Pending Reviews', value: '0', icon: <Clock size={20} />, color: 'var(--warning)' },
  ];

  const recentActivity = dashboardData?.histories?.map(item => ({
    id: item.id,
    name: user?.name || 'Applicant',
    date: 'Recent',
    amount: `$${item.loan_amount?.toLocaleString()}`,
    status: item.prediction === 1 ? 'Approved' : 'Declined'
  })) || [];

  return (
    <div className="dashboard-overview">
      <div className="welcome-banner glass-panel">
        <div className="banner-content">
          <h2>Welcome back, {user?.name || 'User'}!</h2>
          <p>Here's what's happening with your loan decision models today.</p>
          <Link to="/dashboard/predict" className="btn btn-primary" style={{ marginTop: '16px' }}>
            <Zap size={16} /> Run New Prediction
          </Link>
        </div>
        <div className="banner-visual">
          <BrainIcon />
        </div>
      </div>

      <div className="stats-grid dashboard-stats">
        {stats.map((stat, i) => (
          <div key={i} className="stat-card glass-panel" style={{ textAlign: 'left', display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div className="stat-icon" style={{ backgroundColor: `${stat.color}15`, color: stat.color, padding: '16px', borderRadius: '12px' }}>
              {stat.icon}
            </div>
            <div>
              <div className="stat-value" style={{ fontSize: '28px', margin: 0, color: 'var(--text-primary)' }}>
                {loading ? <RefreshCw className="animate-spin" size={20} /> : stat.value}
              </div>
              <div className="stat-label">{stat.label}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="dashboard-grid">
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ fontSize: '18px' }}>Recent Predictions</h3>
            <Link to="/dashboard/history" style={{ fontSize: '14px', color: 'var(--accent-primary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              View All <ArrowRight size={14} />
            </Link>
          </div>
          
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Applicant</th>
                  <th>Amount</th>
                  <th>Date</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan="4" style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)' }}>
                      <RefreshCw className="animate-spin" size={24} style={{ animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
                      Loading recent activity...
                    </td>
                  </tr>
                ) : error ? (
                  <tr>
                    <td colSpan="4" style={{ textAlign: 'center', padding: '40px 0', color: '#ef4444' }}>
                      {error}
                    </td>
                  </tr>
                ) : recentActivity.length === 0 ? (
                  <tr>
                    <td colSpan="4" style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)' }}>
                      No recent predictions found.
                    </td>
                  </tr>
                ) : (
                  recentActivity.map(activity => (
                    <tr key={activity.id}>
                      <td style={{ fontWeight: 500 }}>{activity.name}</td>
                      <td>{activity.amount}</td>
                      <td style={{ color: 'var(--text-secondary)' }}>{activity.date}</td>
                      <td>
                        <span className={`status-badge ${activity.status.toLowerCase()}`}>
                          {activity.status === 'Approved' ? <CheckCircle size={12} /> : <XCircle size={12} />}
                          {activity.status}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

// Simple SVG component for banner
const BrainIcon = () => (
  <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.5 }}>
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
  </svg>
);

export default DashboardOverview;
