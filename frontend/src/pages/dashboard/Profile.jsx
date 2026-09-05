import React, { useState } from 'react';
import { User, Mail, Briefcase, Lock, Bell, Shield, Save } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

const Profile = () => {
  const { user, changePassword } = useAuth();
  
  const [activeTab, setActiveTab] = useState('general');
  const [isSaving, setIsSaving] = useState(false);
  const [passwordData, setPasswordData] = useState({ oldPassword: '', newPassword: '', confirmPassword: '' });
  const [passwordMessage, setPasswordMessage] = useState({ text: '', type: '' });

  const handleSaveGeneral = (e) => {
    e.preventDefault();
    setIsSaving(true);
    setTimeout(() => setIsSaving(false), 1000);
  };

  const handleSavePassword = async (e) => {
    e.preventDefault();
    setPasswordMessage({ text: '', type: '' });
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setPasswordMessage({ text: "New passwords don't match", type: 'error' });
      return;
    }
    
    setIsSaving(true);
    try {
      await changePassword(passwordData);
      setPasswordMessage({ text: "Password updated successfully", type: 'success' });
      setPasswordData({ oldPassword: '', newPassword: '', confirmPassword: '' });
    } catch (err) {
      setPasswordMessage({ text: err.message || "Failed to update password", type: 'error' });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="profile-page">
      <div className="profile-layout" style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '32px' }}>
        
        {/* Sidebar Nav */}
        <div className="profile-nav glass-panel" style={{ padding: '20px', alignSelf: 'start' }}>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <li>
              <button 
                className={`profile-nav-btn ${activeTab === 'general' ? 'active' : ''}`}
                onClick={() => setActiveTab('general')}
              >
                <User size={18} /> General Settings
              </button>
            </li>
            <li>
              <button 
                className={`profile-nav-btn ${activeTab === 'security' ? 'active' : ''}`}
                onClick={() => setActiveTab('security')}
              >
                <Shield size={18} /> Security & Password
              </button>
            </li>
            <li>
              <button 
                className={`profile-nav-btn ${activeTab === 'notifications' ? 'active' : ''}`}
                onClick={() => setActiveTab('notifications')}
              >
                <Bell size={18} /> Notifications
              </button>
            </li>
          </ul>
        </div>

        {/* Content Area */}
        <div className="profile-content glass-panel" style={{ padding: '32px' }}>
          
          {activeTab === 'general' && (
            <div className="animate-fade-in">
              <h3 style={{ fontSize: '20px', marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid var(--glass-border)' }}>
                General Settings
              </h3>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '24px', marginBottom: '32px' }}>
                <div className="avatar-large" style={{ 
                  width: '80px', height: '80px', borderRadius: '50%', 
                  background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '32px', fontWeight: 'bold'
                }}>
                  {user?.name?.charAt(0).toUpperCase() || 'U'}
                </div>
                <div>
                  <button className="btn btn-secondary" style={{ padding: '8px 16px', fontSize: '13px' }}>
                    Upload New Avatar
                  </button>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginTop: '8px' }}>
                    Recommended size 256x256px
                  </p>
                </div>
              </div>

              <form onSubmit={handleSaveGeneral}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Full Name</label>
                    <div className="input-with-icon">
                      <User className="icon" size={16} />
                      <input type="text" className="form-control" defaultValue={user?.name || 'John Doe'} />
                    </div>
                  </div>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Email Address</label>
                    <div className="input-with-icon">
                      <Mail className="icon" size={16} />
                      <input type="email" className="form-control" defaultValue={user?.email || 'john@company.com'} readOnly style={{ opacity: 0.7 }} />
                    </div>
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Company / Organization</label>
                  <div className="input-with-icon">
                    <Briefcase className="icon" size={16} />
                    <input type="text" className="form-control" defaultValue={user?.company_name || 'Acme Financial'} />
                  </div>
                </div>
                


                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '32px' }}>
                  <button type="submit" className="btn btn-primary" disabled={isSaving}>
                    {isSaving ? 'Saving...' : <><Save size={16} /> Save Changes</>}
                  </button>
                </div>
              </form>
            </div>
          )}

          {activeTab === 'security' && (
            <div className="animate-fade-in">
              <h3 style={{ fontSize: '20px', marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid var(--glass-border)' }}>
                Security & Password
              </h3>
              
              <form onSubmit={handleSavePassword}>
                {passwordMessage.text && (
                  <div className={passwordMessage.type === 'error' ? 'alert-danger' : 'alert-success'} style={{ 
                    color: passwordMessage.type === 'error' ? '#ef4444' : '#10b981', 
                    backgroundColor: passwordMessage.type === 'error' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)', 
                    padding: '12px', borderRadius: '8px', marginBottom: '16px', fontSize: '14px' 
                  }}>
                    {passwordMessage.text}
                  </div>
                )}
                <div className="form-group">
                  <label className="form-label">Current Password</label>
                  <div className="input-with-icon">
                    <Lock className="icon" size={16} />
                    <input type="password" name="oldPassword" value={passwordData.oldPassword} onChange={(e) => setPasswordData({...passwordData, oldPassword: e.target.value})} className="form-control" placeholder="••••••••" required />
                  </div>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">New Password</label>
                    <div className="input-with-icon">
                      <Lock className="icon" size={16} />
                      <input type="password" name="newPassword" value={passwordData.newPassword} onChange={(e) => setPasswordData({...passwordData, newPassword: e.target.value})} className="form-control" placeholder="••••••••" required />
                    </div>
                  </div>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Confirm New Password</label>
                    <div className="input-with-icon">
                      <Lock className="icon" size={16} />
                      <input type="password" name="confirmPassword" value={passwordData.confirmPassword} onChange={(e) => setPasswordData({...passwordData, confirmPassword: e.target.value})} className="form-control" placeholder="••••••••" required />
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '32px' }}>
                  <button type="submit" className="btn btn-primary" disabled={isSaving}>
                    {isSaving ? 'Updating...' : <><Save size={16} /> Update Password</>}
                  </button>
                </div>
              </form>

              <div style={{ marginTop: '40px', padding: '20px', borderRadius: '12px', border: '1px solid rgba(239, 68, 68, 0.3)', backgroundColor: 'rgba(239, 68, 68, 0.05)' }}>
                <h4 style={{ color: 'var(--danger)', marginBottom: '8px' }}>Two-Factor Authentication</h4>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  Add an extra layer of security to your account. We recommend enabling 2FA for all administrative accounts.
                </p>
                <button className="btn btn-outline" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>
                  Enable 2FA
                </button>
              </div>
            </div>
          )}

          {activeTab === 'notifications' && (
            <div className="animate-fade-in">
              <h3 style={{ fontSize: '20px', marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid var(--glass-border)' }}>
                Notification Preferences
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h5 style={{ fontSize: '15px', marginBottom: '4px' }}>Email Notifications</h5>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Receive daily summaries of loan decisions</p>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h5 style={{ fontSize: '15px', marginBottom: '4px' }}>System Alerts</h5>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Get notified about model updates and maintenance</p>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h5 style={{ fontSize: '15px', marginBottom: '4px' }}>High Risk Alerts</h5>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Immediate notification for high-risk applications</p>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Profile;
