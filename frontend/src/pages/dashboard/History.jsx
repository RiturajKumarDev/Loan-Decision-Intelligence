import React, { useState, useEffect } from 'react';
import { Search, Filter, CheckCircle, XCircle, ChevronLeft, ChevronRight, Download, RefreshCw } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { apiCall } from '../../api';
import { useAuth } from '../../context/AuthContext';

const History = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { user } = useAuth();

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const data = await apiCall('/history/histories');
        const formattedData = data.map((item, index) => {
          const confidence = item.probability ? Math.max(...item.probability) : 0;
          return {
            id: `LD-${1000 + index}`,
            age: item.age,
            income: `$${item.annual_income?.toLocaleString()}`,
            amount: `$${item.loan_amount?.toLocaleString()}`,
            score: item.credit_score,
            employment: item.employment_years,
            education: item.education_level,
            housing: item.housing_status,
            status: item.prediction === 1 ? 'Approved' : 'Declined',
            confidence: `${(confidence * 100).toFixed(1)}%`
          };
        });
        setHistoryData(formattedData);
      } catch (err) {
        if (err.message === "No histories found!") {
          setHistoryData([]);
        } else {
          setError(err.message || 'Failed to fetch prediction history');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [user]);

  const filteredData = historyData.filter(item =>
    item.education.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.housing.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.status.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleExportPDF = () => {
    const doc = new jsPDF('landscape');
    
    // Add title
    doc.setFontSize(18);
    doc.text("Loan Prediction History", 14, 22);
    
    // Add subtitle / date
    doc.setFontSize(11);
    doc.setTextColor(100);
    doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 14, 30);
    
    // Define table columns
    const tableColumn = ["Age", "Income", "Loan Amt", "Credit", "Emp Yrs", "Education", "Housing", "Confidence", "Status"];
    
    // Map data to table rows
    const tableRows = [];
    filteredData.forEach(item => {
      const rowData = [
        item.age,
        item.income,
        item.amount,
        item.score,
        item.employment,
        item.education,
        item.housing,
        item.confidence,
        item.status
      ];
      tableRows.push(rowData);
    });
    
    // Generate the table
    autoTable(doc, {
      head: [tableColumn],
      body: tableRows,
      startY: 40,
      styles: { fontSize: 10 },
      headStyles: { fillColor: [99, 102, 241] }
    });
    
    // Save the PDF
    doc.save(`loan-history-${new Date().toISOString().split('T')[0]}.pdf`);
  };

  return (
    <div className="history-page">
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div className="table-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h3 style={{ fontSize: '20px', marginBottom: '4px' }}>Prediction History</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>View and export past loan assessment results.</p>
          </div>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', flex: '1 1 auto', justifyContent: 'flex-end' }}>
            <div className="input-with-icon" style={{ flex: '1 1 200px', maxWidth: '400px' }}>
              <Search className="icon" size={16} />
              <input
                type="text"
                className="form-control"
                placeholder="Search by Education, Housing..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{ padding: '10px 16px 10px 40px', fontSize: '14px', width: '100%' }}
              />
            </div>

            <div style={{ display: 'flex', gap: '12px', flexWrap: 'nowrap' }}>
              <button className="btn btn-secondary" style={{ padding: '10px 16px', whiteSpace: 'nowrap' }}>
                <Filter size={16} /> Filter
              </button>
              <button className="btn btn-primary" onClick={handleExportPDF} style={{ padding: '10px 16px', whiteSpace: 'nowrap' }}>
                <Download size={16} /> Export PDF
              </button>
            </div>
          </div>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Age</th>
                <th>Income</th>
                <th>Loan Amt</th>
                <th>Credit Score</th>
                <th>Employment (Yrs)</th>
                <th>Education</th>
                <th>Housing</th>
                <th>Confidence</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="9" style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)' }}>
                    <RefreshCw className="animate-spin" size={24} style={{ animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
                    Loading history...
                  </td>
                </tr>
              ) : error ? (
                <tr>
                  <td colSpan="9" style={{ textAlign: 'center', padding: '40px 0', color: '#ef4444' }}>
                    {error}
                  </td>
                </tr>
              ) : filteredData.length === 0 ? (
                <tr>
                  <td colSpan="9" style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)' }}>
                    No records found matching your search.
                  </td>
                </tr>
              ) : (
                filteredData.map(item => (
                  <tr key={item.id}>
                    <td>{item.age}</td>
                    <td>{item.income}</td>
                    <td>{item.amount}</td>
                    <td>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        backgroundColor: item.score >= 700 ? 'rgba(16, 185, 129, 0.1)' : item.score >= 650 ? 'rgba(245, 158, 11, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        color: item.score >= 700 ? 'var(--success)' : item.score >= 650 ? 'var(--warning)' : 'var(--danger)'
                      }}>
                        {item.score}
                      </span>
                    </td>
                    <td>{item.employment}</td>
                    <td>{item.education}</td>
                    <td>{item.housing}</td>
                    <td>{item.confidence}</td>
                    <td>
                      <span className={`status-badge ${item.status.toLowerCase()}`}>
                        {item.status === 'Approved' ? <CheckCircle size={12} /> : <XCircle size={12} />}
                        {item.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="pagination" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '24px', paddingTop: '20px', borderTop: '1px solid var(--glass-border)', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
            Showing 1 to {filteredData.length} of {historyData.length} entries
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button className="btn btn-secondary" style={{ padding: '6px 12px' }} disabled>
              <ChevronLeft size={16} /> Prev
            </button>
            <button className="btn btn-primary" style={{ padding: '6px 12px' }}>
              1
            </button>
            <button className="btn btn-secondary" style={{ padding: '6px 12px' }} disabled>
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default History;
