import React, { useState, useEffect, useRef, useCallback } from 'react';

// Get API base URL from Vite environment or default to proxy/relative path
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export default function App() {
  // Navigation / Workflow State: 'home' | 'preview' | 'analyzing' | 'results'
  const [stage, setStage] = useState('home');

  // Client ownership scoping for lightweight persistent history
  const [clientId] = useState(() => {
    let id = localStorage.getItem('crackgauge_client_id');
    if (!id || !/^[a-zA-Z0-9_-]{1,64}$/.test(id)) {
      id = 'op_' + Math.random().toString(36).substring(2, 10);
      localStorage.setItem('crackgauge_client_id', id);
    }
    return id;
  });

  // Input & Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Calibration & Configuration Options
  const [showConfig, setShowConfig] = useState(false);
  const [calibrationMode, setCalibrationMode] = useState('none'); // 'none' | 'aruco' | 'scale'
  const [markerSizeMm, setMarkerSizeMm] = useState('50.0');
  const [mmPerPixel, setMmPerPixel] = useState('0.15');
  const [segmentationProfile, setSegmentationProfile] = useState('baseline');

  // Backend Health & Connectivity
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [serverVersion, setServerVersion] = useState(null);

  // Analysis Result State
  const [result, setResult] = useState(null);
  const [activeVisualTab, setActiveVisualTab] = useState('overlay');

  // Viewer Zoom & Pan State
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  // History Drawer State
  const [showHistoryDrawer, setShowHistoryDrawer] = useState(false);
  const [historyItems, setHistoryItems] = useState([]);
  const [historyCount, setHistoryCount] = useState(0);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Reporting State
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [isExportingCsv, setIsExportingCsv] = useState(false);

  // File Input Refs
  const cameraInputRef = useRef(null);
  const fileInputRef = useRef(null);

  // ---------------------------------------------------------------------------
  // Check Backend Health & Load Initial History
  // ---------------------------------------------------------------------------
  const checkHealth = useCallback(() => {
    const url = API_BASE ? `${API_BASE}/api/health` : '/api/health';
    fetch(url)
      .then((res) => {
        if (res.ok) return res.json();
        throw new Error('Offline');
      })
      .then((data) => {
        setBackendStatus('online');
        setServerVersion(data.version || '0.1.0');
      })
      .catch(() => {
        setBackendStatus('offline');
      });
  }, []);

  const fetchHistory = useCallback(() => {
    const url = API_BASE ? `${API_BASE}/api/history` : '/api/history';
    return fetch(url)
      .then((res) => {
        if (res.ok) return res.json();
        throw new Error('Failed to load history');
      })
      .then((data) => {
        setHistoryItems(data.analyses || []);
        setHistoryCount(data.total || 0);
      })
      .catch((err) => {
        console.warn('Failed to fetch history:', err);
      })
      .finally(() => {
        setIsLoadingHistory(false);
      });
  }, []);

  useEffect(() => {
    checkHealth();
    fetchHistory();
  }, [checkHealth, fetchHistory]);

  const handleTabChange = (tabId) => {
    setActiveVisualTab(tabId);
    setZoomLevel(1.0);
    setPanPosition({ x: 0, y: 0 });
  };

  // ---------------------------------------------------------------------------
  // File Selection Handlers
  // ---------------------------------------------------------------------------
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Basic client-side validation
    const validExtensions = /\.(jpe?g|png|bmp|webp)$/i;
    if (!validExtensions.test(file.name) && !file.type.startsWith('image/')) {
      setErrorMsg('Please select a valid image file (JPEG, PNG, WebP, or BMP).');
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setErrorMsg('File exceeds 20 MB size limit.');
      return;
    }

    setErrorMsg(null);
    setSelectedFile(file);

    // Create local object URL for preview
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    setStage('preview');
  };

  const triggerCamera = () => {
    if (cameraInputRef.current) {
      cameraInputRef.current.value = '';
      cameraInputRef.current.click();
    }
  };

  const triggerFilePicker = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  // ---------------------------------------------------------------------------
  // Run Crack Analysis (POST /api/analyze)
  // ---------------------------------------------------------------------------
  const runAnalysis = async () => {
    if (!selectedFile) return;

    setStage('analyzing');
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', selectedFile, selectedFile.name);
    formData.append('segmentation_profile', segmentationProfile);

    // Add calibration parameters if configured
    if (calibrationMode === 'aruco') {
      const parsed = parseFloat(markerSizeMm);
      if (!isNaN(parsed) && parsed > 0) {
        formData.append('marker_size_mm', parsed.toString());
      }
    } else if (calibrationMode === 'scale') {
      const parsed = parseFloat(mmPerPixel);
      if (!isNaN(parsed) && parsed > 0) {
        formData.append('mm_per_pixel', parsed.toString());
      }
    }

    try {
      const targetUrl = API_BASE ? `${API_BASE}/api/analyze` : '/api/analyze';
      const response = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'X-Client-Id': clientId,
        },
        body: formData,
      });

      if (!response.ok) {
        let errDetail = 'Failed to analyze crack image.';
        try {
          const errJson = await response.json();
          if (errJson.detail) {
            errDetail = typeof errJson.detail === 'string'
              ? errJson.detail
              : errJson.detail.message || JSON.stringify(errJson.detail);
          }
        } catch {
          errDetail = `Server returned HTTP ${response.status} ${response.statusText}`;
        }
        throw new Error(errDetail);
      }

      const data = await response.json();
      setResult(data);
      setStage('results');
      setActiveVisualTab('overlay');
      setZoomLevel(1.0);
      setPanPosition({ x: 0, y: 0 });

      // Refresh history list so the new scan immediately reflects
      fetchHistory();
    } catch (err) {
      console.error('Analysis failed:', err);
      let userFriendlyMessage = err.message || 'Network connection failed.';
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        userFriendlyMessage = `Cannot connect to API server at ${API_BASE || 'localhost:8000'}. Make sure the backend server is running.`;
      }
      setErrorMsg(userFriendlyMessage);
      setStage('preview');
    }
  };

  // ---------------------------------------------------------------------------
  // Load Historical Scan
  // ---------------------------------------------------------------------------
  const loadHistoricalAnalysis = async (analysisId) => {
    try {
      const url = API_BASE ? `${API_BASE}/api/history/${analysisId}` : `/api/history/${analysisId}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to load analysis record (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
      setStage('results');
      setShowHistoryDrawer(false);
      setActiveVisualTab('overlay');
      setZoomLevel(1.0);
      setPanPosition({ x: 0, y: 0 });
    } catch (err) {
      setErrorMsg(`Could not load past scan: ${err.message}`);
    }
  };

  // ---------------------------------------------------------------------------
  // Delete Historical Scan
  // ---------------------------------------------------------------------------
  const deleteHistoricalAnalysis = async (e, analysisId) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this scan from history?')) {
      return;
    }
    try {
      const url = API_BASE ? `${API_BASE}/api/history/${analysisId}` : `/api/history/${analysisId}`;
      const res = await fetch(url, {
        method: 'DELETE',
      });
      if (res.ok) {
        fetchHistory();
        if (result?.analysis_id === analysisId) {
          handleReset();
        }
      }
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  // ---------------------------------------------------------------------------
  // PDF Report Download
  // ---------------------------------------------------------------------------
  const downloadPdfReport = async (targetResult = result) => {
    if (!targetResult) return;
    setIsGeneratingPdf(true);
    try {
      let res;
      if (targetResult.analysis_id) {
        const url = API_BASE
          ? `${API_BASE}/api/history/${targetResult.analysis_id}/pdf`
          : `/api/history/${targetResult.analysis_id}/pdf`;
        res = await fetch(url);
      }

      if (!res || !res.ok) {
        // Fallback to direct payload generation
        const postUrl = API_BASE ? `${API_BASE}/api/report/pdf` : '/api/report/pdf';
        res = await fetch(postUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(targetResult),
        });
      }

      if (!res.ok) {
        throw new Error('Failed to generate PDF report from server.');
      }

      const blob = await res.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `crackgauge_report_${targetResult.image_name || 'inspection'}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error('PDF generation error:', err);
      alert(`Could not download PDF: ${err.message}`);
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  // ---------------------------------------------------------------------------
  // CSV Export Download
  // ---------------------------------------------------------------------------
  const downloadCsvMeasurements = async (targetResult = result) => {
    if (!targetResult) return;
    setIsExportingCsv(true);
    try {
      let csvText = '';
      if (targetResult.analysis_id) {
        const url = API_BASE
          ? `${API_BASE}/api/history/${targetResult.analysis_id}/csv`
          : `/api/history/${targetResult.analysis_id}/csv`;
        const res = await fetch(url);
        if (res.ok) {
          csvText = await res.text();
        }
      }

      if (!csvText) {
        // Fallback client-side CSV formatting
        const isCal = Boolean(targetResult.summary?.calibration?.is_calibrated);
        const headers = [
          'analysis_id', 'image_name', 'timestamp', 'segmentation_profile',
          'is_calibrated', 'calibration_status', 'mm_per_pixel',
          'max_width_px', 'max_width_mm', 'mean_width_px', 'mean_width_mm',
          'total_length_px', 'total_length_mm', 'dominant_angle_deg', 'dominant_type',
          'crack_coverage_percent', 'crack_pixels', 'component_count'
        ];
        const row = [
          targetResult.analysis_id || '',
          targetResult.image_name || '',
          targetResult.timestamp || '',
          targetResult.segmentation_profile || 'baseline',
          isCal ? 'True' : 'False',
          targetResult.summary?.calibration?.status || 'not_calibrated',
          isCal ? (targetResult.summary?.calibration?.mm_per_pixel || '') : '',
          targetResult.summary?.max_width_px ?? '',
          isCal ? (targetResult.summary?.max_width_mm ?? '') : '',
          targetResult.summary?.mean_width_px ?? '',
          isCal ? (targetResult.summary?.mean_width_mm ?? '') : '',
          targetResult.summary?.total_length_px ?? '',
          isCal ? (targetResult.summary?.total_length_mm ?? '') : '',
          targetResult.summary?.dominant_angle_deg ?? '',
          `"${targetResult.summary?.dominant_type || ''}"`,
          targetResult.summary?.crack_coverage_percent ?? '',
          targetResult.summary?.crack_pixels ?? '',
          targetResult.summary?.component_count ?? 1
        ];
        csvText = headers.join(',') + '\n' + row.join(',') + '\n';
      }

      const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `crackgauge_${targetResult.image_name || 'inspection'}_measurements.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error('CSV export error:', err);
      alert(`Could not export CSV: ${err.message}`);
    } finally {
      setIsExportingCsv(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Export All History CSV
  // ---------------------------------------------------------------------------
  const exportAllHistory = async () => {
    try {
      const url = API_BASE ? `${API_BASE}/api/history/export/csv` : '/api/history/export/csv';
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to export history CSV.');
      const csvText = await res.text();
      const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = 'crackgauge_history.csv';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    }
  };

  // ---------------------------------------------------------------------------
  // Download Currently Active Visual PNG
  // ---------------------------------------------------------------------------
  const downloadActiveImage = () => {
    if (!result?.visuals?.[activeVisualTab]) return;
    const dataUrl = result.visuals[activeVisualTab];
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `${result.image_name || 'crack'}_${activeVisualTab}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // ---------------------------------------------------------------------------
  // Zoom & Pan Handlers
  // ---------------------------------------------------------------------------
  const handleZoomIn = () => {
    setZoomLevel((prev) => Math.min(3.0, Number((prev + 0.25).toFixed(2))));
  };

  const handleZoomOut = () => {
    setZoomLevel((prev) => {
      const next = Math.max(0.5, Number((prev - 0.25).toFixed(2)));
      if (next <= 1.0) setPanPosition({ x: 0, y: 0 });
      return next;
    });
  };

  const handleFitScreen = () => {
    setZoomLevel(1.0);
    setPanPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e) => {
    if (zoomLevel <= 1.0) return;
    setIsDragging(true);
    dragStartRef.current = {
      x: e.clientX - panPosition.x,
      y: e.clientY - panPosition.y,
    };
  };

  const handleMouseMove = (e) => {
    if (!isDragging || zoomLevel <= 1.0) return;
    setPanPosition({
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleTouchStart = (e) => {
    if (zoomLevel <= 1.0 || e.touches.length !== 1) return;
    setIsDragging(true);
    dragStartRef.current = {
      x: e.touches[0].clientX - panPosition.x,
      y: e.touches[0].clientY - panPosition.y,
    };
  };

  const handleTouchMove = (e) => {
    if (!isDragging || zoomLevel <= 1.0 || e.touches.length !== 1) return;
    setPanPosition({
      x: e.touches[0].clientX - dragStartRef.current.x,
      y: e.touches[0].clientY - dragStartRef.current.y,
    });
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
  };

  // ---------------------------------------------------------------------------
  // Reset Workflow
  // ---------------------------------------------------------------------------
  const handleReset = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg(null);
    setZoomLevel(1.0);
    setPanPosition({ x: 0, y: 0 });
    setStage('home');
  };

  // Helper formatting functions
  const fmt = (num, decimals = 2) => {
    if (num === null || num === undefined || isNaN(num)) return '—';
    return Number(num).toFixed(decimals);
  };

  // Format date readable
  const formatDate = (isoStr) => {
    if (!isoStr) return '—';
    try {
      const dt = new Date(isoStr);
      return dt.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoStr;
    }
  };

  // ---------------------------------------------------------------------------
  // Available Visual Tabs Definition
  // ---------------------------------------------------------------------------
  const visualTabs = [
    { id: 'original', label: 'Original Image' },
    { id: 'overlay', label: 'Crack Overlay' },
    { id: 'crack_mask', label: 'Binary Mask' },
    { id: 'width_heatmap', label: 'Width Heatmap' },
    { id: 'length_map', label: 'Length Skeleton' },
    { id: 'orientation_map', label: 'PCA Orientation' },
  ];
  if (result?.visuals?.rectified_marker) {
    visualTabs.push({ id: 'rectified_marker', label: 'ArUco Marker' });
  }

  return (
    <div className="app-container">
      {/* --------------------------------------------------------------------
          Hidden File Inputs (Camera & Gallery)
      --------------------------------------------------------------------- */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="file-input-hidden"
        onChange={handleFileChange}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/bmp,image/webp"
        className="file-input-hidden"
        onChange={handleFileChange}
      />

      {/* --------------------------------------------------------------------
          Sticky Top Header
      --------------------------------------------------------------------- */}
      <header className="app-header">
        <div className="brand-wrapper" onClick={handleReset} style={{ cursor: 'pointer' }}>
          <div className="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12h20M12 2v20M8 8l8 8M16 8l-8 8" />
            </svg>
          </div>
          <div className="brand-text">
            <h1>CrackGauge</h1>
            <span className="brand-badge">CV FIELD SCANNER</span>
          </div>
        </div>

        <div className="header-actions">
          {/* History Drawer Toggle Button */}
          <button
            type="button"
            className="btn-history-toggle"
            onClick={() => {
              setShowHistoryDrawer(true);
              fetchHistory();
            }}
            title="View persistent inspection history"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            <span>History</span>
            <span className="history-count-badge">{historyCount}</span>
          </button>

          {/* Liveness Indicator */}
          <div className="health-status" onClick={checkHealth} title="Click to refresh connection">
            <span className={`status-dot ${backendStatus === 'online' ? 'online' : 'offline'}`} />
            <span>{backendStatus === 'online' ? `Online ${serverVersion ? `v${serverVersion}` : ''}` : 'Offline'}</span>
          </div>
        </div>
      </header>

      {/* --------------------------------------------------------------------
          Main Content View Router
      --------------------------------------------------------------------- */}
      <main className="main-content">
        {/* Error Notification Banner */}
        {errorMsg && (
          <div className="error-card">
            <div className="error-header">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span>Inspection Alert</span>
            </div>
            <p className="error-message">{errorMsg}</p>
          </div>
        )}

        {/* ==================================================================
            STAGE 1: HOME SCREEN
        =================================================================== */}
        {stage === 'home' && (
          <>
            <div className="hero-box">
              <div className="hero-subtitle">Structural Health Inspection</div>
              <h2 className="hero-title">Concrete Crack Measurement</h2>
              <p className="hero-desc">
                High-precision width, length, and orientation measurements powered by Frangi ridge filtering and computer vision.
              </p>

              {/* Primary Prominent Action: Camera Scan */}
              <button
                type="button"
                className="btn-primary"
                onClick={triggerCamera}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                  <circle cx="12" cy="13" r="4" />
                </svg>
                <span>Scan Crack with Camera</span>
              </button>
            </div>

            {/* Secondary Option: File Picker */}
            <div className="action-grid">
              <button
                type="button"
                className="btn-secondary"
                onClick={triggerFilePicker}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span>Upload from Device Gallery</span>
              </button>
            </div>

            {/* Field Calibration Settings Drawer */}
            <div className="calibration-drawer">
              <div
                className="calibration-header"
                onClick={() => setShowConfig(!showConfig)}
              >
                <div className="calibration-title-row">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                  </svg>
                  <span>Field Calibration Options ({calibrationMode === 'none' ? 'Pixels Only' : calibrationMode === 'aruco' ? 'ArUco Marker' : 'Manual Scale'})</span>
                </div>
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  style={{ transform: showConfig ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s ease' }}
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </div>

              {showConfig && (
                <div className="calibration-body">
                  <div className="form-group">
                    <label className="form-label">Physical Calibration Source</label>
                    <select
                      className="form-select"
                      value={calibrationMode}
                      onChange={(e) => setCalibrationMode(e.target.value)}
                    >
                      <option value="none">None (Pixel-only measurements)</option>
                      <option value="aruco">Automatic ArUco Marker (mm)</option>
                      <option value="scale">Direct Scale (mm/pixel)</option>
                    </select>
                  </div>

                  {calibrationMode === 'aruco' && (
                    <div className="form-group">
                      <label className="form-label">ArUco Marker Physical Size (mm)</label>
                      <input
                        type="number"
                        step="0.5"
                        min="1"
                        className="form-input"
                        value={markerSizeMm}
                        onChange={(e) => setMarkerSizeMm(e.target.value)}
                        placeholder="e.g. 50.0"
                      />
                    </div>
                  )}

                  {calibrationMode === 'scale' && (
                    <div className="form-group">
                      <label className="form-label">Scale Factor (mm per pixel)</label>
                      <input
                        type="number"
                        step="0.001"
                        min="0.001"
                        className="form-input"
                        value={mmPerPixel}
                        onChange={(e) => setMmPerPixel(e.target.value)}
                        placeholder="e.g. 0.150"
                      />
                    </div>
                  )}

                  <div className="form-group">
                    <label className="form-label">Segmentation Algorithm Profile</label>
                    <select
                      className="form-select"
                      value={segmentationProfile}
                      onChange={(e) => setSegmentationProfile(e.target.value)}
                    >
                      <option value="baseline">Baseline (Frangi Primary Filter)</option>
                      <option value="adaptive_recovery">Adaptive Recovery (Low contrast)</option>
                      <option value="guided_recovery">Guided Recovery (Faint cracks)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* ==================================================================
            STAGE 2: PREVIEW BEFORE ANALYSIS
        =================================================================== */}
        {stage === 'preview' && (
          <div className="preview-card card">
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Inspect Captured Image</h3>
            <div className="preview-media-wrap">
              <img src={previewUrl} alt="Preview of selected crack" />
            </div>

            <div className="file-meta-row">
              <span>{selectedFile?.name || 'capture.jpg'}</span>
              <span>{(selectedFile?.size / 1024).toFixed(1)} KB</span>
            </div>

            <button
              type="button"
              className="btn-primary"
              onClick={runAnalysis}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <span>Analyze Crack Now</span>
            </button>

            <button
              type="button"
              className="btn-secondary"
              onClick={handleReset}
            >
              <span>Retake / Choose Another</span>
            </button>
          </div>
        )}

        {/* ==================================================================
            STAGE 3: SCANNING / LOADING STATE
        =================================================================== */}
        {stage === 'analyzing' && (
          <div className="loading-box">
            <div className="radar-spinner" />
            <div>
              <h3 className="loading-title">Processing Crack Analysis</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '4px' }}>
                Running multi-stage Hessian ridge filter and geometry extraction...
              </p>
            </div>

            <div className="loading-steps">
              <div className="step-item active">
                <span>1. CLAHE Grayscale &amp; Adaptive Filter</span>
              </div>
              <div className="step-item active">
                <span>2. Multi-Scale Frangi Ridge Detection</span>
              </div>
              <div className="step-item active">
                <span>3. Distance Transform Width Profiling</span>
              </div>
              <div className="step-item active">
                <span>4. Skeletonization &amp; PCA Orientation</span>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================
            STAGE 4: RESULTS DASHBOARD
        =================================================================== */}
        {stage === 'results' && result && (
          <div className="results-wrapper">
            {/* Classification & Orientation Highlight */}
            <div className="classification-banner">
              <div className="classification-info">
                <span className="classification-label">Crack Pattern</span>
                <span className="classification-value">{result.summary.dominant_type}</span>
              </div>
              <div className="classification-badge">
                {fmt(result.summary.dominant_angle_deg, 1)}° Angle
              </div>
            </div>

            {/* Visual Viewer with Tabs and Zoom/Download Toolbar */}
            <div className="visual-viewer-card">
              {/* Tabs for all required views */}
              <div className="visual-tabs-scroll">
                {visualTabs.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={`tab-btn ${activeVisualTab === tab.id ? 'active' : ''}`}
                    onClick={() => handleTabChange(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Viewer Control Toolbar: Zoom & Download */}
              <div className="viewer-controls-bar">
                <div className="viewer-zoom-group">
                  <button
                    type="button"
                    className="control-btn"
                    onClick={handleZoomOut}
                    disabled={zoomLevel <= 0.5}
                    title="Zoom Out"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  </button>
                  <span className="zoom-indicator">{Math.round(zoomLevel * 100)}%</span>
                  <button
                    type="button"
                    className="control-btn"
                    onClick={handleZoomIn}
                    disabled={zoomLevel >= 3.0}
                    title="Zoom In"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    className={`control-btn ${zoomLevel === 1.0 && panPosition.x === 0 && panPosition.y === 0 ? 'active' : ''}`}
                    onClick={handleFitScreen}
                    title="Reset to 100% / Fit to Screen"
                  >
                    Fit Screen
                  </button>
                </div>

                <button
                  type="button"
                  className="control-btn"
                  onClick={downloadActiveImage}
                  title="Download Current Visual as PNG"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Download PNG</span>
                </button>
              </div>

              {/* Canvas with Pan & Zoom Transform */}
              <div
                className={`viewer-canvas ${zoomLevel > 1.0 ? (isDragging ? 'is-dragging' : 'is-zoomed') : ''}`}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
              >
                {result.visuals?.[activeVisualTab] ? (
                  <img
                    src={result.visuals[activeVisualTab]}
                    alt={`Visualization: ${activeVisualTab}`}
                    className="viewer-canvas-img"
                    style={{
                      transform: `scale(${zoomLevel}) translate(${panPosition.x / zoomLevel}px, ${panPosition.y / zoomLevel}px)`,
                      transition: isDragging ? 'none' : 'transform 0.15s ease-out',
                    }}
                    draggable={false}
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)', padding: '40px 0' }}>
                    Visual artifact not available for this mode
                  </div>
                )}
              </div>

              <div className="viewer-caption">
                <span>Resolution: {result.dimensions?.width} × {result.dimensions?.height} px</span>
                <span style={{ textTransform: 'capitalize' }}>
                  Tab: {activeVisualTab.replace('_', ' ')}
                  {zoomLevel > 1.0 && ` (${Math.round(zoomLevel * 100)}% - drag to pan)`}
                </span>
              </div>
            </div>

            {/* Calibration Status Badge & Notice */}
            <div className="calibration-status-card">
              <div className="calibration-badge-row">
                <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Calibration Status</span>
                {result.summary.calibration?.is_calibrated ? (
                  <span className="badge-calibrated">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    Calibrated in mm
                  </span>
                ) : (
                  <span className="badge-uncalibrated">
                    Pixel Measurements Only
                  </span>
                )}
              </div>
              <p className="calibration-detail-text">
                {result.summary.calibration?.is_calibrated ? (
                  <>
                    Scale: <strong>{fmt(result.summary.calibration.mm_per_pixel, 4)} mm/px</strong>
                    {result.summary.calibration.marker_id !== null && ` (ArUco ID ${result.summary.calibration.marker_id})`}
                  </>
                ) : (
                  'No physical scale or ArUco reference marker detected. Measurements are reported strictly in pixels.'
                )}
              </p>
            </div>

            {/* Core Metric Grid (Respecting physical units rule) */}
            <div className="metrics-grid">
              {/* Max Crack Width */}
              <div className="metric-card highlight">
                <div className="metric-header">
                  <span>Max Crack Width</span>
                </div>
                <div>
                  <div className="metric-primary-val">
                    {result.summary.calibration?.is_calibrated && result.summary.max_width_mm !== undefined ? (
                      <>
                        {fmt(result.summary.max_width_mm, 2)}
                        <span className="metric-unit">mm</span>
                      </>
                    ) : (
                      <>
                        {fmt(result.summary.max_width_px, 1)}
                        <span className="metric-unit">px</span>
                      </>
                    )}
                  </div>
                  <div className="metric-secondary-val">
                    {result.summary.calibration?.is_calibrated && result.summary.max_width_mm !== undefined ? (
                      `Pixel size: ${fmt(result.summary.max_width_px, 1)} px`
                    ) : (
                      'Uncalibrated scale'
                    )}
                  </div>
                </div>
              </div>

              {/* Total Crack Length */}
              <div className="metric-card">
                <div className="metric-header">
                  <span>Total Length</span>
                </div>
                <div>
                  <div className="metric-primary-val">
                    {result.summary.calibration?.is_calibrated && result.summary.total_length_mm !== undefined ? (
                      <>
                        {fmt(result.summary.total_length_mm, 1)}
                        <span className="metric-unit">mm</span>
                      </>
                    ) : (
                      <>
                        {fmt(result.summary.total_length_px, 1)}
                        <span className="metric-unit">px</span>
                      </>
                    )}
                  </div>
                  <div className="metric-secondary-val">
                    {result.summary.calibration?.is_calibrated && result.summary.total_length_mm !== undefined ? (
                      `Pixel length: ${fmt(result.summary.total_length_px, 0)} px`
                    ) : (
                      `${result.summary.component_count || 1} continuous branch(es)`
                    )}
                  </div>
                </div>
              </div>

              {/* Mean Width */}
              <div className="metric-card">
                <div className="metric-header">
                  <span>Mean Width</span>
                </div>
                <div>
                  <div className="metric-primary-val">
                    {result.summary.calibration?.is_calibrated && result.summary.mean_width_mm !== undefined ? (
                      <>
                        {fmt(result.summary.mean_width_mm, 2)}
                        <span className="metric-unit">mm</span>
                      </>
                    ) : (
                      <>
                        {fmt(result.summary.mean_width_px, 1)}
                        <span className="metric-unit">px</span>
                      </>
                    )}
                  </div>
                  <div className="metric-secondary-val">
                    Median:{' '}
                    {result.summary.calibration?.is_calibrated && result.summary.median_width_mm !== undefined
                      ? `${fmt(result.summary.median_width_mm, 2)} mm`
                      : `${fmt(result.summary.median_width_px, 1)} px`}
                  </div>
                </div>
              </div>

              {/* Surface Coverage */}
              <div className="metric-card">
                <div className="metric-header">
                  <span>Surface Area</span>
                </div>
                <div>
                  <div className="metric-primary-val">
                    {fmt(result.summary.crack_coverage_percent, 2)}
                    <span className="metric-unit">%</span>
                  </div>
                  <div className="metric-secondary-val">
                    {Number(result.summary.crack_pixels || 0).toLocaleString()} crack pixels
                  </div>
                </div>
              </div>
            </div>

            {/* Report Generation Actions: Professional PDF & CSV Export */}
            <div className="report-action-card">
              <button
                type="button"
                className="btn-report btn-pdf"
                onClick={() => downloadPdfReport(result)}
                disabled={isGeneratingPdf}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
                <span>{isGeneratingPdf ? 'Generating PDF...' : 'Download PDF Report'}</span>
              </button>

              <button
                type="button"
                className="btn-report btn-csv"
                onClick={() => downloadCsvMeasurements(result)}
                disabled={isExportingCsv}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                <span>{isExportingCsv ? 'Exporting...' : 'Export CSV'}</span>
              </button>
            </div>

            {/* Components Detail Card if multiple fragments */}
            {result.components && result.components.length > 1 && (
              <div className="card" style={{ marginTop: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>
                    Continuous Segments ({result.components.length})
                  </span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Gap-closed branches
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {result.components.slice(0, 5).map((c, idx) => (
                    <div
                      key={c.id || idx}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '8px 10px',
                        backgroundColor: 'var(--bg-card-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.82rem',
                      }}
                    >
                      <span style={{ fontWeight: 600 }}>Branch #{c.id || idx + 1}</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>
                        {result.summary?.calibration?.is_calibrated && c.length_mm !== undefined
                          ? `${fmt(c.length_mm, 1)} mm`
                          : `${fmt(c.length_px, 1)} px`}
                      </span>
                    </div>
                  ))}
                  {result.components.length > 5 && (
                    <div style={{ textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', paddingTop: '4px' }}>
                      + {result.components.length - 5} smaller segment(s)
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* --------------------------------------------------------------------
          Sticky Bottom Action Bar in Results Stage
      --------------------------------------------------------------------- */}
      {stage === 'results' && (
        <div className="bottom-bar">
          <button
            type="button"
            className="btn-primary"
            onClick={handleReset}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
            </svg>
            <span>Scan New Crack</span>
          </button>
        </div>
      )}

      {/* --------------------------------------------------------------------
          Persistent Inspection History Side Drawer
      --------------------------------------------------------------------- */}
      {showHistoryDrawer && (
        <div className="history-backdrop" onClick={() => setShowHistoryDrawer(false)}>
          <div className="history-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="history-drawer-header">
              <div className="history-title-wrap">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <polyline points="12 6 12 12 16 14" />
                </svg>
                <h2>Inspection History</h2>
              </div>
              <button
                type="button"
                className="history-close-btn"
                onClick={() => setShowHistoryDrawer(false)}
                title="Close drawer"
              >
                &times;
              </button>
            </div>

            <div className="history-toolbar">
              <span>{historyCount} saved inspection{historyCount === 1 ? '' : 's'}</span>
              {historyCount > 0 && (
                <button
                  type="button"
                  className="control-btn"
                  onClick={exportAllHistory}
                  title="Export all scans to CSV"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Export All CSV</span>
                </button>
              )}
            </div>

            <div className="history-drawer-body">
              {isLoadingHistory ? (
                <div style={{ textAlign: 'center', padding: '36px 0', color: 'var(--text-muted)' }}>
                  Loading inspection records...
                </div>
              ) : historyItems.length === 0 ? (
                <div className="history-empty-state">
                  <div className="history-empty-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                      <line x1="8" y1="21" x2="16" y2="21" />
                      <line x1="12" y1="17" x2="12" y2="21" />
                    </svg>
                  </div>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: 'var(--text-main)' }}>No Scans Yet</h3>
                  <p style={{ fontSize: '0.82rem', lineHeight: 1.4 }}>
                    Analyses conducted with the camera or uploaded images are automatically saved to your persistent history database.
                  </p>
                </div>
              ) : (
                historyItems.map((item) => (
                  <div
                    key={item.analysis_id}
                    className="history-item-card"
                    onClick={() => loadHistoricalAnalysis(item.analysis_id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="history-card-top">
                      <div>
                        <div className="history-item-name">{item.image_name}</div>
                        <div className="history-item-time">{formatDate(item.timestamp)}</div>
                      </div>
                      <span className="brand-badge" style={{ fontSize: '0.64rem' }}>
                        {item.segmentation_profile || 'baseline'}
                      </span>
                    </div>

                    <div className="history-metrics-summary">
                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Max Width</span>
                        <span className="history-metric-val">
                          {item.is_calibrated && item.max_width_mm !== null
                            ? `${fmt(item.max_width_mm, 2)} mm`
                            : `${fmt(item.max_width_px, 1)} px`}
                        </span>
                      </div>
                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Length</span>
                        <span className="history-metric-val">
                          {item.is_calibrated && item.total_length_mm !== null
                            ? `${fmt(item.total_length_mm, 1)} mm`
                            : `${fmt(item.total_length_px, 1)} px`}
                        </span>
                      </div>
                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Orientation</span>
                        <span className="history-metric-val">
                          {item.dominant_type} ({fmt(item.dominant_angle_deg, 1)}°)
                        </span>
                      </div>
                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Coverage</span>
                        <span className="history-metric-val">
                          {fmt(item.crack_coverage_percent, 2)}%
                        </span>
                      </div>
                    </div>

                    <div className="history-card-footer">
                      <div>
                        {item.is_calibrated ? (
                          <span className="badge-calibrated" style={{ fontSize: '0.68rem', padding: '2px 7px' }}>
                            Calibrated
                          </span>
                        ) : (
                          <span className="badge-uncalibrated" style={{ fontSize: '0.68rem', padding: '2px 7px' }}>
                            Pixels Only
                          </span>
                        )}
                      </div>

                      <div className="history-card-actions">
                        <button
                          type="button"
                          className="btn-card-action"
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadPdfReport({
                              analysis_id: item.analysis_id,
                              image_name: item.image_name,
                            });
                          }}
                          title="Download PDF report for this scan"
                        >
                          PDF
                        </button>
                        <button
                          type="button"
                          className="btn-card-action"
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadCsvMeasurements({
                              analysis_id: item.analysis_id,
                              image_name: item.image_name,
                            });
                          }}
                          title="Export CSV for this scan"
                        >
                          CSV
                        </button>
                        <button
                          type="button"
                          className="btn-card-action btn-card-delete"
                          onClick={(e) => deleteHistoricalAnalysis(e, item.analysis_id)}
                          title="Delete from history"
                        >
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
