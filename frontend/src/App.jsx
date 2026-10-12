import React, { useState, useEffect, useRef } from 'react';

// Get API base URL from Vite environment or default to proxy/relative path
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export default function App() {
  // Navigation / Workflow State: 'home' | 'preview' | 'analyzing' | 'results'
  const [stage, setStage] = useState('home');

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

  // File Input Refs
  const cameraInputRef = useRef(null);
  const fileInputRef = useRef(null);

  // ---------------------------------------------------------------------------
  // Check Backend Health on Mount
  // ---------------------------------------------------------------------------
  const checkHealth = async () => {
    try {
      const url = API_BASE ? `${API_BASE}/api/health` : '/api/health';
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus('online');
        setServerVersion(data.version || '0.1.0');
      } else {
        setBackendStatus('offline');
      }
    } catch {
      setBackendStatus('offline');
    }
  };

  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        const url = API_BASE ? `${API_BASE}/api/health` : '/api/health';
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
            setBackendStatus('online');
            setServerVersion(data.version || '0.1.0');
          }
        } else if (isMounted) {
          setBackendStatus('offline');
        }
      } catch {
        if (isMounted) {
          setBackendStatus('offline');
        }
      }
    })();
    return () => {
      isMounted = false;
    };
  }, []);

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
    setStage('home');
  };

  // Helper formatting functions
  const fmt = (num, decimals = 2) => {
    if (num === null || num === undefined || isNaN(num)) return '—';
    return Number(num).toFixed(decimals);
  };

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
        <div className="brand-wrapper">
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

        <div className="health-status" onClick={checkHealth} title="Click to refresh connection">
          <span className={`status-dot ${backendStatus === 'online' ? 'online' : 'offline'}`} />
          <span>{backendStatus === 'online' ? `API Online ${serverVersion ? `v${serverVersion}` : ''}` : 'API Offline'}</span>
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
                <span>1. CLAHE Grayscale & Adaptive Filter</span>
              </div>
              <div className="step-item active">
                <span>2. Multi-Scale Frangi Ridge Detection</span>
              </div>
              <div className="step-item active">
                <span>3. Distance Transform Width Profiling</span>
              </div>
              <div className="step-item active">
                <span>4. Skeletonization & PCA Orientation</span>
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

            {/* Visual Viewer with Tabs */}
            <div className="visual-viewer-card">
              <div className="visual-tabs-scroll">
                <button
                  type="button"
                  className={`tab-btn ${activeVisualTab === 'overlay' ? 'active' : ''}`}
                  onClick={() => setActiveVisualTab('overlay')}
                >
                  Crack Overlay
                </button>
                <button
                  type="button"
                  className={`tab-btn ${activeVisualTab === 'width_heatmap' ? 'active' : ''}`}
                  onClick={() => setActiveVisualTab('width_heatmap')}
                >
                  Width Heatmap
                </button>
                <button
                  type="button"
                  className={`tab-btn ${activeVisualTab === 'length_map' ? 'active' : ''}`}
                  onClick={() => setActiveVisualTab('length_map')}
                >
                  Length Skeleton
                </button>
                <button
                  type="button"
                  className={`tab-btn ${activeVisualTab === 'orientation_map' ? 'active' : ''}`}
                  onClick={() => setActiveVisualTab('orientation_map')}
                >
                  PCA Orientation
                </button>
                <button
                  type="button"
                  className={`tab-btn ${activeVisualTab === 'original' ? 'active' : ''}`}
                  onClick={() => setActiveVisualTab('original')}
                >
                  Original Image
                </button>
                {result.visuals?.rectified_marker && (
                  <button
                    type="button"
                    className={`tab-btn ${activeVisualTab === 'rectified_marker' ? 'active' : ''}`}
                    onClick={() => setActiveVisualTab('rectified_marker')}
                  >
                    ArUco Marker
                  </button>
                )}
              </div>

              <div className="viewer-canvas">
                {result.visuals?.[activeVisualTab] ? (
                  <img
                    src={result.visuals[activeVisualTab]}
                    alt={`Visualization: ${activeVisualTab}`}
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)', padding: '40px 0' }}>
                    Visual not available for this mode
                  </div>
                )}
              </div>

              <div className="viewer-caption">
                <span>Resolution: {result.dimensions?.width} × {result.dimensions?.height} px</span>
                <span style={{ textTransform: 'capitalize' }}>Mode: {activeVisualTab.replace('_', ' ')}</span>
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
                  'No physical scale or ArUco reference marker detected. Measurements are reported in pixels.'
                )}
              </p>
            </div>

            {/* Core Metric Grid */}
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
                    Median: {result.summary.calibration?.is_calibrated && result.summary.median_width_mm !== undefined
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
                        {c.length_mm !== undefined ? `${fmt(c.length_mm, 1)} mm` : `${fmt(c.length_px, 1)} px`}
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
    </div>
  );
}
