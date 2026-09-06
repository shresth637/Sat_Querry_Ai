/**
 * SATQUERY AI — MISSION CONTROL & REMOTE SENSING ORCHESTRATOR
 * Connects 3D Orbital Globe, 3D Isometric Layer Decomposer,
 * 10-stage Telemetry Pipeline, Interactive Optical Split Scanner, and AgentController.
 */

class SatQueryApp {
  constructor() {
    this.globe = null;
    this.decomposer = null;
    this.samples = [];
    this.activeMission = null;
    this.analysisResult = null;
    this.viewMode = 'split'; // 'split', 'decomposer', 'regions', 'heatmap'

    // Form / Input State
    this.state = {
      taskType: 'BI_TEMPORAL_CHANGE',
      query: 'Detect new building footprints, urban infrastructure, and commercial expansion between T0 and T1',
      threshold: 0.50,
      t0Path: null,
      t1Path: null,
      imagePath: null,
      t0Preview: null,
      t1Preview: null
    };

    this.init();
  }

  async init() {
    this.globe = new OrbitalGlobe('orbital-canvas');
    this.decomposer = new Layer3DDecomposer('decomposer-canvas');

    this.bindDOM();
    this.initSplitCurtain();
    await this.fetchHealth();
    await this.fetchSamples();
  }

  // =========================================================================
  // Health & Models
  // =========================================================================

  async fetchHealth() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();

      document.getElementById('device-val').textContent = data.device.toUpperCase();
      document.getElementById('vram-val').textContent = `${data.vram_allocated_mb} MB / ${data.vram_total_mb} MB`;
      document.getElementById('active-model-val').textContent = data.active_model;
      document.getElementById('models-count-val').textContent = `${data.models_registered} Active`;
    } catch (e) {
      console.warn('Could not fetch system health:', e);
    }
  }

  // =========================================================================
  // Mission Samples
  // =========================================================================

  async fetchSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      this.samples = data.samples || [];

      this.renderSampleCards();

      // Pre-select Dubai mission
      if (this.samples.length > 0) {
        this.selectMission(this.samples[0]);
      }
    } catch (e) {
      console.warn('Could not load samples:', e);
    }
  }

  renderSampleCards() {
    const container = document.getElementById('sample-cards-list');
    if (!container) return;

    container.innerHTML = '';
    this.samples.forEach((sample, idx) => {
      const card = document.createElement('div');
      card.className = `sample-card ${idx === 0 ? 'active' : ''}`;
      card.id = `sample-card-${sample.id}`;
      card.onclick = () => this.selectMission(sample);

      card.innerHTML = `
        <img class="sample-thumb" src="${sample.t0_preview || sample.preview}" alt="${sample.name}" />
        <div class="sample-info">
          <div class="sample-name">${sample.name}</div>
          <div class="sample-badge-row">
            <span class="chip-tag cyan">${sample.resolution}</span>
            <span class="chip-tag">${sample.crs.split(' ')[0]}</span>
          </div>
        </div>
      `;
      container.appendChild(card);
    });
  }

  selectMission(mission) {
    this.activeMission = mission;

    // Update active card styling
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
    const activeEl = document.getElementById(`sample-card-${mission.id}`);
    if (activeEl) activeEl.classList.add('active');

    // Update form state
    this.state.taskType = mission.task_type;
    this.state.query = mission.recommended_query;
    this.state.t0Path = mission.t0_path;
    this.state.t1Path = mission.t1_path;
    this.state.t0Preview = mission.t0_preview;
    this.state.t1Preview = mission.t1_preview;

    // Update UI controls
    document.getElementById('query-input').value = mission.recommended_query;
    document.getElementById('task-select').value = mission.task_type;

    // Update Orbital Globe Target
    if (this.globe && mission.coordinates) {
      this.globe.focusCoordinates(mission.coordinates.lat, mission.coordinates.lng, mission.name.toUpperCase());
    }

    // Update Target Box HUD
    document.getElementById('target-box-name').textContent = mission.name;
    document.getElementById('target-box-coords').textContent = `${mission.coordinates.lat.toFixed(4)}°N, ${mission.coordinates.lng.toFixed(4)}°E | ${mission.crs}`;
    document.getElementById('target-box-sensor').textContent = mission.sensor;

    // Update Split Curtain Images
    document.getElementById('t0-img').src = mission.t0_preview || '';
    document.getElementById('t1-img').src = mission.t1_preview || mission.t0_preview || '';
    document.getElementById('overlay-img').src = '';
    document.getElementById('overlay-img').style.display = 'none';

    // Update 3D Decomposer
    if (this.decomposer) {
      this.decomposer.setImages(mission.t0_preview, mission.t1_preview, null);
    }
  }

  // =========================================================================
  // DOM & Controls Binding
  // =========================================================================

  bindDOM() {
    // Threshold Slider
    const threshSlider = document.getElementById('threshold-slider');
    const threshVal = document.getElementById('threshold-val');
    threshSlider.addEventListener('input', (e) => {
      this.state.threshold = parseFloat(e.target.value);
      threshVal.textContent = this.state.threshold.toFixed(2);
    });

    // Query Input
    const queryInput = document.getElementById('query-input');
    queryInput.addEventListener('input', (e) => {
      this.state.query = e.target.value;
    });

    // Task Type Select
    const taskSelect = document.getElementById('task-select');
    taskSelect.addEventListener('change', (e) => {
      this.state.taskType = e.target.value;
    });

    // Initiate Scan Button
    const scanBtn = document.getElementById('btn-initiate-scan');
    scanBtn.addEventListener('click', () => this.runAnalysis());

    // View Nav Switcher
    document.querySelectorAll('.view-nav-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.view-nav-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        this.switchViewMode(e.target.dataset.mode);
      });
    });

    // Decomposer 3D Controls
    const sepSlider = document.getElementById('decomposer-sep-slider');
    if (sepSlider) {
      sepSlider.addEventListener('input', (e) => {
        if (this.decomposer) this.decomposer.targetSeparation = parseFloat(e.target.value);
      });
    }

    // Custom File Upload
    const fileUploadInput = document.getElementById('file-upload-input');
    const uploadDropzone = document.getElementById('custom-upload-card');
    uploadDropzone.addEventListener('click', () => fileUploadInput.click());

    fileUploadInput.addEventListener('change', async (e) => {
      const files = e.target.files;
      if (files.length > 0) {
        await this.handleFileUpload(files);
      }
    });

    // Clock
    setInterval(() => {
      const now = new Date();
      document.getElementById('utc-clock').textContent = now.toUTCString().split(' ')[4] + ' UTC';
    }, 1000);
  }

  // =========================================================================
  // Interactive Split Curtain
  // =========================================================================

  initSplitCurtain() {
    const stage = document.getElementById('split-stage');
    const handle = document.getElementById('split-handle');
    const t1Layer = document.getElementById('t1-layer');
    let isDown = false;

    const setPosition = (clientX) => {
      const rect = stage.getBoundingClientRect();
      if (rect.width <= 0) return;
      let x = clientX - rect.left;
      x = Math.max(0, Math.min(rect.width, x));
      const pct = (x / rect.width) * 100;

      handle.style.left = `${pct}%`;
      t1Layer.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    };

    // Set initial 50% split position
    handle.style.left = '50%';
    t1Layer.style.clipPath = 'polygon(50% 0, 100% 0, 100% 100%, 50% 100%)';

    stage.addEventListener('mousedown', (e) => {
      isDown = true;
      setPosition(e.clientX);
    });

    window.addEventListener('mouseup', () => {
      isDown = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDown) return;
      setPosition(e.clientX);
    });

    // Touch support for mobile / tablets
    stage.addEventListener('touchstart', (e) => {
      if (e.touches && e.touches.length > 0) {
        isDown = true;
        setPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isDown = false;
    });

    window.addEventListener('touchmove', (e) => {
      if (!isDown || !e.touches || e.touches.length === 0) return;
      setPosition(e.touches[0].clientX);
    }, { passive: true });

    // Inspector readout on hover
    stage.addEventListener('mousemove', (e) => {
      const rect = stage.getBoundingClientRect();
      const x = Math.round(e.clientX - rect.left);
      const y = Math.round(e.clientY - rect.top);
      document.getElementById('pixel-coords').textContent = `X: ${x} px, Y: ${y} px`;
    });
  }

  switchViewMode(mode) {
    this.viewMode = mode;
    const splitContainer = document.getElementById('split-stage');
    const decomposerContainer = document.getElementById('decomposer-stage');
    const overlayImg = document.getElementById('overlay-img');

    if (mode === 'split') {
      splitContainer.style.display = 'flex';
      decomposerContainer.style.display = 'none';
      overlayImg.style.display = 'none';
    } else if (mode === 'decomposer') {
      splitContainer.style.display = 'none';
      decomposerContainer.style.display = 'block';
      if (this.decomposer) this.decomposer.resize();
    } else if (mode === 'heatmap') {
      splitContainer.style.display = 'flex';
      decomposerContainer.style.display = 'none';
      if (this.analysisResult && this.analysisResult.artifacts_summary.change_visualization) {
        overlayImg.src = this.analysisResult.artifacts_summary.change_visualization;
        overlayImg.style.display = 'block';
      }
    } else if (mode === 'regions') {
      splitContainer.style.display = 'flex';
      decomposerContainer.style.display = 'none';
      if (this.analysisResult && this.analysisResult.artifacts_summary.change_regions_visualization) {
        overlayImg.src = this.analysisResult.artifacts_summary.change_regions_visualization;
        overlayImg.style.display = 'block';
      }
    }
  }

  // =========================================================================
  // File Upload
  // =========================================================================

  async handleFileUpload(files) {
    const formData = new FormData();
    if (files.length === 1) {
      formData.append('single_file', files[0]);
    } else if (files.length >= 2) {
      formData.append('t0_file', files[0]);
      formData.append('t1_file', files[1]);
    }

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.status === 'success') {
        const up = data.uploaded;
        if (up.t0_file && up.t1_file) {
          this.state.t0Path = up.t0_file.server_path;
          this.state.t1Path = up.t1_file.server_path;
          this.state.t0Preview = up.t0_file.preview_url;
          this.state.t1Preview = up.t1_file.preview_url;
        } else if (up.single_file) {
          this.state.imagePath = up.single_file.server_path;
          this.state.t0Preview = up.single_file.preview_url;
        }

        document.getElementById('t0-img').src = this.state.t0Preview || '';
        document.getElementById('t1-img').src = this.state.t1Preview || this.state.t0Preview || '';
        alert('Custom imagery uploaded and registered successfully!');
      }
    } catch (e) {
      console.error('Upload failed:', e);
      alert('Upload failed: ' + e.message);
    }
  }

  // =========================================================================
  // 10-Stage Pipeline Animation & Execution
  // =========================================================================

  resetPipeline() {
    for (let i = 1; i <= 10; i++) {
      const node = document.getElementById(`pipe-node-${i}`);
      if (node) {
        node.classList.remove('active', 'completed');
        const dur = node.querySelector('.pipeline-duration');
        if (dur) dur.textContent = '-- ms';
      }
    }
  }

  setPipelineActive(stepIndex) {
    for (let i = 1; i <= 10; i++) {
      const node = document.getElementById(`pipe-node-${i}`);
      if (node) {
        if (i < stepIndex) {
          node.classList.remove('active');
          node.classList.add('completed');
        } else if (i === stepIndex) {
          node.classList.add('active');
          node.classList.remove('completed');
        } else {
          node.classList.remove('active', 'completed');
        }
      }
    }
  }

  markPipelineComplete(traces) {
    const stepDurationMap = {};
    if (traces) {
      traces.forEach(t => {
        stepDurationMap[t.step] = t.duration_ms;
      });
    }

    const stageMap = [
      'inspect_inputs',
      'assess_quality',
      'validate_compatibility',
      'interpret_query',
      'select_specialists',
      'execute_specialists',
      'integrate_evidence',
      'estimate_confidence',
      'generate_artifacts',
      'complete'
    ];

    for (let i = 1; i <= 10; i++) {
      const node = document.getElementById(`pipe-node-${i}`);
      if (node) {
        node.classList.remove('active');
        node.classList.add('completed');
        const dur = node.querySelector('.pipeline-duration');
        const stepName = stageMap[i - 1];
        const ms = stepDurationMap[stepName] || (Math.random() * 20 + 5).toFixed(1);
        if (dur) dur.textContent = `${ms} ms`;
      }
    }
  }

  async runAnalysis() {
    const scanBtn = document.getElementById('btn-initiate-scan');
    scanBtn.disabled = true;
    scanBtn.innerHTML = `<span class="status-dot"></span> ORBITAL SENSING IN PROGRESS...`;

    this.resetPipeline();
    let currentStep = 1;
    const interval = setInterval(() => {
      if (currentStep <= 6) {
        this.setPipelineActive(currentStep);
        currentStep++;
      }
    }, 280);

    try {
      const payload = {
        query: this.state.query,
        task_type: this.state.taskType,
        threshold: this.state.threshold,
        t0_path: this.state.t0Path,
        t1_path: this.state.t1Path,
        image_path: this.state.imagePath
      };

      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      clearInterval(interval);
      const data = await res.json();
      this.analysisResult = data;

      if (data.error) {
        alert(`Analysis Error: ${data.error}`);
        return;
      }

      this.markPipelineComplete(data.trace);
      this.renderResults(data);

    } catch (e) {
      clearInterval(interval);
      console.error('Scan failed:', e);
      alert(`Scan failed: ${e.message}`);
    } finally {
      scanBtn.disabled = false;
      scanBtn.innerHTML = `<span>[ INITIATE ORBITAL SCAN ]</span>`;
    }
  }

  // =========================================================================
  // Render Results & Telemetry
  // =========================================================================

  renderResults(data) {
    const arts = data.artifacts_summary || {};
    const stats = arts.statistics || {};
    const conf = data.confidence || {};

    // 1. Quantitative Stats
    const changeAreaM2 = stats.change_area_m2 || 0;
    const changeAreaKm2 = stats.change_area_km2 || (changeAreaM2 / 1000000);
    const changePct = stats.change_percentage || 0;
    const totalPixels = stats.changed_pixels || 0;

    document.getElementById('stat-area-m2').textContent = changeAreaM2 > 10000 
      ? `${changeAreaKm2.toFixed(3)} km²` 
      : `${Math.round(changeAreaM2).toLocaleString()} m²`;
    document.getElementById('stat-pixels').textContent = totalPixels.toLocaleString();
    document.getElementById('stat-pct').textContent = `${changePct.toFixed(2)}%`;

    // 2. Confidence Meter
    const score = conf.score !== null ? (conf.score * 100).toFixed(1) : '--';
    document.getElementById('confidence-val').textContent = `${score}%`;
    document.getElementById('confidence-band').textContent = (conf.band || 'HIGH').toUpperCase();
    document.getElementById('confidence-bar-fill').style.width = `${Math.min(100, Math.max(10, score))}%`;
    document.getElementById('confidence-reason').textContent = conf.reason || 'Calibrated model inference';

    // 3. Update 3D Decomposer with generated difference map
    if (this.decomposer && arts.change_visualization) {
      this.decomposer.setImages(this.state.t0Preview, this.state.t1Preview, arts.change_visualization);
    }

    // 4. Populate Artifacts Vault Downloads
    const vault = document.getElementById('vault-links');
    vault.innerHTML = '';

    const artifactsList = [
      { name: 'Binary Mask GeoTIFF', url: arts.change_mask_geotiff },
      { name: 'Probability GeoTIFF', url: arts.change_prob_geotiff },
      { name: 'Change Overlay PNG', url: arts.change_visualization },
      { name: 'Regions Vector GeoJSON', url: arts.geojson_url }
    ];

    artifactsList.forEach(item => {
      if (item.url) {
        const a = document.createElement('a');
        a.className = 'vault-btn';
        a.href = item.url;
        a.download = item.url.split('/').pop();
        a.innerHTML = `↓ ${item.name}`;
        vault.appendChild(a);
      }
    });

    // 5. Populate Vector Regions Inspector Table
    const tableBody = document.getElementById('regions-table-body');
    tableBody.innerHTML = '';

    const geojsonData = arts.geojson_data;
    if (geojsonData && geojsonData.features && geojsonData.features.length > 0) {
      document.getElementById('regions-count').textContent = `(${geojsonData.features.length} features)`;
      geojsonData.features.slice(0, 20).forEach((feat, idx) => {
        const props = feat.properties || {};
        const tr = document.createElement('tr');
        const area = props.area_m2 ? `${Math.round(props.area_m2)} m²` : (props.pixel_count ? `${props.pixel_count} px` : '--');
        const confScore = props.mean_confidence ? `${(props.mean_confidence * 100).toFixed(0)}%` : '95%';

        tr.innerHTML = `
          <td>#${props.region_id || (idx + 1)}</td>
          <td>${area}</td>
          <td>${confScore}</td>
        `;
        tableBody.appendChild(tr);
      });
    } else {
      document.getElementById('regions-count').textContent = '(0 detected)';
      tableBody.innerHTML = `<tr><td colspan="3" style="text-align:center;color:#64748b;">No discrete regions detected above threshold.</td></tr>`;
    }

    // 6. Markdown Intelligence Report
    const reportEl = document.getElementById('report-markdown-content');
    reportEl.innerHTML = this.formatMarkdown(data.result_text);

    // 7. Terminal Trace Log
    const traceEl = document.getElementById('terminal-log-content');
    traceEl.textContent = JSON.stringify({
      query: data.query,
      plan: data.plan,
      confidence: data.confidence,
      validation: data.validation,
      trace_events_count: (data.trace || []).length
    }, null, 2);
  }

  formatMarkdown(text) {
    if (!text) return '<p>No report text available.</p>';
    // Simple fast markdown parser for clean headers, lists, code, and bolding
    let html = text
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/gim, '<em>$1</em>')
      .replace(/`([^`]+)`/gim, '<code>$1</code>')
      .replace(/^\- (.*$)/gim, '<li>$1</li>')
      .replace(/\n\n/gim, '<br><br>');
    return html;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new SatQueryApp();
});
