/**
 * SATQUERY AI — 3D ISOMETRIC LAYER DECOMPOSER
 * Exploded 3D spatial layer stack decomposition:
 *   - Layer 1 (Bottom): T0 Baseline Surface Orthomosaic
 *   - Layer 2 (Middle): Neural Feature Probability Field / Heatmap
 *   - Layer 3 (Top): T1 Resurvey Surface with Vector GeoJSON Polygons
 */

class Layer3DDecomposer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;

    // Layer Separation Distance (pixels)
    this.separation = 85;
    this.targetSeparation = 85;

    // 3D Angles
    this.pitch = 0.58; // Tilt
    this.yaw = -0.45;  // Rotation
    this.targetPitch = 0.58;
    this.targetYaw = -0.45;

    // Images
    this.imgT0 = null;
    this.imgT1 = null;
    this.imgDiff = null;

    // Opacity and visibility
    this.opacity = 0.85;
    this.showT0 = true;
    this.showDiff = true;
    this.showT1 = true;

    // Interaction
    this.isDragging = false;
    this.lastX = 0;
    this.lastY = 0;

    this.bindEvents();
    this.resize();
    this.animate();
  }

  bindEvents() {
    window.addEventListener('resize', () => this.resize());

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;

      this.targetYaw += dx * 0.008;
      this.targetPitch += dy * 0.008;
      this.targetPitch = Math.max(0.15, Math.min(1.2, this.targetPitch));

      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;

    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  setImages(t0Url, t1Url, diffUrl) {
    if (t0Url) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { this.imgT0 = img; };
      img.src = t0Url;
    }
    if (t1Url) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { this.imgT1 = img; };
      img.src = t1Url;
    }
    if (diffUrl) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { this.imgDiff = img; };
      img.src = diffUrl;
    }
  }

  // Draw an isometric transformed plane in 3D
  drawIsometricPlane(img, zOffset, label, color, drawGrid = false) {
    const ctx = this.ctx;
    const cx = this.width / 2;
    const cy = this.height / 2 - zOffset;

    const planeW = Math.min(this.width, this.height) * 0.52;
    const planeH = planeW;

    ctx.save();
    ctx.translate(cx, cy);

    // Apply 3D rotation transform matrix
    const cosY = Math.cos(this.yaw);
    const sinY = Math.sin(this.yaw);
    const cosP = Math.cos(this.pitch);

    // 2D affine transform simulating 3D rotation:
    // x' = x * cosY - y * sinY
    // y' = (x * sinY + y * cosY) * sin(pitch)
    const a = cosY;
    const b = sinY * cosP;
    const c = -sinY;
    const d = cosY * cosP;

    ctx.transform(a, b, c, d, 0, 0);

    const hw = planeW / 2;
    const hh = planeH / 2;

    // Draw Plane Border & Glow
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.strokeRect(-hw, -hh, planeW, planeH);
    ctx.shadowBlur = 0;

    // Fill Image
    if (img && img.complete) {
      ctx.globalAlpha = this.opacity;
      ctx.drawImage(img, -hw, -hh, planeW, planeH);
    } else {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
      ctx.fillRect(-hw, -hh, planeW, planeH);
    }

    // Grid wireframe overlay
    if (drawGrid) {
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
      ctx.lineWidth = 1;
      const step = planeW / 8;
      for (let x = -hw; x <= hw; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, -hh);
        ctx.lineTo(x, hh);
        ctx.stroke();
      }
      for (let y = -hh; y <= hh; y += step) {
        ctx.beginPath();
        ctx.moveTo(-hw, y);
        ctx.lineTo(hw, y);
        ctx.stroke();
      }
    }

    // Border corner brackets
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    const cl = 14;
    // Top-Left
    ctx.beginPath(); ctx.moveTo(-hw, -hh + cl); ctx.lineTo(-hw, -hh); ctx.lineTo(-hw + cl, -hh); ctx.stroke();
    // Top-Right
    ctx.beginPath(); ctx.moveTo(hw - cl, -hh); ctx.lineTo(hw, -hh); ctx.lineTo(hw, -hh + cl); ctx.stroke();
    // Bottom-Right
    ctx.beginPath(); ctx.moveTo(hw, hh - cl); ctx.lineTo(hw, hh); ctx.lineTo(hw - cl, hh); ctx.stroke();
    // Bottom-Left
    ctx.beginPath(); ctx.moveTo(-hw + cl, hh); ctx.lineTo(-hw, hh); ctx.lineTo(-hw, hh - cl); ctx.stroke();

    ctx.restore();

    // Draw Monospace Label beside plane
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = color;
    ctx.fillText(`[ ${label} ]`, cx - planeW * 0.45, cy + planeH * 0.35);
  }

  drawSeparationGuides() {
    const ctx = this.ctx;
    const cx = this.width / 2;
    const cy = this.height / 2;
    const sep = this.separation;

    // Connecting dashed vertical pillar lines at the four plane corners
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;

    const offset = Math.min(this.width, this.height) * 0.28;
    // Four corner pillars
    const corners = [
      { x: cx - offset, y: cy },
      { x: cx + offset, y: cy },
      { x: cx, y: cy - offset * 0.6 },
      { x: cx, y: cy + offset * 0.6 }
    ];

    for (let c of corners) {
      ctx.beginPath();
      ctx.moveTo(c.x, c.y + sep);
      ctx.lineTo(c.x, c.y - sep);
      ctx.stroke();
    }

    ctx.setLineDash([]);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    this.pitch += (this.targetPitch - this.pitch) * 0.1;
    this.yaw += (this.targetYaw - this.yaw) * 0.1;
    this.separation += (this.targetSeparation - this.separation) * 0.1;

    this.ctx.clearRect(0, 0, this.width, this.height);

    this.drawSeparationGuides();

    // Draw from bottom to top in Z order
    // 1. Layer 0 (Bottom): T0 Baseline
    if (this.showT0) {
      this.drawIsometricPlane(this.imgT0, -this.separation, "L1: T0 BASELINE ORTHOPHOTO", "#38bdf8", true);
    }

    // 2. Layer 1 (Middle): Neural Difference / Probability
    if (this.showDiff) {
      this.drawIsometricPlane(this.imgDiff, 0, "L2: NEURAL DIFFERENCE PROBABILITY", "#ff2a6d", false);
    }

    // 3. Layer 2 (Top): T1 Resurvey
    if (this.showT1) {
      this.drawIsometricPlane(this.imgT1, this.separation, "L3: T1 RESURVEY ORTHOMOSAIC", "#00ff9d", true);
    }
  }
}

window.Layer3DDecomposer = Layer3DDecomposer;
