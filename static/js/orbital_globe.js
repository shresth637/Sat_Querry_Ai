/**
 * SATQUERY AI — 3D ORBITAL GLOBE & SENSOR SIMULATOR
 * High-performance native WebGL/Canvas3D Earth Observation Simulation.
 * Renders textured Earth sphere, latitude/longitude graticules, satellite orbit,
 * nadir scanning laser cone, and target focus tracking.
 */

class OrbitalGlobe {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    // Globe state
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;
    this.radius = Math.min(this.width, this.height) * 0.38;
    
    // Rotation angles (radians)
    this.rotX = 0.35; // Tilt
    this.rotY = -0.6; // Longitude rotation
    this.targetRotX = 0.35;
    this.targetRotY = -0.6;
    
    // Zoom / Camera
    this.zoom = 1.0;
    this.targetZoom = 1.0;
    
    // Interaction
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    
    // Satellite mechanics
    this.satAngle = 0;
    this.satOrbitRadius = this.radius * 1.35;
    this.targetPoint = { lat: 25.2048, lng: 55.2708, label: "DUBAI RECON" };
    
    // Particle stars
    this.stars = [];
    this.initStars(140);
    
    // Landmass contour data (simplified vector polygons for 100% offline procedural rendering)
    this.initLandmasses();
    
    this.bindEvents();
    this.resize();
    this.animate();
  }

  initStars(count) {
    this.stars = [];
    for (let i = 0; i < count; i++) {
      this.stars.push({
        x: Math.random(),
        y: Math.random(),
        size: Math.random() * 1.8 + 0.5,
        alpha: Math.random() * 0.7 + 0.3,
        twinkleSpeed: Math.random() * 0.03 + 0.01
      });
    }
  }

  initLandmasses() {
    // Key continental anchor points [lat, lng]
    this.continents = [
      // Africa & Europe
      [
        [36, -5], [44, -1], [50, 2], [54, 8], [60, 5], [68, 14], [70, 28],
        [60, 30], [45, 36], [38, 24], [32, 34], [30, 32], [12, 44], [10, 51],
        [-4, 40], [-25, 33], [-34, 18], [-20, 12], [4, 9], [6, -10], [15, -17],
        [28, -13], [36, -5]
      ],
      // Asia & Middle East
      [
        [38, 24], [42, 28], [42, 42], [30, 48], [25, 55], [24, 60], [25, 67],
        [15, 74], [8, 77], [13, 80], [22, 90], [10, 99], [1, 104], [12, 109],
        [22, 114], [30, 122], [40, 120], [40, 128], [55, 136], [60, 150],
        [68, 175], [72, 130], [70, 70], [60, 60], [50, 50], [38, 24]
      ],
      // North America
      [
        [70, -160], [60, -140], [50, -125], [38, -122], [30, -115], [20, -105],
        [16, -92], [22, -97], [30, -84], [25, -80], [35, -75], [45, -65],
        [55, -60], [60, -65], [65, -85], [72, -95], [70, -160]
      ],
      // South America
      [
        [10, -75], [5, -52], [-5, -35], [-20, -40], [-35, -55], [-55, -68],
        [-45, -75], [-20, -70], [-5, -80], [8, -77], [10, -75]
      ],
      // Australia
      [
        [-12, 130], [-15, 136], [-12, 142], [-22, 150], [-34, 151], [-38, 145],
        [-35, 117], [-22, 114], [-16, 123], [-12, 130]
      ]
    ];
  }

  bindEvents() {
    window.addEventListener('resize', () => this.resize());

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastMouseX;
      const dy = e.clientY - this.lastMouseY;
      
      this.targetRotY += dx * 0.006;
      this.targetRotX += dy * 0.006;
      
      // Clamp tilt
      this.targetRotX = Math.max(-1.1, Math.min(1.1, this.targetRotX));
      
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.targetZoom += e.deltaY * -0.001;
      this.targetZoom = Math.max(0.75, Math.min(1.8, this.targetZoom));
    }, { passive: false });
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    this.width = rect.width;
    this.height = rect.height;
    
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.floor(this.width * dpr);
    this.canvas.height = Math.floor(this.height * dpr);
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(dpr, dpr);
    
    this.radius = Math.min(this.width, this.height) * 0.38;
    this.satOrbitRadius = this.radius * 1.35;
  }

  focusCoordinates(lat, lng, label = "TARGET LOCK") {
    // Convert target lat/lng to required globe rotations so target faces camera
    // Target is at camera center when:
    this.targetRotX = (lat * Math.PI) / 180;
    this.targetRotY = -((lng * Math.PI) / 180) - Math.PI / 2;
    this.targetPoint = { lat, lng, label };
    this.targetZoom = 1.25;
  }

  // Project spherical coords (lat, lng in rad) to 3D Cartesian, then rotate, then project to 2D
  project(latRad, lngRad, radiusOffset = 0) {
    const r = (this.radius + radiusOffset) * this.zoom;
    
    // Spherical to 3D Cartesian
    let x = r * Math.cos(latRad) * Math.sin(lngRad);
    let y = -r * Math.sin(latRad);
    let z = r * Math.cos(latRad) * Math.cos(lngRad);
    
    // Rotate Y (longitude spin)
    const cosY = Math.cos(this.rotY);
    const sinY = Math.sin(this.rotY);
    const x1 = x * cosY + z * sinY;
    const z1 = -x * sinY + z * cosY;
    
    // Rotate X (latitude tilt)
    const cosX = Math.cos(this.rotX);
    const sinX = Math.sin(this.rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;
    
    // 2D Screen coordinates centered
    const screenX = this.width / 2 + x1;
    const screenY = this.height / 2 + y2;
    
    return { x: screenX, y: screenY, z: z2, visible: z2 > 0 };
  }

  drawStars() {
    const ctx = this.ctx;
    for (let star of this.stars) {
      star.alpha += Math.sin(Date.now() * star.twinkleSpeed) * 0.01;
      const a = Math.max(0.2, Math.min(0.9, star.alpha));
      ctx.fillStyle = `rgba(200, 230, 255, ${a})`;
      ctx.fillRect(star.x * this.width, star.y * this.height, star.size, star.size);
    }
  }

  drawGlobe() {
    const ctx = this.ctx;
    const cx = this.width / 2;
    const cy = this.height / 2;
    const r = this.radius * this.zoom;

    // Atmospheric Glow Halo (Planetary Limb)
    const haloGrad = ctx.createRadialGradient(cx, cy, r * 0.85, cx, cy, r * 1.25);
    haloGrad.addColorStop(0, 'rgba(0, 240, 255, 0.2)');
    haloGrad.addColorStop(0.5, 'rgba(0, 240, 255, 0.08)');
    haloGrad.addColorStop(1, 'rgba(0, 240, 255, 0)');
    
    ctx.fillStyle = haloGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.25, 0, Math.PI * 2);
    ctx.fill();

    // Ocean Sphere Base with Day/Night Shadow Terminator
    const sphereGrad = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r);
    sphereGrad.addColorStop(0, '#0c1d38');
    sphereGrad.addColorStop(0.7, '#060d1a');
    sphereGrad.addColorStop(1, '#02050b');

    ctx.fillStyle = sphereGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Graticules (Latitude & Longitude grid lines)
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.clip(); // Restrict lines within globe horizon

    ctx.lineWidth = 0.75;
    
    // Parallels (Latitudes)
    for (let lat = -75; lat <= 75; lat += 15) {
      const latRad = (lat * Math.PI) / 180;
      ctx.beginPath();
      let started = false;
      
      for (let lng = -180; lng <= 180; lng += 5) {
        const lngRad = (lng * Math.PI) / 180;
        const p = this.project(latRad, lngRad);
        if (p.visible) {
          if (!started) { ctx.moveTo(p.x, p.y); started = true; }
          else { ctx.lineTo(p.x, p.y); }
        } else {
          started = false;
        }
      }
      ctx.strokeStyle = (lat === 0) ? 'rgba(0, 240, 255, 0.4)' : 'rgba(0, 240, 255, 0.08)';
      ctx.stroke();
    }

    // Meridians (Longitudes)
    for (let lng = -180; lng < 180; lng += 20) {
      const lngRad = (lng * Math.PI) / 180;
      ctx.beginPath();
      let started = false;
      
      for (let lat = -90; lat <= 90; lat += 5) {
        const latRad = (lat * Math.PI) / 180;
        const p = this.project(latRad, lngRad);
        if (p.visible) {
          if (!started) { ctx.moveTo(p.x, p.y); started = true; }
          else { ctx.lineTo(p.x, p.y); }
        } else {
          started = false;
        }
      }
      ctx.strokeStyle = (lng === 0) ? 'rgba(0, 240, 255, 0.3)' : 'rgba(0, 240, 255, 0.08)';
      ctx.stroke();
    }

    // Continental Landmasses
    ctx.fillStyle = 'rgba(0, 240, 255, 0.15)';
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.45)';
    ctx.lineWidth = 1;

    for (let poly of this.continents) {
      ctx.beginPath();
      let started = false;
      for (let pt of poly) {
        const latRad = (pt[0] * Math.PI) / 180;
        const lngRad = (pt[1] * Math.PI) / 180;
        const p = this.project(latRad, lngRad);
        if (p.visible) {
          if (!started) { ctx.moveTo(p.x, p.y); started = true; }
          else { ctx.lineTo(p.x, p.y); }
        }
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();
  }

  drawTargetLock() {
    if (!this.targetPoint) return;
    const ctx = this.ctx;
    const latRad = (this.targetPoint.lat * Math.PI) / 180;
    const lngRad = (this.targetPoint.lng * Math.PI) / 180;
    const p = this.project(latRad, lngRad, 2);

    if (p.visible) {
      // Pulsing target beacon
      const t = Date.now() * 0.004;
      const pulseSize = 12 + Math.sin(t) * 4;

      ctx.strokeStyle = '#ff2a6d';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, pulseSize, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#ff2a6d';
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();

      // Target Label
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#fff';
      ctx.fillText(`[ ${this.targetPoint.label} ]`, p.x + 16, p.y - 8);
      ctx.fillStyle = '#ff2a6d';
      ctx.fillText(`${this.targetPoint.lat.toFixed(2)}°N, ${this.targetPoint.lng.toFixed(2)}°E`, p.x + 16, p.y + 6);
    }
  }

  drawSatelliteAndSensorCone() {
    const ctx = this.ctx;
    this.satAngle += 0.008;

    // Inclined Sun-Synchronous Orbit (Polar orbit ~98°)
    const orbitRadius = this.satOrbitRadius * this.zoom;
    const satLat = Math.sin(this.satAngle) * 1.3;
    const satLng = this.satAngle * 0.3;

    const satPos = this.project(satLat, satLng, orbitRadius - this.radius * this.zoom);

    // Ground track point directly below satellite
    const groundPos = this.project(satLat, satLng, 0);

    if (satPos.visible || groundPos.visible) {
      // Volumetric Nadir Sensor Beam Cone
      if (groundPos.visible) {
        const beamGrad = ctx.createRadialGradient(groundPos.x, groundPos.y, 4, groundPos.x, groundPos.y, 40);
        beamGrad.addColorStop(0, 'rgba(0, 240, 255, 0.45)');
        beamGrad.addColorStop(0.6, 'rgba(0, 240, 255, 0.1)');
        beamGrad.addColorStop(1, 'rgba(0, 240, 255, 0)');

        ctx.fillStyle = beamGrad;
        ctx.beginPath();
        ctx.arc(groundPos.x, groundPos.y, 36, 0, Math.PI * 2);
        ctx.fill();

        // Laser beam line from satellite to ground
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(satPos.x, satPos.y);
        ctx.lineTo(groundPos.x, groundPos.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Satellite body
      ctx.fillStyle = '#00f0ff';
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.rect(satPos.x - 4, satPos.y - 4, 8, 8);
      ctx.fill();

      // Solar Panels
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(satPos.x - 14, satPos.y);
      ctx.lineTo(satPos.x - 4, satPos.y);
      ctx.moveTo(satPos.x + 4, satPos.y);
      ctx.lineTo(satPos.x + 14, satPos.y);
      ctx.stroke();

      ctx.shadowBlur = 0; // Reset
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (!this.canvas || this.width <= 10 || this.height <= 10) {
      this.resize();
      return;
    }

    // Smooth spherical damping to target rotations
    this.rotX += (this.targetRotX - this.rotX) * 0.05;
    this.rotY += (this.targetRotY - this.rotY) * 0.05;
    this.zoom += (this.targetZoom - this.zoom) * 0.08;

    // Slow ambient rotation if not dragging
    if (!this.isDragging) {
      this.targetRotY += 0.0008;
    }

    this.ctx.clearRect(0, 0, this.width, this.height);

    this.drawStars();
    this.drawGlobe();
    this.drawTargetLock();
    this.drawSatelliteAndSensorCone();
  }
}

window.OrbitalGlobe = OrbitalGlobe;
