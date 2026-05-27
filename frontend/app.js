/* ── AI Studio Frontend App ──────────────────────────────────────────────── */
const API = '';  // Same origin

// --- Authentication ---
const urlParams = new URLSearchParams(window.location.search);
const urlToken = urlParams.get('token');
if (urlToken) {
    localStorage.setItem('jwt_token', urlToken);
    window.history.replaceState({}, document.title, window.location.pathname);
}

function updateAuthUI() {
    const token = localStorage.getItem('jwt_token');
    const btnGoogle = document.getElementById('btn-login-google');
    const btnFacebook = document.getElementById('btn-login-facebook');
    const userInfo = document.getElementById('user-info');
    
    if (token) {
        if(btnGoogle) btnGoogle.style.display = 'none';
        if(btnFacebook) btnFacebook.style.display = 'none';
        if(userInfo) userInfo.style.display = 'flex';
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            document.getElementById('user-name').textContent = payload.name || payload.sub || 'User';
        } catch(e) {
            document.getElementById('user-name').textContent = 'Logged In';
        }
    } else {
        if(btnGoogle) btnGoogle.style.display = 'flex';
        if(btnFacebook) btnFacebook.style.display = 'flex';
        if(userInfo) userInfo.style.display = 'none';
    }
}

window.logout = function() {
    localStorage.removeItem('jwt_token');
    updateAuthUI();
};

document.addEventListener('DOMContentLoaded', updateAuthUI);
// ----------------------

/* ── State ──────────────────────────────────────────────────────────────── */
const state = {
  currentTab: 'generate',
  activeJobId: null,
  pollTimer: null,
  width: 512,
  height: 512,
  steps: 20,
  cfg: 7.5,
  seed: -1,
  numImages: 1,

  // LoRA
  selectedLoras: {},   // { "lora_name.safetensors": 0.8 }
  availableLoras: [],

  // CLIP
  clipSkip: 2,

  // Performance & Memory
  performanceMode: 'fast',
  vramMode: 'balanced',
  vramLimitGb: 0,
  ramLimitGb: 0,

  // NSFW lock state. True only after backend password unlock.
  nsfwMode: false,
};


/* ── Random prompts ──────────────────────────────────────────────────────── */
const RANDOM_PROMPTS = [
  "nico robin from one piece, reading a book in a sunlit library, wearing her post-timeskip outfit, beautiful black hair, highly detailed, cinematic lighting, 4k",
  "masterpiece, best quality, nico robin, one piece, using hana hana no mi powers, cherry blossoms flying, dynamic pose, vivid colors, anime style",
  "nico robin, archaeologist, exploring ancient ruins, glowing mysterious poneglyph, dramatic lighting, highly detailed, digital art",
  "portrait of nico robin, one piece, wearing sunglasses and a stylish summer hat, tropical background, bright sunlight, beautiful eyes, 8k, photorealistic",
  "nico robin, one piece, in a stunning evening dress, elegant, holding a wine glass, moonlight, sophisticated atmosphere, masterpiece, detailed face",
  "nico robin, one piece, action scene, giant arms sprouting from the ground, intense expression, epic composition, high quality anime art",
  "nico robin, one piece, wearing her pre-timeskip cowboy hat and purple corset, confident smile, sunny day, detailed character design",
  "1girl, nico robin, one piece, sitting on the deck of the thousand sunny, drinking coffee, peaceful morning, highly detailed, 4k",
  "masterpiece, nico robin, one piece, elegant mature woman, long black hair, piercing blue eyes, realistic rendering, intricate details, artstation",
  "nico robin, one piece, studying an ancient map, candle light, cozy room, shadows, cinematic depth of field, high quality"
];


/* ── Canvas background animation ──────────────────────────────────────────── */
function initCanvas() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let particles = [];
  let animFrame;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = [];
    const count = Math.floor((canvas.width * canvas.height) / 18000);

    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.5 + 0.1,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 120) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(124,92,252,${0.12 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(124,92,252,${p.alpha})`;
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
    });

    animFrame = requestAnimationFrame(draw);
  }

  resize();
  createParticles();
  draw();

  window.addEventListener('resize', () => {
    resize();
    createParticles();
  });
}


/* ── Tab navigation ──────────────────────────────────────────────────────── */
function initTabs() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
    });
  });

  // Sidebar inner tabs
  const btnParams = document.getElementById('btn-sidebar-params');
  const btnSettings = document.getElementById('btn-sidebar-settings');
  const contentParams = document.getElementById('sidebar-params-content');
  const contentSettings = document.getElementById('sidebar-settings-content');

  if (btnParams && btnSettings) {
    btnParams.addEventListener('click', () => {
      btnParams.classList.add('active');
      btnParams.style.background = 'rgba(124,92,252,0.15)';
      btnParams.style.color = 'var(--primary-2)';
      btnParams.style.fontWeight = 'bold';

      btnSettings.classList.remove('active');
      btnSettings.style.background = 'transparent';
      btnSettings.style.color = 'var(--text-2)';
      btnSettings.style.fontWeight = 'normal';

      contentParams.style.display = 'block';
      contentSettings.style.display = 'none';
    });

    btnSettings.addEventListener('click', () => {
      btnSettings.classList.add('active');
      btnSettings.style.background = 'rgba(124,92,252,0.15)';
      btnSettings.style.color = 'var(--primary-2)';
      btnSettings.style.fontWeight = 'bold';

      btnParams.classList.remove('active');
      btnParams.style.background = 'transparent';
      btnParams.style.color = 'var(--text-2)';
      btnParams.style.fontWeight = 'normal';

      contentSettings.style.display = 'block';
      contentParams.style.display = 'none';
    });
  }
}

function switchTab(tab) {
  state.currentTab = tab;

  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });

  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `tab-${tab}`);
  });

  if (tab === 'gallery') loadGallery();
}


/* ── Health check ────────────────────────────────────────────────────────── */
async function checkHealth() {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');

  dot.className = 'status-dot loading';
  text.textContent = 'Connecting...';

  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();

    if (data.model_loaded) {
      dot.className = 'status-dot online';
      text.textContent = `Online · ${data.model_info?.name || 'Model loaded'}`;
    } else {
      dot.className = 'status-dot error';
      text.textContent = 'Model not loaded';
    }
  } catch {
    dot.className = 'status-dot error';
    text.textContent = 'Server offline';
  }
}


/* ── Sliders & Parameters ────────────────────────────────────────────────── */
function initControls() {
  // Steps
  const stepsSlider = document.getElementById('steps-slider');
  const stepsVal = document.getElementById('steps-value');

  stepsSlider.addEventListener('input', () => {
    state.steps = +stepsSlider.value;
    stepsVal.textContent = state.steps;
  });

  // CFG
  const cfgSlider = document.getElementById('cfg-slider');
  const cfgVal = document.getElementById('cfg-value');

  cfgSlider.addEventListener('input', () => {
    state.cfg = (+cfgSlider.value / 10);
    cfgVal.textContent = state.cfg.toFixed(1);
  });

  // Seed
  const seedInput = document.getElementById('seed-input');
  const seedDisplay = document.getElementById('seed-display');

  seedInput.addEventListener('input', () => {
    state.seed = +seedInput.value;
    seedDisplay.textContent = state.seed === -1 ? 'Random' : state.seed;
  });

  // Num images
  const numSlider = document.getElementById('num-images-slider');
  const numVal = document.getElementById('num-images-value');

  numSlider.addEventListener('input', () => {
    state.numImages = +numSlider.value;
    numVal.textContent = state.numImages;
  });

  // CLIP Skip
  const clipSlider = document.getElementById('clip-skip-slider');
  const clipVal = document.getElementById('clip-skip-value');

  clipSlider.addEventListener('input', () => {
    state.clipSkip = +clipSlider.value;
    clipVal.textContent = state.clipSkip;
  });

  // Resolution presets
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      state.width = +btn.dataset.w;
      state.height = +btn.dataset.h;
    });
  });

  // Random prompt
  document.getElementById('btn-random-prompt').addEventListener('click', () => {
    const prompt = RANDOM_PROMPTS[Math.floor(Math.random() * RANDOM_PROMPTS.length)];
    document.getElementById('prompt-input').value = prompt;
  });

  // Enhance prompt
  document.getElementById('btn-enhance-prompt').addEventListener('click', () => {
    const input = document.getElementById('prompt-input');
    const enhancers = ', ultra detailed, 4k, masterpiece, best quality, sharp focus, cinematic lighting, trending on artstation';

    if (input.value && !input.value.includes('ultra detailed')) {
      input.value += enhancers;
      showToast('✨ Prompt enhanced!', 'success');
    }
  });

  // Sample negative prompt
  const btnSampleNegative = document.getElementById('btn-sample-negative');
  if (btnSampleNegative) {
    btnSampleNegative.addEventListener('click', () => {
      const input = document.getElementById('negative-input');
      if (state.nsfwMode) {
        input.value = "censored, mosaic, bar censor, blurry, ugly, deformed, bad anatomy, worst quality, low quality, text, watermark, signature, extra digit, missing limbs";
        showToast('🎲 Đã điền mẫu Negative NSFW', 'info');
      } else {
        input.value = "nsfw, nude, explicit, nipple, porn, blurry, ugly, deformed, bad anatomy, worst quality, low quality, text, watermark, signature, extra digit, missing limbs";
        showToast('🎲 Đã điền mẫu Negative SFW', 'info');
      }
    });
  }

  // Random seed button
  document.getElementById('btn-random-seed').addEventListener('click', () => {
    const newSeed = Math.floor(Math.random() * 4294967295);

    document.getElementById('seed-input').value = newSeed;
    state.seed = newSeed;
    document.getElementById('seed-display').textContent = newSeed;
  });
}


/* ── Performance & Memory Controls ────────────────────────────────────── */
function initPerfControls() {
  // Performance mode
  const perfSelect = document.getElementById('perf-mode-select');
  const perfLabel  = document.getElementById('perf-mode-label');
  const perfLabels = { turbo: 'Turbo', fast: 'Fast', balanced: 'Balanced', quality: 'Quality' };

  perfSelect.addEventListener('change', () => {
    state.performanceMode = perfSelect.value;
    perfLabel.textContent = perfLabels[perfSelect.value] || perfSelect.value;
  });

  // VRAM mode
  const vramSelect = document.getElementById('vram-mode-select');
  const vramLabel  = document.getElementById('vram-mode-label');
  const vramLabels = { max_speed: 'Max Speed', balanced: 'Balanced', low_vram: 'Low VRAM', ultra_low_vram: 'Ultra Low' };

  vramSelect.addEventListener('change', () => {
    state.vramMode = vramSelect.value;
    vramLabel.textContent = vramLabels[vramSelect.value] || vramSelect.value;
  });

  // VRAM Limit slider
  const vramLimitSlider = document.getElementById('vram-limit-slider');
  const vramLimitVal    = document.getElementById('vram-limit-value');

  vramLimitSlider.addEventListener('input', () => {
    const v = +vramLimitSlider.value;
    state.vramLimitGb = v;
    vramLimitVal.textContent = v === 0 ? 'Không giới hạn' : `${v} GB`;
  });

  // RAM Limit slider
  const ramLimitSlider = document.getElementById('ram-limit-slider');
  const ramLimitVal    = document.getElementById('ram-limit-value');

  ramLimitSlider.addEventListener('input', () => {
    const v = +ramLimitSlider.value;
    state.ramLimitGb = v;
    ramLimitVal.textContent = v === 0 ? 'Không giới hạn' : `Cần tối thiểu ${v} GB trống`;
  });
}


/* ── System Info ──────────────────────────────────────────────────────────── */
async function loadSystemInfo() {
  try {
    const res  = await fetch(`${API}/api/system-info`);
    const info = await res.json();

    const gpuEl  = document.getElementById('sys-gpu');
    const vramEl = document.getElementById('sys-vram');
    const ramEl  = document.getElementById('sys-ram');

    if (!gpuEl) return;

    gpuEl.textContent  = info.gpu_name ? `GPU: ${info.gpu_name}` : 'GPU: CPU only';
    vramEl.textContent = info.vram && info.vram.total_gb
      ? `VRAM: ${info.vram.used_gb}/${info.vram.total_gb} GB`
      : 'VRAM: N/A';
    ramEl.textContent  = info.ram && info.ram.available_gb
      ? `RAM: ${info.ram.available_gb} GB free`
      : 'RAM: N/A';

    // Auto-set VRAM slider max to match actual GPU
    if (info.vram && info.vram.total_gb) {
      const slider = document.getElementById('vram-limit-slider');
      if (slider) slider.max = Math.ceil(info.vram.total_gb);
    }
  } catch {
    // silently ignore
  }
}


/* ── Generate ────────────────────────────────────────────────────────────── */
function initGenerate() {
  document.getElementById('btn-generate').addEventListener('click', startGeneration);
}

async function startGeneration() {
  const prompt = document.getElementById('prompt-input').value.trim();

  if (!prompt) {
    showToast('⚠️ Please enter a prompt', 'error');
    return;
  }

  const btn = document.getElementById('btn-generate');
  const btnText = document.getElementById('btn-generate-text');

  // Auto random seed every time
  const newSeed = Math.floor(Math.random() * 4294967295);
  state.seed = newSeed;
  document.getElementById('seed-input').value = newSeed;
  document.getElementById('seed-display').textContent = newSeed;

  btn.disabled = true;
  btnText.textContent = 'Generating...';

  const statusEl = document.getElementById('generation-status');
  const gridEl = document.getElementById('image-grid');
  const emptyEl = document.getElementById('empty-state');

  statusEl.style.display = 'block';
  setProgress(0);

  if (emptyEl) emptyEl.remove();

  for (let i = 0; i < state.numImages; i++) {
    const shimmer = document.createElement('div');
    shimmer.className = 'shimmer image-card';
    shimmer.id = `shimmer-${i}`;
    gridEl.prepend(shimmer);
  }

  try {
    const payload = {
      prompt,
      negative_prompt: document.getElementById('negative-input').value.trim(),
      width: state.width,
      height: state.height,
      steps: state.steps,
      guidance_scale: state.cfg,
      seed: state.seed,
      num_images: state.numImages,
      lora_names: Object.keys(state.selectedLoras),
      lora_scales: Object.values(state.selectedLoras),
      clip_skip: state.clipSkip,
      // Performance & memory controls
      performance_mode: state.performanceMode,
      vram_mode: state.vramMode,
      vram_limit_gb: state.vramLimitGb,
      ram_limit_gb: state.ramLimitGb,
    };

    const headers = { 'Content-Type': 'application/json' };
    const token = localStorage.getItem('jwt_token');
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API}/api/generate`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Generation failed');
    }

    const data = await res.json();
    state.activeJobId = data.job_id;
    pollJob(data.job_id);
  } catch (err) {
    showToast(`❌ ${err.message}`, 'error');
    resetGenerateButton();
    statusEl.style.display = 'none';
    document.querySelectorAll('[id^="shimmer-"]').forEach(el => el.remove());
  }
}

function pollJob(jobId) {
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API}/api/jobs/${jobId}`);
      const job = await res.json();

      setProgress(job.progress);

      document.getElementById('progress-text').textContent =
        `Generating... ${job.progress}%`;

      if (job.status === 'done') {
        clearInterval(state.pollTimer);

        document.getElementById('generation-status').style.display = 'none';
        document.querySelectorAll('[id^="shimmer-"]').forEach(el => el.remove());

        const grid = document.getElementById('image-grid');

        job.images.reverse().forEach(url => {
          const card = createImageCard(url, job.metadata);
          grid.prepend(card);
        });

        resetGenerateButton();
        document.getElementById('output-actions').style.display = 'flex';
        showToast(`✅ Generated ${job.images.length} image(s) in ${job.metadata?.elapsed_seconds}s`, 'success');
      }

      if (job.status === 'error') {
        clearInterval(state.pollTimer);

        document.getElementById('generation-status').style.display = 'none';
        document.querySelectorAll('[id^="shimmer-"]').forEach(el => el.remove());

        resetGenerateButton();
        showToast(`❌ ${job.error}`, 'error');
      }
    } catch {
      clearInterval(state.pollTimer);
      resetGenerateButton();
    }
  }, 800);
}

function setProgress(pct) {
  document.getElementById('progress-fill').style.width = `${pct}%`;
}

function resetGenerateButton() {
  const btn = document.getElementById('btn-generate');
  const btnText = document.getElementById('btn-generate-text');

  btn.disabled = false;
  btnText.textContent = 'Generate Image';
}


/* ── Image cards ─────────────────────────────────────────────────────────── */
function createImageCard(url, meta = {}) {
  const card = document.createElement('div');
  card.className = 'image-card';

  const img = document.createElement('img');
  img.src = url;
  img.alt = 'Generated image';
  img.loading = 'lazy';

  const overlay = document.createElement('div');
  overlay.className = 'image-card-overlay';

  const downloadBtn = document.createElement('button');
  downloadBtn.textContent = '⬇ Save';

  downloadBtn.addEventListener('click', e => {
    e.stopPropagation();
    downloadImage(url);
  });

  const expandBtn = document.createElement('button');
  expandBtn.textContent = '⤢ View';

  expandBtn.addEventListener('click', e => {
    e.stopPropagation();
    openLightbox(url, meta);
  });

  const deleteBtn = document.createElement('button');
  deleteBtn.textContent = '🗑️ Delete';
  deleteBtn.style.background = 'rgba(239, 68, 68, 0.8)';
  
  deleteBtn.addEventListener('click', async e => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this image?')) {
      const filename = url.split('/').pop();
      try {
        const res = await fetch(`${API}/api/gallery/${filename}`, { method: 'DELETE' });
        if (res.ok) {
          card.remove();
          showToast('Image deleted', 'success');
        } else {
          showToast('Failed to delete', 'error');
        }
      } catch {
        showToast('Error deleting image', 'error');
      }
    }
  });

  overlay.append(downloadBtn, expandBtn, deleteBtn);
  card.append(img, overlay);
  card.addEventListener('click', () => openLightbox(url, meta));

  return card;
}


/* ── Download ────────────────────────────────────────────────────────────── */
function downloadImage(url) {
  const a = document.createElement('a');
  a.href = url;
  a.download = `ai-image-${Date.now()}.png`;
  a.click();
}


/* ── Download all ────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-download-all').addEventListener('click', () => {
    document.querySelectorAll('#image-grid .image-card img').forEach((img, i) => {
      setTimeout(() => downloadImage(img.src), i * 300);
    });
  });
});


/* ── Lightbox ────────────────────────────────────────────────────────────── */
function openLightbox(url, meta = {}) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = url;

  const metaEl = document.getElementById('lightbox-meta');

  if (meta && Object.keys(meta).length > 0) {
    metaEl.innerHTML = `Generated in ${meta.elapsed_seconds}s`;
  } else {
    metaEl.innerHTML = '';
  }

  lb.classList.add('open');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}


/* ── Gallery ─────────────────────────────────────────────────────────────── */
async function loadGallery() {
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = '<div class="shimmer" style="height:200px;grid-column:1/-1"></div>';

  try {
    const res = await fetch(`${API}/api/gallery`);
    const images = await res.json();

    grid.innerHTML = '';

    if (images.length === 0) {
      grid.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🖼️</div>
          <p class="empty-title">No images yet</p>
          <p class="empty-sub">Generated images will appear here</p>
        </div>`;
      return;
    }

    images.forEach(img => {
      const card = createImageCard(img.url, img.metadata);
      grid.append(card);
    });
  } catch {
    grid.innerHTML = '<p class="muted">Failed to load gallery.</p>';
  }
}


/* ── Model display helpers ───────────────────────────────────────────────── */
function nsfwBadgeHtml(item) {
  return item?.nsfw
    ? '<span class="lora-nsfw-badge">NSFW</span>'
    : '<span class="lora-nsfw-badge" style="background:rgba(45,212,191,.16);color:#5eead4;border-color:rgba(45,212,191,.35)">SFW</span>';
}

function renderModelItems(items, emptyText) {
  if (!items || items.length === 0) {
    return `<p class="muted">${emptyText}</p>`;
  }

  return items.map(m => `
    <div class="model-item ${m.nsfw ? 'model-item-nsfw' : 'model-item-sfw'}">
      <div style="min-width: 0; flex: 1;">
        <div class="model-item-name"><span class="truncate" title="${m.name}">${m.name}</span> ${nsfwBadgeHtml(m)}</div>
        <div class="model-item-meta">${m.type} · ${m.size_mb} MB · ${m.source || 'models'}</div>
      </div>
    </div>
  `).join('');
}


/* ── Settings ────────────────────────────────────────────────────────────── */
async function loadSettings() {
  // Model info
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();

    const badge = document.getElementById('model-status-badge');

    if (data.model_loaded) {
      badge.textContent = 'Online';
      badge.className = 'badge badge-online';

      document.getElementById('model-name-val').textContent = data.model_info?.name || '—';
      document.getElementById('model-device-val').textContent = data.model_info?.device || '—';
      document.getElementById('model-type-val').textContent = data.model_info?.type || '—';

      const loras = data.model_info?.active_loras || [];
      document.getElementById('model-lora-val').textContent = loras.length ? loras.join(', ') : 'None';
    } else {
      badge.textContent = 'Not loaded';
      badge.className = 'badge badge-offline';
    }
  } catch {
    document.getElementById('model-status-badge').textContent = 'Error';
    document.getElementById('model-status-badge').className = 'badge badge-offline';
  }

  // Models list
  try {
    const res = await fetch(`${API}/api/models`, { credentials: 'include' });
    const models = await res.json();
    const list = document.getElementById('models-list');

    list.innerHTML = renderModelItems(
      models,
      state.nsfwMode
        ? 'No models found in /models directory.'
        : 'No SFW models found. Nhập mật khẩu NSFW để hiện model bị khóa.'
    );
  } catch {
    document.getElementById('models-list').innerHTML = '<p class="muted">Could not load models.</p>';
  }

  // Checkpoints list
  try {
    const res = await fetch(`${API}/api/checkpoints`, { credentials: 'include' });
    const ckpts = await res.json();
    const list = document.getElementById('checkpoints-list');

    list.innerHTML = renderModelItems(
      ckpts,
      state.nsfwMode
        ? 'No checkpoints found in /checkpoints directory.'
        : 'No SFW checkpoints found. Nhập mật khẩu NSFW để hiện checkpoint bị khóa.'
    );
  } catch {
    document.getElementById('checkpoints-list').innerHTML = '<p class="muted">Could not load checkpoints.</p>';
  }

  // Switch model dropdown
  await initSwitchModel();
}


/* ── Toast ───────────────────────────────────────────────────────────────── */
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');

  toast.className = `toast ${type}`;
  toast.textContent = message;

  container.append(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';

    setTimeout(() => toast.remove(), 300);
  }, 3500);
}


/* ── Preferences ─────────────────────────────────────────────────────────── */
function initPreferences() {
  const saved = JSON.parse(localStorage.getItem('ai-studio-prefs') || '{}');

  if (saved.steps) {
    document.getElementById('steps-slider').value = saved.steps;
    document.getElementById('steps-value').textContent = saved.steps;
    state.steps = saved.steps;
  }

  if (saved.cfg) {
    document.getElementById('cfg-slider').value = saved.cfg * 10;
    document.getElementById('cfg-value').textContent = saved.cfg;
    state.cfg = saved.cfg;
  }

  document.getElementById('btn-save-prefs').addEventListener('click', () => {
    const prefs = {
      steps: +document.getElementById('pref-steps').value,
      cfg: +document.getElementById('pref-cfg').value,
    };

    localStorage.setItem('ai-studio-prefs', JSON.stringify(prefs));

    document.getElementById('steps-slider').value = prefs.steps;
    document.getElementById('steps-value').textContent = prefs.steps;
    state.steps = prefs.steps;

    document.getElementById('cfg-slider').value = prefs.cfg * 10;
    document.getElementById('cfg-value').textContent = prefs.cfg.toFixed(1);
    state.cfg = prefs.cfg;

    showToast('✅ Preferences saved!', 'success');
  });
}


/* ── NSFW Password Gate ─────────────────────────────────────────────────── */
async function initNsfwToggle() {
  await refreshNsfwStatus();

  const btn = document.getElementById('nsfw-toggle');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    if (state.nsfwMode) {
      await lockNsfw();
      return;
    }

    const password = window.prompt('Nhập mật khẩu để mở khóa NSFW models/checkpoints:');

    if (!password) return;

    await unlockNsfw(password);
  });
}

async function refreshNsfwStatus() {
  try {
    const res = await fetch(`${API}/api/auth/nsfw-status`, {
      credentials: 'include',
    });

    const data = await res.json();
    state.nsfwMode = !!data.unlocked;
  } catch {
    state.nsfwMode = false;
  }

  updateNsfwUI();
}

async function unlockNsfw(password) {
  try {
    const res = await fetch(`${API}/api/auth/nsfw`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({
        password: password,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Sai mật khẩu NSFW');
    }

    state.nsfwMode = true;
    updateNsfwUI();

    await initLoraPicker();
    await loadSettings();

    showToast('🔞 Đã mở khóa NSFW models/checkpoints', 'success');
  } catch (err) {
    state.nsfwMode = false;
    updateNsfwUI();
    showToast(`❌ ${err.message}`, 'error');
  }
}

async function lockNsfw() {
  try {
    await fetch(`${API}/api/auth/nsfw-logout`, {
      method: 'POST',
      credentials: 'include',
    });
  } catch {
    // Ignore logout network errors
  }

  state.nsfwMode = false;

  const nsfwLoras = state.availableLoras
    .filter(lora => lora.nsfw)
    .map(lora => lora.name);

  nsfwLoras.forEach(name => {
    delete state.selectedLoras[name];
  });

  updateNsfwUI();
  updateLoraActiveBar();

  await initLoraPicker();
  await loadSettings();

  showToast('🔒 Đã khóa NSFW, chỉ hiện SFW models/checkpoints', 'info');
}

function updateNsfwUI() {
  const btn = document.getElementById('nsfw-toggle');
  const icon = document.getElementById('nsfw-icon');
  const label = document.getElementById('nsfw-label');
  const negativeInput = document.getElementById('negative-input');

  if (!btn || !icon || !label) return;

  if (state.nsfwMode) {
    btn.classList.add('nsfw-on');
    icon.textContent = '🔞';
    label.textContent = 'NSFW Unlocked';
    if (negativeInput) negativeInput.placeholder = "censored, blurry, ugly, deformed, bad anatomy, text...";
  } else {
    btn.classList.remove('nsfw-on');
    icon.textContent = '🔒';
    label.textContent = 'SFW Only';
    if (negativeInput) negativeInput.placeholder = "nsfw, nude, explicit, blurry, ugly, deformed, bad anatomy, text...";
  }
}


/* ── LoRA Picker ─────────────────────────────────────────────────────────── */
async function initLoraPicker() {
  try {
    const res = await fetch(`${API}/api/loras`, {
      credentials: 'include',
    });

    state.availableLoras = await res.json();
  } catch {
    state.availableLoras = [];
  }

  renderLoraPicker();
}

function renderLoraPicker() {
  const area = document.getElementById('lora-picker-area');
  const emptyMsg = document.getElementById('lora-empty-msg');
  const countBadge = document.getElementById('lora-count-badge');

  if (state.availableLoras.length === 0) {
    area.innerHTML = `
      <p class="muted">No LoRA files found in <code style="color:var(--primary-2);background:rgba(124,92,252,.1);padding:1px 5px;border-radius:3px">/loras</code></p>
      <p class="hint-text" style="margin-top:6px">Download .safetensors LoRA files from <a href="https://civitai.com" target="_blank" style="color:var(--accent)">CivitAI</a> and place them in the /loras folder.</p>`;

    countBadge.style.display = 'none';
    return;
  }

  countBadge.textContent = state.availableLoras.length;
  countBadge.style.display = 'inline';

  const list = document.createElement('div');
  list.className = 'lora-list';

  state.availableLoras.forEach(lora => {
    const isSelected = !!state.selectedLoras[lora.name];
    const currentScale = state.selectedLoras[lora.name] || lora.recommended_scale || 0.8;
    const isNsfw = !!lora.nsfw;
    const hiddenByMode = isNsfw && !state.nsfwMode;

    const item = document.createElement('div');
    item.className = `lora-item${isSelected ? ' selected' : ''}${hiddenByMode ? ' lora-nsfw-hidden' : ''}`;
    item.id = `lora-item-${lora.stem}`;

    const triggerHtml = lora.trigger_words?.length
      ? `<div class="lora-trigger">trigger: ${lora.trigger_words.join(', ')}</div>`
      : '';

    const nsfwBadge = isNsfw
      ? `<span class="lora-nsfw-badge">NSFW</span>`
      : '';

    item.innerHTML = `
      <label class="lora-item-header">
        <input type="checkbox" class="lora-checkbox" id="lora-cb-${lora.stem}" ${isSelected ? 'checked' : ''} />
        <div class="lora-item-info">
          <div class="lora-name">${lora.stem}${nsfwBadge}</div>
          <div class="lora-meta">${lora.size_mb} MB · ${lora.base_model}${lora.description ? ' · ' + lora.description : ''}</div>
          ${triggerHtml}
        </div>
      </label>
      <div class="lora-scale-row" id="lora-scale-row-${lora.stem}" style="${isSelected ? '' : 'display:none'}">
        <label>Scale</label>
        <input type="range" class="slider" id="lora-scale-${lora.stem}" min="0" max="20" value="${Math.round(currentScale * 10)}" />
        <span class="lora-scale-val" id="lora-scale-val-${lora.stem}">${currentScale.toFixed(1)}</span>
      </div>`;

    const cb = item.querySelector(`#lora-cb-${lora.stem}`);

    cb.addEventListener('change', () => {
      if (cb.checked) {
        state.selectedLoras[lora.name] = currentScale;
        item.classList.add('selected');
        item.querySelector(`#lora-scale-row-${lora.stem}`).style.display = 'flex';
      } else {
        delete state.selectedLoras[lora.name];
        item.classList.remove('selected');
        item.querySelector(`#lora-scale-row-${lora.stem}`).style.display = 'none';
      }

      updateLoraActiveBar();
    });

    const scaleSlider = item.querySelector(`#lora-scale-${lora.stem}`);
    const scaleVal = item.querySelector(`#lora-scale-val-${lora.stem}`);

    scaleSlider.addEventListener('input', () => {
      const val = +(scaleSlider.value / 10).toFixed(1);
      scaleVal.textContent = val.toFixed(1);

      if (state.selectedLoras[lora.name] !== undefined) {
        state.selectedLoras[lora.name] = val;
      }
    });

    list.append(item);
  });

  area.innerHTML = '';
  area.append(list);

  updateLoraActiveBar();
}

function updateLoraActiveBar() {
  const bar = document.getElementById('lora-active-bar');
  const tagsEl = document.getElementById('lora-active-names');
  const activeNames = Object.keys(state.selectedLoras);

  if (activeNames.length === 0) {
    bar.style.display = 'none';
    return;
  }

  bar.style.display = 'flex';

  tagsEl.innerHTML = activeNames.map(name => {
    const stem = name.replace('.safetensors', '');
    const scale = state.selectedLoras[name];

    return `<span class="lora-tag">${stem} ×${scale.toFixed(1)}</span>`;
  }).join('');
}


/* ── Switch Model ────────────────────────────────────────────────────────── */
async function initSwitchModel() {
  try {
    const [modelsRes, ckptRes] = await Promise.all([
      fetch(`${API}/api/models`, { credentials: 'include' }),
      fetch(`${API}/api/checkpoints`, { credentials: 'include' }),
    ]);

    const models = await modelsRes.json();
    const ckpts = await ckptRes.json();
    const all = [...models, ...ckpts];

    const select = document.getElementById('switch-model-select');
    select.innerHTML = '<option value="">— Select model —</option>';

    all.forEach(m => {
      const opt = document.createElement('option');

      opt.value = m.name;
      opt.textContent = `${m.nsfw ? '🔞 ' : '✅ '}${m.name} (${m.size_mb} MB) [${m.source}]`;

      select.append(opt);
    });
  } catch {
    console.warn('Could not load model list for switch');
  }
}


/* ── Boot ────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initCanvas();
  initTabs();
  initControls();
  initPerfControls();
  initGenerate();
  initPreferences();
  initNsfwToggle();
  initLoraPicker();
  loadSystemInfo();
  setInterval(loadSystemInfo, 30000);   // refresh system info every 30s
  loadSettings();

  // Lightbox close
  document.getElementById('lightbox-close').addEventListener('click', closeLightbox);

  document.getElementById('lightbox').addEventListener('click', e => {
    if (e.target.id === 'lightbox') closeLightbox();
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeLightbox();
  });

  // Gallery refresh
  document.getElementById('btn-refresh-gallery').addEventListener('click', loadGallery);

  // Delete all gallery
  const btnDeleteAllGallery = document.getElementById('btn-delete-all-gallery');
  if (btnDeleteAllGallery) {
    btnDeleteAllGallery.addEventListener('click', async () => {
      if (confirm('Are you sure you want to delete ALL generated images? This cannot be undone.')) {
        try {
          const res = await fetch(`${API}/api/gallery`, { method: 'DELETE' });
          if (res.ok) {
            showToast('All images deleted', 'success');
            loadGallery();
          } else {
            showToast('Failed to delete all images', 'error');
          }
        } catch {
          showToast('Error deleting images', 'error');
        }
      }
    });
  }

  // Clear LoRAs
  document.getElementById('btn-clear-loras').addEventListener('click', () => {
    state.selectedLoras = {};
    renderLoraPicker();
    showToast('🧹 LoRAs cleared', 'info');
  });

  // Switch model button
  document.getElementById('btn-switch-model').addEventListener('click', async () => {
    const select = document.getElementById('switch-model-select');
    const modelName = select.value;

    if (!modelName) {
      showToast('⚠️ Select a model first', 'error');
      return;
    }

    const btn = document.getElementById('btn-switch-model');

    btn.textContent = '⏳ Loading...';
    btn.disabled = true;

    try {
      const res = await fetch(`${API}/api/load-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          model_name: modelName,
        }),
      });

      if (!res.ok) {
        throw new Error((await res.json()).detail);
      }

      const data = await res.json();

      showToast(`✅ Loaded: ${data.model_info.name}`, 'success');
      checkHealth();
      loadSettings();
    } catch (err) {
      showToast(`❌ ${err.message}`, 'error');
    } finally {
      btn.textContent = '⚡ Load';
      btn.disabled = false;
    }
  });

  // Download Model
  document.getElementById('btn-download-model').addEventListener('click', async () => {
    const url = document.getElementById('dl-url').value.trim();
    const filename = document.getElementById('dl-filename').value.trim();
    const folder = document.getElementById('dl-folder').value;

    if (!url || !filename) {
      showToast('⚠️ Please enter URL and filename', 'error');
      return;
    }

    const btn = document.getElementById('btn-download-model');

    btn.textContent = '⏳ Starting...';
    btn.disabled = true;

    try {
      const res = await fetch(`${API}/api/models/download?url=${encodeURIComponent(url)}&filename=${encodeURIComponent(filename)}&folder=${folder}`, {
        method: 'POST',
      });

      const data = await res.json();

      showToast(`🚀 ${data.message}`, 'success');

      document.getElementById('dl-url').value = '';
      document.getElementById('dl-filename').value = '';
    } catch {
      showToast('❌ Failed to start download', 'error');
    } finally {
      btn.textContent = '⬇ Download';
      btn.disabled = false;
    }
  });

  // Health check
  checkHealth();
  setInterval(checkHealth, 30000);
});