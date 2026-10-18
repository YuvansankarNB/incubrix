// Scriptify Web Engine Controller (Professional Light UI)

let currentJobId = null;
let currentOutputs = null;
let pollTimer = null;
let cachedManifest = null;
let cachedTimeline = null;
let cachedSrt = null;
let selectedSwapIndex = null;
let captionsEnabled = true;
let captionCues = [];

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initPresets();
  initFormSubmit();
  initModal();
  initInspectTabs();
  initFileUploadUI();
  initCharacterBox();
  initScriptStats();
  initCaptionToggle();
  initVideoTimeUpdate();
});

// Live Script Stats (Word Count, Character Count, Estimated Speech Duration)
function initScriptStats() {
  const textarea = document.getElementById('script-text');
  const wordEl = document.getElementById('stat-words');
  const charEl = document.getElementById('stat-chars');
  const timeEl = document.getElementById('stat-time');
  if (!textarea || !wordEl) return;

  function update() {
    const text = textarea.value.trim();
    const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
    const chars = text.length;
    // Avg narration speed: ~150 words per minute (2.5 words/sec)
    const estSec = Math.round(words / 2.5);
    const estTimeStr = estSec < 60 ? `~${estSec}s speech` : `~${Math.floor(estSec/60)}m ${estSec%60}s speech`;

    wordEl.textContent = `${words} words`;
    charEl.textContent = `${chars} chars`;
    timeEl.textContent = estTimeStr;
  }

  textarea.addEventListener('input', update);
  update();
}

// Character Box — Multi-Selection, Drama Duos, Custom Inputs & Live Summary
function initCharacterBox() {
  const chips = document.querySelectorAll('#char-chips .char-chip');
  const customInput = document.getElementById('character-type-input');
  const summaryBadge = document.getElementById('char-selection-summary');
  const duoBtns = document.querySelectorAll('.char-duo-btn');

  function updateSummary() {
    if (!summaryBadge) return;
    const custom = (customInput?.value || '').trim();
    if (custom) {
      const parts = custom.split(/[,+&]| and /i).map(s => s.trim()).filter(Boolean);
      if (parts.length >= 2) {
        summaryBadge.textContent = `🎭 Drama Mode: ${parts.slice(0, 2).join(' & ')} (${parts.length} Characters)`;
        summaryBadge.classList.add('drama-active');
      } else {
        summaryBadge.textContent = `Custom: ${custom}`;
        summaryBadge.classList.remove('drama-active');
      }
      return;
    }

    const activeChips = Array.from(document.querySelectorAll('#char-chips .char-chip.active'));
    if (activeChips.length === 0) {
      summaryBadge.textContent = "1 Character: Human";
      summaryBadge.classList.remove('drama-active');
    } else if (activeChips.length === 1) {
      const name = activeChips[0].textContent.trim();
      summaryBadge.textContent = `1 Character: ${name}`;
      summaryBadge.classList.remove('drama-active');
    } else {
      const names = activeChips.map(c => c.textContent.trim().split(' ').pop()).join(' & ');
      summaryBadge.textContent = `🎭 Drama Mode: ${names} (${activeChips.length} Characters)`;
      summaryBadge.classList.add('drama-active');
    }
  }

  // Toggle multi-selection on chips
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      // Clear custom text when clicking chips
      if (customInput) customInput.value = '';
      duoBtns.forEach(b => b.classList.remove('active'));

      chip.classList.toggle('active');

      // Ensure at least one chip is active if custom is empty
      const activeCount = document.querySelectorAll('#char-chips .char-chip.active').length;
      if (activeCount === 0) {
        chip.classList.add('active');
      }

      updateSummary();
    });
  });

  // Discussion Duo Quick Presets
  duoBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const duoStr = btn.getAttribute('data-duo'); // e.g. "human,robot"
      const pair = duoStr.split(',');

      duoBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (customInput) customInput.value = '';

      // Activate chips matching pair
      chips.forEach(c => {
        if (pair.includes(c.dataset.char)) {
          c.classList.add('active');
        } else {
          c.classList.remove('active');
        }
      });

      updateSummary();
    });
  });

  // Custom text input
  if (customInput) {
    customInput.addEventListener('input', () => {
      if (customInput.value.trim()) {
        chips.forEach(c => c.classList.remove('active'));
        duoBtns.forEach(b => b.classList.remove('active'));
      } else {
        const humanChip = document.querySelector('#char-chips .char-chip[data-char="human"]');
        if (humanChip) humanChip.classList.add('active');
      }
      updateSummary();
    });
  }

  updateSummary();
}

// 1. Tab Navigation
function initTabs() {
  const segBtns = document.querySelectorAll('.seg-btn');
  segBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      segBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      const target = document.getElementById(btn.getAttribute('data-tab'));
      if (target) target.classList.add('active');
    });
  });
}

// 2. Preset Quick Fill
function initPresets() {
  const selectTab = (tabId) => {
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(c => c.classList.remove('active'));
    const btn = document.querySelector(`.seg-btn[data-tab="${tabId}"]`);
    if (btn) btn.classList.add('active');
    const pane = document.getElementById(tabId);
    if (pane) pane.classList.add('active');
  };

  // Drama Dialogue Preset
  document.getElementById('preset-drama')?.addEventListener('click', () => {
    selectTab('tab-script');
    const scriptEl = document.getElementById('script-text');
    scriptEl.value = 
      "Alex: We need to make a decision on the new launch right now before the deadline passes.\n" +
      "Morgan: I understand the urgency, but rushing without full testing will risk everything we built.\n" +
      "Alex: Look at the competitors, they are moving twice as fast and capturing market share.\n" +
      "Morgan: Quality and customer trust is why our users stay loyal. Let's do a phased rollout instead.\n" +
      "Alex: That is actually a brilliant compromise. Let's execute the phased rollout today.";
    scriptEl.dispatchEvent(new Event('input'));
    document.getElementById('tts-voice').value = "en-US-AriaNeural";

    // Auto-select Human & Robot Discussion Duo
    const duoBtn = document.querySelector('.char-duo-btn[data-duo="human,robot"]');
    if (duoBtn) duoBtn.click();
  });

  document.getElementById('preset-finance')?.addEventListener('click', () => {
    selectTab('tab-script');
    const scriptEl = document.getElementById('script-text');
    scriptEl.value = 
      "In 2025, our corporate sales grew 40% across all markets. When you deposit capital into a high yield savings bank, compound interest works quietly in your favor. Every smart investment begins with patience.";
    scriptEl.dispatchEvent(new Event('input'));
    document.getElementById('tts-voice').value = "en-US-AriaNeural";
  });

  document.getElementById('preset-cloud')?.addEventListener('click', () => {
    selectTab('tab-script');
    const scriptEl = document.getElementById('script-text');
    scriptEl.value = 
      "Modern cloud computing infrastructure powers millions of servers globally. Our distributed clusters guarantee zero downtime, automated failovers, and resilient data backups.";
    scriptEl.dispatchEvent(new Event('input'));
    document.getElementById('tts-voice').value = "en-US-AriaNeural";
  });

  document.getElementById('preset-hindi')?.addEventListener('click', () => {
    selectTab('tab-script');
    const scriptEl = document.getElementById('script-text');
    scriptEl.value = 
      "तकनीक और नवाचार की दुनिया में आपका स्वागत है। हमारा उद्देश्य रचनाकारों को कृत्रिम बुद्धिमत्ता के साथ शक्तिशाली वीडियो बनाने में सहायता करना है।";
    scriptEl.dispatchEvent(new Event('input'));
    document.getElementById('tts-voice').value = "hi-IN-female";
  });

  document.getElementById('preset-tamil')?.addEventListener('click', () => {
    selectTab('tab-script');
    const scriptEl = document.getElementById('script-text');
    scriptEl.value = 
      "செயற்கை நுண்ணறிவு மூலம் காணொளிகளை உருவாக்குவது இப்போது மிக எளிது. உங்கள் படைப்பாற்றலை அடுத்த கட்டத்திற்கு கொண்டு செல்லுங்கள்.";
    scriptEl.dispatchEvent(new Event('input'));
    document.getElementById('tts-voice').value = "ta-IN-female";
  });
}

// File upload box UI helper
function initFileUploadUI() {
  const uploadInput = document.getElementById('audio-upload');
  const label = document.getElementById('upload-filename');
  if (uploadInput && label) {
    uploadInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        label.textContent = e.target.files[0].name;
      }
    });
  }
}

// 3. Form Submission & Job Dispatch
function initFormSubmit() {
  const btn = document.getElementById('btn-generate');
  btn.addEventListener('click', async () => {
    const activeTab = document.querySelector('.seg-btn.active').getAttribute('data-tab');
    const aspectRatios = [];
    if (document.getElementById('ratio-16-9')?.checked) aspectRatios.push("16:9");
    if (document.getElementById('ratio-9-16')?.checked) aspectRatios.push("9:16");
    if (document.getElementById('ratio-1-1')?.checked) aspectRatios.push("1:1");

    if (aspectRatios.length === 0) {
      alert("Please select at least one aspect ratio.");
      return;
    }

    const styleEl = document.getElementById('video-style');
    const style = styleEl ? styleEl.value.trim() : "cinematic-real";

    // Read character type — multi-select chips or custom text input
    const charCustom = (document.getElementById('character-type-input')?.value || '').trim();
    const activeChips = Array.from(document.querySelectorAll('#char-chips .char-chip.active'));
    const selectedChars = activeChips.map(c => c.dataset.char);
    const characterType = charCustom || (selectedChars.length > 0 ? selectedChars.join(', ') : 'human');

    // Generate unique idempotency key
    const idempotencyKey = "req_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);
    setLoadingState(true);

    try {
      let resp;
      if (activeTab === 'tab-audio' && document.getElementById('audio-upload')?.files.length > 0) {
        // Multipart upload
        const file = document.getElementById('audio-upload').files[0];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('aspect_ratios', aspectRatios.join(','));
        formData.append('style', style);
        formData.append('burn_in_captions', true);
        formData.append('character_type', characterType);

        resp = await fetch('/api/v1/broll/jobs/upload', {
          method: 'POST',
          headers: { 'Idempotency-Key': idempotencyKey },
          body: formData
        });
      } else {
        // JSON body
        let payload = {
          aspect_ratios: aspectRatios,
          style: style,
          avoid: [],
          captions: { burn_in: true, srt: true },
          character_type: characterType
        };

        if (activeTab === 'tab-script') {
          payload.script = document.getElementById('script-text').value.trim();
          payload.voice = document.getElementById('tts-voice').value;
          if (!payload.script) {
            alert("Please enter a text script or click a preset.");
            setLoadingState(false);
            return;
          }
        } else if (activeTab === 'tab-audio') {
          payload.audio_url = document.getElementById('audio-url').value.trim();
          if (!payload.audio_url) {
            alert("Please provide an audio URL or upload an audio file.");
            setLoadingState(false);
            return;
          }
        } else if (activeTab === 'tab-scenes') {
          try {
            payload.scenes = JSON.parse(document.getElementById('scenes-json').value);
            payload.voice = document.getElementById('tts-voice').value;
          } catch (e) {
            alert("Invalid JSON format in custom scenes.");
            setLoadingState(false);
            return;
          }
        }

        resp = await fetch('/api/v1/broll/jobs', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Idempotency-Key': idempotencyKey
          },
          body: JSON.stringify(payload)
        });
      }

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error ? data.error.message : "Failed to create job");
      }

      currentJobId = data.job_id;
      startPolling(currentJobId);
    } catch (err) {
      alert("Error submitting job: " + err.message);
      setLoadingState(false);
    }
  });
}

function setLoadingState(loading) {
  const btn = document.getElementById('btn-generate');
  const txt = document.getElementById('btn-text');
  const spinner = document.getElementById('btn-spinner');
  btn.disabled = loading;
  if (loading) {
    txt.textContent = "Processing Job...";
    spinner.classList.remove('hidden');
    document.getElementById('empty-state').classList.add('hidden');
    const badge = document.getElementById('job-status-badge');
    badge.textContent = "Running";
    badge.className = "status-badge badge-running";
  } else {
    txt.textContent = "Generate B-Roll Video";
    spinner.classList.add('hidden');
  }
}

// 4. Polling for Job Completion
function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/v1/jobs/${jobId}`);
      if (!resp.ok) return;

      const job = await resp.json();
      updateProgress(job.progress, job.current_step || job.status);

      if (job.status === 'succeeded') {
        clearInterval(pollTimer);
        setLoadingState(false);
        const badge = document.getElementById('job-status-badge');
        badge.textContent = "Succeeded";
        badge.className = "status-badge badge-success";
        renderResults(job.outputs);
      } else if (job.status === 'failed') {
        clearInterval(pollTimer);
        setLoadingState(false);
        const badge = document.getElementById('job-status-badge');
        badge.textContent = "Failed";
        badge.className = "status-badge badge-danger";
        const errMsg = job.error ? `${job.error.code}: ${job.error.message}` : "Job failed.";
        alert("Job execution failed: " + errMsg);
      }
    } catch (e) {
      console.error("Polling error:", e);
    }
  }, 1000);
}

function updateProgress(percent, label) {
  document.getElementById('progress-bar').style.width = percent + '%';
  document.getElementById('progress-percent').textContent = percent + '%';
  document.getElementById('progress-step-text').textContent = label;
}

// 5. Render Video & Outputs
async function renderResults(outputs) {
  currentOutputs = outputs;
  document.getElementById('video-player-wrapper').classList.remove('hidden');
  document.getElementById('timeline-container').classList.remove('hidden');
  document.getElementById('inspect-section').classList.remove('hidden');

  // Multi-aspect ratio switcher
  const ratioBtnContainer = document.getElementById('ratio-buttons');
  ratioBtnContainer.innerHTML = '';
  const videoKeys = Object.keys(outputs.videos || {});

  videoKeys.forEach((key, idx) => {
    const btn = document.createElement('button');
    btn.className = `ratio-tab-btn ${idx === 0 ? 'active' : ''}`;
    btn.textContent = key;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ratio-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadVideo(outputs.videos[key]);
    });
    ratioBtnContainer.appendChild(btn);
  });

  if (videoKeys.length > 0) {
    loadVideo(outputs.videos[videoKeys[0]]);
  }

  // Set download links
  const srtBtn = document.getElementById('btn-download-srt');
  if (outputs.captions_srt) {
    srtBtn.href = outputs.captions_srt;
    // Load and parse captions for live interactive toggle
    loadCaptionCues(outputs.captions_srt);
  }

  // Fetch and display timeline
  if (outputs.timeline) {
    const tResp = await fetch(outputs.timeline);
    cachedTimeline = await tResp.json();
    renderTimelineCards(cachedTimeline);
  }

  // Fetch manifest
  if (outputs.manifest) {
    const mResp = await fetch(outputs.manifest);
    cachedManifest = await mResp.json();
    document.getElementById('inspect-content').textContent = JSON.stringify(cachedManifest, null, 2);
  }
}

function loadVideo(url) {
  const video = document.getElementById('main-video');
  video.src = url;
  video.load();
  document.getElementById('btn-download-video').href = url;
}

// Captions Toggle & Parser
function initCaptionToggle() {
  const toggleBtn = document.getElementById('btn-caption-toggle');
  const statusTxt = document.getElementById('caption-status-text');
  const overlay = document.getElementById('video-caption-overlay');

  if (!toggleBtn) return;

  toggleBtn.addEventListener('click', () => {
    captionsEnabled = !captionsEnabled;
    if (captionsEnabled) {
      toggleBtn.classList.add('active');
      statusTxt.textContent = "Captions: ON";
    } else {
      toggleBtn.classList.remove('active');
      statusTxt.textContent = "Captions: OFF";
      if (overlay) {
        overlay.textContent = '';
        overlay.classList.remove('visible');
      }
    }
  });
}

function parseSrtTimeToSeconds(timeStr) {
  // 00:00:01,234
  const parts = timeStr.trim().replace(',', '.').split(':');
  if (parts.length === 3) {
    return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
  }
  return 0;
}

async function loadCaptionCues(srtUrl) {
  try {
    const resp = await fetch(srtUrl);
    if (!resp.ok) return;
    const text = await resp.text();
    captionCues = [];

    // Parse standard SRT format
    const blocks = text.trim().split(/\r?\n\r?\n/);
    blocks.forEach(block => {
      const lines = block.split(/\r?\n/);
      if (lines.length >= 3) {
        const timeLine = lines[1];
        const match = timeLine.match(/(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})/);
        if (match) {
          const start = parseSrtTimeToSeconds(match[1]);
          const end = parseSrtTimeToSeconds(match[2]);
          const captionText = lines.slice(2).join(' ');
          captionCues.push({ start, end, text: captionText });
        }
      }
    });
  } catch (e) {
    console.warn("Could not load captions for overlay:", e);
  }
}

function initVideoTimeUpdate() {
  const video = document.getElementById('main-video');
  const overlay = document.getElementById('video-caption-overlay');
  if (!video || !overlay) return;

  video.addEventListener('timeupdate', () => {
    if (!captionsEnabled || captionCues.length === 0) {
      overlay.classList.remove('visible');
      return;
    }

    const cur = video.currentTime;
    const activeCue = captionCues.find(c => cur >= c.start && cur <= c.end);
    if (activeCue) {
      overlay.textContent = activeCue.text;
      overlay.classList.add('visible');
    } else {
      overlay.classList.remove('visible');
    }
  });
}

// 6. Timeline Cards & Provenance Display
function renderTimelineCards(scenes) {
  const list = document.getElementById('scene-list');
  list.innerHTML = '';

  scenes.forEach(s => {
    const clip = s.clip_used || {};
    const card = document.createElement('div');
    card.className = 'scene-item-card';

    const aiChip = clip.is_ai_generated 
      ? `<span class="tag tag-ai">AI Synthesized</span>` 
      : `<span class="tag tag-stock">Stock Footage</span>`;

    card.innerHTML = `
      <div class="time-badge">${s.start_time}s – ${s.end_time}s</div>
      <div class="scene-detail">
        <div class="scene-speech">"${s.narration}"</div>
        <div class="meta-tags">
          ${aiChip}
          <span class="tag tag-score">Relevance: ${clip.relevance_score || '4.5'}/5</span>
          <span class="tag tag-license">${clip.source || 'Commercial Library'}</span>
          <span class="tag tag-license">${clip.license || 'Commercial Free'}</span>
        </div>
      </div>
      <button class="btn-swap" data-index="${s.scene_index}">Swap Clip</button>
    `;
    list.appendChild(card);
  });

  // Attach swap buttons
  document.querySelectorAll('.btn-swap').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(e.target.getAttribute('data-index'));
      openSwapModal(idx, scenes[idx]);
    });
  });
}

// 7. Swap Clip Modal
function initModal() {
  const modal = document.getElementById('swap-modal');
  document.getElementById('btn-close-modal').addEventListener('click', () => modal.classList.add('hidden'));
  document.getElementById('btn-cancel-swap').addEventListener('click', () => modal.classList.add('hidden'));

  document.getElementById('btn-confirm-swap').addEventListener('click', async () => {
    const newQuery = document.getElementById('swap-query').value.trim();
    const source = document.getElementById('swap-source').value;

    if (!newQuery) {
      alert("Please enter a new concept query.");
      return;
    }

    const swapBtn = document.getElementById('btn-confirm-swap');
    swapBtn.disabled = true;
    swapBtn.textContent = "Re-rendering...";

    try {
      const resp = await fetch(`/api/v1/jobs/${currentJobId}/swap-clip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scene_index: selectedSwapIndex,
          new_query: newQuery,
          preferred_source: source === 'auto' ? null : source
        })
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error ? data.error.message : "Failed to swap clip");
      }

      modal.classList.add('hidden');
      renderResults(data.outputs);
    } catch (err) {
      alert("Failed to swap clip: " + err.message);
    } finally {
      swapBtn.disabled = false;
      swapBtn.textContent = "Fast Re-Render Scene";
    }
  });
}

function openSwapModal(index, scene) {
  selectedSwapIndex = index;
  document.getElementById('swap-scene-idx').textContent = index + 1;
  document.getElementById('swap-scene-narration').textContent = `"${scene.narration}"`;
  document.getElementById('swap-query').value = scene.search_query || '';
  document.getElementById('swap-modal').classList.remove('hidden');
}

// 8. Inspector Tabs
function initInspectTabs() {
  const btns = document.querySelectorAll('.inspect-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', async () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const target = btn.getAttribute('data-inspect');
      const contentEl = document.getElementById('inspect-content');

      if (target === 'inspect-manifest') {
        contentEl.textContent = JSON.stringify(cachedManifest, null, 2);
      } else if (target === 'inspect-timeline') {
        contentEl.textContent = JSON.stringify(cachedTimeline, null, 2);
      } else if (target === 'inspect-srt') {
        if (!cachedSrt && currentOutputs?.captions_srt) {
          const resp = await fetch(currentOutputs.captions_srt);
          cachedSrt = await resp.text();
        }
        contentEl.textContent = cachedSrt || "No captions available.";
      }
    });
  });
}
