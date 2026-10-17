// IncuBrix Web Engine Controller (Light Professional UI)

let currentJobId = null;
let currentOutputs = null;
let pollTimer = null;
let cachedManifest = null;
let cachedTimeline = null;
let cachedSrt = null;
let selectedSwapIndex = null;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initPresets();
  initFormSubmit();
  initModal();
  initInspectTabs();
  initFileUploadUI();
});

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

  function setMascot(msg) {
    const el = document.getElementById('mascot-message');
    if (el) el.textContent = msg;
  }

  document.getElementById('preset-finance')?.addEventListener('click', () => {
    selectTab('tab-script');
    document.getElementById('script-text').value = 
      "In 2025, our corporate sales grew 40% across all markets. When you deposit capital into a high yield savings bank, compound interest works quietly in your favor. Every smart investment begins with patience.";
    document.getElementById('tts-voice').value = "en-US-AriaNeural";
    setMascot("📈 Great choice! I will synthesize ascending financial stock graphs, candlestick growth curves, and the animated character host!");
  });

  document.getElementById('preset-cloud')?.addEventListener('click', () => {
    selectTab('tab-script');
    document.getElementById('script-text').value = 
      "Modern cloud computing infrastructure powers millions of servers globally. Our distributed clusters guarantee zero downtime, automated failovers, and resilient data backups.";
    document.getElementById('tts-voice').value = "en-US-AriaNeural";
    setMascot("☁️ High-tech choice! I will synthesize 3D cybernetic perspective grids, server nodes, and pulse circuits!");
  });

  document.getElementById('preset-hindi')?.addEventListener('click', () => {
    selectTab('tab-script');
    document.getElementById('script-text').value = 
      "तकनीक और नवाचार की दुनिया में आपका स्वागत है। हमारा उद्देश्य रचनाकारों को कृत्रिम बुद्धिमत्ता के साथ शक्तिशाली वीडियो बनाने में सहायता करना है।";
    document.getElementById('tts-voice').value = "hi-IN-female";
    setMascot("🇮🇳 बहुत बढ़िया! हिंदी में वॉयसओवर और असीमित एनिमेटेड दृश्य तैयार कर रहा हूँ!");
  });

  document.getElementById('preset-tamil')?.addEventListener('click', () => {
    selectTab('tab-script');
    document.getElementById('script-text').value = 
      "செயற்கை நுண்ணறிவு மூலம் காணொளிகளை உருவாக்குவது இப்போது மிக எளிது. உங்கள் படைப்பாற்றலை அடுத்த கட்டத்திற்கு கொண்டு செல்லுங்கள்.";
    document.getElementById('tts-voice').value = "ta-IN-female";
    setMascot("✨ அருமை! தமிழ் குரல்வழியில் அனிமேஷன் காட்சிகளை உருவாக்குகிறேன்!");
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
    if (document.getElementById('ratio-16-9').checked) aspectRatios.push("16:9");
    if (document.getElementById('ratio-9-16').checked) aspectRatios.push("9:16");
    if (document.getElementById('ratio-1-1').checked) aspectRatios.push("1:1");

    if (aspectRatios.length === 0) {
      alert("Please select at least one aspect ratio.");
      return;
    }

    const style = document.getElementById('video-style').value.trim();
    const avoidRaw = document.getElementById('avoid-keywords').value;
    const avoid = avoidRaw.split(',').map(s => s.trim()).filter(Boolean);
    const burnIn = document.getElementById('burn-in-captions').checked;

    // Generate unique idempotency key
    const idempotencyKey = "req_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);
    setLoadingState(true);

    try {
      let resp;
      if (activeTab === 'tab-audio' && document.getElementById('audio-upload').files.length > 0) {
        // Multipart upload
        const file = document.getElementById('audio-upload').files[0];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('aspect_ratios', aspectRatios.join(','));
        formData.append('style', style);
        formData.append('avoid', avoid.join(','));
        formData.append('burn_in_captions', burnIn);

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
          avoid: avoid,
          captions: { burn_in: burnIn, srt: true }
        };

        if (activeTab === 'tab-script') {
          payload.script = document.getElementById('script-text').value.trim();
          payload.voice = document.getElementById('tts-voice').value;
          if (!payload.script) {
            alert("Please enter a text script or click a quick preset.");
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
    const mascot = document.getElementById('mascot-message');
    if (mascot) mascot.textContent = "⚡ Generating your B-roll video! Transcribing speech with Faster-Whisper, animating the character host, and rendering scene cuts...";
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

  const mascot = document.getElementById('mascot-message');
  if (mascot) mascot.textContent = "🎉 Awesome! Your finished video is ready! Enjoy the playback with animated character host, switch aspect ratios (16:9 / 9:16), or swap any scene clip below!";

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
      <div>
        <button class="btn-swap-clip" onclick="openSwapModal(${s.scene_index}, '${escapeQuotes(s.narration)}')">Swap Clip</button>
      </div>
    `;
    list.appendChild(card);
  });
}

function escapeQuotes(str) {
  return (str || '').replace(/'/g, "\\'");
}

// 7. Clip Swapping Modal Logic
function initModal() {
  const modal = document.getElementById('swap-modal');
  document.getElementById('btn-close-modal').addEventListener('click', () => modal.classList.add('hidden'));
  document.getElementById('btn-cancel-swap').addEventListener('click', () => modal.classList.add('hidden'));

  document.getElementById('btn-confirm-swap').addEventListener('click', async () => {
    if (selectedSwapIndex === null || !currentJobId) return;

    const query = document.getElementById('swap-query').value.trim();
    const source = document.getElementById('swap-source').value;
    const confirmBtn = document.getElementById('btn-confirm-swap');
    confirmBtn.disabled = true;
    confirmBtn.textContent = "Swapping...";

    try {
      const resp = await fetch(`/api/v1/jobs/${currentJobId}/swap-clip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scene_index: selectedSwapIndex,
          new_query: query || null,
          preferred_source: source === 'auto' ? null : source
        })
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error ? data.error.message : "Swap failed");

      modal.classList.add('hidden');
      confirmBtn.disabled = false;
      confirmBtn.textContent = "Fast Re-Render Scene";
      
      // Reload updated outputs
      renderResults(data.outputs);
    } catch (e) {
      alert("Error swapping clip: " + e.message);
      confirmBtn.disabled = false;
      confirmBtn.textContent = "Fast Re-Render Scene";
    }
  });
}

window.openSwapModal = function(sceneIdx, narration) {
  selectedSwapIndex = sceneIdx;
  document.getElementById('swap-scene-idx').textContent = sceneIdx;
  document.getElementById('swap-scene-narration').textContent = narration;
  document.getElementById('swap-query').value = '';
  document.getElementById('swap-modal').classList.remove('hidden');
};

// 8. Inspect Tab Switcher
function initInspectTabs() {
  const btns = document.querySelectorAll('.inspect-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', async () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const type = btn.getAttribute('data-inspect');
      const pre = document.getElementById('inspect-content');

      if (type === 'inspect-manifest' && cachedManifest) {
        pre.textContent = JSON.stringify(cachedManifest, null, 2);
      } else if (type === 'inspect-timeline' && cachedTimeline) {
        pre.textContent = JSON.stringify(cachedTimeline, null, 2);
      } else if (type === 'inspect-srt' && currentOutputs?.captions_srt) {
        if (!cachedSrt) {
          const resp = await fetch(currentOutputs.captions_srt);
          cachedSrt = await resp.text();
        }
        pre.textContent = cachedSrt;
      }
    });
  });
}
