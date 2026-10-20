/**
 * IncuBrix AI B-Roll Studio — Job Processing Controller
 */

import { Api } from './api.js';

const STAGES = [
  'Validating input',
  'Transcribing audio or generating narration',
  'Splitting narration into scenes',
  'Understanding scene meaning',
  'Searching for footage',
  'Checking relevance and licences',
  'Selecting and timing clips',
  'Rendering video',
  'Generating captions',
  'Validating output',
  'Saving results'
];

let pollInterval = null;
let startTime = Date.now();

document.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const jobId = urlParams.get('job_id');

  if (!jobId) {
    document.getElementById('processing-container').innerHTML = `
      <div style="color:var(--danger); text-align:center; padding:40px;">
        <h3>No Job ID specified</h3>
        <p>Please initiate video generation from the Create Video page.</p>
        <a href="/create.html" class="btn btn-primary" style="margin-top:16px;">Go to Create Video</a>
      </div>
    `;
    return;
  }

  document.getElementById('job-id-text').textContent = jobId;
  renderStages();
  startElapsedTimer();
  startPolling(jobId);
});

function renderStages() {
  const container = document.getElementById('stages-container');
  if (!container) return;

  container.innerHTML = STAGES.map((s, i) => `
    <div class="stage-item" id="stage-${i}">
      <div class="stage-indicator">${i + 1}</div>
      <div class="stage-label">${s}</div>
    </div>
  `).join('');
}

function startElapsedTimer() {
  const timerEl = document.getElementById('elapsed-time-text');
  setInterval(() => {
    const sec = Math.floor((Date.now() - startTime) / 1000);
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    if (timerEl) {
      timerEl.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
  }, 1000);
}

function startPolling(jobId) {
  poll();
  pollInterval = setInterval(poll, 1800);

  async function poll() {
    try {
      const data = await Api.getJob(jobId);
      updateUI(data);

      if (data.status === 'succeeded') {
        clearInterval(pollInterval);
        setTimeout(() => {
          window.location.href = `/results.html?job_id=${jobId}`;
        }, 1200);
      } else if (data.status === 'failed') {
        clearInterval(pollInterval);
        handleFailure(data.error);
      }
    } catch (err) {
      console.warn('Poll error:', err);
    }
  }
}

function updateUI(data) {
  const pct = data.progress || 0;
  document.getElementById('progress-percent-text').textContent = `${pct}%`;
  document.getElementById('progress-bar-fill').style.width = `${pct}%`;

  const stepText = data.current_step || 'Processing...';
  document.getElementById('current-step-label').textContent = stepText;

  // Map progress % to 11 stages
  const activeIdx = Math.min(STAGES.length - 1, Math.floor((pct / 100) * STAGES.length));
  STAGES.forEach((_, i) => {
    const el = document.getElementById(`stage-${i}`);
    if (!el) return;
    el.classList.remove('completed', 'active');
    if (i < activeIdx) el.classList.add('completed');
    if (i === activeIdx) el.classList.add('active');
  });
}

function handleFailure(error) {
  const headline = document.getElementById('current-step-label');
  headline.textContent = 'Processing Failed';
  headline.style.color = 'var(--danger)';

  const errBox = document.getElementById('error-box');
  if (errBox) {
    errBox.style.display = 'block';
    errBox.innerHTML = `
      <div style="background:var(--danger-bg); border:1px solid var(--danger); border-radius:var(--radius-md); padding:16px; color:var(--danger); text-align:left;">
        <strong>Error: ${error?.code || 'PIPELINE_ERROR'}</strong>
        <p style="margin-top:4px;">${error?.message || 'An error occurred during video rendering.'}</p>
        <a href="/create.html" class="btn btn-primary" style="margin-top:12px;">Try Again</a>
      </div>
    `;
  }
}
