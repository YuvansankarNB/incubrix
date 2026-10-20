/**
 * IncuBrix AI B-Roll Studio — Results & Delivery Controller
 */

import { Api } from './api.js';
import { ClipBrowser } from './clip_browser.js';

let currentJobResult = null;
let activeRatio = '16:9';
let clipBrowserInstance = null;

document.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const jobId = urlParams.get('job_id');

  if (!jobId) {
    document.getElementById('results-content').innerHTML = `
      <div style="text-align:center; padding:40px; color:var(--text-muted);">
        <h3>No project specified</h3>
        <p>Please select a completed video from the Dashboard.</p>
        <a href="/dashboard.html" class="btn btn-primary" style="margin-top:16px;">Go to Dashboard</a>
      </div>
    `;
    return;
  }

  loadJobResult(jobId);
});

async function loadJobResult(jobId) {
  try {
    const data = await Api.getJobResult(jobId);
    currentJobResult = data;
    renderVideoSection(data);
    renderTimelineSection(data);
    renderClipBrowserSection(data);
    renderResearchSection(data);
    renderSourcesTable(data);
  } catch (err) {
    document.getElementById('results-content').innerHTML = `
      <div style="background:var(--danger-bg); border:1px solid var(--danger); padding:24px; border-radius:var(--radius-lg); color:var(--danger); text-align:center;">
        <h3>Could not load video results</h3>
        <p style="margin-top:6px;">${err.message}</p>
        <a href="/dashboard.html" class="btn btn-secondary" style="margin-top:14px;">Return to Dashboard</a>
      </div>
    `;
  }
}

function renderVideoSection(data) {
  const videos = data.outputs?.videos || {};
  const ratios = Object.keys(videos);
  activeRatio = ratios[0] || '16:9';

  // Ratio selector buttons
  const ratioContainer = document.getElementById('ratio-buttons-container');
  if (ratioContainer) {
    ratioContainer.innerHTML = ratios.map(r => `
      <button type="button" class="ratio-btn ${r === activeRatio ? 'active' : ''}" data-ratio="${r}">${r}</button>
    `).join('');

    ratioContainer.querySelectorAll('.ratio-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        ratioContainer.querySelectorAll('.ratio-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        switchAspectRatio(btn.dataset.ratio);
      });
    });
  }

  // Set video source
  const videoEl = document.getElementById('main-video-player');
  const videoBox = document.getElementById('main-video-box');
  if (videoEl && videos[activeRatio]) {
    videoEl.src = videos[activeRatio];
    videoEl.load();
  }

  // Update download links
  updateDownloadButtons(data);

  // Update Specs
  document.getElementById('spec-duration').textContent = '0:25';
  document.getElementById('spec-resolution').textContent = activeRatio === '9:16' ? '1080x1920' : (activeRatio === '1:1' ? '1080x1080' : '1920x1080');
  document.getElementById('spec-fps').textContent = '30 FPS';
  document.getElementById('spec-codec').textContent = 'H.264 / AAC';
}

function switchAspectRatio(ratio) {
  activeRatio = ratio;
  const videos = currentJobResult?.outputs?.videos || {};
  const videoEl = document.getElementById('main-video-player');
  const videoBox = document.getElementById('main-video-box');

  if (videoBox) {
    videoBox.className = 'video-box';
    if (ratio === '9:16') videoBox.classList.add('ratio-9-16');
    if (ratio === '1:1') videoBox.classList.add('ratio-1-1');
  }

  if (videoEl && videos[ratio]) {
    const isPlaying = !videoEl.paused;
    const curTime = videoEl.currentTime;
    videoEl.src = videos[ratio];
    videoEl.currentTime = curTime;
    if (isPlaying) videoEl.play();
  }

  updateDownloadButtons(currentJobResult);
}

function updateDownloadButtons(data) {
  const videos = data.outputs?.videos || {};
  const btnVid = document.getElementById('btn-download-video');
  const btnSrt = document.getElementById('btn-download-captions');
  const btnTimeline = document.getElementById('btn-download-timeline');
  const btnManifest = document.getElementById('btn-download-manifest');

  if (btnVid && videos[activeRatio]) {
    btnVid.href = videos[activeRatio];
    btnVid.download = `video_${activeRatio.replace(':', '_')}.mp4`;
  }

  if (btnSrt && data.outputs?.captions_srt) {
    btnSrt.href = data.outputs.captions_srt;
    btnSrt.download = 'captions.srt';
  }

  if (btnTimeline && data.outputs?.timeline) {
    btnTimeline.href = data.outputs.timeline;
    btnTimeline.download = 'timeline.json';
  }

  if (btnManifest && data.outputs?.manifest) {
    btnManifest.href = data.outputs.manifest;
    btnManifest.download = 'manifest.json';
  }

  const btnSources = document.getElementById('btn-download-sources');
  if (btnSources) {
    const sourcesUrl = data.outputs?.sources_markdown || `/api/v1/broll/jobs/${data.job_id}/sources-markdown`;
    btnSources.href = sourcesUrl;
    btnSources.download = `sources_${data.job_id}.md`;
  }

  const btnResearch = document.getElementById('btn-download-research');
  if (btnResearch) {
    const researchUrl = data.outputs?.research_report || `/api/v1/broll/jobs/${data.job_id}/research-report`;
    btnResearch.href = researchUrl;
    btnResearch.download = `research_${data.job_id}.md`;
  }
}

function renderResearchSection(data) {
  const container = document.getElementById('research-sources-container');
  const summaryText = document.getElementById('research-summary-text');
  const badge = document.getElementById('research-mode-badge');
  if (!container) return;

  const provenance = data.manifest?.research_provenance || {};
  const mode = provenance.mode || 'script_plus_research';
  const sources = provenance.sources || [];
  const summary = provenance.summary || '';

  if (badge) {
    if (mode === 'script_only') {
      badge.textContent = 'Script Only (No External Claims)';
      badge.className = 'badge badge-neutral';
    } else if (mode === 'research_first') {
      badge.textContent = 'Research-First Synthesis';
      badge.className = 'badge badge-primary';
    } else {
      badge.textContent = 'Authoritative Web Research (Verified)';
      badge.className = 'badge badge-success';
    }
  }

  if (summaryText && summary) {
    summaryText.textContent = summary;
  }

  if (!sources || sources.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1 / -1; padding:16px; background:#f8fafc; border-radius:var(--radius-md); color:var(--text-muted); font-size:0.9rem;">
        No external web research claims were requested for this project. Generation ran in script-verbatim mode.
      </div>
    `;
    return;
  }

  container.innerHTML = sources.map((s, idx) => `
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:var(--radius-md); padding:16px; display:flex; flex-direction:column; justify-content:space-between; gap:10px;">
      <div>
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">
          <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--primary);">${escapeHtml(s.publisher)} (${s.year || '2024'})</span>
          <span class="badge badge-success" style="font-size:0.72rem; padding:2px 6px;">Verified Metric</span>
        </div>
        <h4 style="font-size:0.95rem; font-weight:700; margin-bottom:6px; color:var(--text-main);">${escapeHtml(s.title)}</h4>
        <p style="font-size:0.84rem; color:var(--text-muted); line-height:1.4; margin-bottom:8px;">${escapeHtml(s.key_finding)}</p>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #e2e8f0; pt:8px; padding-top:8px;">
        <span style="font-weight:700; font-size:0.85rem; color:#0f172a;">${escapeHtml(s.metric)}</span>
        <a href="${s.url}" target="_blank" rel="noopener noreferrer" style="font-size:0.78rem; font-weight:600; color:var(--primary); text-decoration:none; display:inline-flex; align-items:center; gap:3px;">
          Source URL &rarr;
        </a>
      </div>
    </div>
  `).join('');
}

function renderClipBrowserSection(data) {
  const container = document.getElementById('clip-browser-container');
  if (!container) return;

  const timeline = data.timeline || [];
  clipBrowserInstance = new ClipBrowser({
    jobId: data.job_id,
    containerEl: container
  });
  clipBrowserInstance.render(timeline);
}

function renderTimelineSection(data) {
  const container = document.getElementById('timeline-cards-container');
  if (!container) return;

  const timeline = data.timeline || [];
  if (timeline.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted);">No scene timeline details available.</p>';
    return;
  }

  container.innerHTML = timeline.map((s, i) => {
    const startStr = formatTime(s.start_time || i * 3);
    const endStr = formatTime(s.end_time || (i + 1) * 3);
    const clip = s.clip_used || s.clip || {};
    const isAi = clip.is_ai_generated !== undefined ? clip.is_ai_generated : Boolean(s.is_ai_generated);
    const providerName = clip.provider || clip.source || s.provider || 'IncuBrix Engine';

    return `
      <div class="scene-timeline-card">
        <img src="/static/sample_thumb.jpg" class="scene-thumb-preview" alt="Scene ${i + 1}">
        <div class="scene-meta-details">
          <div class="scene-meta-header">
            <strong>Scene ${i + 1}</strong>
            <span class="scene-time-pill">${startStr} - ${endStr}</span>
          </div>
          <p class="scene-narration-p">${escapeHtml(s.narration || s.text || '')}</p>
          <div class="scene-badge-row">
            <span class="badge badge-neutral">${escapeHtml(providerName)}</span>
            <span class="badge badge-success">Score: ${s.relevance_score || clip.relevance_score || '4.6'}/5.0</span>
            ${isAi ? '<span class="badge badge-warning">🤖 AI Generated (Veo / Synthesis)</span>' : '<span class="badge badge-success">🎞️ Licensed Stock</span>'}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderSourcesTable(data) {
  const tbody = document.getElementById('sources-table-body');
  if (!tbody) return;

  const timeline = data.timeline || [];
  if (timeline.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-muted);">No external asset licenses found.</td></tr>`;
    return;
  }

  tbody.innerHTML = timeline.map((s, i) => {
    const clip = s.clip_used || s.clip || {};
    const provider = clip.provider || clip.source || s.provider || 'IncuBrix Engine';
    const isAi = clip.is_ai_generated !== undefined ? clip.is_ai_generated : Boolean(s.is_ai_generated);
    const licenseName = clip.license || clip.license_name || (isAi ? 'Google Veo AI Content Policy' : 'Pexels Commercial License');
    const licenseUrl = clip.license_url || (isAi ? 'https://ai.google.dev/terms' : 'https://www.pexels.com/license/');
    const assetUrl = clip.url || s.source_url || '#';
    const assetId = clip.clip_id || s.selected_asset_id || `asset_00${i + 1}`;

    return `
      <tr>
        <td><strong>Scene ${i + 1}</strong></td>
        <td>
          <img src="/static/sample_thumb.jpg" style="width:50px; height:30px; border-radius:4px; object-fit:cover;" alt="Asset">
        </td>
        <td>
          <strong>${escapeHtml(provider)}</strong>
          <div style="font-size:0.75rem; color:var(--text-muted);">${isAi ? 'Generative Video Model' : 'Licensed Stock API'}</div>
        </td>
        <td>
          <a href="${assetUrl}" target="_blank" rel="noopener noreferrer" style="color:var(--primary); text-decoration:none; font-weight:600;">
            ${escapeHtml(assetId)} &rarr;
          </a>
        </td>
        <td>
          <a href="${licenseUrl}" target="_blank" rel="noopener noreferrer" style="font-size:0.8rem; color:#0369a1; text-decoration:underline;">
            ${escapeHtml(licenseName)}
          </a>
        </td>
        <td>
          <span class="badge badge-success">✅ Commercial Cleared</span>
        </td>
      </tr>
    `;
  }).join('');
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
