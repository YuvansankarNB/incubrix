/**
 * IncuBrix AI B-Roll Studio — Scene Timeline Editor Controller
 */

import { Api } from './api.js';

let currentJobId = null;
let currentScenes = [];
let activeSwapSceneIdx = 0;

document.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  currentJobId = urlParams.get('job_id') || 'demo_project_clean_energy';

  initModal();
  loadEditorScenes(currentJobId);
});

async function loadEditorScenes(jobId) {
  const container = document.getElementById('scenes-editor-list');
  if (!container) return;

  try {
    const data = await Api.getJobResult(jobId).catch(() => ({}));
    currentScenes = data.timeline || [
      {
        scene_index: 0,
        start_time: 0.0,
        end_time: 4.5,
        duration: 4.5,
        narration: "Carbon emissions were reduced by 35% across all manufacturing plants.",
        provider: "Pexels Commercial Stock API",
        relevance_score: 4.7,
        license_status: "Verified Commercial",
        selection_reason: "High-confidence stock match for clean energy wind farm at sunset."
      },
      {
        scene_index: 1,
        start_time: 4.5,
        end_time: 9.2,
        duration: 4.7,
        narration: "Next-generation product design relies entirely on biodegradable and circular materials.",
        provider: "Pexels Commercial Stock API",
        relevance_score: 4.6,
        license_status: "Verified Commercial",
        selection_reason: "High-confidence stock match for modern biotechnology laboratory."
      },
      {
        scene_index: 2,
        start_time: 9.2,
        end_time: 14.0,
        duration: 4.8,
        narration: "Lasting ecological change is achieved by fostering green literacy within local communities.",
        provider: "IncuBrix Motion Synthesis",
        relevance_score: 4.5,
        license_status: "Verified Commercial",
        selection_reason: "Dynamic motion synthesis matching community botanical garden workshop."
      }
    ];

    container.innerHTML = currentScenes.map((s, idx) => `
      <div class="scene-editor-item" id="editor-scene-${idx}">
        <div class="editor-thumb-box">
          <img src="/static/sample_thumb.jpg" alt="Scene ${idx + 1}" onerror="this.src='/static/sample_thumb.jpg'">
        </div>
        <div class="editor-content-box">
          <div class="editor-time-range">SCENE ${idx + 1} // ${s.start_time.toFixed(1)}s - ${s.end_time.toFixed(1)}s (${s.duration.toFixed(1)}s)</div>
          <div class="editor-narration">${escapeHtml(s.narration)}</div>
          <div class="editor-reason">${escapeHtml(s.selection_reason)}</div>
          <div class="editor-meta-tags">
            <span class="badge badge-neutral">${s.provider}</span>
            <span class="badge badge-success">Relevance: ${s.relevance_score}/5.0</span>
            <span class="badge badge-success">${s.license_status}</span>
          </div>
        </div>
        <div class="editor-actions">
          <button type="button" class="btn btn-outline btn-swap-clip" data-idx="${idx}">
            Replace Footage
          </button>
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.btn-swap-clip').forEach(btn => {
      btn.addEventListener('click', () => {
        openSwapModal(parseInt(btn.dataset.idx, 10));
      });
    });
  } catch (err) {
    container.innerHTML = `<div style="color:var(--danger); padding:20px;">Failed to load scene editor data.</div>`;
  }
}

function initModal() {
  const modal = document.getElementById('swap-modal');
  const btnClose = document.getElementById('btn-close-modal');
  const btnCancel = document.getElementById('btn-cancel-modal');
  const btnConfirm = document.getElementById('btn-confirm-modal');

  const close = () => modal.classList.add('hidden');
  if (btnClose) btnClose.addEventListener('click', close);
  if (btnCancel) btnCancel.addEventListener('click', close);

  if (btnConfirm) {
    btnConfirm.addEventListener('click', async () => {
      const query = document.getElementById('swap-query-input').value.trim();
      const source = document.getElementById('swap-provider-select').value;
      if (!query) {
        alert('Please enter a replacement search query.');
        return;
      }

      btnConfirm.disabled = true;
      btnConfirm.textContent = 'Re-rendering Scene...';

      try {
        await Api.replaceScene(currentJobId, activeSwapSceneIdx, query, source);
        alert(`Scene ${activeSwapSceneIdx + 1} footage replaced and re-rendered successfully!`);
        close();
        window.location.reload();
      } catch (err) {
        alert(`Error replacing footage: ${err.message}`);
        btnConfirm.disabled = false;
        btnConfirm.textContent = 'Confirm Replacement';
      }
    });
  }
}

function openSwapModal(idx) {
  activeSwapSceneIdx = idx;
  const scene = currentScenes[idx];
  document.getElementById('modal-scene-idx-text').textContent = idx + 1;
  document.getElementById('modal-scene-narration').textContent = scene?.narration || '';
  document.getElementById('swap-query-input').value = '';
  document.getElementById('swap-modal').classList.remove('hidden');
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
