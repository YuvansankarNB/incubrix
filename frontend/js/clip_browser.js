/**
 * IncuBrix Clip Browser — Vanilla JS
 * Per-scene Pixabay clip search, preview, and replacement panel.
 *
 * Usage:
 *   import { ClipBrowser } from './clip_browser.js';
 *   const browser = new ClipBrowser({ jobId, containerEl });
 *   browser.render(scenesTimeline);
 */

import { Api } from './api.js';

function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtDuration(sec) {
  const s = Math.round(sec || 0);
  return s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` : `0:${String(s).padStart(2, '0')}`;
}

function fmtResolution(w, h) {
  if (!w || !h) return '';
  if (w >= 1920) return '4K / FHD';
  if (w >= 1280) return 'HD';
  return `${w}×${h}`;
}

// ──────────────────────────────────────────────────────────────
export class ClipBrowser {
  /**
   * @param {object} opts
   * @param {string}      opts.jobId        — Job ID for context
   * @param {HTMLElement} opts.containerEl  — Target DOM node
   */
  constructor({ jobId, containerEl }) {
    this.jobId = jobId;
    this.container = containerEl;
    this._scenes = [];
    this._panelState = {};   // sceneIndex → { open, query, filter, selectedResult }
    this._loadingSet = new Set();
  }

  // ── Public API ─────────────────────────────────────────────

  render(scenesTimeline) {
    this._scenes = scenesTimeline || [];
    if (!this.container) return;

    if (!this._scenes.length) {
      this.container.innerHTML = `<p style="color:var(--text-muted);font-size:0.9rem;">No scene timeline available yet.</p>`;
      return;
    }

    this.container.innerHTML = this._scenes.map((s, i) => this._renderSceneRow(s, i)).join('');
    this._attachListeners();
  }

  // ── Scene row HTML ─────────────────────────────────────────

  _renderSceneRow(scene, idx) {
    const clip = scene.clip_used || scene.clip || {};
    const isMock = clip.is_mock ?? true;
    const isAi = clip.is_ai_generated ?? Boolean(scene.is_ai_generated);
    const thumbSrc = clip.thumbnail_url || '/static/sample_thumb.jpg';
    const narration = (scene.narration || scene.text || '').slice(0, 90);
    const sourceBadge = isAi ? '🤖 AI' : (isMock ? '📦 Mock' : '📹 Pixabay');

    return `
      <div class="scene-clip-row" data-scene="${idx}">
        <div class="scene-current-thumb">
          <img src="${esc(thumbSrc)}" alt="Scene ${idx + 1} thumbnail" loading="lazy"
               onerror="this.src='/static/sample_thumb.jpg'">
          <span class="thumb-source-badge">${esc(sourceBadge)}</span>
        </div>
        <div class="scene-clip-info" style="flex:1;min-width:0;">
          <div class="scene-label">Scene ${idx + 1}</div>
          <div class="scene-narration-snippet">${esc(narration)}${narration.length >= 90 ? '…' : ''}</div>
          <div class="scene-clip-actions">
            <button class="btn btn-secondary btn-sm btn-search-clips" data-scene="${idx}"
              style="font-size:0.78rem;padding:5px 10px;">
              🔍 Search Clips
            </button>
            ${!isAi && clip.page_url ? `
              <a href="${esc(clip.page_url)}" target="_blank" rel="noopener noreferrer"
                class="btn btn-secondary btn-sm" style="font-size:0.78rem;padding:5px 10px;">
                View Source ↗
              </a>` : ''}
          </div>

          <!-- Inline search panel (hidden by default) -->
          <div class="clip-search-panel" id="clip-panel-${idx}">
            <div class="clip-search-bar">
              <input type="text" class="clip-search-input" id="clip-query-${idx}"
                placeholder="e.g. doctor consultation, classroom, nature…"
                value="${esc(scene.broll_keywords || scene.narration || '').slice(0, 80)}">
              <button class="btn btn-primary btn-sm btn-do-search" data-scene="${idx}"
                style="font-size:0.82rem;">Search</button>
              <button class="btn btn-secondary btn-sm btn-close-panel" data-scene="${idx}"
                style="font-size:0.82rem;">✕ Close</button>
            </div>

            <div class="clip-filter-chips" id="clip-filters-${idx}">
              <span class="filter-chip active" data-filter="all" data-scene="${idx}">All</span>
              <span class="filter-chip" data-filter="16:9" data-scene="${idx}">📺 Landscape</span>
              <span class="filter-chip" data-filter="9:16" data-scene="${idx}">📱 Portrait</span>
              <span class="filter-chip" data-filter="1:1" data-scene="${idx}">⬛ Square</span>
              <span class="filter-chip" data-filter="editors_choice" data-scene="${idx}">⭐ Editor's Choice</span>
            </div>

            <div class="clip-results-grid" id="clip-grid-${idx}">
              <div class="clip-state-message">Click <strong>Search</strong> to find matching clips.</div>
            </div>

            <div class="clip-apply-bar hidden" id="clip-apply-bar-${idx}">
              <span class="apply-label" id="clip-apply-label-${idx}">No clip selected</span>
              <button class="btn btn-primary btn-sm btn-apply-clip" data-scene="${idx}"
                style="font-size:0.82rem;">✅ Apply Replacement</button>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // ── Event binding ──────────────────────────────────────────

  _attachListeners() {
    if (!this.container) return;

    // Open panel
    this.container.querySelectorAll('.btn-search-clips').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.scene, 10);
        this._togglePanel(idx, true);
      });
    });

    // Close panel
    this.container.querySelectorAll('.btn-close-panel').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.scene, 10);
        this._togglePanel(idx, false);
      });
    });

    // Run search on button click
    this.container.querySelectorAll('.btn-do-search').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.scene, 10);
        this._runSearch(idx);
      });
    });

    // Run search on Enter in input
    this.container.querySelectorAll('.clip-search-input').forEach(input => {
      const idx = parseInt(input.id.replace('clip-query-', ''), 10);
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') this._runSearch(idx);
      });
    });

    // Filter chips
    this.container.querySelectorAll('.filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const idx = parseInt(chip.dataset.scene, 10);
        const filterRow = this.container.querySelector(`#clip-filters-${idx}`);
        filterRow?.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const ps = this._panelState[idx] || {};
        ps.filter = chip.dataset.filter;
        this._panelState[idx] = ps;
        this._runSearch(idx);
      });
    });

    // Apply clip
    this.container.querySelectorAll('.btn-apply-clip').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.scene, 10);
        this._applySelectedClip(idx);
      });
    });
  }

  // ── Panel toggle ───────────────────────────────────────────

  _togglePanel(sceneIdx, open) {
    const panel = this.container?.querySelector(`#clip-panel-${sceneIdx}`);
    if (!panel) return;
    if (open) {
      panel.classList.add('open');
    } else {
      panel.classList.remove('open');
    }
    if (!this._panelState[sceneIdx]) {
      this._panelState[sceneIdx] = { open, filter: 'all', selectedResult: null };
    }
    this._panelState[sceneIdx].open = open;
  }

  // ── Search ─────────────────────────────────────────────────

  async _runSearch(sceneIdx) {
    if (this._loadingSet.has(sceneIdx)) return;

    const ps = this._panelState[sceneIdx] || {};
    const queryEl = this.container?.querySelector(`#clip-query-${sceneIdx}`);
    const query = queryEl?.value?.trim() || '';
    const filter = ps.filter || 'all';

    const grid = this.container?.querySelector(`#clip-grid-${sceneIdx}`);
    if (!grid) return;

    grid.innerHTML = `<div class="clip-state-message">
      <span class="clip-loading-spinner"></span> Searching Pixabay…
    </div>`;
    this._loadingSet.add(sceneIdx);

    try {
      let aspectRatio = '16:9';
      let editorsChoice = false;
      if (filter === '9:16') aspectRatio = '9:16';
      else if (filter === '1:1') aspectRatio = '1:1';
      else if (filter === 'editors_choice') { editorsChoice = true; }

      let data;
      if (this.jobId) {
        data = await Api.searchSceneClips(this.jobId, sceneIdx, {
          query: query || undefined,
          perPage: 12,
          aspectRatio,
          editorsChoice,
        });
      } else {
        data = await Api.searchPixabayVideos({
          q: query || 'professional b-roll',
          perPage: 12,
          aspectRatio,
          editorsChoice,
        });
      }

      const results = data.results || [];
      if (!results.length) {
        grid.innerHTML = `<div class="clip-state-message">No clips found for "<strong>${esc(query)}</strong>". Try different keywords.</div>`;
        return;
      }

      grid.innerHTML = results.map(r => this._renderClipCard(r, sceneIdx)).join('');
      this._attachCardListeners(sceneIdx, results);
      this._panelState[sceneIdx] = { ...ps, results, selectedResult: null };
      this._updateApplyBar(sceneIdx, null);

    } catch (err) {
      grid.innerHTML = `<div class="clip-state-message" style="color:#ef4444;">
        ⚠ Search failed: ${esc(err.message)}
      </div>`;
    } finally {
      this._loadingSet.delete(sceneIdx);
    }
  }

  // ── Clip card HTML ─────────────────────────────────────────

  _renderClipCard(result, sceneIdx) {
    const isMock = result.is_mock;
    const thumbSrc = result.thumbnail_url || '/static/sample_thumb.jpg';
    const res = fmtResolution(result.width, result.height);
    const dur = fmtDuration(result.duration);
    const score = result.relevance_score ? `★${result.relevance_score.toFixed(1)}` : '';
    const tags = (result.tags || '').split(',').slice(0, 4).join(', ');

    return `
      <div class="clip-card" data-id="${result.pixabay_id}" data-scene="${sceneIdx}">
        <div class="clip-card-thumb">
          <img src="${esc(thumbSrc)}" alt="${esc(tags)}" loading="lazy"
               onerror="this.src='/static/sample_thumb.jpg'">
          <span class="clip-duration-badge">${esc(dur)}</span>
          ${result.editors_choice ? '<span class="clip-ec-badge">EC</span>' : ''}
          <span class="clip-source-badge${isMock ? ' mock' : ''}">${isMock ? 'Mock' : '📹 Pixabay'}</span>
        </div>
        <div class="clip-card-meta">
          <div class="clip-card-tags">${esc(tags) || 'No tags'}</div>
          <div class="clip-card-stats">
            <span>${esc(res)}</span>
            <span>${esc(score)}</span>
          </div>
          <div class="clip-card-actions">
            ${result.page_url && !isMock ? `
              <a href="${esc(result.page_url)}" target="_blank" rel="noopener noreferrer"
                class="btn btn-secondary" title="View on Pixabay">↗</a>` : ''}
            <button class="btn btn-primary btn-select-clip"
              data-id="${result.pixabay_id}" data-scene="${sceneIdx}">Select</button>
          </div>
        </div>
      </div>
    `;
  }

  _attachCardListeners(sceneIdx, results) {
    const grid = this.container?.querySelector(`#clip-grid-${sceneIdx}`);
    if (!grid) return;

    grid.querySelectorAll('.btn-select-clip').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = parseInt(btn.dataset.id, 10);
        // Deselect all
        grid.querySelectorAll('.clip-card').forEach(c => c.classList.remove('selected'));
        // Select clicked
        btn.closest('.clip-card')?.classList.add('selected');

        const result = results.find(r => r.pixabay_id === id);
        const ps = this._panelState[sceneIdx] || {};
        ps.selectedResult = result;
        this._panelState[sceneIdx] = ps;
        this._updateApplyBar(sceneIdx, result);
      });
    });
  }

  // ── Apply bar ──────────────────────────────────────────────

  _updateApplyBar(sceneIdx, result) {
    const bar = this.container?.querySelector(`#clip-apply-bar-${sceneIdx}`);
    const label = this.container?.querySelector(`#clip-apply-label-${sceneIdx}`);
    if (!bar || !label) return;

    if (!result) {
      bar.classList.add('hidden');
      label.textContent = 'No clip selected';
      return;
    }

    bar.classList.remove('hidden');
    const tags = (result.tags || '').split(',')[0]?.trim() || 'Clip';
    label.textContent = `Selected: "${tags}" (${fmtDuration(result.duration)}, ${fmtResolution(result.width, result.height)})`;
  }

  // ── Apply clip to scene ────────────────────────────────────

  async _applySelectedClip(sceneIdx) {
    const ps = this._panelState[sceneIdx] || {};
    const result = ps.selectedResult;
    if (!result) {
      alert('Please select a clip first.');
      return;
    }

    const bar = this.container?.querySelector(`#clip-apply-bar-${sceneIdx}`);
    const applyBtn = bar?.querySelector('.btn-apply-clip');
    if (applyBtn) {
      applyBtn.textContent = '⏳ Applying…';
      applyBtn.disabled = true;
    }

    try {
      const query = result.tags?.split(',')[0]?.trim() || 'stock video';

      if (this.jobId) {
        await Api.replaceSceneClip(this.jobId, sceneIdx, {
          newQuery: query,
          preferredSource: 'pixabay',
          pixabayVideoId: result.is_mock ? undefined : result.pixabay_id,
        });
      }

      // Update thumb in row
      const thumbImg = this.container?.querySelector(`.scene-clip-row[data-scene="${sceneIdx}"] .scene-current-thumb img`);
      if (thumbImg && result.thumbnail_url) {
        thumbImg.src = result.thumbnail_url;
      }
      const sourceBadge = this.container?.querySelector(`.scene-clip-row[data-scene="${sceneIdx}"] .thumb-source-badge`);
      if (sourceBadge) {
        sourceBadge.textContent = result.is_mock ? '📦 Mock' : '📹 Pixabay';
      }

      if (applyBtn) {
        applyBtn.textContent = '✅ Applied!';
        applyBtn.style.background = '#22c55e';
        setTimeout(() => {
          if (applyBtn) {
            applyBtn.textContent = '✅ Apply Replacement';
            applyBtn.style.background = '';
            applyBtn.disabled = false;
          }
        }, 2500);
      }

      // Close panel after short delay
      setTimeout(() => this._togglePanel(sceneIdx, false), 1500);

    } catch (err) {
      if (applyBtn) {
        applyBtn.textContent = `⚠ Failed: ${err.message.slice(0, 40)}`;
        applyBtn.style.background = '#ef4444';
        applyBtn.disabled = false;
        setTimeout(() => {
          applyBtn.textContent = '✅ Apply Replacement';
          applyBtn.style.background = '';
        }, 3000);
      }
    }
  }
}
