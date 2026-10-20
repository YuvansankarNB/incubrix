/**
 * IncuBrix AI B-Roll Studio — Settings Controller
 */

import { Api } from './api.js';

document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
});

async function loadSettings() {
  try {
    const data = await Api.getSettings();
    const providers = data.providers || {};

    const listEl = document.getElementById('provider-status-list');
    if (listEl) {
      listEl.innerHTML = Object.entries(providers).map(([k, active]) => `
        <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 0; border-bottom:1px solid var(--border-color);">
          <strong style="text-transform:capitalize;">${k.replace(/_/g, ' ')}</strong>
          ${active 
            ? '<span class="badge badge-success">Configured & Active</span>' 
            : '<span class="badge badge-neutral">Demo / Fallback Mode</span>'}
        </div>
      `).join('');
    }

    document.getElementById('storage-mode-text').textContent = data.storage?.mode || 'local';
    document.getElementById('storage-path-text').textContent = data.storage?.local_dir || './data/storage';
    document.getElementById('max-duration-text').textContent = `${data.limits?.max_duration_seconds || 600} seconds`;
  } catch (err) {
    console.warn('Failed to load settings:', err);
  }
}
