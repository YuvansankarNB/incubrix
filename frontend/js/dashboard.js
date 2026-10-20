/**
 * IncuBrix AI B-Roll Studio — Dashboard Controller
 */

import { Api } from './api.js';

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadRecentProjects();
});

async function loadStats() {
  try {
    const stats = await Api.getStats();
    document.getElementById('stat-total-projects').textContent = stats.total_projects;
    document.getElementById('stat-completed-videos').textContent = stats.completed_videos;
    document.getElementById('stat-processing-jobs').textContent = stats.processing_jobs;
    document.getElementById('stat-failed-jobs').textContent = stats.failed_jobs;
  } catch (err) {
    console.warn('Failed to load stats:', err);
  }
}

async function loadRecentProjects() {
  const tbody = document.getElementById('projects-table-body');
  if (!tbody) return;

  try {
    const data = await Api.getProjects();
    const projects = data.projects || [];

    if (projects.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align:center; padding: 40px; color: var(--text-muted);">
            No video projects found. Click <strong>Create New Video</strong> to get started.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = projects.map(p => {
      const dateStr = p.created_at ? new Date(p.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Just now';
      const statusBadge = p.status === 'succeeded'
        ? '<span class="badge badge-success">Completed</span>'
        : (p.status === 'running' || p.status === 'queued')
        ? `<span class="badge badge-warning">${p.status === 'running' ? 'Processing ' + p.progress + '%' : 'Queued'}</span>`
        : '<span class="badge badge-danger">Failed</span>';

      const ratiosHtml = (p.aspect_ratios || ['16:9'])
        .map(r => `<span class="aspect-pill">${r}</span>`)
        .join('');

      const actionLink = p.status === 'succeeded'
        ? `<a href="/results.html?job_id=${p.job_id}" class="btn btn-outline" style="padding: 4px 10px; font-size: 0.8rem;">View Results</a>`
        : (p.status === 'running' || p.status === 'queued')
        ? `<a href="/processing.html?job_id=${p.job_id}" class="btn btn-secondary" style="padding: 4px 10px; font-size: 0.8rem;">View Progress</a>`
        : `<span style="font-size: 0.8rem; color: var(--text-light);">-</span>`;

      return `
        <tr>
          <td>
            <div style="display:flex; align-items:center; gap:12px;">
              <img src="${p.thumbnail_url || '/static/sample_thumb.jpg'}" alt="Thumb" class="project-thumb" onerror="this.src='/static/sample_thumb.jpg'">
              <div>
                <div class="project-name-cell">${escapeHtml(p.name)}</div>
                <div class="project-id-sub">${p.job_id}</div>
              </div>
            </div>
          </td>
          <td><span class="badge badge-neutral">${p.input_type}</span></td>
          <td style="color:var(--text-muted); font-size:0.86rem;">${dateStr}</td>
          <td style="font-family:monospace; font-size:0.86rem;">${p.duration}</td>
          <td>${ratiosHtml}</td>
          <td>${statusBadge}</td>
          <td>${actionLink}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load projects:', err);
    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--danger); text-align:center; padding:20px;">Failed to load project records.</td></tr>`;
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
