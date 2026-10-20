/**
 * IncuBrix AI B-Roll Studio — Global App Shell
 */

import { Api } from './api.js';

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  checkSystemStatus();
});

function initNavigation() {
  const currentPath = window.location.pathname.toLowerCase();
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item');

  navItems.forEach(item => {
    const href = item.getAttribute('href').toLowerCase();
    if (currentPath === href || (currentPath === '/' && href.includes('dashboard'))) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // Mobile menu button if present
  const toggleBtn = document.getElementById('menu-toggle');
  const sidebar = document.querySelector('.sidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
}

async function checkSystemStatus() {
  const statusEl = document.getElementById('system-status');
  if (!statusEl) return;
  try {
    const health = await Api.getHealth();
    if (health.status === 'healthy') {
      statusEl.innerHTML = '<span class="status-dot"></span><span>Engine Active</span>';
      statusEl.className = 'status-badge';
    } else {
      statusEl.innerHTML = '<span class="status-dot" style="background:#f59e0b"></span><span>Degraded</span>';
    }
  } catch (e) {
    statusEl.innerHTML = '<span class="status-dot" style="background:#ef4444"></span><span>Offline</span>';
  }
}
