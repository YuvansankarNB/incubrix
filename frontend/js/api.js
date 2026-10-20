/**
 * IncuBrix AI B-Roll Studio — API Client
 * Wraps Fetch API for all backend communication under /api/v1
 */

const API_BASE = '/api/v1';

function generateIdempotencyKey() {
  return 'idemp_' + Math.random().toString(36).substring(2, 15) + '_' + Date.now();
}

export const Api = {
  async getHealth() {
    const res = await fetch('/health');
    return res.json();
  },

  async getStats() {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) throw new Error('Failed to fetch metrics');
    return res.json();
  },

  async getProjects() {
    const res = await fetch(`${API_BASE}/projects`);
    if (!res.ok) throw new Error('Failed to fetch projects');
    return res.json();
  },

  async getJob(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Job ${jobId} not found`);
    }
    return res.json();
  },

  async getJobResult(jobId) {
    const res = await fetch(`${API_BASE}/broll/jobs/${jobId}/result`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Failed to fetch results for job ${jobId}`);
    }
    return res.json();
  },

  async createJob(payload) {
    const res = await fetch(`${API_BASE}/broll/jobs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Idempotency-Key': generateIdempotencyKey()
      },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error?.message || 'Failed to submit video job');
    }
    return data;
  },

  async createJobUpload(formData) {
    const res = await fetch(`${API_BASE}/broll/jobs/upload`, {
      method: 'POST',
      headers: {
        'Idempotency-Key': generateIdempotencyKey()
      },
      body: formData
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error?.message || 'Failed to upload audio job');
    }
    return data;
  },

  async replaceScene(jobId, sceneId, newQuery, preferredSource = 'auto') {
    const res = await fetch(`${API_BASE}/broll/jobs/${jobId}/scenes/${sceneId}/replace`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        new_query: newQuery,
        preferred_source: preferredSource
      })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error?.message || 'Failed to replace scene clip');
    }
    return data;
  },

  async getSettings() {
    const res = await fetch(`${API_BASE}/settings`);
    if (!res.ok) throw new Error('Failed to fetch settings');
    return res.json();
  },

  async generateScenePlan(payload) {
    const res = await fetch(`${API_BASE}/scene-plan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error?.message || 'Failed to generate scene plan');
    }
    return data;
  },

  // ── Pixabay Stock Video Search ───────────────────────────────
  async searchPixabayVideos({ q, page = 1, perPage = 12, aspectRatio = '16:9', targetDuration = 5.0, editorsChoice = false, jobId = null } = {}) {
    const params = new URLSearchParams({
      q,
      page,
      per_page: perPage,
      aspect_ratio: aspectRatio,
      target_duration: targetDuration,
      editors_choice: editorsChoice,
    });
    if (jobId) params.set('job_id', jobId);
    const res = await fetch(`${API_BASE}/stock/pixabay/videos?${params}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || 'Pixabay search failed');
    }
    return res.json();
  },

  async searchSceneClips(jobId, sceneId, { query, perPage = 12, aspectRatio, targetDuration, editorsChoice = false } = {}) {
    const body = { per_page: perPage, editors_choice: editorsChoice };
    if (query) body.query = query;
    if (aspectRatio) body.aspect_ratio = aspectRatio;
    if (targetDuration) body.target_duration = targetDuration;

    const res = await fetch(`${API_BASE}/jobs/${jobId}/scenes/${sceneId}/search-clips`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error?.message || 'Scene clip search failed');
    return data;
  },

  async replaceSceneClip(jobId, sceneId, { newQuery, preferredSource = 'pixabay', pixabayVideoId } = {}) {
    const body = { preferred_source: preferredSource };
    if (newQuery) body.new_query = newQuery;
    if (pixabayVideoId) body.pixabay_video_id = pixabayVideoId;

    const res = await fetch(`${API_BASE}/jobs/${jobId}/scenes/${sceneId}/replace-clip`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error?.message || 'Clip replacement failed');
    return data;
  },

  async getJobSources(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/sources`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Failed to fetch sources for job ${jobId}`);
    }
    return res.json();
  },
};
