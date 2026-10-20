/**
 * IncuBrix AI B-Roll Studio — Video Creation Wizard Controller
 */

import { Api } from './api.js';
import { Validators } from './validators.js';

let currentStep = 1;
let selectedInputType = 'script'; // 'audio', 'script', 'scenes', 'topic'
let uploadedFile = null;
let cachedScenePlan = null;

document.addEventListener('DOMContentLoaded', () => {
  initInputTypeSelection();
  initStepNavigation();
  initScriptCounter();
  initFileDropzone();
  initSceneBuilder();
  initPresets();
  initScenePlanPreview();
  initCharacterVoiceSync();
});

function initCharacterVoiceSync() {
  const charSelect = document.getElementById('character-select');
  const voiceGenderSelect = document.getElementById('voice-gender-select');
  const voiceSelect = document.getElementById('voice-select');

  if (!charSelect) return;

  const maleRoles = ['doctor', 'patient', 'executive', 'narrator', 'farmer', 'engineer'];
  const femaleRoles = ['female_doctor', 'female_patient', 'teacher', 'student'];

  charSelect.addEventListener('change', () => {
    const val = charSelect.value;
    if (val === 'doctor, patient') {
      if (voiceGenderSelect) voiceGenderSelect.value = 'auto';
    } else if (femaleRoles.includes(val)) {
      if (voiceGenderSelect) voiceGenderSelect.value = 'female';
      if (voiceSelect && (voiceSelect.value.includes('Christopher') || voiceSelect.value.includes('Guy') || voiceSelect.value.includes('en-IN-male') || voiceSelect.value.includes('hi-IN-male') || voiceSelect.value.includes('ta-IN-male'))) {
        voiceSelect.value = 'auto';
      }
    } else if (maleRoles.includes(val)) {
      if (voiceGenderSelect) voiceGenderSelect.value = 'male';
      if (voiceSelect && (voiceSelect.value.includes('Jenny') || voiceSelect.value.includes('Aria') || voiceSelect.value.includes('en-IN-female') || voiceSelect.value.includes('hi-IN-female') || voiceSelect.value.includes('ta-IN-female'))) {
        voiceSelect.value = 'auto';
      }
    }
  });

  if (voiceGenderSelect) {
    voiceGenderSelect.addEventListener('change', () => {
      const g = voiceGenderSelect.value;
      if (voiceSelect && voiceSelect.value !== 'auto') {
        if (g === 'male' && (voiceSelect.value.includes('Jenny') || voiceSelect.value.includes('Aria') || voiceSelect.value.includes('Ava') || voiceSelect.value.includes('female'))) {
          voiceSelect.value = 'en-US-ChristopherNeural';
        } else if (g === 'female' && (voiceSelect.value.includes('Christopher') || voiceSelect.value.includes('Guy') || voiceSelect.value.includes('male'))) {
          voiceSelect.value = 'en-US-JennyNeural';
        }
      }
    });
  }
}

function initInputTypeSelection() {
  const cards = document.querySelectorAll('.type-card');
  cards.forEach(card => {
    card.addEventListener('click', () => {
      cards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      selectedInputType = card.dataset.type;
    });
  });
}

function initStepNavigation() {
  const btnNext = document.getElementById('btn-next-step');
  const btnBack = document.getElementById('btn-back-step');
  const btnSubmit = document.getElementById('btn-submit-job');

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      if (validateStep(currentStep)) {
        goToStep(currentStep + 1);
      }
    });
  }

  if (btnBack) {
    btnBack.addEventListener('click', () => {
      goToStep(currentStep - 1);
    });
  }

  if (btnSubmit) {
    btnSubmit.addEventListener('click', handleJobSubmission);
  }
}

function goToStep(step) {
  currentStep = step;

  // Update wizard progress indicators
  document.querySelectorAll('.step-item').forEach((item, idx) => {
    const s = idx + 1;
    item.classList.remove('active', 'completed');
    if (s < currentStep) item.classList.add('completed');
    if (s === currentStep) item.classList.add('active');
  });

  // Switch step panels
  document.querySelectorAll('.wizard-step-panel').forEach(panel => {
    panel.style.display = 'none';
  });

  const activePanel = document.getElementById(`step-${currentStep}-panel`);
  if (activePanel) activePanel.style.display = 'block';

  // Toggle buttons
  const btnBack = document.getElementById('btn-back-step');
  const btnNext = document.getElementById('btn-next-step');
  const btnSubmit = document.getElementById('btn-submit-job');

  if (btnBack) btnBack.style.display = currentStep > 1 ? 'inline-flex' : 'none';
  if (btnNext) btnNext.style.display = currentStep < 4 ? 'inline-flex' : 'none';
  if (btnSubmit) btnSubmit.style.display = currentStep === 4 ? 'inline-flex' : 'none';

  if (currentStep === 2) {
    document.getElementById('input-section-script').style.display = selectedInputType === 'script' ? 'block' : 'none';
    document.getElementById('input-section-topic').style.display = selectedInputType === 'topic' ? 'block' : 'none';
    document.getElementById('input-section-audio').style.display = selectedInputType === 'audio' ? 'block' : 'none';
    document.getElementById('input-section-scenes').style.display = selectedInputType === 'scenes' ? 'block' : 'none';
  }

  if (currentStep === 4) {
    populateReviewSummary();
  }
}

function validateStep(step) {
  if (step === 1) {
    return true;
  }
  if (step === 2) {
    if (selectedInputType === 'script') {
      const script = document.getElementById('script-text').value;
      const res = Validators.validateScript(script);
      if (!res.valid) {
        alert(res.message);
        return false;
      }
    } else if (selectedInputType === 'topic') {
      const topic = document.getElementById('topic-text')?.value.trim();
      if (!topic || topic.length < 3) {
        alert('Please enter a topic or concept prompt with at least 3 characters.');
        return false;
      }
    } else if (selectedInputType === 'audio') {
      const url = document.getElementById('audio-url').value.trim();
      if (!uploadedFile && !url) {
        alert('Please either upload an audio file or provide a public audio URL.');
        return false;
      }
    } else if (selectedInputType === 'scenes') {
      const scenes = collectScenesData();
      const res = Validators.validateScenes(scenes);
      if (!res.valid) {
        alert(res.message);
        return false;
      }
    }
    return true;
  }
  if (step === 3) {
    const ratios = getSelectedRatios();
    if (ratios.length === 0) {
      alert('Please select at least one aspect ratio.');
      return false;
    }
    return true;
  }
  return true;
}

function initScriptCounter() {
  const textarea = document.getElementById('script-text');
  const stats = document.getElementById('script-stats');
  if (!textarea || !stats) return;

  const update = () => {
    const text = textarea.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const estSec = Math.round(words / 2.5);
    stats.textContent = `${words} words • ${textarea.value.length} chars • ~${estSec}s speech`;
  };

  textarea.addEventListener('input', update);
  update();
}

function initFileDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('audio-upload-input');
  const fileLabel = document.getElementById('dropzone-filename');
  const audioPreview = document.getElementById('audio-preview');

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) {
      handleAudioFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      handleAudioFile(fileInput.files[0]);
    }
  });

  function handleAudioFile(file) {
    const val = Validators.validateAudioFile(file);
    if (!val.valid) {
      alert(val.message);
      return;
    }
    uploadedFile = file;
    fileLabel.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    if (audioPreview) {
      audioPreview.src = URL.createObjectURL(file);
      audioPreview.style.display = 'block';
    }
  }
}

function initSceneBuilder() {
  const container = document.getElementById('scene-list-container');
  const btnAdd = document.getElementById('btn-add-scene');
  if (!container || !btnAdd) return;

  btnAdd.addEventListener('click', () => {
    const count = container.children.length + 1;
    const row = document.createElement('div');
    row.className = 'scene-row-card';
    row.innerHTML = `
      <div class="scene-num-badge">${count}</div>
      <div class="scene-inputs">
        <input type="text" class="form-control scene-text-input" placeholder="Narration text spoken in scene...">
        <input type="text" class="form-control scene-hint-input" placeholder="Visual search hint (e.g. Modern biotechnology laboratory)...">
      </div>
      <button type="button" class="btn btn-secondary btn-remove-scene" style="padding:6px 10px; color:var(--danger);">&times;</button>
    `;
    row.querySelector('.btn-remove-scene').addEventListener('click', () => {
      if (container.children.length > 1) {
        row.remove();
        updateSceneNumbers();
      }
    });
    container.appendChild(row);
  });

  function updateSceneNumbers() {
    container.querySelectorAll('.scene-row-card').forEach((row, i) => {
      row.querySelector('.scene-num-badge').textContent = i + 1;
    });
  }
}

function collectScenesData() {
  const container = document.getElementById('scene-list-container');
  if (!container) return [];
  const scenes = [];
  container.querySelectorAll('.scene-row-card').forEach(row => {
    const text = row.querySelector('.scene-text-input').value.trim();
    const hint = row.querySelector('.scene-hint-input').value.trim();
    if (text) {
      scenes.push({ text: text, visual_hint: hint || null });
    }
  });
  return scenes;
}

function getSelectedRatios() {
  const checked = [];
  if (document.getElementById('ratio-16-9')?.checked) checked.push('16:9');
  if (document.getElementById('ratio-9-16')?.checked) checked.push('9:16');
  if (document.getElementById('ratio-1-1')?.checked) checked.push('1:1');
  return checked;
}

function populateReviewSummary() {
  const sumInput = document.getElementById('summary-input');
  const sumRatios = document.getElementById('summary-ratios');
  const sumStyle = document.getElementById('summary-style');
  const sumAvoid = document.getElementById('summary-avoid');
  const sumCaptions = document.getElementById('summary-captions');

  if (selectedInputType === 'script') {
    const words = document.getElementById('script-text').value.trim().split(/\s+/).length;
    sumInput.textContent = `Text Script (${words} words)`;
  } else if (selectedInputType === 'topic') {
    const topic = document.getElementById('topic-text').value.trim();
    sumInput.textContent = `Topic Research: "${topic.substring(0, 40)}..."`;
  } else if (selectedInputType === 'audio') {
    sumInput.textContent = uploadedFile ? `Uploaded Audio: ${uploadedFile.name}` : `Audio URL`;
  } else {
    sumInput.textContent = `Structured Scene List (${collectScenesData().length} scenes)`;
  }

  const vPref = document.getElementById('visual-source-pref')?.value || 'hybrid';
  const vPrefText = vPref === 'ai_first' ? 'AI-Generated First' : (vPref === 'stock_first' ? 'Licensed Stock First' : 'Hybrid (AI + Stock)');
  sumRatios.textContent = `${getSelectedRatios().join(', ')} • ${vPrefText}`;
  sumStyle.textContent = document.getElementById('video-style').value;
  sumAvoid.textContent = document.getElementById('avoid-input').value || 'None';
  
  const burn = document.getElementById('burn-captions-check').checked;
  sumCaptions.textContent = burn ? 'Burned-in Captions + SRT' : 'SRT File Only';
}

async function handleJobSubmission() {
  const btn = document.getElementById('btn-submit-job');
  btn.disabled = true;
  btn.innerHTML = 'Submitting Job...';

  try {
    const ratios = getSelectedRatios();
    const style = document.getElementById('video-style').value;
    const avoidRaw = document.getElementById('avoid-input').value;
    const avoidList = avoidRaw ? avoidRaw.split(',').map(s => s.trim()).filter(Boolean) : [];
    const burnCaptions = document.getElementById('burn-captions-check').checked;
    const characterType = document.getElementById('character-select').value;
    const researchMode = document.getElementById('research-mode')?.value || 'script_plus_research';
    const isVoiceover = (document.getElementById('voiceover-mode')?.value === 'voiceover');
    const textStyle = document.getElementById('typography-style')?.value || 'context_aware';
    const visualSourcePref = document.getElementById('visual-source-pref')?.value || 'stock_first';
    const voiceGender = document.getElementById('voice-gender-select')?.value || 'auto';
    const durVal = document.getElementById('duration-select')?.value;
    const targetDuration = durVal && durVal !== 'auto' ? parseFloat(durVal) : null;

    let res;
    if (selectedInputType === 'audio' && uploadedFile) {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('aspect_ratios', ratios.join(','));
      formData.append('style', style);
      formData.append('avoid', avoidList.join(','));
      formData.append('burn_in_captions', burnCaptions);
      formData.append('character_type', characterType);
      formData.append('research_mode', researchMode);
      formData.append('is_voiceover', isVoiceover);
      res = await Api.createJobUpload(formData);
    } else {
      const payload = {
        script: selectedInputType === 'script' ? document.getElementById('script-text').value.trim() : null,
        topic: selectedInputType === 'topic' ? document.getElementById('topic-text').value.trim() : null,
        audio_url: selectedInputType === 'audio' ? document.getElementById('audio-url').value.trim() : null,
        scenes: selectedInputType === 'scenes' ? collectScenesData() : null,
        voice: document.getElementById('voice-select').value,
        voice_gender: voiceGender,
        target_duration: targetDuration,
        visual_source_preference: visualSourcePref,
        aspect_ratios: ratios,
        style: style,
        avoid: avoidList,
        captions: { burn_in: burnCaptions, srt: true },
        character_type: characterType,
        research_mode: researchMode,
        is_voiceover: isVoiceover,
        text_style: textStyle
      };
      res = await Api.createJob(payload);
    }

    // Redirect to real processing status page
    window.location.href = `/processing.html?job_id=${res.job_id}`;
  } catch (err) {
    alert(`Error: ${err.message}`);
    btn.disabled = false;
    btn.innerHTML = 'Generate Video';
  }
}

function initScenePlanPreview() {
  const btn = document.getElementById('btn-preview-plan');
  const container = document.getElementById('scene-plan-accordion');
  const list = document.getElementById('scene-plan-cards-list');
  const badge = document.getElementById('plan-scene-count-badge');
  if (!btn || !container || !list) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.innerHTML = 'Planning Scenes...';

    try {
      const durVal = document.getElementById('duration-select')?.value;
      const targetDuration = durVal && durVal !== 'auto' ? parseFloat(durVal) : null;
      const planPayload = {
        script: selectedInputType === 'script' ? document.getElementById('script-text').value.trim() : null,
        topic: selectedInputType === 'topic' ? document.getElementById('topic-text').value.trim() : null,
        scenes: selectedInputType === 'scenes' ? collectScenesData() : null,
        character_type: document.getElementById('character-select')?.value || 'human',
        style: document.getElementById('video-style')?.value || 'bright, modern',
        target_duration: targetDuration,
        research_mode: document.getElementById('research-mode')?.value || 'script_plus_research'
      };

      const result = await Api.generateScenePlan(planPayload);
      const scenes = result.scenes || [];
      cachedScenePlan = scenes;

      list.innerHTML = '';
      badge.textContent = `${scenes.length} Scenes Planned • ${result.total_duration}s`;

      scenes.forEach((sc, idx) => {
        const card = document.createElement('div');
        card.style.background = '#ffffff';
        card.style.border = '1px solid #cbd5e1';
        card.style.borderRadius = '8px';
        card.style.padding = '12px 14px';
        card.style.fontSize = '0.84rem';

        const keywordsStr = (sc.b_roll_search_keywords || []).join(', ');
        const overlay = sc.text_overlay_content || {};

        card.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <strong style="color:var(--primary); font-size:0.92rem;">Scene ${idx + 1} (${sc.duration}s) ${sc.speaker ? `— Speaker: ${sc.speaker}` : ''}</strong>
            <span style="background:#e0e7ff; color:#3730a3; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;">Time: ${sc.audio_timing?.start_time}s – ${sc.audio_timing?.end_time}s</span>
          </div>
          <div style="margin-bottom:6px;"><strong>Narration:</strong> <span style="color:#1e293b;">${sc.narration}</span></div>
          <div style="margin-bottom:6px;"><strong>Visual Description:</strong> <span style="color:#475569;">${sc.visual_description}</span></div>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:6px;">
            <div><strong>Character:</strong> ${sc.character_requirements}</div>
            <div><strong>Action / Expression:</strong> ${sc.character_actions_and_expressions}</div>
          </div>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:6px;">
            <div><strong>Environment:</strong> ${sc.environment_and_background}</div>
            <div><strong>Camera Kinematics:</strong> ${sc.camera_angle_and_movement}</div>
          </div>
          <div style="margin-bottom:6px; background:#f1f5f9; padding:6px 8px; border-radius:4px; font-size:0.8rem;">
            <strong>Veo Prompt:</strong> <em>${sc.video_generation_prompt}</em>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#64748b; flex-wrap:wrap; gap:6px;">
            <span><strong>B-roll Keywords:</strong> ${keywordsStr}</span>
            <span><strong>Overlay:</strong> ${overlay.headline || 'None'}</span>
            <span><strong>Licensing:</strong> Commercial Safe (Model release: ${sc.source_and_licensing_requirements?.model_release_required ? 'Yes' : 'No'})</span>
          </div>
        `;
        list.appendChild(card);
      });

      container.style.display = 'block';
    } catch (err) {
      alert(`Scene planning failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = '📋 Refresh Scene Plan';
    }
  });
}

function initPresets() {
  const presetCleanEnergy = document.getElementById('preset-clean-energy');
  const presetDoctor = document.getElementById('preset-doctor');
  const presetDialogue = document.getElementById('preset-dialogue');
  const presetTeacher = document.getElementById('preset-teacher');
  const presetFarmer = document.getElementById('preset-farmer');

  const presetTopicCardio = document.getElementById('preset-topic-cardio');
  const presetTopicTrees = document.getElementById('preset-topic-trees');
  const presetTopicAi = document.getElementById('preset-topic-ai');

  if (presetCleanEnergy) {
    presetCleanEnergy.addEventListener('click', () => {
      document.getElementById('script-text').value = 
        "Carbon emissions were reduced by 35% across all manufacturing plants. Next-generation product design relies entirely on biodegradable and circular materials. Lasting ecological change is achieved by fostering green literacy within local communities.";
      document.getElementById('character-select').value = 'engineer';
      document.getElementById('video-style').value = 'bright, modern, real-world footage';
      initScriptCounter();
    });
  }

  if (presetDoctor) {
    presetDoctor.addEventListener('click', () => {
      document.getElementById('script-text').value = 
        "Early clinical diagnosis remains the single most effective tool in patient recovery. Our hospital team utilizes advanced biometric scanning and real-time laboratory diagnostics to ensure comprehensive care.";
      document.getElementById('character-select').value = 'doctor';
      initScriptCounter();
    });
  }

  if (presetDialogue) {
    presetDialogue.addEventListener('click', () => {
      document.getElementById('script-text').value = 
        "Doctor: Hello David, your routine blood pressure readings are showing noticeable improvement this month.\nPatient: Thank you Doctor, I have been following the aerobic walking regimen every morning.";
      document.getElementById('character-select').value = 'doctor, patient';
      initScriptCounter();
    });
  }

  if (presetTeacher) {
    presetTeacher.addEventListener('click', () => {
      document.getElementById('script-text').value = 
        "In today's physics lesson, we explore how atmospheric convection currents drive global weather systems. Notice how warm air masses expand and rise, transferring thermal energy upward.";
      document.getElementById('character-select').value = 'teacher';
      initScriptCounter();
    });
  }

  if (presetFarmer) {
    presetFarmer.addEventListener('click', () => {
      document.getElementById('script-text').value = 
        "Regenerative agriculture begins directly beneath our boots in the living soil. By reintroducing organic compost and cover crops, we restore soil biology and maximize moisture retention.";
      document.getElementById('character-select').value = 'farmer';
      initScriptCounter();
    });
  }

  if (presetTopicCardio) {
    presetTopicCardio.addEventListener('click', () => {
      document.getElementById('topic-text').value = "Cardiovascular health, hypertension lifestyle interventions, and aerobic exercise";
      document.getElementById('character-select').value = 'doctor';
    });
  }

  if (presetTopicTrees) {
    presetTopicTrees.addEventListener('click', () => {
      document.getElementById('topic-text').value = "Urban tree canopies and microclimate cooling in sustainable city infrastructure";
      document.getElementById('character-select').value = 'farmer';
    });
  }

  if (presetTopicAi) {
    presetTopicAi.addEventListener('click', () => {
      document.getElementById('topic-text').value = "Developer productivity benchmarks and AI-assisted software refactoring velocity";
      document.getElementById('character-select').value = 'engineer';
    });
  }
}
