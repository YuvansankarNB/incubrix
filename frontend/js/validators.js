/**
 * IncuBrix AI B-Roll Studio — Validation Routines
 */

export const Validators = {
  validateScript(script) {
    if (!script || !script.trim()) {
      return { valid: false, message: 'Script text cannot be empty.' };
    }
    const words = script.trim().split(/\s+/).filter(Boolean);
    if (words.length < 3) {
      return { valid: false, message: 'Script must be at least 3 words long.' };
    }
    return { valid: true, wordCount: words.length, charCount: script.length };
  },

  validateAudioFile(file) {
    if (!file) {
      return { valid: false, message: 'Please select an audio file.' };
    }
    const allowed = ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/m4a', 'audio/x-m4a', 'audio/aac'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp3', 'wav', 'm4a', 'aac'].includes(ext) && !allowed.includes(file.type)) {
      return { valid: false, message: 'Only MP3, WAV, or M4A audio files are supported.' };
    }
    const maxSize = 100 * 1024 * 1024; // 100 MB
    if (file.size > maxSize) {
      return { valid: false, message: 'Audio file exceeds maximum size limit of 100 MB.' };
    }
    return { valid: true };
  },

  validateScenes(scenes) {
    if (!Array.isArray(scenes) || scenes.length === 0) {
      return { valid: false, message: 'At least one scene is required.' };
    }
    for (let i = 0; i < scenes.length; i++) {
      if (!scenes[i].text || !scenes[i].text.trim()) {
        return { valid: false, message: `Scene ${i + 1} narration text cannot be empty.` };
      }
    }
    return { valid: true };
  }
};
