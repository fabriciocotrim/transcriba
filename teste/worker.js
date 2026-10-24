// Teste rápido (E-06): roda o Whisper fora da tela, para a página não travar.
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

const SR = 16000;
const CHUNK = 30 * SR; // trechos de 30 s, como o Whisper espera

let asr = null;
let loadedKey = '';

function dtypeFor(device) {
  return device === 'webgpu'
    ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
    : { encoder_model: 'q8', decoder_model_merged: 'q8' };
}

self.onmessage = async ({ data }) => {
  const { model, device, audio, language } = data;
  try {
    const key = model + '|' + device;
    let loadMs = 0;
    if (key !== loadedKey) {
      const t0 = performance.now();
      asr = await pipeline('automatic-speech-recognition', model, {
        device,
        dtype: dtypeFor(device),
        progress_callback: (p) => {
          if (p.status === 'progress') self.postMessage({ type: 'download', file: p.file, loaded: p.loaded, total: p.total });
        },
      });
      loadMs = performance.now() - t0;
      loadedKey = key;
    }
    const total = Math.ceil(audio.length / CHUNK);
    const parts = [];
    const t1 = performance.now();
    for (let i = 0; i < total; i++) {
      const piece = audio.subarray(i * CHUNK, Math.min(audio.length, (i + 1) * CHUNK));
      const out = await asr(piece, { language, task: 'transcribe', max_new_tokens: 224 });
      parts.push((out.text || '').trim());
      self.postMessage({ type: 'chunk', done: i + 1, total, ms: performance.now() - t1, text: parts[i] });
    }
    self.postMessage({ type: 'done', loadMs, runMs: performance.now() - t1, text: parts.join('\n') });
  } catch (e) {
    self.postMessage({ type: 'error', message: String((e && e.message) || e) });
  }
};
