// Teste rápido (E-06): roda o Whisper fora da tela, para a página não travar.
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

const SR = 16000;
const MAX = 30 * SR;   // o Whisper lê no máximo 30 s por vez
const BUSCA = 4 * SR;  // procura a pausa nos últimos 4 s do trecho
const JANELA = SR / 10; // energia medida em janelas de 100 ms

let asr = null;
let loadedKey = '';

function dtypeFor(device) {
  return device === 'webgpu'
    ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
    : { encoder_model: 'q8', decoder_model_merged: 'q8' };
}

// Corta na pausa mais silenciosa perto do limite de 30 s, para não partir palavra ao meio.
function cortes(audio) {
  const out = [];
  let ini = 0;
  while (audio.length - ini > MAX) {
    let melhor = ini + MAX, menor = Infinity;
    for (let p = ini + MAX - BUSCA; p + JANELA <= ini + MAX; p += JANELA) {
      let e = 0;
      for (let i = p; i < p + JANELA; i++) e += audio[i] * audio[i];
      if (e < menor) { menor = e; melhor = p + JANELA / 2; }
    }
    out.push([ini, melhor]);
    ini = melhor;
  }
  out.push([ini, audio.length]);
  return out;
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
    const partes = cortes(audio);
    const textos = [];
    const t1 = performance.now();
    for (let i = 0; i < partes.length; i++) {
      const [a, b] = partes[i];
      const opts = { task: 'transcribe' };
      if (language) opts.language = language; // vazio: o Whisper detecta, como no desktop
      const out = await asr(audio.slice(a, b), opts);
      textos.push((out.text || '').trim());
      self.postMessage({ type: 'chunk', done: i + 1, total: partes.length, ms: performance.now() - t1, text: textos[i] });
    }
    self.postMessage({ type: 'done', loadMs, runMs: performance.now() - t1, text: textos.join('\n') });
  } catch (e) {
    self.postMessage({ type: 'error', message: String((e && e.message) || e) });
  }
};
