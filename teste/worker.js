// Teste rápido (E-06): roda o Whisper fora da tela, para a página não travar.
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

const SR = 16000;
const MAX = 30 * SR;   // o Whisper lê no máximo 30 s por vez
const BUSCA = 4 * SR;  // procura a pausa nos últimos 4 s do trecho
const JANELA = SR / 10; // energia medida em janelas de 100 ms

let asr = null;
let loadedKey = '';

function dtypeFor(model, device) {
  if (device !== 'webgpu') return { encoder_model: 'q8', decoder_model_merged: 'q8' };
  // o encoder do turbo em fp32 passa de 2 GB; em fp16 cabe na placa
  return { encoder_model: model.includes('turbo') ? 'fp16' : 'fp32', decoder_model_merged: 'q4' };
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

// Idioma do arquivo inteiro, pelo primeiro trecho, como o app desktop faz.
// Sem idioma, o transformers.js assume inglês e traduz tudo: por isso a detecção é feita aqui.
async function detectarIdioma(trecho) {
  const gc = asr.model.generation_config;
  const { input_features } = await asr.processor(trecho);
  const ids = (await asr.model.generate({
    inputs: input_features, decoder_input_ids: [gc.decoder_start_token_id], max_new_tokens: 1,
  })).tolist()[0];
  const id = Number(ids[ids.length - 1]);
  const token = Object.keys(gc.lang_to_id).find((k) => gc.lang_to_id[k] === id);
  return token || '<|pt|>'; // ex.: "<|pt|>"
}

// O Whisper às vezes entra em laço ("network network network…"). Corta a repetição.
function semLaco(texto) {
  return texto.replace(/(\b[\p{L}\p{N}'’-]+(?:\s+[\p{L}\p{N}'’-]+){0,3})(?:[\s,.]+\1\b){3,}/giu, '$1');
}

// Trecho quase sem som: o Whisper inventa texto ("Música", "E aí"). O desktop usa VAD para isso.
function quaseSilencio(trecho) {
  let e = 0;
  for (let i = 0; i < trecho.length; i++) e += trecho[i] * trecho[i];
  return Math.sqrt(e / trecho.length) < 0.002;
}

// Transcreve um trecho dando ao Whisper o texto anterior como contexto, como o desktop
// (faster-whisper) faz. Sem isso, as citações em inglês eram puladas ou traduzidas.
async function transcreverComContexto(trecho, idioma, anterior) {
  const gc = asr.model.generation_config;
  const { input_features } = await asr.processor(trecho);
  const prev = anterior ? Array.from(asr.tokenizer.encode(' ' + anterior, { add_special_tokens: false })).slice(-160) : [];
  const init = [
    ...(prev.length ? [gc.prev_sot_token_id, ...prev] : []),
    gc.decoder_start_token_id, gc.lang_to_id[idioma], gc.task_to_id.transcribe, gc.no_timestamps_token_id,
  ];
  const ids = (await asr.model.generate({
    inputs: input_features, decoder_input_ids: init, max_new_tokens: 440 - init.length,
  })).tolist()[0].map(Number);
  return asr.tokenizer.decode(ids.slice(init.length), { skip_special_tokens: true }).trim();
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
        dtype: dtypeFor(model, device),
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
    const idioma = language ? '<|' + { portuguese: 'pt', english: 'en' }[language] + '|>'
      : await detectarIdioma(audio.slice(partes[0][0], partes[0][1]));
    let contexto = '';
    for (let i = 0; i < partes.length; i++) {
      const [a, b] = partes[i];
      const trecho = audio.slice(a, b);
      const bruto = quaseSilencio(trecho) ? '' : await transcreverComContexto(trecho, idioma, contexto);
      const limpo = semLaco(bruto);
      textos.push(limpo);
      contexto = limpo === bruto ? limpo : ''; // texto em laço não vira contexto, para o laço não se espalhar
      self.postMessage({ type: 'chunk', done: i + 1, total: partes.length, ms: performance.now() - t1, text: textos[i] });
    }
    self.postMessage({ type: 'done', loadMs, runMs: performance.now() - t1, idioma, text: textos.join('\n') });
  } catch (e) {
    self.postMessage({ type: 'error', message: String((e && e.message) || e) });
  }
};
