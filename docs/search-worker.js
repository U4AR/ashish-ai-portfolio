import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;

const MODEL = 'Xenova/all-MiniLM-L6-v2';
const INDEX_DB = 'ashish-portfolio-search-index';
const INDEX_STORE = 'indexes';
const INDEX_TTL = 24 * 60 * 60 * 1000;
let extractor = null;
let projectVectors = null;
let projectIndexes = null;
let projectSignature = '';

self.onmessage = async ({ data }) => {
  if (data.type !== 'search') return;
  try {
    if (!extractor) {
      self.postMessage({ type: 'status', requestId: data.requestId, message: 'Getting best-match search ready…' });
      extractor = await pipeline('feature-extraction', MODEL, {
        device: 'wasm',
        dtype: 'q8',
        progress_callback: (update) => {
          if (update.status === 'progress' || update.status === 'progress_total') {
            self.postMessage({
              type: 'status',
              requestId: data.requestId,
              message: `Getting best-match search ready: ${Math.round(update.progress || 0)}%`,
            });
          }
        },
      });
    }

    const signature = hashText(data.projects.map((project) => `${project.index}:${project.text}`).join('\u0001'));
    if (!projectVectors || projectSignature !== signature) {
      const cached = await readCachedIndex(signature);
      if (cached) {
        self.postMessage({ type: 'status', requestId: data.requestId, message: 'Using the saved project index…' });
        projectVectors = cached.vectors.map((vector) => (
          vector instanceof Float32Array ? vector : new Float32Array(vector)
        ));
        projectIndexes = cached.indexes;
      } else {
        self.postMessage({ type: 'status', requestId: data.requestId, message: 'Comparing all projects…' });
        const output = await extractor(
          data.projects.map((project) => project.text),
          { pooling: 'mean', normalize: true },
        );
        projectVectors = tensorRows(output);
        projectIndexes = data.projects.map((project) => project.index);
        await saveCachedIndex(signature, projectVectors, projectIndexes);
      }
      projectSignature = signature;
    }

    const queryOutput = await extractor(data.query, { pooling: 'mean', normalize: true });
    const queryVector = tensorRows(queryOutput)[0];
    const scores = projectVectors.map((vector, row) => ({
      index: projectIndexes[row],
      score: dot(vector, queryVector),
    })).sort((a, b) => b.score - a.score);

    self.postMessage({ type: 'results', requestId: data.requestId, scores });
  } catch (error) {
    self.postMessage({
      type: 'error',
      requestId: data.requestId,
      message: error?.message || String(error),
    });
  }
};

function tensorRows(tensor) {
  const rows = tensor.dims.length === 1 ? 1 : tensor.dims[0];
  const width = tensor.data.length / rows;
  return Array.from({ length: rows }, (_, row) => (
    Float32Array.from(tensor.data.slice(row * width, (row + 1) * width))
  ));
}

function dot(left, right) {
  let value = 0;
  for (let index = 0; index < left.length; index += 1) value += left[index] * right[index];
  return value;
}

function hashText(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return String(hash >>> 0);
}

async function readCachedIndex(signature) {
  try {
    const database = await openIndexDatabase();
    const value = await new Promise((resolve, reject) => {
      const request = database.transaction(INDEX_STORE, 'readonly').objectStore(INDEX_STORE).get('projects');
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
    database.close();
    if (!value || value.signature !== signature || Date.now() - value.savedAt > INDEX_TTL) return null;
    return value;
  } catch {
    return null;
  }
}

async function saveCachedIndex(signature, vectors, indexes) {
  try {
    const database = await openIndexDatabase();
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(INDEX_STORE, 'readwrite');
      transaction.objectStore(INDEX_STORE).put({
        key: 'projects',
        signature,
        savedAt: Date.now(),
        vectors,
        indexes,
      });
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
    });
    database.close();
  } catch {}
}

function openIndexDatabase() {
  if (!('indexedDB' in self)) return Promise.reject(new Error('IndexedDB unavailable'));
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(INDEX_DB, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(INDEX_STORE)) {
        request.result.createObjectStore(INDEX_STORE, { keyPath: 'key' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
