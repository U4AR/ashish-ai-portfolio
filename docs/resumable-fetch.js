const DB_NAME = 'ashish-portfolio-model-downloads';
const DB_VERSION = 1;
const STORE_NAME = 'chunks';
const CHUNK_SIZE = 8 * 1024 * 1024;
const MAX_RETRIES = 6;

let databasePromise = null;

export function installResumableFetch(env, onProgress) {
  const nativeFetch = globalThis.fetch.bind(globalThis);
  env.fetch = async (input, init = {}) => {
    const request = input instanceof Request ? input : null;
    const url = request?.url || String(input);
    const method = (init.method || request?.method || 'GET').toUpperCase();
    const hasRange = new Headers(init.headers || request?.headers).has('Range');
    if (method !== 'GET' || hasRange || !isExternalModelData(url) || !('indexedDB' in globalThis)) {
      return nativeFetch(input, init);
    }

    try {
      return await resumableResponse(url, nativeFetch, onProgress);
    } catch (error) {
      throw new Error(`Resumable model download paused: ${error?.message || error}. Saved chunks will be reused when you retry.`);
    }
  };
}

export async function clearResumableChunks(modelId) {
  if (!('indexedDB' in globalThis)) return;
  const db = await openDatabase();
  await new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.openCursor();
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      if (cursor.value.url.includes(`/${modelId}/`)) cursor.delete();
      cursor.continue();
    };
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
  });
}

async function resumableResponse(url, nativeFetch, onProgress) {
  const probe = await nativeFetch(url, {
    headers: { Range: 'bytes=0-0' },
    cache: 'no-store',
  });
  if (probe.status !== 206) return nativeFetch(url, { cache: 'no-store' });

  const range = probe.headers.get('Content-Range');
  const total = Number(range?.match(/\/(\d+)$/)?.[1]);
  if (!Number.isFinite(total) || total <= 0) return nativeFetch(url, { cache: 'no-store' });
  const etag = probe.headers.get('ETag') || '';
  const count = Math.ceil(total / CHUNK_SIZE);
  const db = await openDatabase();
  let completedBytes = 0;
  const saved = new Map();

  for (let index = 0; index < count; index += 1) {
    const start = index * CHUNK_SIZE;
    const end = Math.min(total - 1, start + CHUNK_SIZE - 1);
    const expected = end - start + 1;
    const existing = await getRecord(db, chunkId(url, index));
    if (existing?.etag === etag && existing?.total === total && existing?.blob?.size === expected) {
      saved.set(index, existing);
      completedBytes += expected;
    }
  }
  const resumedBytes = completedBytes;
  if (resumedBytes > 0) report(onProgress, url, completedBytes, total, resumedBytes);

  for (let index = 0; index < count; index += 1) {
    const start = index * CHUNK_SIZE;
    const end = Math.min(total - 1, start + CHUNK_SIZE - 1);
    const expected = end - start + 1;
    const id = chunkId(url, index);
    if (saved.has(index)) continue;

    const blob = await downloadChunk(nativeFetch, url, start, end, expected);
    await putRecord(db, { id, url, index, etag, total, blob, updatedAt: Date.now() });
    completedBytes += blob.size;
    report(onProgress, url, completedBytes, total, resumedBytes);
  }

  const blobs = [];
  for (let index = 0; index < count; index += 1) {
    const record = await getRecord(db, chunkId(url, index));
    if (!record?.blob) throw new Error(`saved chunk ${index + 1} is missing`);
    blobs.push(record.blob);
  }

  const body = new Blob(blobs, { type: 'application/octet-stream' });
  return new Response(body, {
    status: 200,
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(total),
      ETag: etag,
    },
  });
}

async function downloadChunk(nativeFetch, url, start, end, expected) {
  let lastError = null;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt += 1) {
    try {
      const response = await nativeFetch(url, {
        headers: { Range: `bytes=${start}-${end}` },
        cache: 'no-store',
      });
      if (response.status !== 206) throw new Error(`server returned HTTP ${response.status}`);
      const blob = await response.blob();
      if (blob.size !== expected) throw new Error(`received ${blob.size} of ${expected} bytes`);
      return blob;
    } catch (error) {
      lastError = error;
      if (attempt < MAX_RETRIES - 1) await delay(Math.min(16000, 1000 * (2 ** attempt)));
    }
  }
  throw new Error(`chunk ${start}-${end} failed after ${MAX_RETRIES} attempts: ${lastError?.message || lastError}`);
}

function report(onProgress, url, loaded, total, resumedBytes) {
  onProgress?.({
    file: url.split('/').pop(),
    loaded,
    total,
    progress: (loaded / total) * 100,
    resumedFrom: (resumedBytes / total) * 100,
  });
}

function isExternalModelData(url) {
  try {
    return new URL(url).pathname.endsWith('.onnx_data');
  } catch {
    return false;
  }
}

function chunkId(url, index) {
  return `${url}::${index}`;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function openDatabase() {
  if (databasePromise) return databasePromise;
  databasePromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return databasePromise;
}

function getRecord(db, id) {
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(id);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function putRecord(db, record) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).put(record);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
  });
}
