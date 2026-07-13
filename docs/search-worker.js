import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;

const MODEL = 'Xenova/all-MiniLM-L6-v2';
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

    const signature = data.projects.map((project) => project.index).join(',');
    if (!projectVectors || projectSignature !== signature) {
      self.postMessage({ type: 'status', requestId: data.requestId, message: 'Comparing all projects…' });
      const output = await extractor(
        data.projects.map((project) => project.text),
        { pooling: 'mean', normalize: true },
      );
      projectVectors = tensorRows(output);
      projectIndexes = data.projects.map((project) => project.index);
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
