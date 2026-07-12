import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;

const MODEL = 'onnx-community/gemma-3-1b-it-ONNX';
let generator = null;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      self.postMessage({
        type: 'progress',
        value: 1,
        message: 'Loading Gemma 3 1B and preparing WebGPU…',
      });

      const progress_callback = (update) => {
        const pct = update.progress
          ?? (update.loaded && update.total ? (update.loaded / update.total) * 100 : 0);
        const downloading = update.status === 'progress' || update.status === 'progress_total';
        self.postMessage({
          type: 'progress',
          value: pct,
          message: downloading
            ? `Downloading ${update.file || 'model data'}: ${Math.round(pct)}%`
            : `Preparing ${update.file || 'Gemma 3 1B q4f16'}…`,
        });
      };

      generator = await pipeline('text-generation', MODEL, {
        device: 'webgpu',
        dtype: 'q4f16',
        progress_callback,
      });

      self.postMessage({ type: 'ready' });
      return;
    }

    if (data.type === 'generate') {
      if (!generator) {
        throw new Error('The local model has not been loaded yet.');
      }

      const context = data.projects.map((project) => (
        `[${project.citation}] ${project.title} (${project.year}; ${project.context}; ${project.category})\n`
        + `${project.description}\n`
        + `Technologies: ${project.tags.join(', ')}\n`
        + `Source: ${project.url || 'portfolio record'}`
      )).join('\n\n');

      const messages = [{
        role: 'user',
        content: `Answer questions about Ashish T Vasant only from the supplied project evidence. Be concise and factual. Cite every factual claim using bracketed project numbers such as [1]. If evidence is insufficient, say so. Do not invent employers, outcomes, metrics, links, or technologies.\n\nQuestion: ${data.question}\n\nRetrieved project evidence:\n${context}\n\nWrite a direct answer with citations, then a short Recommended projects list.`,
      }];
      const result = await generator(messages, {
        max_new_tokens: 280,
        do_sample: false,
        return_full_text: false,
      });

      self.postMessage({
        type: 'answer',
        text: extractText(result),
      });
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error?.message || String(error) });
  }
};

function extractText(result) {
  let value = result?.[0]?.generated_text ?? result?.generated_text ?? result;
  if (Array.isArray(value)) {
    const last = value[value.length - 1];
    value = last?.content ?? last?.text ?? last;
  }
  if (Array.isArray(value)) value = value.map((item) => item?.text ?? String(item)).join('');
  if (typeof value === 'object') value = value?.text ?? value?.content ?? JSON.stringify(value);
  return String(value || 'No answer was generated.').trim();
}
