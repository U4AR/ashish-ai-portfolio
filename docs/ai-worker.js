import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  env,
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
import { clearResumableChunks, installResumableFetch } from './resumable-fetch.js?v=20260712-2';

env.allowLocalModels = false;
installResumableFetch(env, (update) => {
  const pct = Math.round(update.progress || 0);
  const resumed = Math.round(update.resumedFrom || 0);
  self.postMessage({
    type: 'progress',
    value: pct,
    message: `${resumed ? `Resumed from ${resumed}%; saving new chunks` : 'Saving resumable chunks'}: ${update.file} — ${pct}%`,
  });
});

const MODELS = {
  small: {
    id: 'onnx-community/gemma-3-1b-it-ONNX',
    label: 'Gemma 3 1B q4f16',
    kind: 'text-generation',
  },
  large: {
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    label: 'Gemma 4 E2B q4f16',
    kind: 'gemma4',
  },
};

let runGeneration = null;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const selected = MODELS[data.model] || MODELS.small;
      self.postMessage({
        type: 'progress',
        value: 1,
        message: `Loading ${selected.label} and preparing WebGPU…`,
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
            : `Preparing ${update.file || selected.label}…`,
        });
      };

      if (selected.kind === 'text-generation') {
        const generator = await pipeline('text-generation', selected.id, {
          device: 'webgpu',
          dtype: 'q4f16',
          progress_callback,
        });
        runGeneration = async (messages) => extractText(await generator(messages, {
          max_new_tokens: 280,
          do_sample: false,
          return_full_text: false,
        }));
      } else {
        const processor = await AutoProcessor.from_pretrained(selected.id, { progress_callback });
        const model = await Gemma4ForConditionalGeneration.from_pretrained(selected.id, {
          device: 'webgpu',
          dtype: 'q4f16',
          progress_callback,
        });
        runGeneration = async (messages) => {
          const multimodalMessages = messages.map((message) => ({
            role: message.role,
            content: [{ type: 'text', text: message.content }],
          }));
          const prompt = processor.apply_chat_template(multimodalMessages, {
            enable_thinking: false,
            add_generation_prompt: true,
          });
          const inputs = await processor(prompt, null, null, { add_special_tokens: false });
          const outputs = await model.generate({
            ...inputs,
            max_new_tokens: 280,
            do_sample: false,
          });
          return String(processor.batch_decode(
            outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
            { skip_special_tokens: true },
          )[0] || 'No answer was generated.').trim();
        };
      }

      self.postMessage({ type: 'progress', value: 100, message: 'Finalizing the persistent model cache…' });
      await clearResumableChunks(selected.id);
      self.postMessage({ type: 'ready', label: selected.label });
      return;
    }

    if (data.type === 'generate') {
      if (!runGeneration) throw new Error('The local model has not been loaded yet.');

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

      self.postMessage({ type: 'answer', text: await runGeneration(messages) });
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
