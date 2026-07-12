import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;

const MODEL = 'onnx-community/gemma-4-E2B-it-qat-mobile-ONNX';
let processor = null;
let model = null;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      self.postMessage({
        type: 'progress',
        value: 1,
        message: 'Loading the Gemma 4 processor and preparing WebGPU…',
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
            : `Preparing ${update.file || 'Gemma 4 E2B QAT'}…`,
        });
      };

      processor = await AutoProcessor.from_pretrained(MODEL, { progress_callback });
      model = await Gemma4ForConditionalGeneration.from_pretrained(MODEL, {
        device: 'webgpu',
        progress_callback,
      });

      self.postMessage({ type: 'ready' });
      return;
    }

    if (data.type === 'generate') {
      if (!processor || !model) {
        throw new Error('The local model has not been loaded yet.');
      }

      const context = data.projects.map((project) => (
        `[${project.citation}] ${project.title} (${project.year}; ${project.context}; ${project.category})\n`
        + `${project.description}\n`
        + `Technologies: ${project.tags.join(', ')}\n`
        + `Source: ${project.url || 'portfolio record'}`
      )).join('\n\n');

      const messages = [
        {
          role: 'system',
          content: [{
            type: 'text',
            text: 'You answer questions about Ashish T Vasant only from the supplied project evidence. Be concise and factual. Cite every factual claim using bracketed project numbers such as [1]. If evidence is insufficient, say so. Do not invent employers, outcomes, metrics, links, or technologies.',
          }],
        },
        {
          role: 'user',
          content: [{
            type: 'text',
            text: `Question: ${data.question}\n\nRetrieved project evidence:\n${context}\n\nWrite a direct answer with citations, then a short Recommended projects list.`,
          }],
        },
      ];

      const prompt = processor.apply_chat_template(messages, {
        enable_thinking: false,
        add_generation_prompt: true,
      });
      const inputs = await processor(prompt, null, null, { add_special_tokens: false });
      const outputs = await model.generate({
        ...inputs,
        max_new_tokens: 280,
        do_sample: false,
      });
      const promptLength = inputs.input_ids.dims.at(-1);
      const decoded = processor.batch_decode(
        outputs.slice(null, [promptLength, null]),
        { skip_special_tokens: true },
      );

      self.postMessage({
        type: 'answer',
        text: String(decoded[0] || 'No answer was generated.').trim(),
      });
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error?.message || String(error) });
  }
};
