import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
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
        runGeneration = async (messages, options = {}) => {
          let streamed = '';
          const streamer = options.stream ? new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            callback_function: (text) => {
              streamed += text;
              self.postMessage({ type: 'token', requestId: options.requestId, text });
            },
          }) : undefined;
          const result = await generator(messages, {
          max_new_tokens: options.maxNewTokens || 320,
          do_sample: false,
          return_full_text: false,
          ...(streamer ? { streamer } : {}),
          });
          return streamed.trim() || extractText(result);
        };
      } else {
        const processor = await AutoProcessor.from_pretrained(selected.id, { progress_callback });
        const model = await Gemma4ForConditionalGeneration.from_pretrained(selected.id, {
          device: 'webgpu',
          dtype: 'q4f16',
          progress_callback,
        });
        runGeneration = async (messages, options = {}) => {
          const multimodalMessages = messages.map((message) => ({
            role: message.role,
            content: [{ type: 'text', text: message.content }],
          }));
          const prompt = processor.apply_chat_template(multimodalMessages, {
            enable_thinking: false,
            add_generation_prompt: true,
          });
          const inputs = await processor(prompt, null, null, { add_special_tokens: false });
          let streamed = '';
          const streamer = options.stream && processor.tokenizer ? new TextStreamer(processor.tokenizer, {
            skip_prompt: true,
            callback_function: (text) => {
              streamed += text;
              self.postMessage({ type: 'token', requestId: options.requestId, text });
            },
          }) : undefined;
          const outputs = await model.generate({
            ...inputs,
            max_new_tokens: options.maxNewTokens || 320,
            do_sample: false,
            ...(streamer ? { streamer } : {}),
          });
          return streamed.trim() || String(processor.batch_decode(
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

    if (data.type === 'plan') {
      if (!runGeneration) throw new Error('The local model has not been loaded yet.');
      const messages = [{
        role: 'user',
        content: `You are a tool-calling planner for Ashish T Vasant's portfolio. Rephrase the visitor's question into a concise project-search query, preserving names, technologies, dates and domains. Then call the only available tool. Return ONLY one JSON object and no Markdown or explanation.\n\nTool: search_projects(query: string)\nRequired format: {"tool":"search_projects","query":"concise search query"}\n\nVisitor question: ${data.question}`,
      }];
      const planText = await runGeneration(messages, { maxNewTokens: 54 });
      const toolCall = parseToolCall(planText, data.question);
      self.postMessage({ type: 'tool_call', requestId: data.requestId, ...toolCall });
      return;
    }

    if (data.type === 'generate') {
      if (!runGeneration) throw new Error('The local model has not been loaded yet.');

      const context = data.projects.map((project) => (
        `[${project.citation}] ${project.title} (${project.year}; ${project.context}; ${project.category})\n`
        + `${project.description}\n`
        + `Technologies: ${project.tags.join(', ')}\n`
        + `${project.details ? `Additional resume or project context: ${project.details}\n` : ''}`
        + `Source: ${project.url || 'portfolio record'}`
      )).join('\n\n');
      const messages = [{
        role: 'user',
        content: `Answer questions about Ashish T Vasant only from the supplied evidence.

Rules:
- Begin with a direct answer in 1 or 2 sentences.
- Start with the specific answer, not an introduction such as "Here is a summary".
- After the direct answer, add an "Evidence" list with no more than 3 relevant projects.
- Use at most 130 words including the evidence list.
- For a simple question, use plain paragraphs with no heading.
- Mention no more than the 3 most relevant projects unless the visitor explicitly asks for a longer or complete list.
- Prefer concrete project names and actions over broad claims about Ashish's background.
- Do not repeat the question, describe the search process, or add a generic summary.
- Use a short Markdown list only when it makes the answer clearer.
- Every factual project or resume claim must include its bracketed evidence number, for example [1].
- If the evidence is insufficient, say so briefly. Do not invent employers, outcomes, metrics, links, technologies or dates.

Question: ${data.question}

Retrieved evidence:
${context}

Required order: direct answer first, then the short cited evidence list.`,
      }];

      self.postMessage({ type: 'answer_start', requestId: data.requestId });
      const generated = await runGeneration(messages, {
        maxNewTokens: data.lowMemory ? 140 : 160,
        stream: true,
        requestId: data.requestId,
      });
      self.postMessage({
        type: 'answer',
        requestId: data.requestId,
        text: formatAnswerWithEvidence(generated, data.projects, data.question),
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

function ensureProjectCitations(text, projects) {
  let cited = text;
  for (const project of projects) {
    const marker = `[${project.citation}]`;
    if (cited.includes(marker)) continue;
    const escapedTitle = project.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    cited = cited.replace(new RegExp(escapedTitle, 'i'), (title) => `${title} ${marker}`);
  }
  return cited;
}

function formatAnswerWithEvidence(text, projects, question = '') {
  if (!projects.length) return 'The portfolio does not contain enough evidence to answer that question.';
  const specific = projects.filter((project) => !/resume|career profile/i.test(project.title));
  const selected = (specific.length ? specific : projects).slice(0, 3);
  const directAnswer = extractDirectAnswer(text, question, selected);
  return [
    directAnswer,
    '',
    '**Evidence**',
    '',
    ...selected.map((project) => `- **${project.title}** — ${project.description} [${project.citation}]`),
  ].join('\n');
}

function extractDirectAnswer(text, question, projects) {
  const citations = projects.map((project) => `[${project.citation}]`).join('');
  let cleaned = ensureProjectCitations(String(text || ''), projects)
    .replace(/\r/g, '')
    .split(/\n\s*(?:#{1,4}\s*)?(?:\*\*)?(?:evidence|relevant projects|matching projects|projects)(?:\*\*)?\s*:?/i)[0]
    .split(/\n\s*[-*]\s+/)[0]
    .replace(/here(?:'s| is) (?:a )?(?:concise )?summary[^.!?]*[.!?]/gi, '')
    .replace(/\[(?:\d+\s*,\s*)+\d+\]|\[\d+\]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  const sentences = cleaned.match(/[^.!?]+[.!?]+/g) || [];
  cleaned = sentences.slice(0, 2).join(' ').trim();
  const wordCount = cleaned.split(/\s+/).filter(Boolean).length;
  const vague = /based on the (?:provided|supplied) evidence|strong background|extensive experience|proven track record|various domains/i.test(cleaned);
  if (!cleaned || wordCount < 5 || wordCount > 60 || vague) {
    cleaned = buildQuestionLead(question, projects);
  }
  return `${cleaned.replace(/\s+$/, '')} ${citations}`.trim();
}

function buildQuestionLead(question, projects) {
  if (/\b(compare|comparison|difference|versus|vs\.?)\b/i.test(question) && projects.length >= 2) {
    return `${projects[0].title} is ${lowercaseFirst(projects[0].description)} ${projects[1].title}, by contrast, is ${lowercaseFirst(projects[1].description)}`;
  }
  const experience = question.match(/\bwhat\s+(.{2,60}?)\s+experience\s+(?:does|do)\b/i)?.[1]?.trim();
  const titles = projects.map((project) => project.title).join(', ').replace(/, ([^,]*)$/, ' and $1');
  if (experience) return `Ashish has practical ${experience} experience demonstrated through ${titles}.`;
  return `Ashish's most relevant work for this question includes ${titles}.`;
}

function lowercaseFirst(value) {
  const text = String(value || '').trim();
  return text ? `${text.charAt(0).toLowerCase()}${text.slice(1)}` : '';
}

function parseToolCall(text, originalQuestion) {
  const objectMatch = String(text || '').match(/\{[\s\S]*?\}/);
  if (objectMatch) {
    try {
      const parsed = JSON.parse(objectMatch[0]);
      const query = String(parsed.query || '').trim();
      if (parsed.tool === 'search_projects' && query) {
        return { tool: 'search_projects', query: query.slice(0, 220) };
      }
    } catch {}
  }
  const cleaned = String(text || '')
    .replace(/```(?:json)?|```/gi, '')
    .replace(/search_projects|tool|query/gi, '')
    .replace(/[{}\[\]":]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return {
    tool: 'search_projects',
    query: (cleaned.length >= 3 && cleaned.length <= 220 ? cleaned : originalQuestion).slice(0, 220),
  };
}
