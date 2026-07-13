(() => {
  const byId = (id) => document.getElementById(id);
  const form = byId('ragForm');
  const query = byId('ragQuery');
  const searchMode = byId('searchMode');
  const output = byId('ragOutput');
  const evidence = byId('ragEvidence');
  const semanticStatus = byId('semanticStatus');
  const openDialog = byId('openAnswerDialog');
  const dialog = byId('answerDialog');
  const closeDialog = byId('closeAnswerDialog');
  const headerAiChat = byId('headerAiChat');
  const chatForm = byId('chatForm');
  const chatQuery = byId('chatQuery');
  const chatSubmit = byId('chatSubmit');
  const chatProgress = byId('chatProgress');
  const chatStage = byId('chatStage');
  const chatLog = byId('chatLog');
  const chatToolResults = byId('chatToolResults');
  const answer = byId('ragAnswer');
  const enable = byId('enableAI');
  const choice = byId('localModelChoice');
  const status = byId('modelStatus');
  const progress = byId('modelProgress');
  const githubDialog = byId('githubDialog');
  const openGitHubDialog = byId('openGitHubDialog');
  const closeGitHubDialog = byId('closeGitHubDialog');

  let modelWorker = null;
  let searchWorker = null;
  let modelReady = false;
  let modelLoading = false;
  let retrieved = [];
  let chatRetrieved = [];
  let pendingQuestion = '';
  let currentQuestion = '';
  let chatToolQuery = '';
  let streamedAnswer = '';
  let streamRenderScheduled = false;
  let requestSequence = 0;
  let pageSearchRequestId = 0;
  let chatSearchRequestId = 0;
  let chatRequestId = 0;
  const CHAT_SEARCH_CACHE_KEY = 'ashish-portfolio-chat-search-v2';
  const ONE_DAY_MS = 24 * 60 * 60 * 1000;
  const lowMemoryMode = matchMedia('(max-width: 760px)').matches
    || (Number(navigator.deviceMemory || 8) <= 6);

  const stop = new Set([
    'the', 'a', 'an', 'and', 'or', 'for', 'of', 'to', 'in', 'on', 'with', 'what',
    'which', 'show', 'find', 'project', 'projects', 'work', 'ashish', 'experience',
    'has', 'have', 'built', 'did', 'about', 'vasant', 'who', 'is', 'does',
  ]);
  const expansions = {
    ai: ['llm', 'model', 'agent', 'vision', 'rag'],
    local: ['offline', 'on-device', 'privacy', 'gemma'],
    robotics: ['robot', 'drone', 'ros', 'slam', 'sensor', 'lidar'],
    language: ['multilingual', 'translation', 'malayalam', 'indian-language'],
    deployment: ['cloud', 'gcp', 'docker', 'api', 'server', 'on-premise'],
    voice: ['speech', 'audio', 'gemini live', 'keyboard'],
    evaluation: ['evals', 'benchmark', 'livebench', 'terminal-bench'],
    spatial: ['ar', 'vr', 'hololens', 'digital twin', 'unity'],
    resume: ['career', 'education', 'skills', 'bel', 'profile'],
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = query.value.trim();
    if (!text) return;
    retrieved = searchMode.value === 'keyword' ? exactKeywordRetrieve(text) : keywordRetrieve(text);
    showEvidence();
    if (searchMode.value === 'keyword') {
      semanticStatus.textContent = 'Showing matches for the words in your search.';
      return;
    }
    semanticStatus.textContent = 'Finding the closest matches…';
    runSemanticSearch(text, 'page');
  });

  function openChat(prefill = '') {
    dialog.showModal();
    if (prefill) chatQuery.value = prefill;
    requestAnimationFrame(() => chatQuery.focus());
  }

  openDialog.addEventListener('click', () => openChat(query.value.trim()));
  headerAiChat.addEventListener('click', () => {
    openChat();
    if (!modelReady && !modelLoading) requestAnimationFrame(() => startModelLoad());
  });
  closeDialog.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });
  openGitHubDialog.addEventListener('click', () => githubDialog.showModal());
  closeGitHubDialog.addEventListener('click', () => githubDialog.close());
  githubDialog.addEventListener('click', (event) => {
    if (event.target === githubDialog) githubDialog.close();
  });
  githubDialog.querySelector('.github-all-link').addEventListener('click', () => githubDialog.close());

  chatForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = chatQuery.value.trim();
    if (!text) return;
    beginChat(text);
  });
  enable.addEventListener('click', () => startModelLoad());

  function beginChat(questionText) {
    currentQuestion = questionText;
    pendingQuestion = questionText;
    chatRequestId = ++requestSequence;
    streamedAnswer = '';
    chatRetrieved = [];
    answer.hidden = true;
    answer.innerHTML = '';
    chatLog.innerHTML = '';
    chatToolResults.innerHTML = '';
    chatProgress.hidden = false;
    chatProgress.classList.remove('is-done');
    chatSubmit.disabled = true;
    addLog('Question received.');
    if (modelReady) {
      requestToolPlan(questionText);
    } else {
      setStage(modelLoading ? 'Waiting for AI setup…' : 'AI answers need to be enabled first…');
      startModelLoad(questionText);
    }
  }

  async function startModelLoad(questionAfterReady = '') {
    if (questionAfterReady) pendingQuestion = questionAfterReady;
    if (modelReady) {
      if (pendingQuestion) requestToolPlan(pendingQuestion);
      return;
    }
    if (modelLoading) return;
    if (!('gpu' in navigator)) {
      status.textContent = 'AI answers are not supported in this browser. Project search still works.';
      setStage('AI answers are unavailable in this browser.', true);
      chatSubmit.disabled = false;
      return;
    }

    modelLoading = true;
    enable.disabled = true;
    choice.disabled = true;
    enable.textContent = 'Checking saved AI…';
    status.textContent = 'Checking for the saved AI model…';

    const selected = choice.value;
    const size = selected === 'large' ? '3.38 GB' : '0.8 GB';
    const cached = await isModelCached(selected);
    if (!cached && !confirm(`AI answers need a one-time download of about ${size}. Continue?`)) {
      modelLoading = false;
      enable.disabled = false;
      choice.disabled = false;
      enable.textContent = 'Enable AI answers';
      status.textContent = 'AI answers are off until you enable them.';
      setStage('AI setup was not started.', true);
      chatSubmit.disabled = false;
      pendingQuestion = '';
      return;
    }

    navigator.storage?.persist?.().catch(() => {});
    enable.textContent = cached ? 'Opening saved AI…' : 'Preparing AI answers…';
    progress.hidden = false;
    status.textContent = cached
      ? 'Opening the saved AI model…'
      : 'Preparing AI answers. You can retry if the connection is interrupted.';
    setStage(cached ? 'Opening saved AI…' : 'Preparing AI answers…');
    addLog(cached ? 'Saved AI model found.' : 'AI setup started.');

    modelWorker = new Worker('ai-worker.js?v=20260713-4', { type: 'module' });
    modelWorker.onmessage = handleModelMessage;
    modelWorker.onerror = () => handleModelFailure();
    modelWorker.postMessage({ type: 'init', model: selected });
  }

  function handleModelMessage({ data }) {
    if (data.type === 'progress') {
      const pct = Math.max(0, Math.min(100, Math.round(data.value || 0)));
      progress.value = pct;
      status.textContent = `Preparing AI answers: ${pct}%`;
      setStage(`Preparing AI answers: ${pct}%`);
    }
    if (data.type === 'ready') {
      modelReady = true;
      modelLoading = false;
      progress.value = 100;
      progress.hidden = true;
      enable.textContent = 'AI answers enabled';
      status.textContent = 'AI answers are ready to use.';
      markModelCached(choice.value);
      addLog('AI answers ready.');
      if (pendingQuestion) requestToolPlan(pendingQuestion);
      else setStage('Ready for a question.', true);
    }
    if (data.type === 'tool_call' && data.requestId === chatRequestId) {
      chatToolQuery = data.query;
      addLog(`Question rephrased as “${data.query}”.`);
      addLog(`Search tool called: ${data.tool}.`);
      addLog(`Searching for “${data.query}”…`);
      setStage(`Searching projects for “${data.query}”…`);
      runFastChatSearch(data.query);
    }
    if (data.type === 'answer_start' && data.requestId === chatRequestId) {
      streamedAnswer = '';
      streamRenderScheduled = false;
      answer.hidden = false;
      answer.innerHTML = '<p>Starting answer…</p>';
      answer.classList.add('is-streaming');
      addLog('Answer generation started.');
      setStage('Writing the answer…');
    }
    if (data.type === 'token' && data.requestId === chatRequestId) {
      streamedAnswer += data.text || '';
      answer.hidden = false;
      scheduleStreamRender();
    }
    if (data.type === 'answer' && data.requestId === chatRequestId) {
      streamedAnswer = data.text || streamedAnswer;
      answer.hidden = false;
      answer.innerHTML = renderMarkdown(streamedAnswer);
      answer.classList.remove('is-streaming');
      addLog('Answer completed.');
      setStage('Answer ready.', true);
      chatSubmit.disabled = false;
      pendingQuestion = '';
    }
    if (data.type === 'error') {
      if (!modelReady) handleModelFailure();
      else {
        const fallbackMatches = chatRetrieved.length ? chatRetrieved : keywordRetrieve(currentQuestion).slice(0, 4);
        answer.hidden = false;
        answer.classList.remove('is-streaming');
        answer.innerHTML = renderEvidenceFallback(fallbackMatches);
        status.textContent = 'Showing the matched project evidence.';
        setStage('Search results ready. This device could not finish the generated summary.', true);
        addLog('Generated summary stopped; matched evidence is shown instead.');
        chatSubmit.disabled = false;
      }
    }
  }

  function handleModelFailure() {
    modelReady = false;
    modelLoading = false;
    status.textContent = 'Setup paused. Retry to continue where it stopped.';
    setStage('AI setup paused. Retry when ready.', true);
    addLog('AI setup paused.');
    enable.disabled = false;
    choice.disabled = false;
    enable.textContent = 'Retry AI setup';
    progress.hidden = true;
    chatSubmit.disabled = false;
    modelWorker?.terminate();
    modelWorker = null;
  }

  function requestToolPlan(questionText) {
    pendingQuestion = '';
    currentQuestion = questionText;
    setStage('Understanding and rephrasing your question…');
    addLog('Model is interpreting the question.');
    modelWorker.postMessage({ type: 'plan', requestId: chatRequestId, question: questionText });
  }

  function runSemanticSearch(text, purpose) {
    ensureSearchWorker();
    const requestId = ++requestSequence;
    if (purpose === 'chat') chatSearchRequestId = requestId;
    else pageSearchRequestId = requestId;
    searchWorker.postMessage({
      type: 'search',
      requestId,
      query: text,
      projects: projects.map((project, index) => ({
        index,
        text: `${project.title}. ${project.description}. ${project.details || ''}. ${project.broad}. ${project.category}. ${project.tags.join(', ')}. ${project.context}.`,
      })),
    });
  }

  function runFastChatSearch(text) {
    const saved = readSavedChatSearch(text);
    if (saved.length) {
      addLog('Loaded matching projects from the saved local index.');
      finishChatSearch(saved);
      return;
    }
    const matches = keywordRetrieve(text);
    saveChatSearch(text, matches);
    addLog('Fast local project index searched.');
    finishChatSearch(matches);
  }

  function ensureSearchWorker() {
    if (searchWorker) return;
    searchWorker = new Worker('search-worker.js?v=20260713-3', { type: 'module' });
    searchWorker.onmessage = ({ data }) => {
      if (data.type === 'status') {
        if (data.requestId === chatSearchRequestId) setStage(data.message);
        if (data.requestId === pageSearchRequestId) semanticStatus.textContent = data.message;
      }
      if (data.type === 'results') {
        const matches = data.scores.slice(0, 7).map(({ index, score }) => ({
          ...projects[index],
          relevance: Math.max(0, score * 100),
        }));
        if (data.requestId === chatSearchRequestId) finishChatSearch(matches);
        if (data.requestId === pageSearchRequestId) {
          retrieved = matches;
          semanticStatus.textContent = 'Best matches ready.';
          showEvidence();
        }
      }
      if (data.type === 'error') {
        if (data.requestId === chatSearchRequestId) finishChatSearch(keywordRetrieve(chatToolQuery || currentQuestion));
        if (data.requestId === pageSearchRequestId) {
          semanticStatus.textContent = 'Best-match search is unavailable right now. Word matches are shown instead.';
        }
      }
    };
    searchWorker.onerror = () => {
      if (chatSearchRequestId) finishChatSearch(keywordRetrieve(chatToolQuery || currentQuestion));
      semanticStatus.textContent = 'Best-match search is unavailable right now. Word matches are shown instead.';
      searchWorker = null;
    };
  }

  function finishChatSearch(matches) {
    chatSearchRequestId = 0;
    chatRetrieved = matches.slice(0, lowMemoryMode ? 4 : 7);
    addLog(`Search found ${chatRetrieved.length} relevant result${chatRetrieved.length === 1 ? '' : 's'}.`);
    renderChatResults(chatRetrieved);
    setStage(`Found ${chatRetrieved.length} results. Using them to write the answer…`);
    modelWorker.postMessage({
      type: 'generate',
      requestId: chatRequestId,
      question: currentQuestion,
      lowMemory: lowMemoryMode,
      projects: chatRetrieved.map((project, index) => ({
        citation: index + 1,
        title: project.title,
        description: project.description,
        details: (project.details || '').slice(0, lowMemoryMode ? 1400 : 3200),
        year: project.year,
        context: project.context,
        category: project.category,
        tags: project.tags,
        url: project.url,
      })),
    });
  }

  function renderChatResults(matches) {
    chatToolResults.innerHTML = matches.length
      ? `<strong>Search results</strong>${matches.map((project, index) => (
        `<div id="chat-result-${index + 1}" class="chat-tool-result"><span>[${index + 1}] ${escapeHtml(project.title)}</span>`
        + `${project.url ? `<a href="${escapeAttribute(project.url)}" target="_blank" rel="noreferrer">Open</a>` : ''}</div>`
      )).join('')}`
      : '<p>No strong project match was found.</p>';
  }

  function setStage(message, done = false) {
    chatProgress.hidden = false;
    chatProgress.classList.toggle('is-done', done);
    chatStage.querySelector('span:last-child').textContent = message;
  }

  function scheduleStreamRender() {
    if (streamRenderScheduled) return;
    streamRenderScheduled = true;
    requestAnimationFrame(() => {
      streamRenderScheduled = false;
      answer.innerHTML = renderMarkdown(streamedAnswer);
    });
  }

  function renderEvidenceFallback(matches) {
    const markdown = [
      '### Matching project evidence',
      '',
      ...matches.map((project, index) => (
        `- **${project.title}** [${index + 1}] — ${project.description}`
      )),
    ].join('\n');
    return renderMarkdown(markdown);
  }

  async function isModelCached(selected) {
    const modelPath = selected === 'large'
      ? 'onnx-community/gemma-4-E2B-it-ONNX'
      : 'onnx-community/gemma-3-1b-it-ONNX';
    try {
      if (!('caches' in window)) return hasRecentModelMarker(selected);
      const cache = await caches.open('transformers-cache');
      const requests = await cache.keys();
      const hasCompleteBulkWeights = requests.some((request) => {
        const url = decodeURIComponent(request.url);
        return url.includes(modelPath) && /\.onnx_data(?:\?|$)/.test(url);
      });
      return hasCompleteBulkWeights || hasRecentModelMarker(selected);
    } catch {
      return hasRecentModelMarker(selected);
    }
  }

  function hasRecentModelMarker(selected) {
    const savedAt = Number(localStorage.getItem(`ashish-ai-ready-${selected}`) || 0);
    return savedAt > 0 && Date.now() - savedAt < 30 * ONE_DAY_MS;
  }

  function markModelCached(selected) {
    try {
      localStorage.setItem(`ashish-ai-ready-${selected}`, String(Date.now()));
    } catch {}
  }

  function addLog(message) {
    const item = document.createElement('li');
    item.textContent = message;
    chatLog.appendChild(item);
  }

  function keywordRetrieve(text) {
    const base = tokenize(text);
    const terms = [...new Set(base.flatMap((term) => [term, ...(expansions[term] || [])]))];
    return projects.map((project) => {
      const fields = {
        title: project.title.toLowerCase(),
        tags: project.tags.join(' ').toLowerCase(),
        category: `${project.broad} ${project.category}`.toLowerCase(),
        description: `${project.description} ${project.details || ''}`.toLowerCase(),
      };
      let matchScore = 0;
      for (const term of terms) {
        if (fields.title.includes(term)) matchScore += 8;
        if (fields.tags.includes(term)) matchScore += 5;
        if (fields.category.includes(term)) matchScore += 4;
        if (fields.description.includes(term)) matchScore += 2;
      }
      if (fields.title.includes(text.toLowerCase())) matchScore += 12;
      return { ...project, matchScore, relevance: matchScore + project.significance / 100 };
    }).filter((project) => project.matchScore > 0)
      .sort((left, right) => right.relevance - left.relevance)
      .slice(0, 7);
  }

  function exactKeywordRetrieve(text) {
    const terms = tokenize(text);
    return projects.map((project) => {
      const haystack = `${project.title} ${project.description} ${project.details || ''} ${project.broad} ${project.category} ${project.context} ${project.tags.join(' ')}`.toLowerCase();
      const matches = terms.filter((term) => haystack.includes(term)).length;
      return { ...project, relevance: matches * 10 + project.significance / 100 };
    }).filter((project) => project.relevance > 1)
      .sort((left, right) => right.relevance - left.relevance)
      .slice(0, 7);
  }

  function readSavedChatSearch(text) {
    try {
      const saved = JSON.parse(localStorage.getItem(CHAT_SEARCH_CACHE_KEY) || '{}');
      const key = text.trim().toLowerCase();
      const entry = saved[key];
      if (!entry || Date.now() - entry.savedAt > ONE_DAY_MS) return [];
      return entry.titles.map((title) => projects.find((project) => project.title === title)).filter(Boolean);
    } catch {
      return [];
    }
  }

  function saveChatSearch(text, matches) {
    try {
      const saved = JSON.parse(localStorage.getItem(CHAT_SEARCH_CACHE_KEY) || '{}');
      const now = Date.now();
      for (const [key, entry] of Object.entries(saved)) {
        if (!entry?.savedAt || now - entry.savedAt > ONE_DAY_MS) delete saved[key];
      }
      saved[text.trim().toLowerCase()] = {
        savedAt: now,
        titles: matches.map((project) => project.title),
      };
      localStorage.setItem(CHAT_SEARCH_CACHE_KEY, JSON.stringify(saved));
    } catch {}
  }

  function tokenize(text) {
    return (text.toLowerCase().match(/[a-z0-9+#.-]+/g) || [])
      .filter((token) => token.length > 1 && !stop.has(token));
  }

  function showEvidence() {
    output.hidden = false;
    openDialog.disabled = !retrieved.length;
    evidence.innerHTML = retrieved.length
      ? retrieved.map((project, index) => (
        `<article class="evidence-item"><h4>${index + 1}. ${escapeHtml(project.title)}</h4>`
        + `<p>${escapeHtml(project.description)}</p>`
        + `${project.url ? `<a href="${escapeAttribute(project.url)}" target="_blank" rel="noreferrer">Open project →</a>` : ''}</article>`
      )).join('')
      : '<p>No strong match found. Try different terms.</p>';
  }

  function renderMarkdown(markdown) {
    const lines = String(markdown || '').replace(/\r/g, '').split('\n');
    let html = '';
    let list = '';
    let inCode = false;
    let code = '';
    const closeList = () => {
      if (list) html += `</${list}>`;
      list = '';
    };
    for (const rawLine of lines) {
      if (rawLine.trim().startsWith('```')) {
        closeList();
        if (inCode) {
          html += `<pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`;
          code = '';
        }
        inCode = !inCode;
        continue;
      }
      if (inCode) {
        code += `${rawLine}\n`;
        continue;
      }
      const heading = rawLine.match(/^(#{1,4})\s+(.+)$/);
      const bullet = rawLine.match(/^\s*[-*]\s+(.+)$/);
      const numbered = rawLine.match(/^\s*\d+[.)]\s+(.+)$/);
      if (heading) {
        closeList();
        const level = Math.min(4, heading[1].length + 1);
        html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
      } else if (bullet || numbered) {
        const wanted = bullet ? 'ul' : 'ol';
        if (list !== wanted) {
          closeList();
          list = wanted;
          html += `<${list}>`;
        }
        html += `<li>${inlineMarkdown((bullet || numbered)[1])}</li>`;
      } else if (!rawLine.trim()) {
        closeList();
      } else {
        closeList();
        html += `<p>${inlineMarkdown(rawLine)}</p>`;
      }
    }
    closeList();
    if (inCode && code) html += `<pre><code>${escapeHtml(code)}</code></pre>`;
    return html || '<p>Waiting for answer…</p>';
  }

  function inlineMarkdown(value) {
    return escapeHtml(value)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/__([^_]+)__/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
      .replace(/\[(\d+)\]/g, '<a href="#chat-result-$1">[$1]</a>');
  }

  function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = String(value || '');
    return div.innerHTML;
  }

  function escapeAttribute(value) {
    return escapeHtml(value).replace(/"/g, '&quot;');
  }
})();
