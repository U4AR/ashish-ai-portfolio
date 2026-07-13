(() => {
  const byId = (id) => document.getElementById(id);
  const form = byId('ragForm');
  const query = byId('ragQuery');
  const searchMode = byId('searchMode');
  const output = byId('ragOutput');
  const evidence = byId('ragEvidence');
  const answer = byId('ragAnswer');
  const openDialog = byId('openAnswerDialog');
  const dialog = byId('answerDialog');
  const closeDialog = byId('closeAnswerDialog');
  const enable = byId('enableAI');
  const choice = byId('localModelChoice');
  const generate = byId('generateAnswer');
  const status = byId('modelStatus');
  const semanticStatus = byId('semanticStatus');
  const progress = byId('modelProgress');
  const headerAiChat = byId('headerAiChat');
  const chatSearchHint = byId('chatSearchHint');
  const githubDialog = byId('githubDialog');
  const openGitHubDialog = byId('openGitHubDialog');
  const closeGitHubDialog = byId('closeGitHubDialog');

  let modelWorker = null;
  let searchWorker = null;
  let modelReady = false;
  let retrieved = [];
  let searchRequestId = 0;

  const stop = new Set([
    'the', 'a', 'an', 'and', 'or', 'for', 'of', 'to', 'in', 'on', 'with', 'what',
    'which', 'show', 'find', 'project', 'projects', 'work', 'ashish', 'experience',
    'has', 'have', 'built', 'did', 'about',
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
    runSemanticSearch(text);
  });

  openDialog.addEventListener('click', () => {
    if (retrieved.length) dialog.showModal();
  });
  headerAiChat.addEventListener('click', () => dialog.showModal());
  closeDialog.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });
  openGitHubDialog.addEventListener('click', () => githubDialog.showModal());
  closeGitHubDialog.addEventListener('click', () => githubDialog.close());
  githubDialog.addEventListener('click', (event) => {
    if (event.target === githubDialog) githubDialog.close();
  });
  byId('githubDialog').querySelector('.github-all-link').addEventListener('click', () => githubDialog.close());

  enable.addEventListener('click', () => {
    if (modelWorker || modelReady) return;
    if (!('gpu' in navigator)) {
      status.textContent = 'AI answers are not supported in this browser. Project search still works.';
      return;
    }

    const selected = choice.value;
    const details = selected === 'large'
      ? { label: 'Gemma 4 E2B q4f16', size: '3.38 GB' }
      : { label: 'Gemma 3 1B q4f16', size: '0.8 GB' };
    const approved = confirm(
      `AI answers need a one-time download of about ${details.size}. Continue?`,
    );
    if (!approved) return;

    navigator.storage?.persist?.().catch(() => {});
    enable.disabled = true;
    choice.disabled = true;
    enable.textContent = 'Preparing AI answers…';
    progress.hidden = false;
    status.textContent = 'Preparing AI answers. You can retry if the connection is interrupted.';

    modelWorker = new Worker('ai-worker.js?v=20260712-9', { type: 'module' });
    modelWorker.onmessage = ({ data }) => {
      if (data.type === 'progress') {
        const pct = Math.max(0, Math.min(100, Math.round(data.value || 0)));
        progress.value = pct;
        status.textContent = `Preparing AI answers: ${pct}%`;
      }
      if (data.type === 'ready') {
        modelReady = true;
        progress.value = 100;
        progress.hidden = true;
        enable.textContent = 'AI answers enabled';
        status.textContent = 'AI answers are ready to use.';
        generate.disabled = !retrieved.length;
      }
      if (data.type === 'answer') {
        answer.hidden = false;
        answer.innerHTML = `<h3>Answer</h3>${escapeHtml(data.text)}`;
        generate.disabled = false;
        generate.textContent = 'Answer using matched projects';
      }
      if (data.type === 'error') {
        status.textContent = 'Setup paused. Retry to continue where it stopped.';
        progress.hidden = true;
        enable.disabled = false;
        choice.disabled = false;
        enable.textContent = 'Retry AI setup';
        generate.disabled = true;
        modelWorker?.terminate();
        modelWorker = null;
      }
    };
    modelWorker.onerror = (event) => {
      status.textContent = 'Setup paused. Retry to continue where it stopped.';
      enable.disabled = false;
      choice.disabled = false;
      enable.textContent = 'Retry AI setup';
      progress.hidden = true;
      modelWorker = null;
    };
    modelWorker.postMessage({ type: 'init', model: selected });
  });

  generate.addEventListener('click', () => {
    if (!modelReady || !retrieved.length) return;
    generate.disabled = true;
    generate.textContent = 'Writing answer…';
    answer.hidden = false;
    answer.innerHTML = '<h3>Answer</h3>Reviewing the matched projects…';
    modelWorker.postMessage({
      type: 'generate',
      question: query.value.trim(),
      projects: retrieved.map((project, index) => ({
        citation: index + 1,
        title: project.title,
        description: project.description,
        year: project.year,
        context: project.context,
        category: project.category,
        tags: project.tags,
        url: project.url,
      })),
    });
  });

  function runSemanticSearch(text) {
    if (!searchWorker) {
      searchWorker = new Worker('search-worker.js?v=20260712-2', { type: 'module' });
      searchWorker.onmessage = ({ data }) => {
        if (data.type === 'status') semanticStatus.textContent = data.message;
        if (data.type === 'results' && data.requestId === searchRequestId) {
          retrieved = data.scores.slice(0, 7).map(({ index, score }) => ({
            ...projects[index],
            relevance: Math.max(0, score * 100),
          }));
          semanticStatus.textContent = 'Best matches ready.';
          showEvidence();
        }
        if (data.type === 'error' && data.requestId === searchRequestId) {
          semanticStatus.textContent = 'Best-match search is unavailable right now. Word matches are shown instead.';
        }
      };
      searchWorker.onerror = (event) => {
        semanticStatus.textContent = 'Best-match search is unavailable right now. Word matches are shown instead.';
        searchWorker = null;
      };
    }

    searchRequestId += 1;
    searchWorker.postMessage({
      type: 'search',
      requestId: searchRequestId,
      query: text,
      projects: projects.map((project, index) => ({
        index,
        text: `${project.title}. ${project.description}. ${project.broad}. ${project.category}. ${project.tags.join(', ')}. ${project.context}.`,
      })),
    });
  }

  function keywordRetrieve(text) {
    const base = tokenize(text);
    const terms = [...new Set(base.flatMap((term) => [term, ...(expansions[term] || [])]))];
    return projects.map((project) => {
      const fields = {
        title: project.title.toLowerCase(),
        tags: project.tags.join(' ').toLowerCase(),
        category: `${project.broad} ${project.category}`.toLowerCase(),
        description: project.description.toLowerCase(),
      };
      let score = 0;
      for (const term of terms) {
        if (fields.title.includes(term)) score += 8;
        if (fields.tags.includes(term)) score += 5;
        if (fields.category.includes(term)) score += 4;
        if (fields.description.includes(term)) score += 2;
      }
      if (fields.title.includes(text.toLowerCase())) score += 12;
      score += project.significance / 100;
      return { ...project, relevance: score };
    }).filter((project) => project.relevance > 0.5)
      .sort((left, right) => right.relevance - left.relevance)
      .slice(0, 7);
  }

  function exactKeywordRetrieve(text) {
    const terms = tokenize(text);
    return projects.map((project) => {
      const haystack = `${project.title} ${project.description} ${project.broad} ${project.category} ${project.context} ${project.tags.join(' ')}`.toLowerCase();
      const matches = terms.filter((term) => haystack.includes(term)).length;
      return { ...project, relevance: matches * 10 + project.significance / 100 };
    }).filter((project) => project.relevance > 1)
      .sort((left, right) => right.relevance - left.relevance)
      .slice(0, 7);
  }

  function tokenize(text) {
    return (text.toLowerCase().match(/[a-z0-9+#.-]+/g) || [])
      .filter((token) => token.length > 1 && !stop.has(token));
  }

  function showEvidence() {
    output.hidden = false;
    answer.hidden = true;
    openDialog.disabled = !retrieved.length;
    chatSearchHint.hidden = retrieved.length > 0;
    generate.disabled = !modelReady || !retrieved.length;
    evidence.innerHTML = retrieved.length
      ? retrieved.map((project, index) => (
        `<article class="evidence-item"><h4>${index + 1}. ${project.title}</h4>`
        + `<p>${project.description}</p>`
        + `${project.url ? `<a href="${project.url}" target="_blank" rel="noreferrer">Open project →</a>` : ''}</article>`
      )).join('')
      : '<p>No strong match found. Try different terms.</p>';
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
  }
})();
