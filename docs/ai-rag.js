(()=>{
  const byId=id=>document.getElementById(id), form=byId('ragForm'), query=byId('ragQuery'), output=byId('ragOutput'), evidence=byId('ragEvidence'), answer=byId('ragAnswer'), enable=byId('enableAI'), generate=byId('generateAnswer'), status=byId('modelStatus'), progress=byId('modelProgress');
  let worker=null, modelReady=false, retrieved=[];
  const stop=new Set(['the','a','an','and','or','for','of','to','in','on','with','what','which','show','find','project','projects','work','ashish','experience','has','have','built','did','about']);
  const tokenize=text=>(text.toLowerCase().match(/[a-z0-9+#.-]+/g)||[]).filter(x=>x.length>1&&!stop.has(x));
  const expansions={ai:['llm','model','agent','vision','rag'],local:['offline','on-device','privacy','gemma'],robotics:['robot','drone','ros','slam','sensor','lidar'],language:['multilingual','translation','malayalam','indian-language'],deployment:['cloud','gcp','docker','api','server','on-premise'],voice:['speech','audio','gemini live','keyboard'],evaluation:['evals','benchmark','livebench','terminal-bench'],spatial:['ar','vr','hololens','digital twin','unity']};
  function retrieve(text){const base=tokenize(text), terms=[...new Set(base.flatMap(t=>[t,...(expansions[t]||[])]))];return projects.map(p=>{const fields={title:p.title.toLowerCase(),tags:p.tags.join(' ').toLowerCase(),category:`${p.broad} ${p.category}`.toLowerCase(),description:p.description.toLowerCase()};let score=0;for(const t of terms){if(fields.title.includes(t))score+=8;if(fields.tags.includes(t))score+=5;if(fields.category.includes(t))score+=4;if(fields.description.includes(t))score+=2}if(fields.title.includes(text.toLowerCase()))score+=12;score+=p.significance/100;return{...p,relevance:score}}).filter(p=>p.relevance>0.5).sort((a,b)=>b.relevance-a.relevance).slice(0,7)}
  function showEvidence(){output.hidden=false;answer.hidden=true;evidence.innerHTML=retrieved.length?retrieved.map((p,i)=>`<article class="evidence-item"><h4>${i+1}. ${p.title}<span class="evidence-score">relevance ${p.relevance.toFixed(1)}</span></h4><p>${p.description}</p>${p.url?`<a href="${p.url}" target="_blank" rel="noreferrer">Source →</a>`:''}</article>`).join(''):'<p>No strong match found. Try different terms.</p>';generate.disabled=!modelReady||!retrieved.length}
  form.addEventListener('submit',event=>{event.preventDefault();retrieved=retrieve(query.value.trim());showEvidence()});
  enable.addEventListener('click',()=>{
    if(worker||modelReady)return;
    if(!('gpu' in navigator)){status.textContent='WebGPU is unavailable in this browser. Retrieval still works without the model.';return}
    const approved=confirm('This will download about 0.8 GB for the Gemma 3 1B q4f16 model from Hugging Face and allocate GPU memory. Nothing has been downloaded yet. Continue?');
    if(!approved)return;
    enable.disabled=true;enable.textContent='Loading Gemma…';progress.hidden=false;status.textContent='Starting local model worker…';
    worker=new Worker('ai-worker.js?v=20260712-5',{type:'module'});worker.onmessage=({data})=>{
      if(data.type==='progress'){const pct=Math.max(0,Math.min(100,Math.round(data.value||0)));progress.value=pct;status.textContent=data.message||`Downloading model: ${pct}%`}
      if(data.type==='ready'){modelReady=true;progress.value=100;progress.hidden=true;enable.textContent='Gemma 3 1B q4f16 enabled';status.textContent='Local model ready. Model data remains in this browser session/cache.';generate.disabled=!retrieved.length}
      if(data.type==='answer'){answer.hidden=false;answer.innerHTML=`<h3>Local answer</h3>${escapeHtml(data.text)}`;generate.disabled=false;generate.textContent='Generate cited answer locally'}
      if(data.type==='error'){status.textContent=`Local model could not load: ${data.message}. Retrieval remains available.`;progress.hidden=true;enable.disabled=false;enable.textContent='Retry Gemma 3 1B q4f16 download';generate.disabled=true;worker?.terminate();worker=null}
    };worker.onerror=event=>{status.textContent=`Local model worker failed: ${event.message||'unknown error'}. Retrieval remains available.`;enable.disabled=false;enable.textContent='Retry Gemma 3 1B q4f16 download';progress.hidden=true;worker=null};worker.postMessage({type:'init'});
  });
  generate.addEventListener('click',()=>{if(!modelReady||!retrieved.length)return;generate.disabled=true;generate.textContent='Generating locally…';answer.hidden=false;answer.innerHTML='<h3>Local answer</h3>Generating from the retrieved project evidence…';worker.postMessage({type:'generate',question:query.value.trim(),projects:retrieved.map((p,i)=>({citation:i+1,title:p.title,description:p.description,year:p.year,context:p.context,category:p.category,tags:p.tags,url:p.url}))})});
  function escapeHtml(text){const div=document.createElement('div');div.textContent=text;return div.innerHTML.replace(/\n/g,'<br>')}
})();
