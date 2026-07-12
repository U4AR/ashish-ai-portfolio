import {pipeline,env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
env.allowLocalModels=false;
const MODEL='onnx-community/gemma-4-E2B-it-qat-mobile-ONNX';let generator=null;
self.onmessage=async({data})=>{try{
  if(data.type==='init'){
    self.postMessage({type:'progress',value:1,message:'Loading Transformers.js and preparing WebGPU…'});
    generator=await pipeline('any-to-any',MODEL,{device:'webgpu',progress_callback:update=>{const pct=update.progress??(update.loaded&&update.total?update.loaded/update.total*100:0);self.postMessage({type:'progress',value:pct,message:update.status==='progress'?`Downloading ${update.file||'model data'}: ${Math.round(pct)}%`:`Preparing ${update.file||'Gemma 4 E2B QAT'}…`})}});
    self.postMessage({type:'ready'});return;
  }
  if(data.type==='generate'){
    const context=data.projects.map(p=>`[${p.citation}] ${p.title} (${p.year}; ${p.context}; ${p.category})\n${p.description}\nTechnologies: ${p.tags.join(', ')}\nSource: ${p.url||'portfolio record'}`).join('\n\n');
    const messages=[{role:'system',content:[{type:'text',text:'You answer questions about Ashish T Vasant only from the supplied project evidence. Be concise and factual. Cite every factual claim using bracketed project numbers such as [1]. If evidence is insufficient, say so. Do not invent employers, outcomes, metrics, links, or technologies.'}]},{role:'user',content:[{type:'text',text:`Question: ${data.question}\n\nRetrieved project evidence:\n${context}\n\nWrite a direct answer with citations, then a short Recommended projects list.`}]}];
    const result=await generator(messages,{max_new_tokens:280,do_sample:false});self.postMessage({type:'answer',text:extractText(result)});
  }
}catch(error){self.postMessage({type:'error',message:error?.message||String(error)})}};
function extractText(result){let value=result?.[0]?.generated_text??result?.generated_text??result;if(Array.isArray(value)){const last=value[value.length-1];value=last?.content??last?.text??last}if(Array.isArray(value))value=value.map(x=>x?.text??String(x)).join('');if(typeof value==='object')value=value?.text??value?.content??JSON.stringify(value);return String(value||'No answer was generated.').replace(/<\|channel\>thought[\s\S]*?<channel\|>/g,'').trim()}
