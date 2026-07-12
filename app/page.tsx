"use client";

import { useMemo, useState } from "react";

const projects = [
  {
    id: "glm",
    number: "01",
    title: "GLM-5.2, fast on 2×H100",
    account: "U4AR",
    url: "https://github.com/U4AR/glm52-fast-2xh100",
    date: "Jun–Jul 2026",
    year: "2026",
    context: "Research",
    subjects: ["LLM Systems", "Performance"],
    stack: ["Python", "SGLang", "KTransformers", "CUDA"],
    metric: "40.5",
    unit: "tok/s",
    accent: "acid",
    summary: "Serving a 754B MoE model on one dual-H100 box through heterogeneous CPU/GPU execution, packed INT4 kernels, top-2 expert substitution and MTP speculative decode.",
    details: ["2.75× over the documented plain baseline", "~250 GB CPU RAM for expert weights", "OpenAI-compatible API + reproducible benchmarks"],
  },
  {
    id: "voice",
    number: "02",
    title: "Voice Keyboard",
    account: "voicekeyboarddev",
    url: "https://github.com/voicekeyboarddev/voicekeyboard",
    date: "May 2026",
    year: "2026",
    context: "Open Source",
    subjects: ["Local AI", "Voice", "Desktop"],
    stack: ["Rust", "Tauri", "Gemma", "llama.cpp"],
    metric: "100%",
    unit: "on-device",
    accent: "cyan",
    summary: "A private Windows voice-to-keystroke app. Hold the mouse, speak naturally, and inject context-aware text or real keyboard shortcuts into any focused field.",
    details: ["No cloud calls; audio never leaves the machine", "Windows UI Automation + SendInput", "Diagnostics and paired dataset capture"],
  },
  {
    id: "livelingo",
    number: "03",
    title: "LiveLingo",
    account: "ashishvasant",
    url: "https://github.com/ashishvasant/LiveLingo",
    date: "Mar–Jul 2026",
    year: "2026",
    context: "Hackathon",
    subjects: ["Agents", "Multilingual", "Mobile"],
    stack: ["Flutter", "FastAPI", "Gemini Live", "Google ADK"],
    metric: "2",
    unit: "conversation modes",
    accent: "orange",
    summary: "A real-time language agent that can speak and negotiate for you—or coach you with native script, meaning, familiar-script transliteration and pronunciation.",
    details: ["Streaming PCM16 over WebSockets", "Echo, user and counterpart classification", "Parallel translation and transliteration"],
  },
  {
    id: "doless",
    number: "04",
    title: "DoLess-Agents",
    account: "faithful-little",
    url: "https://github.com/faithful-little/DoLess-Agents",
    date: "Feb 2026",
    year: "2026",
    context: "Hobby",
    subjects: ["Agents", "Automation", "Browser"],
    stack: ["JavaScript", "Chrome", "Gemini", "Ollama"],
    metric: "12",
    unit: "integrated tools",
    accent: "pink",
    summary: "A Chrome extension that records browser demonstrations, replays tasks and turns repeated computer-use workflows into reusable AI-generated functions.",
    details: ["Cloud Gemini or local Ollama", "Semantic search, scraping and scheduling", "Optional Docker function registry"],
  },
];

const filters = {
  Context: ["All", "Research", "Open Source", "Hackathon", "Hobby"],
  Subject: ["All", "LLM Systems", "Local AI", "Agents", "Automation", "Multilingual"],
  Date: ["All", "Feb 2026", "Mar–Jul 2026", "May 2026", "Jun–Jul 2026"],
};

export default function Home() {
  const [active, setActive] = useState({ Context: "All", Subject: "All", Date: "All" });
  const [query, setQuery] = useState("");

  const visible = useMemo(() => projects.filter((project) => {
    const contextMatch = active.Context === "All" || project.context === active.Context;
    const subjectMatch = active.Subject === "All" || project.subjects.includes(active.Subject);
    const dateMatch = active.Date === "All" || project.date === active.Date;
    const haystack = `${project.title} ${project.account} ${project.summary} ${project.stack.join(" ")} ${project.subjects.join(" ")}`.toLowerCase();
    return contextMatch && subjectMatch && dateMatch && haystack.includes(query.toLowerCase());
  }), [active, query]);

  const setFilter = (group: keyof typeof active, value: string) => setActive((current) => ({ ...current, [group]: value }));

  return (
    <main>
      <header className="nav shell">
        <a className="mark" href="#top" aria-label="Ashish T Vasant home">ATV<span>◆</span></a>
        <nav aria-label="Primary navigation">
          <a href="#work">Work</a><a href="#about">About</a><a href="mailto:ashish.t.vasant@gmail.com">Contact</a>
        </nav>
      </header>

      <section className="hero shell" id="top">
        <p className="eyebrow">Applied AI · Vision · Spatial systems</p>
        <h1>I build AI that<br/><em>touches reality.</em></h1>
        <div className="hero-bottom">
          <p>Engineer and rapid prototyper working across frontier-model infrastructure, private local AI, real-time agents, computer vision and spatial computing.</p>
          <div className="coordinates"><span>Kottayam, India</span><span>09.5916° N / 76.5222° E</span></div>
        </div>
        <div className="signal" aria-hidden="true"><i/><i/><i/><i/><i/><i/><i/><i/></div>
      </section>

      <section className="work shell" id="work">
        <div className="section-heading">
          <div><p className="eyebrow">Selected open-source work</p><h2>Four repositories.<br/>One engineering thread.</h2></div>
          <p className="intro">Fast, private, useful AI—shipped across servers, desktops, browsers and phones.</p>
        </div>

        <div className="filter-panel" aria-label="Project filters">
          <label className="search"><span>Search</span><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="technology, topic, project…" /></label>
          {Object.entries(filters).map(([group, options]) => (
            <div className="filter-group" key={group}>
              <span>{group}</span>
              <div>{options.map((option) => <button key={option} className={active[group as keyof typeof active] === option ? "active" : ""} onClick={() => setFilter(group as keyof typeof active, option)}>{option}</button>)}</div>
            </div>
          ))}
          <p className="result-count"><b>{visible.length}</b> / {projects.length} projects</p>
        </div>

        <div className="project-grid">
          {visible.map((project) => (
            <article className={`project ${project.accent}`} key={project.id}>
              <div className="project-top"><span>{project.number}</span><span>{project.date}</span></div>
              <div className="metric"><strong>{project.metric}</strong><small>{project.unit}</small></div>
              <div className="project-copy">
                <div className="chips">{project.subjects.map((tag) => <span key={tag}>{tag}</span>)}</div>
                <h3>{project.title}</h3>
                <p>{project.summary}</p>
                <ul>{project.details.map((detail) => <li key={detail}>{detail}</li>)}</ul>
              </div>
              <div className="project-footer"><div>{project.stack.map((item) => <span key={item}>{item}</span>)}</div><a href={project.url} target="_blank" rel="noreferrer">GitHub ↗<small>@{project.account}</small></a></div>
            </article>
          ))}
        </div>
        {visible.length === 0 && <div className="empty">No project matches this combination. Try widening a filter.</div>}
      </section>

      <section className="about shell" id="about">
        <p className="eyebrow">The connecting idea</p>
        <blockquote>“Frontier capability matters only when it survives contact with hardware, latency, imperfect data—and a real person.”</blockquote>
        <div className="about-grid">
          <p>Ashish T Vasant is an applied AI and spatial-computing engineer with nearly 5.5 years at Bharat Electronics Limited. His work spans LLM inference, computer vision, AR/VR, drones, digital twins, robotics and local-language systems.</p>
          <div><span>Current focus</span><b>AI systems that are local, low-latency and physically useful.</b></div>
        </div>
      </section>

      <footer className="shell">
        <div><p>Have a difficult prototype?</p><a href="mailto:ashish.t.vasant@gmail.com">Let’s make it real. ↗</a></div>
        <div className="footer-links"><a href="https://github.com/U4AR">U4AR</a><a href="https://github.com/voicekeyboarddev">voicekeyboarddev</a><a href="https://github.com/ashishvasant">ashishvasant</a><a href="https://github.com/faithful-little">faithful-little</a></div>
        <p className="fineprint">© 2026 Ashish T Vasant · Built from verified public repository data</p>
      </footer>
    </main>
  );
}
