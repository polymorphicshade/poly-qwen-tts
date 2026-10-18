// Poly Qwen3-TTS web UI. Every action goes through the REST API under /api.
"use strict";

const STORAGE_KEY = "polytts.workspaces.v1";

const MODES = [
  { id: "design", label: "Voice Design", hint: "Describe a brand-new voice in plain language: timbre, age, accent, emotion, pace and style." },
  { id: "control", label: "Emotion & Prosody", hint: "Fine-grained control over emotion, intensity, delivery (e.g. whisper), pace, pitch, intonation and pauses. Inline markup works too." },
  { id: "clone", label: "Voice Clone", hint: "Clone any voice from a reference clip of 3 seconds or more. The clip's transcript gives the most faithful clone; if you leave it empty, Whisper writes it for you." },
  { id: "cross", label: "Cross-lingual", hint: "Clone a voice from one language and have it speak another, keeping its timbre and accent." },
  { id: "voice", label: "Voice Library", hint: "Pick a ready-made voice: 9 native speakers, 41 designed presets and your saved clones." },
  { id: "style", label: "Style", hint: "Combine a library voice with a natural-language style instruction." },
];

const DESIGN_EXAMPLES = [
  "Deep male voice with a slight rasp, speaking in a calm, authoritative narrator tone",
  "Cheerful young woman with a bright, bubbly voice, talking fast and excitedly",
  "Elderly British gentleman, soft and wise, with a warm, slow storytelling cadence",
  "Nervous teenage boy whose voice cracks, speaking hesitantly",
  "Sultry, low female jazz-club host with a smooth, relaxed delivery",
];
const STYLE_EXAMPLES = [
  "Sound incredulous, as if you can't believe what you just heard",
  "Very excited, almost shouting with joy",
  "Whisper it like a secret",
  "Sad and tired, speaking slowly with sighs",
  "Speak like a sports commentator at the final whistle",
];
const LANG_SAMPLES = {
  Chinese: "今天的天气真好，我们一起去公园散步吧。",
  English: "The weather is wonderful today, let's go for a walk in the park.",
  Japanese: "今日はとてもいい天気ですね。一緒に公園を散歩しましょう。",
  Korean: "오늘 날씨가 정말 좋네요. 같이 공원에 산책하러 가요.",
  German: "Das Wetter ist heute wunderbar, lass uns im Park spazieren gehen.",
  French: "Il fait un temps magnifique aujourd'hui, allons nous promener au parc.",
  Russian: "Сегодня прекрасная погода, давай прогуляемся в парке.",
  Portuguese: "O tempo está maravilhoso hoje, vamos passear no parque.",
  Spanish: "Hace un tiempo estupendo hoy, vamos a pasear por el parque.",
  Italian: "Oggi il tempo è splendido, andiamo a fare una passeggiata nel parco.",
};

const app = {
  info: null,
  voices: [],
  workspaces: [],   // persisted state objects
  runtime: {},      // id -> { el, refFile, refUrl, refSeconds, results, busy, recorder }
  activeId: null,
};

// ------------------------------------------------------------------ utilities

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const uid = () => Math.random().toString(36).slice(2, 10);
const humanize = (s) => String(s).replace(/_/g, " ");

function getPath(obj, path) { return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj); }
function setPath(obj, path, value) {
  const keys = path.split(".");
  let o = obj;
  for (const k of keys.slice(0, -1)) o = o[k] ??= {};
  o[keys.at(-1)] = value;
}

function newWorkspace(name) {
  return {
    id: uid(), name: name || `Workspace ${app.workspaces.length + 1}`, mode: "design",
    text: "Hello! This is Qwen3-TTS. [pause 400ms] What would you like me to say?", language: "Auto",
    voiceDescription: DESIGN_EXAMPLES[0], voiceId: "aiden", controlSource: "library",
    controls: {}, instruct: "", styleInstruct: STYLE_EXAMPLES[0],
    refText: "", xVectorOnly: false, saveName: "",
    crossSource: "upload", crossVoiceId: "", targetLanguage: "French", sourceLanguage: "",
    voiceFilter: { q: "", kind: "", language: "" }, gen: {},
  };
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ workspaces: app.workspaces, activeId: app.activeId }));
  } catch { /* storage unavailable: state just isn't remembered */ }
}

function load() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (data && Array.isArray(data.workspaces) && data.workspaces.length) {
      app.workspaces = data.workspaces.map((w) => ({ ...newWorkspace(), ...w }));
      app.activeId = data.activeId;
    }
  } catch { /* ignore */ }
  if (!app.workspaces.length) app.workspaces.push(newWorkspace("Workspace 1"));
  if (!app.workspaces.some((w) => w.id === app.activeId)) app.activeId = app.workspaces[0].id;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail, null, 1);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res;
}

// ------------------------------------------------------------------ tabs

function renderTabs() {
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  for (const ws of app.workspaces) {
    const rt = app.runtime[ws.id];
    const tab = document.createElement("div");
    tab.className = "tab" + (ws.id === app.activeId ? " active" : "") + (rt?.busy ? " busy" : "");
    tab.dataset.id = ws.id;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", ws.id === app.activeId);
    tab.title = "Double-click to rename, middle-click to close";
    tab.innerHTML = `<span class="busy-dot"></span><span class="name">${esc(ws.name)}</span>
      <button class="close" title="Close workspace" aria-label="Close ${esc(ws.name)}">×</button>`;
    tab.addEventListener("click", (e) => { if (!e.target.closest(".close")) activate(ws.id); });
    tab.addEventListener("auxclick", (e) => { if (e.button === 1) closeWorkspace(ws.id); });
    tab.addEventListener("dblclick", () => startRename(tab, ws));
    $(".close", tab).addEventListener("click", () => closeWorkspace(ws.id));
    tabs.appendChild(tab);
  }
}

// Update active/busy state without rebuilding the tabs (rebuilding would break double-click rename).
function syncTabs() {
  document.querySelectorAll("#tabs .tab").forEach((tab) => {
    const active = tab.dataset.id === app.activeId;
    tab.classList.toggle("active", active);
    tab.classList.toggle("busy", !!app.runtime[tab.dataset.id]?.busy);
    tab.setAttribute("aria-selected", active);
  });
}

function startRename(tab, ws) {
  const nameEl = $(".name", tab);
  const input = document.createElement("input");
  input.className = "rename";
  input.value = ws.name;
  nameEl.replaceWith(input);
  input.focus();
  input.select();
  const done = (commit) => {
    if (commit && input.value.trim()) ws.name = input.value.trim();
    save();
    renderTabs();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") done(true);
    if (e.key === "Escape") done(false);
  });
  input.addEventListener("blur", () => done(true));
}

function addWorkspace() {
  const ws = newWorkspace();
  app.workspaces.push(ws);
  mountWorkspace(ws);
  app.activeId = ws.id;
  renderTabs();
  activate(ws.id);
}

function closeWorkspace(id) {
  const rt = app.runtime[id];
  if (rt?.results.length && !confirm("Close this workspace? Its generated audio will be discarded.")) return;
  const idx = app.workspaces.findIndex((w) => w.id === id);
  app.workspaces.splice(idx, 1);
  if (rt) {
    rt.results.forEach((r) => r.url && URL.revokeObjectURL(r.url));
    if (rt.refUrl) URL.revokeObjectURL(rt.refUrl);
    rt.el.remove();
    delete app.runtime[id];
  }
  if (!app.workspaces.length) {
    const ws = newWorkspace("Workspace 1");
    app.workspaces.push(ws);
    mountWorkspace(ws);
  }
  if (app.activeId === id) app.activeId = app.workspaces[Math.min(idx, app.workspaces.length - 1)].id;
  renderTabs();
  activate(app.activeId);
}

function activate(id) {
  app.activeId = id;
  for (const [wid, rt] of Object.entries(app.runtime)) rt.el.classList.toggle("active", wid === id);
  syncTabs();
  save();
}

// ------------------------------------------------------------------ workspace

function mountWorkspace(ws) {
  const el = $("#workspace-template").content.firstElementChild.cloneNode(true);
  el.dataset.id = ws.id;
  $("#workspaces").appendChild(el);
  app.runtime[ws.id] = { el, refFile: null, refUrl: null, refSeconds: null, results: [], busy: false, recorder: null, transcribing: false };
  $(".clear-results", el).addEventListener("click", () => {
    const rt = app.runtime[ws.id];
    rt.results.forEach((r) => r.url && URL.revokeObjectURL(r.url));
    rt.results = [];
    renderResults(ws);
  });
  renderModes(ws);
  renderForm(ws);
  renderResults(ws);
}

function renderModes(ws) {
  const el = app.runtime[ws.id].el;
  const modes = $(".modes", el);
  modes.innerHTML = MODES.map((m) =>
    `<button type="button" class="mode" role="radio" aria-checked="${m.id === ws.mode}" data-mode="${m.id}">${m.label}</button>`).join("");
  modes.querySelectorAll(".mode").forEach((b) => b.addEventListener("click", () => {
    ws.mode = b.dataset.mode;
    save();
    renderModes(ws);
    renderForm(ws);
  }));
  $(".mode-hint", el).textContent = MODES.find((m) => m.id === ws.mode).hint;
}

// ---------- form building blocks (return HTML; bindForm wires data-k inputs to state)

function languageOptions(selected, { auto = true } = {}) {
  const langs = (app.info?.languages || ["Auto", "English"]).filter((l) => auto || l !== "Auto");
  return langs.map((l) => `<option ${l === selected ? "selected" : ""}>${l}</option>`).join("");
}

function textField(ws) {
  return `<div class="field"><label>Text</label>
    <textarea class="main-text" data-k="text" placeholder="What should be said?">${esc(ws.text)}</textarea>
    <div class="help">Markup: <code>[pause 800ms]</code> inserts silence, <code>[style: whispering]…[/style]</code> styles a span (built-in voices only).</div></div>`;
}

function languageField(ws, key = "language", label = "Language", opts) {
  return `<div class="field"><label>${label}</label><select data-k="${key}">${languageOptions(ws[key], opts)}</select></div>`;
}

function voicePicker(ws, key, { styleOnly = false, label = "Voice" } = {}) {
  const f = ws.voiceFilter;
  const q = (f.q || "").toLowerCase();
  let voices = app.voices.filter((v) => (!f.kind || v.kind === f.kind) && (!f.language || v.language === f.language)
    && (!q || `${v.id} ${v.name} ${v.description} ${v.tags.join(" ")}`.toLowerCase().includes(q)));
  if (styleOnly) voices = voices.filter((v) => v.supports_style);
  const groups = { native: "Native speakers", designed: "Designed presets", cloned: "My cloned voices" };
  const options = Object.entries(groups).map(([kind, title]) => {
    const vs = voices.filter((v) => v.kind === kind);
    if (!vs.length) return "";
    return `<optgroup label="${title} (${vs.length})">` + vs.map((v) =>
      `<option value="${esc(v.id)}" ${v.id === ws[key] ? "selected" : ""}>${esc(v.name)} — ${esc(v.language)} · ${v.gender} · ${v.age}</option>`).join("") + "</optgroup>";
  }).join("");
  const current = app.voices.find((v) => v.id === ws[key]);
  const langs = [...new Set(app.voices.map((v) => v.language))];
  return `<div class="field voice-picker"><label>${label}</label>
    <div class="voice-filter">
      <input type="text" data-filter="q" placeholder="Search ${app.voices.length} voices…" value="${esc(f.q)}">
      <select data-filter="kind"><option value="">All kinds</option>${Object.entries(groups).map(([k, t]) => `<option value="${k}" ${f.kind === k ? "selected" : ""}>${t}</option>`).join("")}</select>
      <select data-filter="language"><option value="">All languages</option>${langs.map((l) => `<option ${f.language === l ? "selected" : ""}>${l}</option>`).join("")}</select>
    </div>
    <select size="8" data-k="${key}" aria-label="${label}">${options || "<option disabled>No matching voices</option>"}</select>
    <div class="voice-desc"><span class="grow">${current ? esc(current.description) + (current.supports_style ? "" : " <em>(cloned: no style instructions)</em>") : "Select a voice."}</span>
      ${current?.kind === "cloned" ? `<button type="button" class="link danger" data-action="delete-voice">Delete</button>` : ""}</div>
  </div>`;
}

function controlsField(ws, open = true) {
  const opts = app.info?.controls || {};
  const sel = (name) => `<div><label class="label">${humanize(name)}</label><select data-k="controls.${name}">
      <option value="">—</option>${(opts[name] || []).map((o) => `<option value="${o}" ${ws.controls[name] === o ? "selected" : ""}>${humanize(o)}</option>`).join("")}
    </select></div>`;
  return `<details ${open ? "open" : ""}><summary>Emotion & prosody controls</summary>
    <div class="row">${["emotion", "intensity", "delivery"].map(sel).join("")}</div>
    <div class="row" style="margin-top:10px">${["pace", "pitch", "intonation", "pauses"].map(sel).join("")}</div>
    <div class="help">Compiled into an instruction for the model. Unset controls are left to the model.</div>
  </details>`;
}

function chips(list, target) {
  return `<div class="chips">${list.map((t) => `<button type="button" class="chip" data-chip="${target}" data-value="${esc(t)}">${esc(t.length > 48 ? t.slice(0, 46) + "…" : t)}</button>`).join("")}</div>`;
}

function refAudioField(ws) {
  const rt = app.runtime[ws.id];
  const min = app.info?.reference_audio?.min_seconds ?? 3;
  let info = "No reference yet. Upload a clip or record one (3 to 15 seconds of clean speech works best).";
  let bad = false;
  if (rt.refFile) {
    info = `${esc(rt.refFile.name)}` + (rt.refSeconds != null ? ` · ${rt.refSeconds.toFixed(1)} s` : "");
    if (rt.refSeconds != null && rt.refSeconds < min) { info += ` (needs at least ${min} s)`; bad = true; }
  }
  return `<div class="field"><label>Reference audio</label><div class="ref-box">
      <div class="ref-actions">
        <input type="file" accept="audio/*" data-action="ref-file">
        <button type="button" class="btn rec ${rt.recorder ? "recording" : ""}" data-action="record">${rt.recorder ? "■ Stop recording" : "● Record"}</button>
      </div>
      <div class="ref-info ${bad ? "bad" : ""}">${info}</div>
      ${rt.refUrl ? `<audio controls src="${rt.refUrl}"></audio>` : ""}
    </div></div>
    <div class="field"><div class="label-row"><label>Reference transcript <span class="help">(left empty, Whisper fills it in)</span></label>
        ${rt.refFile ? `<button type="button" class="link" data-action="transcribe" ${rt.transcribing ? "disabled" : ""}>Transcribe with Whisper</button>` : ""}</div>
      <textarea data-k="refText" ${rt.transcribing ? "readonly" : ""} placeholder="${rt.transcribing ? "Transcribing with Whisper…" : "Exactly what is said in the reference clip"}">${esc(ws.refText)}</textarea>
      ${rt.transcribing ? `<div class="help"><span class="spinner"></span> Transcribing with Whisper…</div>` : ""}
      <label class="checkbox help"><input type="checkbox" data-k="xVectorOnly" ${ws.xVectorOnly ? "checked" : ""}> Speaker-embedding only (no transcript needed, less faithful)</label>
    </div>`;
}

function generationField(ws) {
  const g = ws.gen;
  // Labels, ranges, defaults and tuning advice come from the server (GET /api/info → generation_options).
  const opts = app.info?.generation_options || {};
  const fields = Object.entries(opts).map(([k, o]) => {
    const id = `${ws.id}-gen-${k}`;
    return `<div class="gen-option"><label class="label" for="${id}">${esc(o.label)}</label>
      <input id="${id}" type="number" step="${o.step}" min="${o.min}" ${o.max != null ? `max="${o.max}"` : ""} data-k="gen.${k}" data-num
        value="${g[k] ?? ""}" placeholder="${o.default ?? "random"}" aria-describedby="${id}-help">
      <div class="help" id="${id}-help">${esc(o.description)}${o.default != null ? ` <span class="default">Default: ${o.default}.</span>` : ""}</div></div>`;
  }).join("");
  return `<details><summary>Generation options</summary>
    <p class="help gen-intro">These change how the model samples speech. Leave a box empty to use the model default (shown in grey).
      Set a seed first, then change one option at a time to hear what it does.</p>
    <div class="gen-grid">${fields || `<p class="help">Generation options are unavailable (the API could not be reached).</p>`}</div>
    <button type="button" class="link" data-action="reset-gen">Reset all to defaults</button></details>`;
}

function renderForm(ws) {
  const rt = app.runtime[ws.id];
  const form = $(".form", rt.el);
  let html = "";
  switch (ws.mode) {
    case "design":
      html = `<div class="field"><label>Voice description</label>
          <textarea data-k="voiceDescription" placeholder="Describe the voice…">${esc(ws.voiceDescription)}</textarea>
          ${chips(DESIGN_EXAMPLES, "voiceDescription")}</div>
        ${textField(ws)}${languageField(ws)}${controlsField(ws, false)}`;
      break;
    case "control":
      html = `<div class="field"><span class="label">Voice source</span><div class="segmented">
          <button type="button" data-seg="controlSource" data-value="library" class="${ws.controlSource === "library" ? "on" : ""}">Library voice</button>
          <button type="button" data-seg="controlSource" data-value="description" class="${ws.controlSource === "description" ? "on" : ""}">Describe a voice</button>
        </div></div>
        ${ws.controlSource === "library" ? voicePicker(ws, "voiceId", { styleOnly: true })
          : `<div class="field"><label>Voice description</label><textarea data-k="voiceDescription">${esc(ws.voiceDescription)}</textarea>${chips(DESIGN_EXAMPLES, "voiceDescription")}</div>`}
        ${controlsField(ws)}
        <div class="field"><label>Extra instruction <span class="help">(optional)</span></label>
          <input type="text" data-k="instruct" value="${esc(ws.instruct)}" placeholder="e.g. voice cracks on the last word"></div>
        ${textField(ws)}${languageField(ws)}`;
      break;
    case "clone":
      html = `${refAudioField(ws)}${textField(ws)}${languageField(ws)}
        <details><summary>Save to voice library</summary>
          <div class="field"><label>Name</label><input type="text" data-k="saveName" value="${esc(ws.saveName)}" placeholder="e.g. My narrator"></div>
          <button type="button" class="btn" data-action="save-voice">Save cloned voice</button>
          <div class="help">Saved voices appear in the library and can be reused in every mode.</div>
        </details>`;
      break;
    case "cross":
      html = `<div class="field"><span class="label">Voice to clone</span><div class="segmented">
          <button type="button" data-seg="crossSource" data-value="upload" class="${ws.crossSource === "upload" ? "on" : ""}">Upload / record</button>
          <button type="button" data-seg="crossSource" data-value="library" class="${ws.crossSource === "library" ? "on" : ""}">Library voice</button>
        </div></div>
        ${ws.crossSource === "upload" ? refAudioField(ws) + languageField(ws, "sourceLanguage", "Reference language") : voicePicker(ws, "crossVoiceId")}
        ${languageField(ws, "targetLanguage", "Target language", { auto: false })}
        <div class="field"><label>Text <span class="help">(in the target language)</span></label>
          <textarea class="main-text" data-k="text">${esc(ws.text)}</textarea>
          <button type="button" class="link" data-action="sample-text">Insert sample sentence in target language</button></div>`;
      break;
    case "voice":
      html = `${voicePicker(ws, "voiceId")}${textField(ws)}${languageField(ws)}`;
      break;
    case "style":
      html = `${voicePicker(ws, "voiceId", { styleOnly: true })}
        <div class="field"><label>Style instruction</label>
          <input type="text" data-k="styleInstruct" value="${esc(ws.styleInstruct)}">${chips(STYLE_EXAMPLES, "styleInstruct")}</div>
        ${textField(ws)}${languageField(ws)}`;
      break;
  }
  html += generationField(ws) + `<div class="actions">
      <button type="submit" class="primary" ${rt.busy ? "disabled" : ""}>${rt.busy ? '<span class="spinner"></span> Generating…' : "Generate"}</button>
      <span class="kbd">Ctrl+Enter</span></div><div class="error" role="alert"></div>`;
  // Keep collapsible sections open/closed across re-renders.
  const openState = new Map([...form.querySelectorAll("details")].map((d) => [d.querySelector("summary").textContent, d.open]));
  form.innerHTML = html;
  form.querySelectorAll("details").forEach((d) => {
    const prev = openState.get(d.querySelector("summary").textContent);
    if (prev !== undefined) d.open = prev;
  });
  bindForm(ws, form);
}

function bindForm(ws, form) {
  const rt = app.runtime[ws.id];
  form.onsubmit = (e) => { e.preventDefault(); generate(ws); };
  form.onkeydown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(ws); } };

  form.querySelectorAll("[data-k]").forEach((input) => {
    const key = input.dataset.k;
    const handler = () => {
      let value = input.type === "checkbox" ? input.checked : input.value;
      if (input.hasAttribute("data-num")) value = value === "" ? null : Number(value);
      setPath(ws, key, value);
      if (key.startsWith("controls.") && !value) delete ws.controls[key.split(".")[1]];
      save();
      if (input.tagName === "SELECT" && (key === "voiceId" || key === "crossVoiceId")) renderForm(ws);
    };
    input.addEventListener(input.tagName === "SELECT" || input.type === "checkbox" ? "change" : "input", handler);
  });
  form.querySelectorAll("[data-filter]").forEach((input) => {
    input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
      ws.voiceFilter[input.dataset.filter] = input.value;
      save();
      const pos = input.selectionStart;
      renderForm(ws);
      const again = form.querySelector(`[data-filter="${input.dataset.filter}"]`);
      again.focus();
      if (pos != null && again.setSelectionRange) again.setSelectionRange(pos, pos);
    });
  });
  form.querySelectorAll("[data-chip]").forEach((chip) => chip.addEventListener("click", () => {
    ws[chip.dataset.chip] = chip.dataset.value;
    save();
    renderForm(ws);
  }));
  form.querySelectorAll("[data-seg]").forEach((b) => b.addEventListener("click", () => {
    ws[b.dataset.seg] = b.dataset.value;
    save();
    renderForm(ws);
  }));
  form.querySelector("[data-action='ref-file']")?.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) setReference(ws, file);
  });
  form.querySelector("[data-action='record']")?.addEventListener("click", () => toggleRecording(ws));
  form.querySelector("[data-action='transcribe']")?.addEventListener("click", () => transcribeReference(ws, true));
  form.querySelector("[data-action='reset-gen']")?.addEventListener("click", () => {
    ws.gen = {};
    save();
    renderForm(ws);
  });
  form.querySelector("[data-action='save-voice']")?.addEventListener("click", () => saveVoice(ws));
  form.querySelector("[data-action='delete-voice']")?.addEventListener("click", () => deleteVoice(ws));
  form.querySelector("[data-action='sample-text']")?.addEventListener("click", () => {
    ws.text = LANG_SAMPLES[ws.targetLanguage] || ws.text;
    save();
    renderForm(ws);
  });
  rt.errorEl = form.querySelector(".error");
}

// ------------------------------------------------------------------ reference audio

async function setReference(ws, file) {
  const rt = app.runtime[ws.id];
  if (rt.refUrl) URL.revokeObjectURL(rt.refUrl);
  rt.refFile = file;
  rt.refUrl = URL.createObjectURL(file);
  rt.refSeconds = null;
  renderForm(ws);
  try {
    const ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    rt.refSeconds = buf.duration;
    ctx.close();
  } catch { /* the server validates anyway */ }
  renderForm(ws);
  if (!ws.refText.trim() && !ws.xVectorOnly) transcribeReference(ws);
}

// Fill the reference transcript with Whisper. Without `force`, a transcript the user typed is kept.
async function transcribeReference(ws, force = false) {
  const rt = app.runtime[ws.id];
  if (!rt.refFile || rt.transcribing) return;
  const file = rt.refFile;
  const fd = new FormData();
  fd.append("audio", file);
  fd.append("language", ws.mode === "cross" ? (ws.sourceLanguage || "Auto") : ws.language);
  rt.transcribing = true;
  renderForm(ws);
  let error = "";
  try {
    const { text } = await (await api("/api/transcribe", { method: "POST", body: fd })).json();
    if (rt.refFile === file && (force || !ws.refText.trim())) {
      ws.refText = text;
      save();
    }
  } catch (e) {
    error = `Auto-transcription failed: ${e.message}. Type the transcript, or tick speaker-embedding only.`;
  } finally {
    rt.transcribing = false;
    renderForm(ws);
    if (error) showError(ws, error);
  }
}

function encodeWav(buffer) {
  // Downmix to mono 16-bit PCM WAV so the server can always decode it.
  const ch = buffer.numberOfChannels, len = buffer.length, sr = buffer.sampleRate;
  const mono = new Float32Array(len);
  for (let c = 0; c < ch; c++) { const d = buffer.getChannelData(c); for (let i = 0; i < len; i++) mono[i] += d[i] / ch; }
  const out = new DataView(new ArrayBuffer(44 + len * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); out.setUint32(4, 36 + len * 2, true); str(8, "WAVE"); str(12, "fmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, sr, true); out.setUint32(28, sr * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  str(36, "data"); out.setUint32(40, len * 2, true);
  for (let i = 0; i < len; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, mono[i])) * 0x7fff, true);
  return new Blob([out], { type: "audio/wav" });
}

async function toggleRecording(ws) {
  const rt = app.runtime[ws.id];
  if (rt.recorder) { rt.recorder.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    showError(ws, `Microphone unavailable: ${e.message}`);
    return;
  }
  const chunks = [];
  const recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    rt.recorder = null;
    try {
      const ctx = new AudioContext();
      const buf = await ctx.decodeAudioData(await new Blob(chunks).arrayBuffer());
      ctx.close();
      await setReference(ws, new File([encodeWav(buf)], "recording.wav", { type: "audio/wav" }));
    } catch (e) {
      showError(ws, `Could not process recording: ${e.message}`);
      renderForm(ws);
    }
  };
  rt.recorder = recorder;
  recorder.start();
  renderForm(ws);
}

// ------------------------------------------------------------------ voices

async function loadVoices() {
  app.voices = await (await api("/api/voices")).json();
}

async function refreshVoices() {
  await loadVoices();
  for (const ws of app.workspaces) renderForm(ws);
}

async function saveVoice(ws) {
  const rt = app.runtime[ws.id];
  showError(ws, "");
  if (!rt.refFile) return showError(ws, "Add reference audio first.");
  if (!ws.saveName.trim()) return showError(ws, "Give the voice a name.");
  const fd = new FormData();
  fd.append("name", ws.saveName.trim());
  fd.append("ref_audio", rt.refFile);
  if (ws.refText.trim()) fd.append("ref_text", ws.refText.trim());
  fd.append("language", ws.language);
  try {
    const voice = await (await api("/api/voices", { method: "POST", body: fd })).json();
    ws.voiceId = voice.id;
    ws.saveName = "";
    save();
    await refreshVoices();
    alert(`Saved "${voice.name}" to the voice library (id: ${voice.id}).`);
  } catch (e) {
    showError(ws, e.message);
  }
}

async function deleteVoice(ws) {
  const key = ws.mode === "cross" ? "crossVoiceId" : "voiceId";
  const voice = app.voices.find((v) => v.id === ws[key]);
  if (!voice || !confirm(`Delete cloned voice "${voice.name}"?`)) return;
  try {
    await api(`/api/voices/${encodeURIComponent(voice.id)}`, { method: "DELETE" });
    await refreshVoices();
  } catch (e) {
    showError(ws, e.message);
  }
}

// ------------------------------------------------------------------ generation

function cleanGen(gen) {
  const out = {};
  for (const [k, v] of Object.entries(gen || {})) if (v !== null && v !== "" && !Number.isNaN(v)) out[k] = v;
  return Object.keys(out).length ? out : null;
}
function cleanControls(c) {
  const out = {};
  for (const [k, v] of Object.entries(c || {})) if (v) out[k] = v;
  return Object.keys(out).length ? out : null;
}
const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function buildRequest(ws) {
  const rt = app.runtime[ws.id];
  const generation = cleanGen(ws.gen);
  const base = { text: ws.text, language: ws.language, generation };
  const voiceName = (id) => app.voices.find((v) => v.id === id)?.name || id;
  switch (ws.mode) {
    case "design":
      return { url: "/api/tts/design", init: json({ ...base, voice_description: ws.voiceDescription, controls: cleanControls(ws.controls) }), label: "Voice design" };
    case "control": {
      const body = { ...base, controls: cleanControls(ws.controls), instruct: ws.instruct || null };
      if (ws.controlSource === "library") body.voice_id = ws.voiceId; else body.voice_description = ws.voiceDescription;
      return { url: "/api/tts/control", init: json(body), label: `Control · ${ws.controlSource === "library" ? voiceName(ws.voiceId) : "designed"}` };
    }
    case "voice":
      return { url: "/api/tts/voice", init: json({ ...base, voice_id: ws.voiceId }), label: voiceName(ws.voiceId) };
    case "style":
      return { url: "/api/tts/style", init: json({ ...base, voice_id: ws.voiceId, instruct: ws.styleInstruct }), label: `Style · ${voiceName(ws.voiceId)}` };
    case "clone":
    case "cross": {
      const fd = new FormData();
      fd.append("text", ws.text);
      const useLibrary = ws.mode === "cross" && ws.crossSource === "library";
      if (useLibrary) {
        if (!ws.crossVoiceId) throw new Error("Pick a voice to clone.");
        fd.append("voice_id", ws.crossVoiceId);
      } else {
        if (!rt.refFile) throw new Error("Add reference audio (upload or record) first.");
        fd.append("ref_audio", rt.refFile);
        if (ws.refText.trim()) fd.append("ref_text", ws.refText.trim());
        fd.append("x_vector_only", ws.xVectorOnly ? "true" : "false");
      }
      if (generation) fd.append("generation", JSON.stringify(generation));
      if (ws.mode === "clone") {
        fd.append("language", ws.language);
        return { url: "/api/tts/clone", init: { method: "POST", body: fd }, label: `Clone · ${rt.refFile.name}` };
      }
      fd.append("target_language", ws.targetLanguage);
      if (!useLibrary && ws.sourceLanguage && ws.sourceLanguage !== "Auto") fd.append("source_language", ws.sourceLanguage);
      return { url: "/api/tts/cross-lingual", init: { method: "POST", body: fd },
        label: `Cross-lingual → ${ws.targetLanguage}${useLibrary ? " · " + voiceName(ws.crossVoiceId) : ""}` };
    }
  }
}

function showError(ws, msg) {
  const el = app.runtime[ws.id].errorEl;
  if (el) el.textContent = msg;
}

async function generate(ws) {
  const rt = app.runtime[ws.id];
  if (rt.busy) return;
  showError(ws, "");
  let req;
  try {
    req = buildRequest(ws);
  } catch (e) {
    return showError(ws, e.message);
  }
  const entry = { id: uid(), label: req.label, text: ws.text, pending: true, started: Date.now() };
  rt.results.unshift(entry);
  rt.busy = true;
  renderResults(ws);
  renderForm(ws);
  syncTabs();
  try {
    const res = await api(req.url, req.init);
    const meta = JSON.parse(decodeURIComponent(res.headers.get("X-TTS-Metadata") || "%7B%7D"));
    const blob = await res.blob();
    Object.assign(entry, { pending: false, url: URL.createObjectURL(blob), meta });
  } catch (e) {
    Object.assign(entry, { pending: false, error: e.message });
  } finally {
    rt.busy = false;
    renderResults(ws);
    renderForm(ws);
    syncTabs();
  }
}

function renderResults(ws) {
  const rt = app.runtime[ws.id];
  const list = $(".result-list", rt.el);
  if (!rt.results.length) {
    list.innerHTML = `<p class="empty">Generated audio appears here.</p>`;
    return;
  }
  list.innerHTML = rt.results.map((r, i) => {
    if (r.pending) return `<div class="result pending"><div class="title"><span>${esc(r.label)}</span><span class="spinner"></span></div>
      <div class="text">${esc(r.text)}</div><div class="meta">Generating… the first request may take a while to load the model.</div></div>`;
    if (r.error) return `<div class="result"><div class="title"><span>${esc(r.label)}</span>
      <button class="link" data-remove="${r.id}">remove</button></div><div class="text">${esc(r.text)}</div><div class="error">${esc(r.error)}</div></div>`;
    const m = r.meta || {};
    const file = `qwen-tts-${m.mode || "audio"}-${rt.results.length - i}.wav`;
    return `<div class="result"><div class="title"><span>${esc(r.label)}</span><button class="link" data-remove="${r.id}">remove</button></div>
      <div class="text">${esc(r.text)}</div>
      <audio controls preload="metadata" src="${r.url}" ${i === 0 && !r.played ? "autoplay" : ""}></audio>
      <div class="meta"><span>${m.duration ?? "?"} s audio</span><span>${m.elapsed ?? "?"} s to generate</span><span>${esc(m.language || "")}</span>${m.segments > 1 ? `<span>${m.segments} segments</span>` : ""}</div>
      ${m.instruct ? `<div class="instruct">“${esc(m.instruct)}”</div>` : ""}
      ${m.ref_text_auto ? `<div class="instruct">Reference transcript (Whisper): “${esc(m.ref_text_auto)}”</div>` : ""}
      ${(m.warnings || []).map((w) => `<div class="warn">${esc(w)}</div>`).join("")}
      <div class="links"><a href="${r.url}" download="${file}">Download WAV</a></div></div>`;
  }).join("");
  // Autoplay only once per result.
  rt.results.forEach((r) => { if (r.url) r.played = true; });
  list.querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", () => {
    const idx = rt.results.findIndex((r) => r.id === b.dataset.remove);
    const [r] = rt.results.splice(idx, 1);
    if (r?.url) URL.revokeObjectURL(r.url);
    renderResults(ws);
  }));
}

// ------------------------------------------------------------------ boot

async function boot() {
  load();
  const status = $("#status");
  try {
    app.info = await (await api("/api/info")).json();
    await loadVoices();
    const b = app.info.backend;
    status.textContent = `${b.backend}${b.gpu ? " · " + b.gpu : ""} · ${app.voices.length} voices`;
    status.className = "status ok";
  } catch (e) {
    status.textContent = `API error: ${e.message}`;
    status.className = "status err";
  }
  app.workspaces.forEach(mountWorkspace);
  renderTabs();
  activate(app.activeId);
  $("#new-tab").addEventListener("click", addWorkspace);
  document.addEventListener("keydown", (e) => {
    if (e.altKey && e.key.toLowerCase() === "n") { e.preventDefault(); addWorkspace(); }
    if (e.altKey && e.key.toLowerCase() === "w") { e.preventDefault(); closeWorkspace(app.activeId); }
  });
}

boot();
