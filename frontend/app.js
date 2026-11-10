// CareerCompass frontend: plain JavaScript, no build step. Talks to the FastAPI service at window.API_URL.
// Every value from the API is inserted with textContent (never innerHTML), so nothing it returns can inject HTML.

const API = (window.API_URL || "http://localhost:8000").replace(/\/$/, "");
const MIN_SKILLS = 5;
const MAX_SKILLS = 30; // the API rejects more (schemas.py: PredictRequest.skills)
const PREFILL_TOTAL = 15; // resume pre-fill: all Skills-section skills, topped up to this many
const $ = (id) => document.getElementById(id);

const state = { skills: [], suggestions: [], active: -1, resumeReady: false };

// ── tiny helpers ──────────────────────────────────────────────────────────────

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

async function api(path, options = {}, timeoutMs = 30000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(API + path, { ...options, signal: ctrl.signal });
    const body = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, body };
  } finally {
    clearTimeout(timer);
  }
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

const pct = (x) => `${Math.round(x * 100)}%`;

// ── connect (the free server sleeps when idle, so the first call can take a few minutes) ─────

async function connect() {
  const started = Date.now();
  for (;;) {
    try {
      const { ok, body } = await api("/health", {}, 10000);
      if (ok) {
        $("model-info").textContent =
          `Model: ${body.model} · version ${body.version} · ${body.algorithm ?? "unknown algorithm"}`;
        state.resumeReady = Boolean(body.features?.resume_upload);
        if (!state.resumeReady) {
          $("resume").disabled = true;
          $("resume-status").textContent = "Resume upload is not available on this server.";
        }
        setStatus("");
        return;
      }
    } catch {
      /* not up yet */
    }
    const waited = Math.round((Date.now() - started) / 1000);
    if (waited > 300) {
      setStatus("Can't reach the server. Please try again later.", true);
      return;
    }
    setStatus(`Waking up the server (free hosting sleeps when idle)… ${waited}s`);
    await new Promise((r) => setTimeout(r, 5000));
  }
}

// ── skills: chips + autocomplete ─────────────────────────────────────────────

function renderChips() {
  $("chips").replaceChildren(
    ...state.skills.map((s) =>
      el("span", { class: "chip" }, s,
        el("button", { type: "button", "aria-label": `Remove ${s}`, onclick: () => removeSkill(s) }, "×")),
    ),
  );
  const missing = MIN_SKILLS - state.skills.length;
  $("analyze").disabled = missing > 0;
  $("analyze").textContent = missing > 0 ? `Add ${missing} more skill${missing > 1 ? "s" : ""}` : "Analyze";
}

function addSkill(skill) {
  const s = skill.trim().toLowerCase();
  if (!s || state.skills.includes(s)) return;
  if (state.skills.length >= MAX_SKILLS) {
    setStatus(`Up to ${MAX_SKILLS} skills. Remove one to add another.`, true);
    return;
  }
  state.skills.push(s);
  renderChips();
}

function removeSkill(skill) {
  state.skills = state.skills.filter((s) => s !== skill);
  renderChips();
}

function renderSuggestions() {
  const list = $("suggestions");
  const items = state.suggestions.filter((s) => !state.skills.includes(s));
  list.replaceChildren(
    ...items.map((s, i) =>
      el("li", { role: "option", "aria-selected": String(i === state.active), onmousedown: () => pick(s) }, s),
    ),
  );
  list.hidden = items.length === 0;
  $("skill-input").setAttribute("aria-expanded", String(!list.hidden));
}

function pick(skill) {
  addSkill(skill);
  $("skill-input").value = "";
  state.suggestions = [];
  state.active = -1;
  renderSuggestions();
  $("skill-input").focus();
}

let searchTimer;
$("skill-input").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  if (!q) {
    state.suggestions = [];
    renderSuggestions();
    return;
  }
  searchTimer = setTimeout(async () => {
    try {
      const { ok, body } = await api(`/skills?q=${encodeURIComponent(q)}&limit=8`);
      if (ok && $("skill-input").value.trim() === q) {
        state.suggestions = body.matches;
        state.active = body.matches.length ? 0 : -1;
        renderSuggestions();
      }
    } catch {
      /* ignore: next keystroke retries */
    }
  }, 150);
});

$("skill-input").addEventListener("keydown", (e) => {
  const visible = state.suggestions.filter((s) => !state.skills.includes(s));
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (!visible.length) return;
    const step = e.key === "ArrowDown" ? 1 : -1;
    state.active = (state.active + step + visible.length) % visible.length;
    renderSuggestions();
  } else if (e.key === "Enter" || e.key === ",") {
    e.preventDefault();
    // Only skills the model knows can be added, so pick from the suggestions.
    if (visible.length) pick(visible[Math.max(state.active, 0)]);
  } else if (e.key === "Backspace" && !e.target.value && state.skills.length) {
    removeSkill(state.skills[state.skills.length - 1]);
  } else if (e.key === "Escape") {
    state.suggestions = [];
    renderSuggestions();
  }
});
$("skill-input").addEventListener("blur", () => setTimeout(() => { $("suggestions").hidden = true; }, 100));

// ── resume upload ────────────────────────────────────────────────────────────

$("resume").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("resume-status").textContent = "Reading your resume…";
  const form = new FormData();
  form.append("file", file);
  try {
    const { ok, status, body } = await api("/extract-skills", { method: "POST", body: form }, 60000);
    if (ok) {
      // Every skill from the resume's Skills section (reliable), plus the best of the rest up to 15 in total:
      // skills picked up from sentences are noisier, and noise makes predictions worse.
      const listed = new Set(body.from_skills_section ?? []);
      const prefill = [];
      for (const s of body.skills) {
        if (prefill.length >= MAX_SKILLS) break;
        if (listed.has(s) || prefill.length < PREFILL_TOTAL) prefill.push(s);
      }
      prefill.forEach(addSkill);
      const note = body.warnings?.length ? ` ${body.warnings.join(" ")}` : "";
      $("resume-status").textContent = prefill.length
        ? `Added ${prefill.length} skills from your resume. Check them below and remove any that are wrong.${note}`
        : (body.message ?? "No skills found in this file. Add them by hand below.");
    } else {
      $("resume-status").textContent =
        typeof body.detail === "string" ? body.detail : `Couldn't read that file (error ${status}).`;
    }
  } catch {
    $("resume-status").textContent = "Upload failed. Please try again.";
  }
  e.target.value = "";
});

// ── analyze + results ────────────────────────────────────────────────────────

$("analyze").addEventListener("click", async () => {
  const exp = $("experience").value;
  setStatus("Analyzing…");
  $("analyze").disabled = true;
  try {
    const { ok, body } = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skills: state.skills, experience_years: exp === "" ? null : Number(exp), top_k: 3 }),
    });
    if (!ok) {
      const d = body.detail;
      setStatus(typeof d === "object" && d?.message ? `${d.message}. Unknown: ${d.unknown_skills.join(", ")}` : "Something went wrong.", true);
      return;
    }
    setStatus("");
    renderResults(body);
  } catch {
    setStatus("The server didn't answer. It may be waking up; try again in a minute.", true);
  } finally {
    renderChips();
  }
});

function meter(label, value) {
  return el("div", {},
    el("div", { class: "meter-label" }, el("span", {}, label), el("strong", {}, pct(value))),
    el("div", { class: "bar", role: "img", "aria-label": `${label} ${pct(value)}` },
      el("span", { style: `width:${Math.max(2, value * 100)}%` })),
  );
}

function trend(growth) {
  if (growth >= 1.15) return "growing in demand";
  if (growth <= 0.87) return "less in demand than in 2019";
  return "steady demand";
}

// Course links come from the API's data; only http(s) URLs become links.
function courseLinks(courses = []) {
  const safe = courses.filter((c) => /^https?:\/\//i.test(c.url));
  if (!safe.length) return "";
  return el("ul", { class: "courses" }, safe.map((c) =>
    el("li", {},
      el("a", { href: c.url, target: "_blank", rel: "noopener noreferrer" }, c.title),
      el("span", { class: "provider" }, ` · ${c.provider}`))));
}

function roleCard(r) {
  const exp = r.typical_experience_years;
  return el("article", { class: "card" },
    el("div", { class: "role-head" },
      el("h3", {}, `${r.rank}. ${r.role}`),
      el("span", { class: "family" }, r.family ?? "")),
    el("div", { class: "meters" }, meter("Match confidence", r.confidence), meter("Readiness", r.readiness)),
    exp ? el("p", { class: "muted" }, `Typical experience asked: ${exp.p25}–${exp.p75} years`) : "",
    r.have.length ? [el("p", { class: "subhead" }, "Core skills you have"),
      el("ul", { class: "skill-list have" }, r.have.map((s) => el("li", {}, s)))] : "",
    r.missing.length ? [el("p", { class: "subhead" }, "Core skills you're missing"),
      el("ul", { class: "skill-list missing" }, r.missing.map((s) => el("li", {}, s)))] : "",
    r.learning_path.length ? [el("p", { class: "subhead" }, "Learn next, in this order"),
      el("ol", { class: "path" }, r.learning_path.slice(0, 5).map((step) =>
        el("li", {}, el("strong", {}, step.skill), " ",
          el("span", { class: "why" }, `asked in ${pct(step.share_of_role_postings)} of these jobs · ${trend(step.market_growth)}`),
          courseLinks(step.courses))))] : "",
  );
}

function renderResults(data) {
  $("low-confidence").hidden = !data.low_confidence;
  $("result-cards").replaceChildren(...data.recommendations.map(roleCard));
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

renderChips();
connect();
