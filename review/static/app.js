const el = id => document.getElementById(id);
let data, current, revision = 0, windowMode = false, dirty = false;
const video = el("video");

function summary() {
  const entries = Object.values(data.annotations);
  const reviewed = entries.filter(a => a.status === "reviewed").length;
  el("stats").replaceChildren();
  for (const [value, label] of [
    [data.source_counts.candidate_links, "source links"],
    [data.source_counts.candidate_downloads, "of those downloaded"],
    [data.source_counts.user_supplied_local_clips, "user-supplied local clips"],
    [data.source_counts.experimental_clips, "clips in this experiment"],
    [reviewed, "reviewed observations"],
    [data.exportable_count, "exportable windows"]
  ]) {
    const card = document.createElement("div");
    card.className = "stat";
    const strong = document.createElement("strong");
    const span = document.createElement("span");
    strong.textContent = value;
    span.textContent = label;
    card.append(strong, span);
    el("stats").append(card);
  }
}

function show(index) {
  video.pause();
  windowMode = false;
  current = data.windows[index];
  const a = data.annotations[current.id] || {};
  revision = a.revision || 0;
  el("window").value = String(index);
  el("quality").textContent = current.quality_eligible ? "Numerical quality gate passed" : "Indeterminate features — not exportable";
  el("quality").className = current.quality_eligible ? "" : "warn";
  el("time").textContent = `${current.start_s.toFixed(2)}–${current.end_s.toFixed(2)} seconds`;
  el("event").textContent = current.event_group;
  el("proposal").textContent = `Proposal only: ${current.proposal}`;
  el("rights").textContent = `Source: ${data.clips[current.clip_id].filename}.`;
  const source = `/media/${current.clip_id}`;
  if (video.getAttribute("src") !== source) {
    video.src = source;
    video.load();
  } else if (video.readyState >= 1) {
    video.currentTime = current.start_s;
  }
  el("behaviour").value = a.behaviour_label || "";
  el("status").value = a.status || "draft";
  el("reviewer").value = a.reviewer || "";
  el("note").value = a.note || "";
  el("message").textContent = "";
  dirty = false;
  el("previous").disabled = index === 0;
  el("next").disabled = index === data.windows.length - 1;
}

function navigate(index) {
  if (index < 0 || index >= data.windows.length) return;
  if (dirty && !confirm("Discard unsaved form changes?")) {
    el("window").value = String(data.windows.indexOf(current));
    return;
  }
  show(index);
}

video.addEventListener("loadedmetadata", () => { if (current) video.currentTime = current.start_s; });
video.addEventListener("timeupdate", () => {
  if (windowMode && video.currentTime >= current.end_s) {
    video.pause();
    windowMode = false;
  }
});
el("play").onclick = async () => {
  video.currentTime = current.start_s;
  windowMode = true;
  try { await video.play(); }
  catch (error) { el("message").textContent = `Playback unavailable: ${error.message}`; }
};
el("previous").onclick = () => navigate(data.windows.indexOf(current) - 1);
el("next").onclick = () => navigate(data.windows.indexOf(current) + 1);
el("window").onchange = () => navigate(Number(el("window").value));
el("annotation").addEventListener("input", () => { dirty = true; });
window.addEventListener("beforeunload", event => {
  if (dirty) { event.preventDefault(); event.returnValue = ""; }
});
el("annotation").onsubmit = async event => {
  event.preventDefault();
  const payload = {
    expected_revision: revision,
    status: el("status").value,
    behaviour_label: el("behaviour").value || null,
    reviewer: el("reviewer").value,
    note: el("note").value
  };
  try {
    const response = await fetch(`/api/annotations/${current.id}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Check the fields and try again");
    data.annotations[current.id] = result;
    revision = result.revision;
    dirty = false;
    const latest = await (await fetch("/api/review")).json();
    data.exportable_count = latest.exportable_count;
    summary();
    el("message").textContent = "Saved locally. This is a behaviour observation, not a risk label.";
  } catch (error) { el("message").textContent = error.message; }
};

(async () => {
  try {
    const response = await fetch("/api/review");
    if (!response.ok) throw new Error("Review package unavailable");
    data = await response.json();
    for (const [index, item] of data.windows.entries()) {
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `${item.clip_id} · ${item.start_s.toFixed(2)}–${item.end_s.toFixed(2)}s${item.quality_eligible ? "" : " · indeterminate"}`;
      el("window").append(option);
    }
    summary();
    show(0);
  } catch (error) { el("message").textContent = error.message; }
})();
