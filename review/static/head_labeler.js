const elements = Object.fromEntries([
  "previous", "next", "frame-select", "zoom", "zoom-value", "undo", "clear",
  "frame-info", "image-canvas", "count", "status", "reviewer", "note",
  "save", "download", "message"
].map(id => [id, document.getElementById(id)]));

const state = { frames: [], annotations: {}, index: 0, image: null, points: [], dirty: false };

function currentFrame() { return state.frames[state.index]; }
function message(text) { elements.message.textContent = text; }

function render(canvas, zoom) {
  if (!state.image) return;
  canvas.width = Math.round(state.image.naturalWidth * zoom);
  canvas.height = Math.round(state.image.naturalHeight * zoom);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, zoom * 3);
  state.points.forEach(([x, y], index) => {
    ctx.beginPath();
    ctx.arc(x * zoom, y * zoom, Math.max(4, zoom * 9), 0, Math.PI * 2);
    ctx.strokeStyle = "#ff2828";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = `${Math.max(12, zoom * 20)}px sans-serif`;
    ctx.fillText(String(index + 1), x * zoom + 8, y * zoom - 8);
  });
}

function draw() {
  if (!state.image) return;
  render(elements["image-canvas"], Number(elements.zoom.value) / 100);
  elements.count.textContent = String(state.points.length);
  elements["zoom-value"].textContent = `${elements.zoom.value}%`;
}

function loadFrame(index) {
  if (state.dirty && !confirm("Discard unsaved changes to this frame?")) {
    elements["frame-select"].value = String(state.index);
    return;
  }
  state.index = index;
  const frame = currentFrame();
  const saved = state.annotations[frame.id] || {};
  state.points = (saved.points || []).map(point => [...point]);
  elements.status.value = saved.status || "draft";
  elements.reviewer.value = saved.reviewer || "";
  elements.note.value = saved.note || "";
  elements["frame-select"].value = String(index);
  elements["frame-info"].textContent = `${frame.video}, frame ${frame.frame} (${frame.timestamp_s}s), ${frame.width}×${frame.height}. YOLO predicted ${frame.detected_people}.`;
  state.dirty = false;
  state.image = null;
  message("");
  const image = new Image();
  image.onload = () => {
    if (currentFrame().id !== frame.id) return;
    state.image = image;
    draw();
  };
  image.onerror = () => message("Could not load this frame.");
  image.src = `/media/${encodeURIComponent(frame.id)}`;
}

elements["image-canvas"].addEventListener("click", event => {
  if (!state.image || elements.status.value === "uncountable") return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = Math.min(state.image.naturalWidth - 1, Math.max(0,
    Math.round((event.clientX - rect.left) * state.image.naturalWidth / rect.width)));
  const y = Math.min(state.image.naturalHeight - 1, Math.max(0,
    Math.round((event.clientY - rect.top) * state.image.naturalHeight / rect.height)));
  if (!state.points.some(([px, py]) => px === x && py === y)) {
    state.points.push([x, y]);
    state.dirty = true;
    draw();
  }
});

elements.zoom.addEventListener("input", draw);
elements.undo.addEventListener("click", () => { state.points.pop(); state.dirty = true; draw(); });
elements.clear.addEventListener("click", () => { state.points = []; state.dirty = true; draw(); });
elements.previous.addEventListener("click", () => loadFrame(Math.max(0, state.index - 1)));
elements.next.addEventListener("click", () => loadFrame(Math.min(state.frames.length - 1, state.index + 1)));
elements["frame-select"].addEventListener("change", event => loadFrame(Number(event.target.value)));
for (const id of ["status", "reviewer", "note"]) {
  elements[id].addEventListener("input", () => { state.dirty = true; });
}

elements.save.addEventListener("click", async () => {
  const frame = currentFrame();
  const previous = state.annotations[frame.id] || {};
  const annotation = {
    expected_revision: previous.revision || 0,
    status: elements.status.value,
    points: state.points,
    reviewer: elements.reviewer.value,
    note: elements.note.value
  };
  try {
    const response = await fetch(`/api/head-labels/${encodeURIComponent(frame.id)}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotation)
    });
    const result = await response.json();
    if (!response.ok) { message(`Not saved: ${result.detail}`); return; }
    state.annotations[frame.id] = result;
    state.dirty = false;
    message(`Saved ${state.points.length} head points (${result.status}).`);
  } catch (error) { message(`Not saved: ${error}`); }
});

elements.download.addEventListener("click", () => {
  if (!state.image) { message("Image not loaded yet."); return; }
  const frame = currentFrame();
  const canvas = document.createElement("canvas");
  render(canvas, 1);
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = `${frame.id}_marked.png`;
  link.click();
  message("Marked image downloaded. Save head labels separately for training.");
});

async function start() {
  const response = await fetch("/api/head-frames");
  if (!response.ok) { message("Could not load audit frames."); return; }
  const data = await response.json();
  state.frames = data.frames;
  state.annotations = data.annotations;
  state.frames.forEach((frame, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1}. ${frame.video} · frame ${frame.frame}`;
    elements["frame-select"].appendChild(option);
  });
  loadFrame(0);
}
start();
