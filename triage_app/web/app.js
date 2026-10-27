// ==========================================================================
// EDGE-POWERED MULTIMODAL EMERGENCY TRIAGE - CLIENT CONTROLLER
// ==========================================================================

let presets = {};
let currentPresetKey = "fire_outbreak";
let customImageBase64 = null;
let rawIotMode = false;
let audioCtx = null;

// DOM Elements
const scenarioChipsEl = document.getElementById("scenario-chips");
const incidentImgEl = document.getElementById("incident-image");
const catBadgeEl = document.getElementById("cat-badge");
const visSliderEl = document.getElementById("vis-slider");
const visValDisplayEl = document.getElementById("vis-val-display");
const imgSourceTag = document.getElementById("img-source-tag");

// Upload Elements
const imageUploadInput = document.getElementById("image-upload");
const btnTriggerUpload = document.getElementById("btn-trigger-upload");
const btnClearImg = document.getElementById("btn-clear-img");

// Text Elements
const transcriptInputEl = document.getElementById("transcript-input");
const keywordCloudEl = document.getElementById("keyword-cloud");

// IoT Telemetry Elements
const tempSliderEl = document.getElementById("temp-slider");
const tempDisplayEl = document.getElementById("temp-display");
const smokeSliderEl = document.getElementById("smoke-slider");
const smokeDisplayEl = document.getElementById("smoke-display");
const gpsDisplayEl = document.getElementById("gps-display");
const hospitalDisplayEl = document.getElementById("hospital-display");
const btnToggleRawIot = document.getElementById("btn-toggle-raw-iot");
const sensorModeBadge = document.getElementById("sensor-mode-badge");

// Modality Ablation
const modVisCheck = document.getElementById("mod-vis");
const modTxtCheck = document.getElementById("mod-txt");
const modIotCheck = document.getElementById("mod-iot");
const btnRunTriage = document.getElementById("btn-run-triage");

// Result Elements
const liveLatencyEl = document.getElementById("live-latency");
const latencyValEl = document.getElementById("latency-val");
const priorityHeroCardEl = document.getElementById("priority-hero-card");
const priorityBadgeBigEl = document.getElementById("priority-badge-big");
const priorityNameEl = document.getElementById("priority-name");
const priorityDescEl = document.getElementById("priority-desc");
const confidenceBadgeEl = document.getElementById("confidence-badge");

const probP1El = document.getElementById("prob-p1");
const probP2El = document.getElementById("prob-p2");
const probP3El = document.getElementById("prob-p3");
const probP4El = document.getElementById("prob-p4");
const probBarP1 = document.getElementById("prob-bar-p1");
const probBarP2 = document.getElementById("prob-bar-p2");
const probBarP3 = document.getElementById("prob-bar-p3");
const probBarP4 = document.getElementById("prob-bar-p4");

const shareVisVal = document.getElementById("share-vis-val");
const shareTxtVal = document.getElementById("share-txt-val");
const shareIotVal = document.getElementById("share-iot-val");
const shareVisBar = document.getElementById("share-vis-bar");
const shareTxtBar = document.getElementById("share-txt-bar");
const shareIotBar = document.getElementById("share-iot-bar");

const dispatchCodeEl = document.getElementById("dispatch-code");
const dispatchActionEl = document.getElementById("dispatch-action");
const dispatchHospitalEl = document.getElementById("dispatch-hospital");
const dispatchEtaEl = document.getElementById("dispatch-eta");
const dispatchedUnitsListEl = document.getElementById("dispatched-units-list");

const btnCopyLog = document.getElementById("btn-copy-log");
const btnSiren = document.getElementById("btn-siren");

// Terminal Console Elements
const terminalScreenEl = document.getElementById("terminal-screen");
const termStatusMsgEl = document.getElementById("term-status-msg");
const btnTermClear = document.getElementById("btn-term-clear");
const btnTermCopy = document.getElementById("btn-term-copy");

// Initialize System
async function init() {
  try {
    const res = await fetch("/api/presets");
    presets = await res.json();
    renderScenarioPills();
    loadScenario("fire_outbreak");
  } catch (err) {
    console.warn("Could not load /api/presets, fallback to default", err);
  }
  setupCustomInputHandlers();
}

// Render Scenario Buttons (Includes Blank / Custom Test)
function renderScenarioPills() {
  scenarioChipsEl.innerHTML = "";

  // Custom Blank Test Pill
  const customBtn = document.createElement("button");
  customBtn.className = "scenario-btn";
  customBtn.style.borderColor = "var(--accent-purple)";
  customBtn.style.color = "var(--accent-purple)";
  customBtn.innerHTML = "✨ <strong>+ Custom / Blank Test</strong>";
  customBtn.onclick = () => {
    document.querySelectorAll(".scenario-btn").forEach(b => b.classList.remove("active"));
    customBtn.classList.add("active");
    loadBlankCustom();
  };
  scenarioChipsEl.appendChild(customBtn);

  // Preset Scenario Pills
  for (const [key, scenario] of Object.entries(presets)) {
    const btn = document.createElement("button");
    btn.className = `scenario-btn ${key === currentPresetKey ? "active" : ""}`;
    btn.textContent = scenario.title;
    btn.onclick = () => {
      document.querySelectorAll(".scenario-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      loadScenario(key);
    };
    scenarioChipsEl.appendChild(btn);
  }
}

// Blank Custom Incident for Free-Form Testing
function loadBlankCustom() {
  currentPresetKey = "custom";
  customImageBase64 = null;
  incidentImgEl.src = "https://images.unsplash.com/photo-1582139329536-e7284fece509?w=600&auto=format&fit=crop&q=80";
  imgSourceTag.textContent = "CUSTOM TESTING";
  catBadgeEl.textContent = "Free-Form Input";

  // Empty text ready for user typing
  transcriptInputEl.value = "";
  transcriptInputEl.placeholder = "Type ANY short text (e.g. 'fire on roof', 'car hit tree', 'chest pain', 'sprained ankle')...";
  keywordCloudEl.innerHTML = "<span style='font-size:0.75rem; color:#64748b;'>Ready for your input...</span>";

  // Default normal ambient conditions (Zero Rich IoT)
  visSliderEl.value = 50;
  visValDisplayEl.textContent = "50%";

  tempSliderEl.value = 24;
  tempDisplayEl.textContent = "24 °C";
  tempDisplayEl.className = "meter-val text-emerald";

  smokeSliderEl.value = 0;
  smokeDisplayEl.textContent = "0 PPM";
  smokeDisplayEl.className = "meter-val text-emerald";

  gpsDisplayEl.textContent = "12.9240° N, 77.6101° E (User Device Location)";
  hospitalDisplayEl.textContent = "Nearest Emergency Center (1.5 km)";

  runTriage();
}

// Load a specific scenario into the UI
function loadScenario(key) {
  currentPresetKey = key;
  customImageBase64 = null;
  btnClearImg.style.display = "none";
  imgSourceTag.textContent = "LIVE FEED";

  const sc = presets[key];
  if (!sc) return;

  incidentImgEl.src = sc.image_url;
  catBadgeEl.textContent = sc.category;
  transcriptInputEl.value = sc.transcript;

  renderKeywords(sc.transcript);

  // Sliders
  tempSliderEl.value = sc.iot.temperature_c;
  tempDisplayEl.textContent = `${sc.iot.temperature_c} °C`;

  smokeSliderEl.value = sc.iot.smoke_ppm;
  smokeDisplayEl.textContent = `${sc.iot.smoke_ppm} PPM`;

  const initialVis = sc.expected_class === 0 ? 88 : sc.expected_class === 1 ? 68 : sc.expected_class === 2 ? 40 : 15;
  visSliderEl.value = initialVis;
  visValDisplayEl.textContent = `${initialVis}%`;

  gpsDisplayEl.textContent = `${sc.gps.lat.toFixed(4)}° N, ${sc.gps.lng.toFixed(4)}° E (${sc.gps.address})`;
  hospitalDisplayEl.textContent = sc.nearest_facility;

  runTriage();
}

// Setup Custom Image Upload & Short Text Handlers
function setupCustomInputHandlers() {
  // Trigger file upload on button click
  btnTriggerUpload.addEventListener("click", () => {
    imageUploadInput.click();
  });

  // Handle image file selection
  imageUploadInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      customImageBase64 = event.target.result;
      incidentImgEl.src = customImageBase64;
      imgSourceTag.textContent = "CUSTOM USER IMAGE";
      catBadgeEl.textContent = `Uploaded: ${file.name.substring(0, 16)}`;
      btnClearImg.style.display = "inline-block";

      // Auto-trigger prediction with uploaded image
      runTriage();
    };
    reader.readAsDataURL(file);
  });

  // Reset image
  btnClearImg.addEventListener("click", () => {
    customImageBase64 = null;
    btnClearImg.style.display = "none";
    if (presets[currentPresetKey]) {
      loadScenario(currentPresetKey);
    } else {
      loadBlankCustom();
    }
  });

  // Short test prompt buttons
  document.querySelectorAll(".chip-prompt").forEach(btn => {
    btn.addEventListener("click", () => {
      transcriptInputEl.value = btn.dataset.text;
      renderKeywords(btn.dataset.text);
      runTriage();
    });
  });

  // Toggle "No Rich IoT Data" Mode
  btnToggleRawIot.addEventListener("click", () => {
    rawIotMode = !rawIotMode;
    btnToggleRawIot.classList.toggle("active", rawIotMode);

    if (rawIotMode) {
      tempSliderEl.value = 24;
      tempDisplayEl.textContent = "24 °C (Normal)";
      tempDisplayEl.className = "meter-val text-emerald";

      smokeSliderEl.value = 0;
      smokeDisplayEl.textContent = "0 PPM (Normal)";
      smokeDisplayEl.className = "meter-val text-emerald";

      sensorModeBadge.textContent = "Raw / No Sensors";
      btnToggleRawIot.querySelector("#raw-iot-text").textContent = "✓ 'No Rich IoT Data' Mode ACTIVE (Baseline Room Ambient)";
    } else {
      sensorModeBadge.textContent = "32-D Telemetry";
      btnToggleRawIot.querySelector("#raw-iot-text").textContent = "Simulate 'Zero Rich IoT Data' (Normal 24°C, 0 Smoke)";
      if (presets[currentPresetKey]) {
        tempSliderEl.value = presets[currentPresetKey].iot.temperature_c;
        tempDisplayEl.textContent = `${presets[currentPresetKey].iot.temperature_c} °C`;
        smokeSliderEl.value = presets[currentPresetKey].iot.smoke_ppm;
        smokeDisplayEl.textContent = `${presets[currentPresetKey].iot.smoke_ppm} PPM`;
      }
    }
    runTriage();
  });

  // Real-time keyword tagging as user types short texts
  transcriptInputEl.addEventListener("input", () => {
    renderKeywords(transcriptInputEl.value);
  });
}

function renderKeywords(text) {
  const watchwords = ["mayday", "trapped", "fire", "smoke", "rupture", "pileup", "crumpled", "unresponsive", "bleeding", "collapse", "severe", "flood", "submerged", "fracture", "cut", "pain", "gas", "burn", "chest", "crash", "choking"];
  const words = text.toLowerCase().split(/\W+/);
  const matched = [...new Set(words.filter(w => watchwords.includes(w)))];

  keywordCloudEl.innerHTML = "";
  if (matched.length === 0 && text.trim().length > 0) {
    const hint = document.createElement("span");
    hint.className = "keyword-pill";
    hint.style.background = "rgba(168, 85, 247, 0.15)";
    hint.style.borderColor = "var(--accent-purple)";
    hint.style.color = "#d8b4fe";
    hint.textContent = `📝 Text Tokens: ${words.filter(Boolean).length} words`;
    keywordCloudEl.appendChild(hint);
  } else {
    matched.forEach(w => {
      const pill = document.createElement("span");
      pill.className = "keyword-pill";
      pill.textContent = `⚡ ${w.toUpperCase()}`;
      keywordCloudEl.appendChild(pill);
    });
  }
}

// Live Sliders Event Listeners
visSliderEl.addEventListener("input", (e) => {
  visValDisplayEl.textContent = `${e.target.value}%`;
});

tempSliderEl.addEventListener("input", (e) => {
  tempDisplayEl.textContent = `${e.target.value} °C`;
  if (e.target.value > 70) {
    tempDisplayEl.className = "meter-val text-red";
  } else if (e.target.value > 40) {
    tempDisplayEl.className = "meter-val text-amber";
  } else {
    tempDisplayEl.className = "meter-val text-emerald";
  }
});

smokeSliderEl.addEventListener("input", (e) => {
  smokeDisplayEl.textContent = `${e.target.value} PPM`;
  if (e.target.value > 500) {
    smokeDisplayEl.className = "meter-val text-red";
  } else if (e.target.value > 200) {
    smokeDisplayEl.className = "meter-val text-amber";
  } else {
    smokeDisplayEl.className = "meter-val text-emerald";
  }
});

// Run Multimodal Triage API Request
async function runTriage() {
  btnRunTriage.classList.add("executing");

  const payload = {
    text: transcriptInputEl.value,
    category: catBadgeEl.textContent,
    image_b64: customImageBase64,
    image_hazard_override: parseFloat(visSliderEl.value) / 100.0,
    iot_data: {
      temperature_c: parseFloat(tempSliderEl.value),
      smoke_ppm: parseFloat(smokeSliderEl.value),
      gas_leak_ppm: parseFloat(smokeSliderEl.value) * 0.4,
      structural_vibration: parseFloat(visSliderEl.value) / 100.0
    },
    active_modalities: {
      visual: modVisCheck.checked,
      textual: modTxtCheck.checked,
      iot: modIotCheck.checked
    }
  };

  try {
    const res = await fetch("/api/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await res.json();
    updateDashboardResults(result);
  } catch (err) {
    console.error("Triage prediction error:", err);
  } finally {
    btnRunTriage.classList.remove("executing");
  }
}

// Update UI with Model Output
function updateDashboardResults(result) {
  liveLatencyEl.textContent = `${result.latency_ms} ms`;
  latencyValEl.textContent = `${result.latency_ms} ms`;

  // Priority Hero Card Styling
  const color = result.color;
  priorityHeroCardEl.style.borderColor = color;
  priorityHeroCardEl.style.boxShadow = `0 0 25px ${color}33`;
  priorityHeroCardEl.style.background = `linear-gradient(135deg, ${color}22, rgba(15, 23, 42, 0.85))`;

  priorityBadgeBigEl.textContent = result.priority;
  priorityBadgeBigEl.style.background = color;
  priorityBadgeBigEl.style.boxShadow = `0 0 20px ${color}88`;

  priorityNameEl.textContent = result.priority_label;
  priorityNameEl.style.color = color;
  priorityDescEl.textContent = result.description;

  confidenceBadgeEl.textContent = `${result.confidence}% CONFIDENCE`;
  confidenceBadgeEl.style.color = color;
  confidenceBadgeEl.style.borderColor = color;
  confidenceBadgeEl.style.background = `${color}22`;

  // Probabilities
  const p = result.class_probabilities;
  probP1El.textContent = `${p.P1}%`;
  probBarP1.style.width = `${p.P1}%`;

  probP2El.textContent = `${p.P2}%`;
  probBarP2.style.width = `${p.P2}%`;

  probP3El.textContent = `${p.P3}%`;
  probBarP3.style.width = `${p.P3}%`;

  probP4El.textContent = `${p.P4}%`;
  probBarP4.style.width = `${p.P4}%`;

  // Modality Contributions
  const shares = result.modality_shares;
  shareVisVal.textContent = `${shares.visual}%`;
  shareVisBar.style.width = `${shares.visual}%`;

  shareTxtVal.textContent = `${shares.textual}%`;
  shareTxtBar.style.width = `${shares.textual}%`;

  shareIotVal.textContent = `${shares.iot_spatial}%`;
  shareIotBar.style.width = `${shares.iot_spatial}%`;

  // Dispatch Report
  const disp = result.dispatch_report;
  dispatchCodeEl.textContent = disp.incident_code;
  dispatchActionEl.textContent = disp.response_status;
  dispatchHospitalEl.textContent = disp.hospital_routing;
  dispatchEtaEl.textContent = disp.estimated_first_responder_eta;

  dispatchedUnitsListEl.innerHTML = "";
  disp.dispatched_units.forEach(u => {
    const chip = document.createElement("span");
    chip.className = "unit-tag";
    chip.textContent = u;
    dispatchedUnitsListEl.appendChild(chip);
  });

  // Stream execution logs to the Live Edge Terminal Console
  if (result.execution_logs && result.execution_logs.length) {
    streamExecutionLogs(result.execution_logs);
  }
}

// Terminal Console Functions
function addTerminalLine(rawText, forceTag = null) {
  if (!terminalScreenEl) return;
  
  const lineDiv = document.createElement("div");
  lineDiv.className = "term-line";

  // Parse pattern: "[12:34:56] [ViT-BASE] Extracted 768-D..."
  const timeMatch = rawText.match(/^\[(\d{2}:\d{2}:\d{2})\]/);
  let timeStr = "";
  let rest = rawText;
  if (timeMatch) {
    timeStr = timeMatch[1];
    rest = rawText.substring(timeMatch[0].length).trim();
  }

  const tagMatch = rest.match(/^\[(.*?)\]/);
  let tagStr = forceTag || "SYS";
  let contentStr = rest;
  if (tagMatch) {
    tagStr = tagMatch[1];
    contentStr = rest.substring(tagMatch[0].length).trim();
  }

  // Determine tag CSS styling
  const tagUpper = tagStr.toUpperCase();
  let tagClass = "tag-sys";
  if (tagUpper.includes("VIT") || tagUpper.includes("VISION")) tagClass = "tag-vit";
  else if (tagUpper.includes("BERT") || tagUpper.includes("DISTILBERT") || tagUpper.includes("NLP")) tagClass = "tag-bert";
  else if (tagUpper.includes("IOT") || tagUpper.includes("GIS")) tagClass = "tag-iot";
  else if (tagUpper.includes("FUSION")) tagClass = "tag-fusion";
  else if (tagUpper.includes("SOFTMAX")) tagClass = "tag-softmax";
  else if (tagUpper.includes("ENSEMBLE")) tagClass = "tag-ensemble";
  else if (tagUpper.includes("DISPATCH")) tagClass = "tag-dispatch";
  else if (tagUpper.includes("INGEST")) tagClass = "tag-ingest";
  else if (tagUpper.includes("READY") || tagUpper.includes("STATUS")) tagClass = "tag-ready";
  else if (tagUpper.includes("MODELS")) tagClass = "tag-mod";

  let html = "";
  if (timeStr) {
    html += `<span class="term-time">${timeStr}</span>`;
  }
  html += `<span class="term-tag ${tagClass}">[${tagStr}]</span>`;
  html += `<span>${contentStr}</span>`;

  lineDiv.innerHTML = html;
  terminalScreenEl.appendChild(lineDiv);

  // Auto-scroll to latest stdout
  terminalScreenEl.scrollTop = terminalScreenEl.scrollHeight;
}

let activeLogTimeouts = [];

function streamExecutionLogs(logs) {
  if (!logs || !logs.length) return;
  
  // Clear any pending timeouts
  activeLogTimeouts.forEach(t => clearTimeout(t));
  activeLogTimeouts = [];

  if (termStatusMsgEl) {
    termStatusMsgEl.textContent = `Processing ${logs.length} edge pipeline telemetry events...`;
  }
  
  // Stream with slight delay for realistic pipeline feel
  logs.forEach((logLine, index) => {
    const t = setTimeout(() => {
      addTerminalLine(logLine);
      if (index === logs.length - 1 && termStatusMsgEl) {
        termStatusMsgEl.textContent = `Inference cycle complete &bull; Edge socket ready.`;
      }
    }, index * 45);
    activeLogTimeouts.push(t);
  });
}

function playSiren() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();

  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(800, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(450, audioCtx.currentTime + 0.35);
  osc.frequency.exponentialRampToValueAtTime(800, audioCtx.currentTime + 0.7);

  gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.7);

  osc.connect(gain);
  gain.connect(audioCtx.destination);

  osc.start();
  osc.stop(audioCtx.currentTime + 0.75);
}

// Event Listeners
btnRunTriage.addEventListener("click", () => {
  runTriage();
});

[modVisCheck, modTxtCheck, modIotCheck].forEach(chk => {
  chk.addEventListener("change", () => {
    runTriage();
  });
});

btnSiren.addEventListener("click", () => {
  playSiren();
});

btnCopyLog.addEventListener("click", () => {
  const logText = `
=== EMERGENCY TRIAGE DISPATCH LOG ===
Incident Code: ${dispatchCodeEl.textContent}
Priority Level: ${priorityNameEl.textContent} (${confidenceBadgeEl.textContent})
Protocol Directive: ${dispatchActionEl.textContent}
Medical Facility: ${dispatchHospitalEl.textContent}
Estimated ETA: ${dispatchEtaEl.textContent}
Dispatched Units: ${Array.from(dispatchedUnitsListEl.children).map(c => c.textContent).join(", ")}
=====================================
`;
  navigator.clipboard.writeText(logText).then(() => {
    const original = btnCopyLog.innerHTML;
    btnCopyLog.innerHTML = "<span>✓ Copied to Clipboard!</span>";
    setTimeout(() => { btnCopyLog.innerHTML = original; }, 2000);
  });
});

// Terminal Action Event Listeners
if (btnTermClear) {
  btnTermClear.addEventListener("click", () => {
    activeLogTimeouts.forEach(t => clearTimeout(t));
    activeLogTimeouts = [];
    terminalScreenEl.innerHTML = "";
    const now = new Date().toLocaleTimeString();
    addTerminalLine(`[${now}] [STATUS] Terminal screen cleared. Edge loop active on localhost:8080.`, "READY");
    if (termStatusMsgEl) termStatusMsgEl.textContent = "Screen cleared &bull; awaiting next trigger";
  });
}

if (btnTermCopy) {
  btnTermCopy.addEventListener("click", () => {
    const allLines = terminalScreenEl.innerText;
    navigator.clipboard.writeText(allLines).then(() => {
      const orig = btnTermCopy.innerHTML;
      btnTermCopy.innerHTML = "<span>✓ Copied!</span>";
      setTimeout(() => { btnTermCopy.innerHTML = orig; }, 1500);
    });
  });
}

// Launch
init();
