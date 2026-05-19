const pollIntervalMs = 750;

const state = {
  scan: null,
  selectedReport: null,
  currentOutputDir: "",
  preparingInput: false,
};

const elements = {
  inputDir: document.getElementById("input-dir"),
  outputDir: document.getElementById("output-dir"),
  mainEnvName: document.getElementById("main-env-name"),
  downstreamEnvName: document.getElementById("downstream-env-name"),
  downstreamEnvField: document.getElementById("downstream-env-field"),
  gpu: document.getElementById("gpu"),
  umPerPixel: document.getElementById("um-per-pixel"),
  frameRate: document.getElementById("frame-rate"),
  llmMode: document.getElementById("llm-mode"),
  datasetType: document.getElementById("dataset-type"),
  apiKeyField: document.getElementById("api-key-field"),
  openaiApiKey: document.getElementById("openai-api-key"),
  scanButton: document.getElementById("scan-button"),
  startButton: document.getElementById("start-button"),
  loadReportsButton: document.getElementById("load-reports-button"),
  selectAllButton: document.getElementById("select-all-button"),
  clearAllButton: document.getElementById("clear-all-button"),
  flash: document.getElementById("flash"),
  subfolderList: document.getElementById("subfolder-list"),
  scanSummary: document.getElementById("scan-summary"),
  jobStatusBadge: document.getElementById("job-status-badge"),
  jobMessage: document.getElementById("job-message"),
  startedAt: document.getElementById("started-at"),
  finishedAt: document.getElementById("finished-at"),
  jobPid: document.getElementById("job-pid"),
  returnCode: document.getElementById("return-code"),
  commandPreview: document.getElementById("command-preview"),
  logOutput: document.getElementById("log-output"),
  reportTree: document.getElementById("report-tree"),
  reportFrame: document.getElementById("report-frame"),
  viewerPath: document.getElementById("viewer-path"),
  openReportLink: document.getElementById("open-report-link"),
  deploymentNote: document.getElementById("deployment-note"),
};

function setFlash(message, tone = "neutral") {
  elements.flash.className = `flash flash-${tone}`;
  elements.flash.textContent = message;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

async function getJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

function isLocalBrowserHost() {
  const host = (window.location.hostname || "").toLowerCase();
  return host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "";
}

async function renderDeploymentContext() {
  if (!elements.deploymentNote) {
    return;
  }
  try {
    const info = await getJson("/api/server-info");
    const localAccess = isLocalBrowserHost();
    elements.deploymentNote.classList.toggle("hidden", localAccess);
    if (!localAccess) {
      elements.deploymentNote.innerHTML = `
        <strong>Web deployment path rule</strong>
        <span>
          Input and output paths are resolved on the NeuroPilot UI server, not on this browser's computer.
          Local paths such as <code>D:\\data\\study</code> only work if that exact path exists on the server.
          Copy, upload, or mount your TIFF data on the server first, then enter the server-side path.
        </span>
        <small>Server root: ${info.server_root || "unknown"}</small>
      `;
      elements.inputDir.placeholder = "Server path, e.g. /data/neuropilot/input";
      elements.outputDir.placeholder = "Server path, e.g. /data/neuropilot/runs/study_001";
    }
  } catch (error) {
    elements.deploymentNote.classList.remove("hidden");
    elements.deploymentNote.textContent = `Could not read server deployment context: ${error.message}`;
  }
}

function toggleConditionalFields() {
  const llmMode = elements.llmMode.value;
  const datasetType = elements.datasetType.value;
  elements.apiKeyField.classList.toggle("hidden", llmMode !== "apply");
  elements.downstreamEnvField.classList.toggle("hidden", datasetType !== "cell-data");
}

function formatList(items) {
  if (!items || !items.length) {
    return "None";
  }
  return items.join(", ");
}

function renderSubfolders(scan) {
  if (!scan || !scan.subfolders || !scan.subfolders.length) {
    elements.subfolderList.className = "subfolder-list empty-state";
    elements.subfolderList.textContent = "No dataset subfolders were detected in the selected input directory.";
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "subfolder-grid";

  scan.subfolders.forEach((item) => {
    const card = document.createElement("label");
    card.className = `subfolder-card ${item.has_tiffs ? "is-valid" : "is-invalid"}`;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = item.name;
    checkbox.className = "subfolder-checkbox";
    checkbox.checked = scan.default_selected_subfolders.includes(item.name);
    checkbox.disabled = !item.has_tiffs;

    const meta = document.createElement("div");
    meta.className = "subfolder-meta";
    meta.innerHTML = `
      <strong>${item.name}</strong>
      <span>${item.tif_count} TIFF file(s)</span>
      <small>${item.has_tiffs ? formatList(item.sample_tifs) : "No TIFF files were found inside this subfolder."}</small>
    `;

    card.appendChild(checkbox);
    card.appendChild(meta);
    wrapper.appendChild(card);
  });

  elements.subfolderList.className = "subfolder-list";
  elements.subfolderList.innerHTML = "";
  elements.subfolderList.appendChild(wrapper);
}

function renderScanSummary(scan) {
  if (!scan) {
    elements.scanSummary.className = "scan-summary empty-state";
    elements.scanSummary.textContent = "No input directory has been scanned yet.";
    return;
  }

  const readiness = scan.can_run ? "Ready for launch" : "Needs attention";
  const prepareHtml = scan.should_prepare_input
    ? `
      <div class="prepare-callout">
        <p><strong>Suggested preprocessing step</strong></p>
        <p>
          Flat TIFF files were detected directly under the selected input directory.
          Use the preparation button below to create dataset subfolders in place while keeping the original flat TIFF files untouched.
        </p>
        <div class="prepare-actions">
          <button id="prepare-input-button" class="button button-ghost" type="button">Prepare flat TIFFs in place</button>
        </div>
        <pre class="inline-command">${scan.prepare_command}</pre>
      </div>
    `
    : "";
  const summaryHtml = `
    <div class="summary-grid">
      <article class="summary-card">
        <span class="summary-label">Readiness</span>
        <strong>${readiness}</strong>
      </article>
      <article class="summary-card">
        <span class="summary-label">Valid subfolders</span>
        <strong>${scan.valid_subfolders.length || 0}</strong>
      </article>
      <article class="summary-card">
        <span class="summary-label">Invalid subfolders</span>
        <strong>${scan.invalid_subfolders.length || 0}</strong>
      </article>
      <article class="summary-card">
        <span class="summary-label">Loose root TIFF files</span>
        <strong>${scan.loose_tifs.length || 0}</strong>
      </article>
    </div>
    <div class="summary-list">
      <p><strong>Input directory</strong><span>${scan.input_dir}</span></p>
      <p><strong>Server-resolved path</strong><span>${scan.server_resolved_input_dir || scan.input_dir}</span></p>
      <p><strong>Server root</strong><span>${scan.server_root || "Unknown"}</span></p>
      <p><strong>Valid names</strong><span>${scan.valid_subfolders.length ? scan.valid_subfolders.join(", ") : "None"}</span></p>
      <p><strong>Invalid names</strong><span>${scan.invalid_subfolders.length ? scan.invalid_subfolders.join(", ") : "None"}</span></p>
      <p><strong>Loose TIFF files</strong><span>${scan.loose_tifs.length ? scan.loose_tifs.join(", ") : "None"}</span></p>
      <p><strong>Prepared root TIFF files</strong><span>${scan.prepared_loose_tifs && scan.prepared_loose_tifs.length ? scan.prepared_loose_tifs.join(", ") : "None"}</span></p>
      <p><strong>Unprepared root TIFF files</strong><span>${scan.unprepared_loose_tifs && scan.unprepared_loose_tifs.length ? scan.unprepared_loose_tifs.join(", ") : "None"}</span></p>
    </div>
    ${prepareHtml}
    <ul class="message-list">${scan.messages.map((item) => `<li>${item}</li>`).join("")}</ul>
  `;

  elements.scanSummary.className = "scan-summary";
  elements.scanSummary.innerHTML = summaryHtml;
  attachPrepareButton();
}

function attachPrepareButton() {
  const button = document.getElementById("prepare-input-button");
  if (!button) {
    return;
  }
  button.disabled = state.preparingInput;
  button.textContent = state.preparingInput
    ? "Preparing input structure..."
    : "Prepare flat TIFFs in place";
  button.addEventListener("click", prepareInputDir);
}

function collectSelectedSubfolders() {
  return Array.from(document.querySelectorAll(".subfolder-checkbox"))
    .filter((node) => node.checked && !node.disabled)
    .map((node) => node.value);
}

function selectAllValidSubfolders(checked) {
  document.querySelectorAll(".subfolder-checkbox").forEach((node) => {
    if (!node.disabled) {
      node.checked = checked;
    }
  });
}

function buildJobPayload() {
  return {
    input_dir: elements.inputDir.value.trim(),
    output_dir: elements.outputDir.value.trim(),
    main_env_name: elements.mainEnvName.value.trim(),
    downstream_env_name: elements.downstreamEnvName.value.trim(),
    gpu: elements.gpu.value.trim(),
    um_per_pixel: elements.umPerPixel.value.trim(),
    frame_rate: elements.frameRate.value.trim(),
    llm_mode: elements.llmMode.value,
    openai_api_key: elements.openaiApiKey.value,
    dataset_type: elements.datasetType.value,
    subfolders: collectSelectedSubfolders(),
  };
}

async function scanInputDir() {
  try {
    const data = await postJson("/api/scan-input", {
      input_dir: elements.inputDir.value.trim(),
    });
    state.scan = data;
    renderSubfolders(data);
    renderScanSummary(data);
    if (data.can_run) {
      if (data.prepared_loose_tifs && data.prepared_loose_tifs.length) {
        setFlash("Prepared dataset subfolders were detected for the root-level TIFF files. The originals were preserved, and the directory is ready for launch.", "success");
      } else {
        setFlash("Input structure check passed. The selected directory is ready for launch.", "success");
      }
    } else if ((data.total_tif_count || 0) === 0) {
      setFlash(`No TIFF files were found on the UI server at ${data.server_resolved_input_dir || data.input_dir}. If the files are on your browser computer, copy/upload/mount them to the server first; preparation is not available for an empty server path.`, "danger");
    } else if (data.should_prepare_input) {
      setFlash("Flat TIFF files were detected at the input root. Use the prepare button shown below before launching the pipeline.", "warning");
    } else {
      setFlash("Input structure still needs attention. Resolve the reported issues before launching a run.", "warning");
    }
  } catch (error) {
    setFlash(error.message, "danger");
  }
}

async function prepareInputDir() {
  if (state.preparingInput) {
    return;
  }
  try {
    state.preparingInput = true;
    attachPrepareButton();
    const data = await postJson("/api/prepare-input", {
      input_dir: elements.inputDir.value.trim(),
    });
    state.scan = data.scan;
    renderSubfolders(data.scan);
    renderScanSummary(data.scan);
    setFlash(
      `Preparation completed. ${data.summary.files_found} TIFF file(s) were processed into dataset subfolders, and the original flat TIFF files were preserved.`,
      "success",
    );
  } catch (error) {
    setFlash(error.message, "danger");
  } finally {
    state.preparingInput = false;
    attachPrepareButton();
  }
}

function renderJobSnapshot(job) {
  const status = (job.status || "idle").toLowerCase();
  if (elements.jobStatusBadge) {
    elements.jobStatusBadge.className = `badge badge-${status}`;
    elements.jobStatusBadge.textContent = status.toUpperCase();
  }
  if (elements.jobMessage) {
    elements.jobMessage.textContent = job.message || "-";
  }
  elements.startedAt.textContent = job.started_at || "-";
  elements.finishedAt.textContent = job.finished_at || "-";
  elements.jobPid.textContent = job.pid ?? "-";
  elements.returnCode.textContent = job.returncode ?? "-";
  elements.commandPreview.textContent = job.command && job.command.length
    ? job.command.join(" ")
    : "No job has been launched yet.";
  const wasNearBottom = (elements.logOutput.scrollHeight - elements.logOutput.scrollTop - elements.logOutput.clientHeight) < 48;
  const nextLogText = job.logs && job.logs.length
    ? job.logs.join("\n")
    : "Waiting for a run...";
  if (elements.logOutput.textContent !== nextLogText) {
    elements.logOutput.textContent = nextLogText;
  }
  if (wasNearBottom) {
    elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
  }
  elements.startButton.disabled = Boolean(job.is_running);

  if (job.config && job.config.output_dir) {
    state.currentOutputDir = job.config.output_dir;
  }
}

async function refreshJobSnapshot() {
  try {
    const job = await getJson("/api/job");
    renderJobSnapshot(job);
    if (state.currentOutputDir) {
      await refreshReports(state.currentOutputDir, true);
    }
  } catch (error) {
    setFlash(error.message, "danger");
  }
}

function renderReportTree(data, outputDir) {
  if (!data.datasets || !data.datasets.length) {
    elements.reportTree.className = "report-tree empty-state";
    elements.reportTree.textContent = data.messages && data.messages.length
      ? data.messages.join(" ")
      : "No report artifacts were found under the selected output directory.";
    elements.reportFrame.removeAttribute("src");
    elements.viewerPath.textContent = "No report selected";
    elements.openReportLink.classList.add("disabled-link");
    elements.openReportLink.href = "#";
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "report-groups";
  let firstReport = null;

  data.datasets.forEach((dataset) => {
    const group = document.createElement("section");
    group.className = "report-group";

    const title = document.createElement("h3");
    title.textContent = dataset.dataset_name;
    group.appendChild(title);

    if (dataset.errors && dataset.errors.length) {
      const errorBox = document.createElement("div");
      errorBox.className = "report-errors";
      errorBox.innerHTML = `<strong>Error logs</strong><ul>${dataset.errors.map((item) => `<li>${item}</li>`).join("")}</ul>`;
      group.appendChild(errorBox);
    }

    dataset.stacks.forEach((stack) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "report-item";
      button.dataset.reportPath = stack.report_path;
      button.textContent = stack.stack_name;
      button.addEventListener("click", () => {
        selectReport(outputDir, stack.report_path);
      });
      group.appendChild(button);
      if (!firstReport) {
        firstReport = stack.report_path;
      }
    });

    wrapper.appendChild(group);
  });

  elements.reportTree.className = "report-tree";
  elements.reportTree.innerHTML = "";
  elements.reportTree.appendChild(wrapper);

  const selectedStillExists = state.selectedReport && data.datasets.some((dataset) =>
    dataset.stacks.some((stack) => stack.report_path === state.selectedReport)
  );

  if (selectedStillExists) {
    selectReport(outputDir, state.selectedReport);
  } else if (firstReport) {
    selectReport(outputDir, firstReport);
  }
}

function selectReport(outputDir, reportPath) {
  state.selectedReport = reportPath;
  const url = `/report?output_dir=${encodeURIComponent(outputDir)}&report_path=${encodeURIComponent(reportPath)}`;
  if (elements.reportFrame.getAttribute("src") !== url) {
    elements.reportFrame.src = url;
  }
  elements.viewerPath.textContent = reportPath;
  elements.openReportLink.href = url;
  elements.openReportLink.classList.remove("disabled-link");
  document.querySelectorAll(".report-item").forEach((node) => {
    node.classList.toggle("is-active", node.dataset.reportPath === reportPath);
  });
}

async function refreshReports(outputDir, keepSelection = false) {
  const target = outputDir.trim();
  if (!target) {
    return;
  }
  try {
    if (!keepSelection) {
      state.selectedReport = null;
    }
    const data = await getJson(`/api/reports?output_dir=${encodeURIComponent(target)}`);
    renderReportTree(data, target);
  } catch (error) {
    setFlash(error.message, "danger");
  }
}

async function startJob() {
  try {
    const payload = buildJobPayload();
    if (!payload.subfolders.length && state.scan && state.scan.default_selected_subfolders) {
      payload.subfolders = [...state.scan.default_selected_subfolders];
    }
    await postJson("/api/start-job", payload);
    state.currentOutputDir = payload.output_dir;
    setFlash("Job submitted. The console is now validating environments and launching the pipeline.", "success");
    await refreshJobSnapshot();
  } catch (error) {
    setFlash(error.message, "danger");
  }
}

elements.datasetType.addEventListener("change", toggleConditionalFields);
elements.llmMode.addEventListener("change", toggleConditionalFields);
elements.scanButton.addEventListener("click", scanInputDir);
elements.startButton.addEventListener("click", startJob);
elements.loadReportsButton.addEventListener("click", () => {
  state.currentOutputDir = elements.outputDir.value.trim();
  refreshReports(state.currentOutputDir, false);
});
elements.selectAllButton.addEventListener("click", () => selectAllValidSubfolders(true));
elements.clearAllButton.addEventListener("click", () => selectAllValidSubfolders(false));

toggleConditionalFields();
renderDeploymentContext();
refreshJobSnapshot();
setInterval(refreshJobSnapshot, pollIntervalMs);
