document.addEventListener('DOMContentLoaded', () => {
    
    // ─── STATE MANAGEMENT ──────────────────────────────────────────────────
    let activeTab = 'data-prep';
    let isTrainingActive = false;
    let statusInterval = null;
    let runsList = [];
    let selectedRunDetails = null;

    const TAB_INFO = {
        'data-prep': {
            title: 'Dataset Preparation & Viewer',
            desc: 'Convert CSV dataset files into SkillOpt compatible JSON splits and preview records.'
        },
        'configure': {
            title: 'Configure & Launch Loop',
            desc: 'Customize model endpoints, credentials, hyperparameters, and launch the training loop.'
        },
        'monitor': {
            title: 'ReflACT Active Monitor',
            desc: 'Detailed real-time tracking of training progress, current active optimizer stage, and logs.'
        },
        'results': {
            title: 'Outputs & Rollout Viewer',
            desc: 'Inspect finalized prompt guidelines, token metrics, and run case-by-case discrepancy analysis.'
        }
    };


    // ─── TAB NAVIGATION ────────────────────────────────────────────────────
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const tabTitleEl = document.getElementById('current-tab-title');
    const tabDescEl = document.getElementById('current-tab-desc');

    function switchTab(tabId) {
        activeTab = tabId;
        
        // Update nav buttons
        navButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update tab views
        tabContents.forEach(content => {
            if (content.id === `tab-${tabId}`) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });

        // Update titles
        if (TAB_INFO[tabId]) {
            tabTitleEl.innerText = TAB_INFO[tabId].title;
            tabDescEl.innerText = TAB_INFO[tabId].desc;
        }

        // Specific actions on tab load
        if (tabId === 'results') {
            loadResultsList();
        }
    }

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            switchTab(btn.getAttribute('data-tab'));
        });
    });


    // ─── TAB 1: CSV TO JSON CONVERSION & DATA PREVIEW ──────────────────────
    const csvConvertForm = document.getElementById('csv-convert-form');
    const btnConvertCsv = document.getElementById('btn-convert-csv');
    const csvSuccessAlert = document.getElementById('csv-success-alert');
    const csvSuccessMsg = document.getElementById('csv-success-msg');
    const csvErrorAlert = document.getElementById('csv-error-alert');
    const csvErrorMsg = document.getElementById('csv-error-msg');
    
    const btnLoadDataset = document.getElementById('btn-load-dataset');
    const datasetPreviewPathInput = document.getElementById('dataset-preview-path');
    const datasetPreviewTable = document.getElementById('dataset-preview-table');
    const datasetPreviewTbody = document.getElementById('dataset-preview-tbody');

    const csvFileInput = document.getElementById('csv-file');
    const columnSelectionPanel = document.getElementById('column-selection-panel');
    const inputPillsContainer = document.getElementById('input-pills-container');
    const targetPillsContainer = document.getElementById('target-pills-container');
    const uploadTextEl = document.querySelector('.upload-text');

    let uploadedFilename = null;
    let selectedInputCols = [];
    let selectedTargetCols = [];

    // Parse headers on file selection
    csvFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadTextEl.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-indigo"></i> Uploading and parsing headers...`;
        csvSuccessAlert.classList.add('hidden');
        csvErrorAlert.classList.add('hidden');
        columnSelectionPanel.classList.add('hidden');
        btnConvertCsv.disabled = true;
        btnConvertCsv.innerHTML = '<i class="fa-solid fa-gears"></i> Select columns to convert';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/parse-csv-headers', {
                method: 'POST',
                body: formData
            });
            const result = await resp.json();

            if (result.success) {
                uploadedFilename = result.filename;
                uploadTextEl.innerHTML = `<i class="fa-solid fa-circle-check text-green"></i> <strong>${file.name}</strong> loaded successfully.`;
                
                // Automatically compute output JSON path
                const baseName = file.name.replace(/\.[^/.]+$/, "");
                const autoJsonPath = `train/${baseName}.json`;
                document.getElementById('json-path').value = autoJsonPath;
                setSelectedValue(datasetPreviewPathInput, autoJsonPath);
                
                // Automatically prefill training config data path to match
                const cfgDataPath = document.getElementById('cfg-env-data-path');
                if (cfgDataPath) {
                    setSelectedValue(cfgDataPath, autoJsonPath);
                    autoSelectEnvForDataPath();
                }

                renderColumnPills(result.headers);
                columnSelectionPanel.classList.remove('hidden');
                
                // Automatically trigger the convert submit to automate conversion
                setTimeout(() => {
                    csvConvertForm.dispatchEvent(new Event('submit'));
                }, 500);
            } else {
                uploadTextEl.innerHTML = `Click to choose or drag CSV file`;
                csvErrorMsg.innerText = result.message || 'Failed to parse CSV.';
                csvErrorAlert.classList.remove('hidden');
            }
        } catch (err) {
            uploadTextEl.innerHTML = `Click to choose or drag CSV file`;
            csvErrorMsg.innerText = `Error: ${err.message}`;
            csvErrorAlert.classList.remove('hidden');
        }
    });

    function renderColumnPills(headers) {
        selectedInputCols = [];
        selectedTargetCols = [];
        
        let inputHtml = '';
        let targetHtml = '';

        headers.forEach(header => {
            inputHtml += `<div class="pill-checkbox" data-col="${header}" data-type="input">${header}</div>`;
            targetHtml += `<div class="pill-checkbox" data-col="${header}" data-type="target">${header}</div>`;
        });

        inputPillsContainer.innerHTML = inputHtml;
        targetPillsContainer.innerHTML = targetHtml;

        // Bind click events for input pills
        inputPillsContainer.querySelectorAll('.pill-checkbox').forEach(pill => {
            pill.addEventListener('click', () => {
                const col = pill.getAttribute('data-col');
                pill.classList.toggle('active');
                if (pill.classList.contains('active')) {
                    selectedInputCols.push(col);
                } else {
                    selectedInputCols = selectedInputCols.filter(c => c !== col);
                }
                validateForm();
            });
        });

        // Bind click events for target pills
        targetPillsContainer.querySelectorAll('.pill-checkbox').forEach(pill => {
            pill.addEventListener('click', () => {
                const col = pill.getAttribute('data-col');
                pill.classList.toggle('active-target');
                if (pill.classList.contains('active-target')) {
                    selectedTargetCols.push(col);
                } else {
                    selectedTargetCols = selectedTargetCols.filter(c => c !== col);
                }
                validateForm();
            });
        });
        
        // Auto-select first column as input, other columns as targets by default for a friendly UX
        if (headers.length > 0) {
            const firstInputPill = inputPillsContainer.querySelector('.pill-checkbox');
            firstInputPill.click();
            
            const targetPills = targetPillsContainer.querySelectorAll('.pill-checkbox');
            targetPills.forEach((pill, idx) => {
                if (idx > 0) {
                    pill.click();
                }
            });
        }
    }

    function validateForm() {
        if (selectedInputCols.length > 0 && selectedTargetCols.length > 0) {
            btnConvertCsv.disabled = false;
            btnConvertCsv.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Convert & Prepare JSON';
        } else {
            btnConvertCsv.disabled = true;
            btnConvertCsv.innerHTML = '<i class="fa-solid fa-gears"></i> Select columns to convert';
        }
    }

    // Convert CSV submit
    csvConvertForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!uploadedFilename || selectedInputCols.length === 0 || selectedTargetCols.length === 0) return;
        
        btnConvertCsv.disabled = true;
        btnConvertCsv.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Converting Dataset...';
        csvSuccessAlert.classList.add('hidden');
        csvErrorAlert.classList.add('hidden');

        const payload = {
            csv_filename: uploadedFilename,
            json_path: document.getElementById('json-path').value,
            input_cols: selectedInputCols,
            target_cols: selectedTargetCols,
            split_ratio: document.getElementById('split-ratio').value
        };

        try {
            const resp = await fetch('/api/convert-csv', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await resp.json();

            if (result.success) {
                csvSuccessMsg.innerText = result.message;
                csvSuccessAlert.classList.remove('hidden');
                
                // Prefill preview path and load it
                setSelectedValue(datasetPreviewPathInput, payload.json_path);
                renderDatasetPreview(result.preview);

                // Update config form's dataset path and auto-resolve environment
                const cfgEnvDataPathEl = document.getElementById('cfg-env-data-path');
                if (cfgEnvDataPathEl) {
                    setSelectedValue(cfgEnvDataPathEl, payload.json_path);
                    autoSelectEnvForDataPath();
                }

                // Automatically update and select the newly generated initial skill path
                const cfgEnvInitSkillEl = document.getElementById('cfg-env-init-skill');
                if (cfgEnvInitSkillEl && result.skill_path) {
                    setSelectedValue(cfgEnvInitSkillEl, result.skill_path);
                }

                // Switch to Configure tab and highlight launch button
                setTimeout(() => {
                    switchTab('configure');
                    const launchBtn = document.getElementById('btn-launch-training');
                    if (launchBtn) {
                        launchBtn.classList.add('btn-launch-ready');
                    }
                }, 1000);
            } else {
                csvErrorMsg.innerText = result.message || 'Conversion failed.';
                csvErrorAlert.classList.remove('hidden');
            }
        } catch (err) {
            csvErrorMsg.innerText = `Network error: ${err.message}`;
            csvErrorAlert.classList.remove('hidden');
        } finally {
            validateForm();
        }
    });

    // Load dataset preview manually
    btnLoadDataset.addEventListener('click', async () => {
        const path = datasetPreviewPathInput.value;
        if (!path) return;

        btnLoadDataset.disabled = true;
        btnLoadDataset.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';

        try {
            const resp = await fetch(`/api/dataset?path=${encodeURIComponent(path)}`);
            const result = await resp.json();
            if (result.success) {
                renderDatasetPreview(result.data);
            } else {
                alert(`Error: ${result.message}`);
            }
        } catch (err) {
            alert(`Error fetching dataset: ${err.message}`);
        } finally {
            btnLoadDataset.disabled = false;
            btnLoadDataset.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Load';
        }
    });

    function renderDatasetPreview(items) {
        if (!items || items.length === 0) {
            datasetPreviewTbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">No items in dataset split.</td></tr>';
            return;
        }

        let html = '';
        // Limit preview render to 50 items for speed
        const previewLimit = Math.min(items.length, 50);
        
        for (let i = 0; i < previewLimit; i++) {
            const item = items[i];
            
            // Format ground truth badges
            let gtHtml = '';
            if (item.ground_truth && typeof item.ground_truth === 'object') {
                for (const [k, v] of Object.entries(item.ground_truth)) {
                    gtHtml += `<span class="param-badge"><strong>${k}:</strong> ${v}</span>`;
                }
            } else {
                gtHtml = `<span class="text-muted">${JSON.stringify(item.ground_truth)}</span>`;
            }

            html += `
                <tr>
                    <td><code>${item.id}</code></td>
                    <td><div style="max-height: 80px; overflow-y: auto;">${escapeHtml(item.input)}</div></td>
                    <td><div style="max-height: 80px; overflow-y: auto;">${gtHtml}</div></td>
                </tr>
            `;
        }

        if (items.length > 50) {
            html += `<tr><td colspan="3" class="text-center text-muted font-weight-bold">... and ${items.length - 50} more records (preview capped at 50) ...</td></tr>`;
        }

        datasetPreviewTbody.innerHTML = html;
    }


    // ─── TAB 2: CONFIGURE & LAUNCH ──────────────────────────────────────────
    const activeConfigDropdown = document.getElementById('active-config-path');
    const trainingConfigForm = document.getElementById('training-config-form');
    const btnLaunchTraining = document.getElementById('btn-launch-training');
    const btnStopTraining = document.getElementById('btn-stop-training');
    const panelStatusText = document.querySelector('.status-text-panel');

    // Discover YAML configs
    async function loadConfigDropdown() {
        try {
            const resp = await fetch('/api/configs');
            const result = await resp.json();
            if (result.success && result.configs.length > 0) {
                let html = '';
                result.configs.forEach(cfg => {
                    const isSelected = cfg === 'train/config.yaml' ? 'selected' : '';
                    html += `<option value="${cfg}" ${isSelected}>${cfg}</option>`;
                });
                activeConfigDropdown.innerHTML = html;
                
                // Fetch default config contents
                loadConfigParameters(activeConfigDropdown.value);
            }
        } catch (err) {
            console.error('Error discovering configurations:', err);
        }
    }

    activeConfigDropdown.addEventListener('change', () => {
        btnLaunchTraining.classList.remove('btn-launch-ready');
        loadConfigParameters(activeConfigDropdown.value);
    });

    async function loadConfigParameters(path) {
        panelStatusText.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Loading template variables...`;
        try {
            const resp = await fetch(`/api/configs/active?path=${encodeURIComponent(path)}`);
            const result = await resp.json();
            if (result.success && result.config) {
                prefillConfigForm(result.config);
                panelStatusText.innerHTML = `<i class="fa-solid fa-circle-info text-indigo"></i> Config parameters prefilled from ${path}`;
            }
        } catch (err) {
            panelStatusText.innerHTML = `<i class="fa-solid fa-circle-xmark text-red"></i> Error loading config variables: ${err.message}`;
        }
    }

    function prefillConfigForm(cfg) {
        // Model section
        const model = cfg.model || {};
        document.getElementById('cfg-model-target').value = model.target || 'gpt-5.4';
        document.getElementById('cfg-model-optimizer').value = model.optimizer || 'gpt-5.4';
        
        // Backend mapping
        const backendSelect = document.getElementById('cfg-model-backend');
        const bType = model.backend || cfg.model_backend || 'azure_openai';
        setSelectedValue(backendSelect, bType);
        
        document.getElementById('cfg-model-endpoint').value = model.azure_openai_endpoint || cfg.azure_openai_endpoint || '';
        document.getElementById('cfg-model-api-key').value = model.azure_openai_api_key || cfg.azure_openai_api_key || '';
        document.getElementById('cfg-model-api-version').value = model.azure_openai_api_version || cfg.azure_openai_api_version || '';
        
        const reasoningSelect = document.getElementById('cfg-model-reasoning');
        setSelectedValue(reasoningSelect, model.reasoning_effort || '');

        // Train Section
        const train = cfg.train || {};
        document.getElementById('cfg-train-epochs').value = train.num_epochs || cfg.num_epochs || 4;
        document.getElementById('cfg-train-batch-size').value = train.batch_size || cfg.batch_size || 20;
        document.getElementById('cfg-train-accumulation').value = train.accumulation || cfg.accumulation || 1;
        document.getElementById('cfg-train-seed').value = train.seed || cfg.seed || 42;

        // Gradient Section
        const gradient = cfg.gradient || {};
        document.getElementById('cfg-gradient-minibatch').value = gradient.minibatch_size || cfg.minibatch_size || 4;
        document.getElementById('cfg-gradient-workers').value = gradient.analyst_workers || cfg.analyst_workers || 4;
        
        const failOnlySelect = document.getElementById('cfg-gradient-failure-only');
        setSelectedValue(failOnlySelect, String(gradient.failure_only !== false));

        // Optimizer Section
        const optimizer = cfg.optimizer || {};
        document.getElementById('cfg-opt-lr').value = optimizer.learning_rate || cfg.edit_budget || 3;
        
        const schedSelect = document.getElementById('cfg-opt-scheduler');
        setSelectedValue(schedSelect, optimizer.lr_scheduler || cfg.lr_scheduler || 'cosine');
        
        const updateModeSelect = document.getElementById('cfg-opt-update-mode');
        setSelectedValue(updateModeSelect, optimizer.skill_update_mode || cfg.skill_update_mode || 'patch');
        
        document.getElementById('cfg-opt-slow-update').checked = optimizer.use_slow_update !== false;
        document.getElementById('cfg-opt-meta-skill').checked = optimizer.use_meta_skill !== false;

        // Env/Evaluation Section
        const evaluation = cfg.evaluation || {};
        document.getElementById('cfg-eval-gate').checked = evaluation.use_gate !== false;
        document.getElementById('cfg-eval-sel-num').value = evaluation.sel_env_num || cfg.sel_env_num || 15;
        document.getElementById('cfg-eval-test-num').value = evaluation.test_env_num || cfg.test_env_num || 15;

        const env = cfg.env || {};
        setSelectedValue(document.getElementById('cfg-env-name'), env.name || cfg.env || 'ceramic_capacitors');
        setSelectedValue(document.getElementById('cfg-env-data-path'), env.data_path || cfg.data_path || 'train/ceramic_capacitors.json');
        document.getElementById('cfg-env-workers').value = env.workers || cfg.workers || 8;
        setSelectedValue(document.getElementById('cfg-env-init-skill'), env.skill_init || cfg.skill_init || 'train/initial.md');
    }

    function setSelectedValue(selectObj, val) {
        if (!selectObj) return;
        let found = false;
        for (let i = 0; i < selectObj.options.length; i++) {
            if (selectObj.options[i].value === val) {
                selectObj.selectedIndex = i;
                found = true;
                break;
            }
        }
        if (!found && val) {
            const opt = document.createElement('option');
            opt.value = val;
            opt.innerText = val;
            selectObj.appendChild(opt);
            selectObj.value = val;
        }
    }

    // Launch Training Submit
    trainingConfigForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        btnLaunchTraining.disabled = true;
        btnLaunchTraining.classList.remove('btn-launch-ready');
        btnLaunchTraining.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Launching Loop...';
        panelStatusText.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Writing overrides and launching scripts/train.py...`;

        const configPath = activeConfigDropdown.value;

        // Build overrides dictionary matching yaml nesting structure
        const overrides = {
            'model.target': document.getElementById('cfg-model-target').value,
            'model.optimizer': document.getElementById('cfg-model-optimizer').value,
            'model.backend': document.getElementById('cfg-model-backend').value,
            'model.azure_openai_endpoint': document.getElementById('cfg-model-endpoint').value,
            'model.azure_openai_api_version': document.getElementById('cfg-model-api-version').value,
            'model.reasoning_effort': document.getElementById('cfg-model-reasoning').value,
            
            'train.num_epochs': document.getElementById('cfg-train-epochs').value,
            'train.batch_size': document.getElementById('cfg-train-batch-size').value,
            'train.accumulation': document.getElementById('cfg-train-accumulation').value,
            'train.seed': document.getElementById('cfg-train-seed').value,
            
            'gradient.minibatch_size': document.getElementById('cfg-gradient-minibatch').value,
            'gradient.analyst_workers': document.getElementById('cfg-gradient-workers').value,
            'gradient.failure_only': document.getElementById('cfg-gradient-failure-only').value === 'true',
            
            'optimizer.learning_rate': document.getElementById('cfg-opt-lr').value,
            'optimizer.lr_scheduler': document.getElementById('cfg-opt-scheduler').value,
            'optimizer.skill_update_mode': document.getElementById('cfg-opt-update-mode').value,
            'optimizer.use_slow_update': document.getElementById('cfg-opt-slow-update').checked,
            'optimizer.use_meta_skill': document.getElementById('cfg-opt-meta-skill').checked,
            
            'evaluation.use_gate': document.getElementById('cfg-eval-gate').checked,
            'evaluation.sel_env_num': document.getElementById('cfg-eval-sel-num').value,
            'evaluation.test_env_num': document.getElementById('cfg-eval-test-num').value,
            
            'env.name': document.getElementById('cfg-env-name').value,
            'env.data_path': document.getElementById('cfg-env-data-path').value,
            'env.workers': document.getElementById('cfg-env-workers').value,
            'env.skill_init': document.getElementById('cfg-env-init-skill').value
        };

        // Include API Key if present
        const apiKey = document.getElementById('cfg-model-api-key').value;
        if (apiKey) {
            overrides['model.azure_openai_api_key'] = apiKey;
        }

        try {
            const resp = await fetch('/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config_path: configPath, overrides: overrides })
            });
            const result = await resp.json();
            
            if (result.success) {
                panelStatusText.innerHTML = `<i class="fa-solid fa-circle-check text-green"></i> Loop launched! Redirecting to Monitor...`;
                
                // Show stop button and hide launch button
                btnLaunchTraining.classList.add('hidden');
                btnStopTraining.classList.remove('hidden');

                // Start monitor polling and switch tab
                isTrainingActive = true;
                startMonitoringPoll();
                setTimeout(() => {
                    switchTab('monitor');
                }, 1000);
            } else {
                panelStatusText.innerHTML = `<i class="fa-solid fa-circle-xmark text-red"></i> Error starting training: ${result.message}`;
            }
        } catch (err) {
            panelStatusText.innerHTML = `<i class="fa-solid fa-circle-xmark text-red"></i> Network error: ${err.message}`;
        } finally {
            btnLaunchTraining.disabled = false;
            btnLaunchTraining.innerHTML = '<i class="fa-solid fa-circle-play"></i> Launch Reflective Loop';
        }
    });

    // Reset Form Defaults
    document.getElementById('btn-reset-config').addEventListener('click', () => {
        btnLaunchTraining.classList.remove('btn-launch-ready');
        loadConfigParameters(activeConfigDropdown.value);
    });

    // Stop Training Action
    btnStopTraining.addEventListener('click', async () => {
        btnStopTraining.disabled = true;
        btnStopTraining.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Stopping Process...';

        try {
            const resp = await fetch('/api/train/stop', { method: 'POST' });
            const result = await resp.json();
            if (result.success) {
                panelStatusText.innerHTML = `<i class="fa-solid fa-circle-info text-amber"></i> ${result.message}`;
                btnStopTraining.classList.add('hidden');
                btnLaunchTraining.classList.remove('hidden');
                isTrainingActive = false;
            } else {
                alert(`Error stopping: ${result.message}`);
            }
        } catch (err) {
            alert(`Error stopping process: ${err.message}`);
        } finally {
            btnStopTraining.disabled = false;
            btnStopTraining.innerHTML = '<i class="fa-solid fa-circle-stop"></i> Stop Training Run';
        }
    });


    // ─── TAB 3: LIVE MONITOR ────────────────────────────────────────────────
    const monitorBadgeStatus = document.getElementById('monitor-badge-status');
    const monitorEpochVal = document.getElementById('monitor-epoch-val');
    const monitorStepVal = document.getElementById('monitor-step-val');
    const monitorPercentVal = document.getElementById('monitor-percent-val');
    const monitorProgressFill = document.getElementById('monitor-progress-fill');
    
    const logTerminal = document.getElementById('log-terminal');
    const terminalAutoscrollCheckbox = document.getElementById('terminal-autoscroll');
    const btnClearTerminal = document.getElementById('btn-clear-terminal');
    const navStatusDot = document.getElementById('nav-status-dot');

    const PIPELINE_NODES = ['Rollout', 'Reflect', 'Aggregate', 'Select', 'Update', 'Gate'];

    function startMonitoringPoll() {
        if (statusInterval) clearInterval(statusInterval);
        
        // Initial call
        pollStatus();
        
        // Poll every 2 seconds
        statusInterval = setInterval(pollStatus, 2000);
    }

    async function pollStatus() {
        try {
            const resp = await fetch('/api/train/status');
            const status = await resp.json();

            // Update UI status badges
            isTrainingActive = status.running;
            
            if (isTrainingActive) {
                navStatusDot.className = 'status-dot-nav running';
                btnLaunchTraining.classList.add('hidden');
                btnStopTraining.classList.remove('hidden');
            } else {
                navStatusDot.className = 'status-dot-nav idle';
                btnStopTraining.classList.add('hidden');
                btnLaunchTraining.classList.remove('hidden');
            }

            // Update status text badge
            monitorBadgeStatus.innerText = status.stage;
            if (isTrainingActive) {
                monitorBadgeStatus.className = 'progress-badge text-green';
            } else if (status.stage === 'Finished') {
                monitorBadgeStatus.className = 'progress-badge text-green';
                if (statusInterval) clearInterval(statusInterval);
            } else if (status.stage.includes('Error')) {
                monitorBadgeStatus.className = 'progress-badge text-red';
                if (statusInterval) clearInterval(statusInterval);
            } else {
                monitorBadgeStatus.className = 'progress-badge';
            }

            // Update progress details
            monitorEpochVal.innerText = status.total_epochs > 0 ? `Epoch: ${status.epoch}/${status.total_epochs}` : 'Epoch: —';
            monitorStepVal.innerText = status.total_steps > 0 ? `Step: ${status.step}/${status.total_steps}` : 'Step: —';
            
            const pct = (status.progress * 100).toFixed(1);
            monitorPercentVal.innerText = `${pct}%`;
            monitorProgressFill.style.width = `${pct}%`;

            // Update visual pipeline stages
            updatePipelineVisualFlow(status.stage);

            // Stream logs to console
            if (status.logs && status.logs.length > 0) {
                renderLogsToTerminal(status.logs);
            } else if (!isTrainingActive) {
                logTerminal.innerHTML = '<span class="text-muted">Awaiting training launch. Configured settings are ready.</span>';
            }

        } catch (err) {
            console.error('Error polling status:', err);
        }
    }

    function updatePipelineVisualFlow(activeStage) {
        // Clear all active states first
        PIPELINE_NODES.forEach(node => {
            const el = document.getElementById(`node-${node}`);
            if (el) el.classList.remove('active-node');
            const conn = document.getElementById(`conn-${node}`);
            if (conn) conn.classList.remove('active-connector');
        });

        if (!activeStage || activeStage === 'Idle' || activeStage === 'Starting' || activeStage === 'Finished') {
            return;
        }

        // Match active node
        let matchedIndex = -1;
        PIPELINE_NODES.forEach((node, idx) => {
            if (activeStage.toLowerCase().includes(node.toLowerCase())) {
                matchedIndex = idx;
            }
        });

        // Slow updates or other stages fallback
        if (activeStage === 'Slow Update') matchedIndex = 4; // Highlight update step
        if (activeStage === 'Meta Skill') matchedIndex = 4;
        if (activeStage === 'Baseline') matchedIndex = 0; // Highlight rollout

        if (matchedIndex !== -1) {
            // Set active node
            const activeNodeName = PIPELINE_NODES[matchedIndex];
            const nodeEl = document.getElementById(`node-${activeNodeName}`);
            if (nodeEl) nodeEl.classList.add('active-node');

            // Highlight all prior connectors
            for (let i = 0; i <= matchedIndex; i++) {
                const connEl = document.getElementById(`conn-${PIPELINE_NODES[i]}`);
                if (connEl) connEl.classList.add('active-connector');
            }
        }
    }

    function renderLogsToTerminal(logs) {
        let html = '';
        logs.forEach(line => {
            // Rebrand logs: reflact -> skillopt
            const displayLine = line
                .replace(/reflact/gi, 'skillopt')
                .replace(/ReflACT/gi, 'SkillOpt');

            const escaped = escapeHtml(displayLine.trimEnd());
            const lineLower = line.toLowerCase();
            
            // Format colors based on output
            let colorClass = '';
            let isBold = false;
            
            if (lineLower.includes('[epoch')) {
                colorClass = 'text-amber';
                isBold = true;
            } else if (lineLower.includes('[step')) {
                colorClass = 'text-indigo';
                isBold = true;
            } else if (lineLower.includes('rollout]') || lineLower.includes('1/6')) {
                colorClass = 'text-indigo';
            } else if (lineLower.includes('reflect') || lineLower.includes('2/6')) {
                colorClass = 'text-amber';
            } else if (lineLower.includes('aggregate') || lineLower.includes('3/6') || lineLower.includes('merge')) {
                colorClass = 'text-green';
            } else if (lineLower.includes('select') || lineLower.includes('4/6')) {
                colorClass = 'text-amber';
            } else if (lineLower.includes('update') || lineLower.includes('5/6')) {
                colorClass = 'text-green';
            } else if (lineLower.includes('gate') || lineLower.includes('6/6')) {
                colorClass = 'text-red';
            } else if (lineLower.includes('[rollout]') && lineLower.includes('hard=1')) {
                colorClass = 'text-green';
            } else if (lineLower.includes('[rollout]') && lineLower.includes('hard=0')) {
                colorClass = 'text-red';
            } else if (lineLower.includes('error') || lineLower.includes('fail') || lineLower.includes('exception')) {
                colorClass = 'text-red';
                isBold = true;
            }

            const style = isBold ? 'font-weight:700;' : '';
            html += `<div class="${colorClass}" style="${style}">${escaped}</div>`;
        });

        logTerminal.innerHTML = html;

        // Auto-scroll logic
        if (terminalAutoscrollCheckbox.checked) {
            logTerminal.scrollTop = logTerminal.scrollHeight;
        }
    }

    btnClearTerminal.addEventListener('click', () => {
        logTerminal.innerHTML = '<span class="text-muted">Terminal cleared.</span>';
    });


    // ─── TAB 4: RESULTS & ROLLOUTS EXPLORER ─────────────────────────────────
    const btnRefreshResults = document.getElementById('btn-refresh-results');
    const runsContainer = document.getElementById('runs-container');
    const selectedRunContainer = document.getElementById('selected-run-container');
    const runDetailsPlaceholder = document.getElementById('run-details-placeholder');
    
    // Stats elements
    const statBaseline = document.getElementById('stat-baseline');
    const statBest = document.getElementById('stat-best');
    const statTest = document.getElementById('stat-test');
    const statDuration = document.getElementById('stat-duration');
    
    // Info elements
    const infoRunId = document.getElementById('info-run-id');
    const infoEnvName = document.getElementById('info-env-name');
    const infoTargetModel = document.getElementById('info-target-model');
    const infoOptModel = document.getElementById('info-opt-model');
    
    const runStepsTbody = document.getElementById('run-steps-tbody');
    const promptViewerContent = document.getElementById('prompt-viewer-content');
    const btnCopySkill = document.getElementById('btn-copy-skill');

    // Detail Tab Buttons
    const detailTabBtns = document.querySelectorAll('.detail-tab-btn');
    const detailTabContents = document.querySelectorAll('.detail-tab-content');

    detailTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-detail-tab');
            
            detailTabBtns.forEach(b => b.classList.remove('active'));
            detailTabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`detail-tab-${target}`).classList.add('active');
        });
    });

    async function loadResultsList() {
        runsContainer.innerHTML = '<div class="text-center text-muted p-20"><i class="fa-solid fa-spinner fa-spin"></i> Scanning outputs...</div>';
        try {
            const resp = await fetch('/api/results');
            const result = await resp.json();
            if (result.success) {
                runsList = result.results;
                renderRunsList(runsList);
            } else {
                runsContainer.innerHTML = `<div class="text-center text-red p-20">Error: ${result.message}</div>`;
            }
        } catch (err) {
            runsContainer.innerHTML = `<div class="text-center text-red p-20">Network Error: ${err.message}</div>`;
        }
    }

    btnRefreshResults.addEventListener('click', loadResultsList);

    function renderRunsList(runs) {
        if (!runs || runs.length === 0) {
            runsContainer.innerHTML = '<div class="text-center text-muted p-20">No output runs found.</div>';
            return;
        }

        let html = '';
        runs.forEach(run => {
            const isBestPercent = typeof run.best_score === 'number' ? `${(run.best_score * 100).toFixed(0)}%` : '—';
            
            html += `
                <div class="run-item" data-run-id="${run.id}">
                    <div class="run-item-header">
                        <span>${run.env}</span>
                        <span class="text-indigo">${isBestPercent}</span>
                    </div>
                    <div class="run-item-meta">
                        <div>ID: <code>${run.name.substring(9, 25)}...</code></div>
                        <div>Date: ${run.timestamp}</div>
                    </div>
                </div>
            `;
        });
        runsContainer.innerHTML = html;

        // Bind clicks
        const items = runsContainer.querySelectorAll('.run-item');
        items.forEach(item => {
            item.addEventListener('click', () => {
                items.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                loadRunDetails(item.getAttribute('data-run-id'));
            });
        });
    }

    async function loadRunDetails(runId) {
        runDetailsPlaceholder.innerHTML = '<div class="text-center p-50"><i class="fa-solid fa-spinner fa-spin placeholder-icon"></i><h3>Loading details...</h3></div>';
        
        try {
            const resp = await fetch(`/api/results/${runId}`);
            const result = await resp.json();
            if (result.success) {
                selectedRunDetails = result.details;
                renderRunDetails(result.details);
            } else {
                alert(`Error loading details: ${result.message}`);
            }
        } catch (err) {
            alert(`Network Error: ${err.message}`);
        }
    }

    function renderRunDetails(details) {
        // Toggle view
        runDetailsPlaceholder.classList.add('hidden');
        selectedRunContainer.classList.remove('hidden');

        const summary = details.summary || {};
        const config = summary.config || {};

        // 1. Update stats row
        const formatScore = (val) => typeof val === 'number' ? `${(val * 100).toFixed(1)}%` : '—';
        statBaseline.innerText = formatScore(summary.baseline_selection_hard);
        statBest.innerText = formatScore(summary.best_selection_hard);
        statTest.innerText = formatScore(summary.test_hard);
        
        // Duration formatting
        const seconds = summary.total_wall_time_s || 0;
        if (seconds > 0) {
            const mins = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(0);
            statDuration.innerText = `${mins}m ${secs}s`;
        } else {
            statDuration.innerText = '—';
        }

        // 2. Info panel
        infoRunId.innerText = details.run_id;
        infoEnvName.innerText = config.env || 'unknown';
        infoTargetModel.innerText = `${config.target_model || '—'} (${config.target_backend || '—'})`;
        infoOptModel.innerText = `${config.optimizer_model || '—'} (${config.optimizer_backend || '—'})`;

        // 3. Steps Table
        if (details.history && details.history.length > 0) {
            let stepsHtml = '';
            details.history.forEach(step => {
                const isBestStep = step.step === summary.best_step;
                const rowClass = isBestStep ? 'style="background-color:rgba(16,185,129,0.04);"' : '';
                const star = isBestStep ? ' <i class="fa-solid fa-star text-amber" title="Best Step"></i>' : '';
                
                stepsHtml += `
                    <tr ${rowClass}>
                        <td><strong>Step ${step.step}</strong>${star}</td>
                        <td>Epoch ${step.epoch}</td>
                        <td>${(step.rollout_hard * 100).toFixed(1)}%</td>
                        <td>${(step.rollout_soft * 100).toFixed(1)}%</td>
                        <td><code>${step.action || 'skip'}</code></td>
                        <td>${formatScore(step.current_score)}</td>
                    </tr>
                `;
            });
            runStepsTbody.innerHTML = stepsHtml;
        } else {
            runStepsTbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No step history recorded.</td></tr>';
        }

        // 4. Prompt Viewer
        promptViewerContent.innerText = details.best_skill || 'No skill document saved for this run.';

        // 5. Rollouts Explorer & Discrepancies
        renderRolloutsExplorer(details.rollout_results);
    }

    // Copy prompt text helper
    btnCopySkill.addEventListener('click', () => {
        const text = promptViewerContent.innerText;
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            btnCopySkill.innerHTML = '<i class="fa-solid fa-check text-green"></i> Copied!';
            setTimeout(() => {
                btnCopySkill.innerHTML = '<i class="fa-solid fa-copy"></i> Copy Prompt';
            }, 2000);
        });
    });

    // Rollout exploration failure mapping
    let activeRolloutFilter = 'all';
    const rolloutItemsContainer = document.getElementById('rollout-items-container');
    const filterRadios = document.querySelectorAll('input[name="rollout-filter"]');
    
    filterRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            activeRolloutFilter = e.target.value;
            if (selectedRunDetails && selectedRunDetails.rollout_results) {
                filterAndRenderRollouts(selectedRunDetails.rollout_results);
            }
        });
    });

    function renderRolloutsExplorer(rollouts) {
        if (!rollouts || rollouts.length === 0) {
            rolloutItemsContainer.innerHTML = '<div class="text-center text-muted p-20">No rollout result items found. Check if training ran any validation gate rollouts.</div>';
            document.getElementById('count-rollout-all').innerText = '0';
            document.getElementById('count-rollout-correct').innerText = '0';
            document.getElementById('count-rollout-failed').innerText = '0';
            return;
        }

        // Calculate counts
        const all = rollouts.length;
        const correct = rollouts.filter(r => r.hard === 1).length;
        const failed = all - correct;

        document.getElementById('count-rollout-all').innerText = all;
        document.getElementById('count-rollout-correct').innerText = correct;
        document.getElementById('count-rollout-failed').innerText = failed;

        filterAndRenderRollouts(rollouts);
    }

    function filterAndRenderRollouts(rollouts) {
        let filtered = rollouts;
        if (activeRolloutFilter === 'correct') {
            filtered = rollouts.filter(r => r.hard === 1);
        } else if (activeRolloutFilter === 'failed') {
            filtered = rollouts.filter(r => r.hard === 0);
        }

        if (filtered.length === 0) {
            rolloutItemsContainer.innerHTML = `<div class="text-center text-muted p-20">No matching rollout items found for filter "${activeRolloutFilter}".</div>`;
            return;
        }

        let html = '';
        filtered.forEach((rollout, idx) => {
            const isCorrect = rollout.hard === 1;
            const cardClass = isCorrect ? 'is-correct' : 'is-failed';
            const badgeClass = isCorrect ? 'badge-success' : 'badge-danger';
            const badgeText = isCorrect ? 'Correct' : 'Failed';
            
            // Format gold answers
            let goldFormatted = '';
            try {
                const gold = Array.isArray(rollout.gold_answers) ? rollout.gold_answers[0] : rollout.gold_answers;
                goldFormatted = JSON.stringify(gold, null, 2);
            } catch (err) {
                goldFormatted = String(rollout.gold_answers);
            }

            // Failure reason
            const failReasonHtml = rollout.fail_reason ? `<div class="text-red" style="margin-bottom: 8px;"><strong>Discrepancy:</strong> ${escapeHtml(rollout.fail_reason)}</div>` : '';

            html += `
                <div class="rollout-item-card ${cardClass}" id="rollout-card-${idx}">
                    <div class="rollout-item-header" onclick="toggleRolloutBody(${idx})">
                        <span>ID: <code>${rollout.id}</code> &nbsp; <span style="font-weight: normal; color: var(--text-secondary);">${escapeHtml(rollout.description.substring(0, 75))}...</span></span>
                        <span class="rollout-item-badge ${badgeClass}">${badgeText}</span>
                    </div>
                    <div class="rollout-item-body hidden" id="rollout-body-${idx}">
                        ${failReasonHtml}
                        <div class="rollout-body-row">
                            <div>
                                <div class="response-box-title">LLM Full Response</div>
                                <pre class="response-box">${escapeHtml(rollout.response || '—')}</pre>
                            </div>
                            <div>
                                <div class="response-box-title">Gold (Expected) Parameters vs Predicted JSON</div>
                                <div class="grid-2col-nested">
                                    <div>
                                        <span class="text-secondary" style="font-size: 0.72rem; font-weight:700;">EXPECTED (GOLD):</span>
                                        <pre class="response-box" style="background:#0f172a; max-height: 180px;">${escapeHtml(goldFormatted)}</pre>
                                    </div>
                                    <div>
                                        <span class="text-secondary" style="font-size: 0.72rem; font-weight:700;">PREDICTED:</span>
                                        <pre class="response-box" style="background:#0f172a; max-height: 180px;">${escapeHtml(formatJsonStr(rollout.predicted_answer))}</pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        rolloutItemsContainer.innerHTML = html;
    }

    // Helper functions
    window.toggleRolloutBody = function(index) {
        const body = document.getElementById(`rollout-body-${index}`);
        if (body) {
            body.classList.toggle('hidden');
        }
    };

    function formatJsonStr(str) {
        if (!str) return '—';
        try {
            const parsed = typeof str === 'string' ? json.loads(str) : str;
            return JSON.stringify(parsed, null, 2);
        } catch (err) {
            try {
                const parsed = JSON.parse(str);
                return JSON.stringify(parsed, null, 2);
            } catch (e) {
                return str;
            }
        }
    }

    function escapeHtml(string) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(string).replace(/[&<>"']/g, function(m) { return map[m]; });
    }

    async function loadAllDropdowns() {
        // Fetch and populate datasets
        try {
            const resp = await fetch('/api/datasets');
            const result = await resp.json();
            if (result.success && result.datasets) {
                populateSelectOptions('cfg-env-data-path', result.datasets, 'train/ceramic_capacitors.json');
                populateSelectOptions('dataset-preview-path', result.datasets, 'train/ceramic_capacitors.json');
            }
        } catch (err) {
            console.error('Error fetching datasets:', err);
        }

        // Fetch and populate skills
        try {
            const resp = await fetch('/api/skills');
            const result = await resp.json();
            if (result.success && result.skills) {
                populateSelectOptions('cfg-env-init-skill', result.skills, 'train/initial.md');
            }
        } catch (err) {
            console.error('Error fetching skills:', err);
        }

        // Fetch and populate environments
        try {
            const resp = await fetch('/api/envs');
            const result = await resp.json();
            if (result.success && result.envs) {
                populateSelectOptions('cfg-env-name', result.envs, 'ceramic_capacitors');
            }
        } catch (err) {
            console.error('Error fetching environments:', err);
        }

        // Load configs and prefill the initial config after other dropdowns are ready
        await loadConfigDropdown();
    }

    function populateSelectOptions(selectId, optionsList, defaultValue) {
        const selectEl = document.getElementById(selectId);
        if (!selectEl) return;
        
        let html = '';
        optionsList.forEach(opt => {
            const isSelected = opt === defaultValue ? 'selected' : '';
            html += `<option value="${opt}" ${isSelected}>${opt}</option>`;
        });
        selectEl.innerHTML = html;
    }

    function autoSelectEnvForDataPath() {
        const cfgDataPathEl = document.getElementById('cfg-env-data-path');
        const cfgEnvNameEl = document.getElementById('cfg-env-name');
        if (!cfgDataPathEl || !cfgEnvNameEl) return;
        
        const path = cfgDataPathEl.value.toLowerCase();
        let envName = 'generic_csv'; // default fallback
        
        if (path.includes('ceramic_capacitors')) {
            envName = 'ceramic_capacitors';
        } else if (path.includes('searchqa')) {
            envName = 'searchqa';
        } else if (path.includes('alfworld')) {
            envName = 'alfworld';
        } else if (path.includes('docvqa')) {
            envName = 'docvqa';
        } else if (path.includes('spreadsheet')) {
            envName = 'spreadsheetbench';
        } else if (path.includes('mathematician')) {
            envName = 'livemathematicianbench';
        } else if (path.includes('officeqa')) {
            envName = 'officeqa';
        } else if (path.includes('sealqa')) {
            envName = 'sealqa';
        } else if (path.includes('mathverse')) {
            envName = 'mathverse';
        } else if (path.includes('babyvision')) {
            envName = 'babyvision';
        } else if (path.includes('mmrb')) {
            envName = 'mmrb';
        }
        
        setSelectedValue(cfgEnvNameEl, envName);
    }


    // ─── INITIALIZATION ────────────────────────────────────────────────────
    loadAllDropdowns();
    
    const cfgDataPathEl = document.getElementById('cfg-env-data-path');
    if (cfgDataPathEl) {
        cfgDataPathEl.addEventListener('change', autoSelectEnvForDataPath);
    }
    
    // Check status immediately
    pollStatus();
    // Periodically poll training status
    startMonitoringPoll();
});
