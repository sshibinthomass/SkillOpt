import os
import sys
import json
import csv
import yaml
import subprocess
import threading
import time
import signal
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FLASK_DIR = PROJECT_ROOT / "train" / "flask"
FLASK_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER = FLASK_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Global Training Manager
class TrainingManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.process = None
        self.log_lines = []
        self.stage = "Idle"
        self.step = 0
        self.total_steps = 0
        self.epoch = 0
        self.total_epochs = 0
        self.running = False
        self.output_dir = None
        self.error_message = None

    def start(self, config_path, overrides):
        with self._lock:
            if self.running:
                return False, "Training is already running."

        # Setup paths
        abs_config_path = PROJECT_ROOT / config_path
        if not abs_config_path.exists():
            return False, f"Config path {config_path} does not exist."

        # Create temporary config with overrides
        try:
            with open(abs_config_path, 'r', encoding='utf-8') as f:
                cfg_data = yaml.safe_load(f) or {}

            # Convert relative _base_ config path to absolute path relative to the template config's directory
            if "_base_" in cfg_data:
                base_ref = cfg_data["_base_"]
                if not os.path.isabs(base_ref):
                    abs_base_path = (abs_config_path.parent / base_ref).resolve()
                    cfg_data["_base_"] = str(abs_base_path.as_posix())

            # Apply overrides
            for key, val in overrides.items():
                if val is not None and val != "":
                    # Handle nested overrides (e.g. optimizer.learning_rate)
                    parts = key.split('.')
                    target = cfg_data
                    for p in parts[:-1]:
                        target = target.setdefault(p, {})
                    try:
                        # Attempt to parse as int/float/bool if possible
                        if str(val).lower() == 'true':
                            parsed_val = True
                        elif str(val).lower() == 'false':
                            parsed_val = False
                        else:
                            try:
                                parsed_val = int(val)
                            except ValueError:
                                try:
                                    parsed_val = float(val)
                                except ValueError:
                                    parsed_val = val
                        target[parts[-1]] = parsed_val
                    except Exception:
                        target[parts[-1]] = val

            # Always write temp config to FLASK_DIR
            temp_cfg_path = FLASK_DIR / "temp_config.yaml"
            with open(temp_cfg_path, 'w', encoding='utf-8') as f:
                yaml.dump(cfg_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            return False, f"Failed to build configuration overrides: {e}"

        # Resolve python interpreter
        python_exe = sys.executable
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            python_exe = str(venv_python)

        cmd = [
            python_exe, "scripts/train.py",
            "--config", str(temp_cfg_path)
        ]

        # Load environment secrets
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        secrets_dir = PROJECT_ROOT / ".secrets"
        if secrets_dir.is_dir():
            for env_file in sorted(secrets_dir.glob("*.env")):
                try:
                    for line in env_file.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env[k] = v
                except Exception:
                    pass

        # Propagate OPTIMIZER_* keys to AZURE_OPENAI_*
        for suffix in ["ENDPOINT", "API_VERSION", "AUTH_MODE", "MANAGED_IDENTITY_CLIENT_ID", "AD_SCOPE", "API_KEY"]:
            base_key = f"AZURE_OPENAI_{suffix}"
            optimizer_key = f"OPTIMIZER_AZURE_OPENAI_{suffix}"
            if not env.get(base_key) and env.get(optimizer_key):
                env[base_key] = env[optimizer_key]

        try:
            # Run from PROJECT_ROOT
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
        except Exception as e:
            return False, f"Failed to launch training process: {e}"

        with self._lock:
            self.process = proc
            self.log_lines = [f"$ {' '.join(cmd)}\n"]
            self.stage = "Starting"
            self.step = 0
            self.total_steps = 0
            self.epoch = 0
            self.total_epochs = 0
            self.running = True
            self.output_dir = None
            self.error_message = None

        # Start thread to read outputs
        thread = threading.Thread(target=self._read_output, daemon=True)
        thread.start()

        return True, "Training started successfully."

    def _read_output(self):
        for line in self.process.stdout:
            with self._lock:
                self.log_lines.append(line)
                self._parse_line(line)
                # Cap log lines buffer size
                if len(self.log_lines) > 5000:
                    self.log_lines = self.log_lines[-4000:]
        exit_code = self.process.wait()
        with self._lock:
            self.running = False
            if exit_code == 0:
                self.stage = "Finished"
            else:
                self.stage = f"Error (exit={exit_code})"
                self.error_message = f"Process exited with error code {exit_code}."

    def _parse_line(self, line):
        line_lower = line.lower()
        # Parse active stage
        if "1/6 rollout" in line_lower or ("rollout" in line_lower and "worker" in line_lower):
            self.stage = "Rollout"
        elif "2/6 reflect" in line_lower or ("reflect" in line_lower and "patch" in line_lower):
            self.stage = "Reflect"
        elif "3/6 aggregate" in line_lower or "merge" in line_lower:
            self.stage = "Aggregate"
        elif "4/6 select" in line_lower:
            self.stage = "Select"
        elif "5/6 update" in line_lower:
            self.stage = "Update"
        elif "6/6" in line_lower or ("gate" in line_lower and "score" in line_lower):
            self.stage = "Gate"
        elif "slow update" in line_lower:
            self.stage = "Slow Update"
        elif "meta skill" in line_lower:
            self.stage = "Meta Skill"
        elif "baseline" in line_lower and "evaluate" in line_lower:
            self.stage = "Baseline"

        # Parse epoch and step numbers
        if "[step" in line_lower:
            try:
                parts = line.split("[STEP")[1].split("]")[0].split("/")
                self.step = int(parts[0].strip())
                self.total_steps = int(parts[1].strip())
            except Exception:
                pass
        if "[epoch" in line_lower:
            try:
                parts = line.split("[EPOCH")[1].split("]")[0].split("/")
                self.epoch = int(parts[0].strip())
                self.total_epochs = int(parts[1].strip())
            except Exception:
                pass

        # Capture output directory location
        if "output saved to:" in line_lower:
            try:
                self.output_dir = line.split("output saved to:")[1].strip()
            except Exception:
                pass

    def stop(self):
        with self._lock:
            if not self.running or not self.process:
                return False, "No training run is currently active."

        try:
            if os.name == 'nt':
                # Force recursive kill of process group on Windows
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception as e:
            try:
                self.process.terminate()
            except Exception:
                pass

        self.process.wait()
        with self._lock:
            self.running = False
            self.stage = "Stopped"

        return True, "Training process terminated."

    def get_status(self):
        with self._lock:
            progress = 0.0
            if self.total_steps > 0:
                progress = min(self.step / self.total_steps, 1.0)
            elif self.total_epochs > 0 and self.epoch > 0:
                progress = min(self.epoch / self.total_epochs, 1.0)

            return {
                "running": self.running,
                "stage": self.stage,
                "epoch": self.epoch,
                "total_epochs": self.total_epochs,
                "step": self.step,
                "total_steps": self.total_steps,
                "progress": progress,
                "output_dir": self.output_dir,
                "error_message": self.error_message,
                "logs": self.log_lines[-200:] # Latest 200 lines
            }

manager = TrainingManager()


# ─── API Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/parse-csv-headers', methods=['POST'])
def parse_csv_headers():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        # Append unique timestamp to avoid name collisions
        name_parts = os.path.splitext(filename)
        unique_filename = f"{name_parts[0]}_{int(time.time())}{name_parts[1]}"
        dest_path = UPLOAD_FOLDER / unique_filename
        file.save(dest_path)
        
        try:
            try:
                with open(dest_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
            except UnicodeDecodeError:
                with open(dest_path, 'r', encoding='latin-1') as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                
            if not headers:
                return jsonify({"success": False, "message": "CSV file is empty or missing headers."}), 400
                
            # Clean headers
            headers = [h.strip() for h in headers if h.strip()]
            return jsonify({
                "success": True, 
                "headers": headers, 
                "filename": unique_filename
            })
        except Exception as e:
            return jsonify({"success": False, "message": f"Error parsing CSV headers: {e}"}), 500
    else:
        return jsonify({"success": False, "message": "Invalid file type. Please upload a .csv file."}), 400



@app.route('/api/convert-csv', methods=['POST'])
def convert_csv():
    data = request.json or {}
    csv_file_path = data.get('csv_path', 'train/ceramic_capacitors.csv')
    csv_filename = data.get('csv_filename') # if uploaded
    json_file_path = data.get('json_path', 'train/ceramic_capacitors.json')
    split_ratio_str = data.get('split_ratio', '6:2:2') # Train:Val:Test

    # Determine CSV source path
    if csv_filename:
        abs_csv = UPLOAD_FOLDER / csv_filename
    else:
        abs_csv = PROJECT_ROOT / csv_file_path

    if not abs_csv.exists():
        return jsonify({"success": False, "message": "CSV file not found."}), 400

    try:
        # Read CSV file
        try:
            with open(abs_csv, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
                rows = list(reader)
        except UnicodeDecodeError:
            with open(abs_csv, 'r', encoding='latin-1') as f:
                reader = csv.DictReader(f)
                fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
                rows = list(reader)

        if not fieldnames:
            return jsonify({"success": False, "message": "CSV file contains no headers or columns."}), 400

        # Input columns list
        input_cols = data.get('input_cols', [])
        if not input_cols and data.get('input_col'):
            input_cols = [data.get('input_col')]
        if isinstance(input_cols, str):
            input_cols = [c.strip() for c in input_cols.split(',') if c.strip()]
        
        # Output/Target columns list
        target_cols = data.get('target_cols', [])
        if not target_cols and data.get('target_cols_str'):
            target_cols = [data.get('target_cols_str')]
        if isinstance(target_cols, str):
            if target_cols == '*' or target_cols == ['*']:
                target_cols = [col for col in fieldnames if col not in input_cols]
            else:
                target_cols = [c.strip() for c in target_cols.split(',') if c.strip()]
        elif isinstance(target_cols, list) and (not target_cols or target_cols == ['*']):
            target_cols = [col for col in fieldnames if col not in input_cols]

        # Clean selected columns against actual fieldnames
        input_cols = [c for c in input_cols if c in fieldnames]
        target_cols = [c for c in target_cols if c in fieldnames]

        if not input_cols:
            return jsonify({"success": False, "message": "No valid input columns selected."}), 400
        if not target_cols:
            return jsonify({"success": False, "message": "No valid target/output columns selected."}), 400

        # Parse ratio
        try:
            parts = [int(p.strip()) for p in split_ratio_str.split(':')]
            if len(parts) != 3 or any(p <= 0 for p in parts):
                raise ValueError
        except Exception:
            return jsonify({"success": False, "message": "Invalid split ratio format. Must be formatted like 6:2:2."}), 400

        # Construct JSON entries
        items = []
        for idx, row in enumerate(rows):
            # Build input description (join multi-fields if length > 1)
            if len(input_cols) == 1:
                description = row.get(input_cols[0], "").strip()
            else:
                description = "\n".join(f"{ic}: {str(row.get(ic, '')).strip()}" for ic in input_cols if row.get(ic) is not None)

            ground_truth = {}
            for col in target_cols:
                val = row.get(col, "")
                if val is not None:
                    val = val.strip()
                    # Try to parse as JSON if it is structured
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, (dict, list)):
                            ground_truth[col] = parsed
                        else:
                            ground_truth[col] = val
                    except Exception:
                        ground_truth[col] = val

            items.append({
                "id": str(idx + 1),
                "input": description,
                "ground_truth": ground_truth
            })

        # Save output JSON
        abs_json = PROJECT_ROOT / json_file_path
        abs_json.parent.mkdir(parents=True, exist_ok=True)
        with open(abs_json, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

        # Create a clean initial skill template for this new dataset if it doesn't exist
        json_path_obj = Path(json_file_path)
        base_name = json_path_obj.stem
        skill_filename = f"{base_name}_initial.md"
        abs_skill = PROJECT_ROOT / "train" / skill_filename
        
        if not abs_skill.exists():
            columns_bullets = "\n".join(f"- {col}" for col in target_cols)
            has_mpn = any("mpn" in str(col).lower() for col in target_cols)
            
            if has_mpn:
                skill_content = (
                    f"# Parameter Extraction Guidelines\n\n"
                    f"To extract parameters from the input description:\n"
                    f"1. Identify the Manufacturer Part Number (MPN) or device identifier.\n"
                    f"2. Extract the values for the following fields:\n"
                    f"{columns_bullets}\n\n"
                    f"## Rules\n\n"
                    f"- Use only values explicitly present or reliably attributable to the identified part number/family.\n"
                    f"- If any target value cannot be determined, return `null`.\n"
                    f"- Do not hallucinate, estimate, or copy unrelated package/body text as electrical specs.\n"
                    f"- Do not add extra commentary, confidence, or alternate values.\n"
                )
            else:
                skill_content = (
                    f"# Parameter Extraction Guidelines\n\n"
                    f"To extract parameters from the input description:\n"
                    f"1. Analyze the description details.\n"
                    f"2. Extract the values for the following fields:\n"
                    f"{columns_bullets}\n\n"
                    f"## Rules\n\n"
                    f"- Use only values explicitly present or reliably attributable to the identified part number/family.\n"
                    f"- If any target value cannot be determined, return `null`.\n"
                    f"- Do not hallucinate, estimate, or copy unrelated package/body text as electrical specs.\n"
                    f"- Do not add extra commentary, confidence, or alternate values.\n"
                )
                
            with open(abs_skill, 'w', encoding='utf-8') as sf:
                sf.write(skill_content)

        return jsonify({
            "success": True,
            "message": f"Successfully converted {len(items)} CSV rows and saved to {json_file_path}",
            "preview": items[:5],
            "total_items": len(items),
            "skill_path": f"train/{skill_filename}"
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error converting CSV: {e}"}), 500


@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    data_path = request.args.get('path', 'train/ceramic_capacitors.json')
    abs_path = PROJECT_ROOT / data_path
    if not abs_path.exists():
        return jsonify({"success": False, "message": f"Dataset file not found at {data_path}"}), 404
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading dataset: {e}"}), 500

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    try:
        datasets = []
        train_dir = PROJECT_ROOT / "train"
        data_dir = PROJECT_ROOT / "data"
        for root_dir in [train_dir, data_dir]:
            if root_dir.exists():
                for path in root_dir.glob("**/*.json"):
                    if not any(exclude in path.name for exclude in ["runtime_state", "summary", "history", "config"]):
                        rel_path = os.path.relpath(path, PROJECT_ROOT)
                        datasets.append(rel_path.replace("\\", "/"))
        return jsonify({"success": True, "datasets": sorted(list(set(datasets)))})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error scanning datasets: {e}"}), 500

@app.route('/api/skills', methods=['GET'])
def list_skills():
    try:
        skills = []
        train_dir = PROJECT_ROOT / "train"
        configs_dir = PROJECT_ROOT / "configs"
        skillopt_dir = PROJECT_ROOT / "skillopt"
        for root_dir in [train_dir, configs_dir, skillopt_dir]:
            if root_dir.exists():
                for path in root_dir.glob("**/*.md"):
                    rel_path = os.path.relpath(path, PROJECT_ROOT)
                    rel_path_clean = rel_path.replace("\\", "/")
                    if not any(exclude in rel_path_clean.lower() for exclude in ["readme", "license", "contributing", "security", "changelog"]):
                        skills.append(rel_path_clean)
        return jsonify({"success": True, "skills": sorted(list(set(skills)))})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error scanning skills: {e}"}), 500


@app.route('/api/envs', methods=['GET'])
def list_envs():
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.train import _register_builtins, _ENV_REGISTRY
        _register_builtins()
        envs = list(_ENV_REGISTRY.keys())
        if not envs:
            envs = ["ceramic_capacitors", "generic_csv"]
        return jsonify({"success": True, "envs": sorted(envs)})
    except Exception as e:
        # Robust fallback containing standard envs
        return jsonify({"success": True, "envs": [
            "alfworld",
            "babyvision",
            "ceramic_capacitors",
            "docvqa",
            "generic_csv",
            "livemathematicianbench",
            "mathverse",
            "mmrb",
            "officeqa",
            "sealqa",
            "searchqa",
            "spreadsheetbench"
        ]})


@app.route('/api/configs', methods=['GET'])
def get_configs():
    try:
        # Discover yaml configs
        configs = []
        configs_dir = PROJECT_ROOT / "configs"
        train_dir = PROJECT_ROOT / "train"
        
        # Scan configs/ and train/ for yaml files
        for root_dir in [configs_dir, train_dir, FLASK_DIR]:
            if root_dir.exists():
                for path in root_dir.glob("**/*.yaml"):
                    if "_base_" not in str(path) and "temp_config" not in str(path):
                        rel_path = os.path.relpath(path, PROJECT_ROOT)
                        configs.append(rel_path.replace("\\", "/"))
        
        return jsonify({"success": True, "configs": sorted(list(set(configs)))})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error fetching configs: {e}"}), 500


@app.route('/api/configs/active', methods=['GET'])
def get_active_config():
    config_path = request.args.get('path', 'train/config.yaml')
    abs_path = PROJECT_ROOT / config_path
    if not abs_path.exists():
        return jsonify({"success": False, "message": f"Config path {config_path} does not exist."}), 404
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            cfg_data = yaml.safe_load(f) or {}
        return jsonify({"success": True, "config": cfg_data})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading config: {e}"}), 500


@app.route('/api/train/start', methods=['POST'])
def start_training():
    data = request.json or {}
    config_path = data.get('config_path', 'train/config.yaml')
    overrides = data.get('overrides', {})
    success, msg = manager.start(config_path, overrides)
    return jsonify({"success": success, "message": msg})


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    success, msg = manager.stop()
    return jsonify({"success": success, "message": msg})


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    return jsonify(manager.get_status())


@app.route('/api/results', methods=['GET'])
def get_results_list():
    try:
        results = []
        outputs_dir = PROJECT_ROOT / "outputs"
        if outputs_dir.exists():
            for entry in outputs_dir.iterdir():
                if entry.is_dir():
                    summary_path = entry / "summary.json"
                    config_path = entry / "config.json"
                    
                    name = entry.name
                    env_name = "unknown"
                    best_score = "N/A"
                    baseline_score = "N/A"
                    test_score = "N/A"
                    timestamp = "unknown"
                    
                    # Try parsing summary
                    if summary_path.exists():
                        try:
                            with open(summary_path, 'r', encoding='utf-8') as f:
                                summary = json.load(f)
                            best_score = summary.get("best_selection_hard", "N/A")
                            baseline_score = summary.get("baseline_selection_hard", "N/A")
                            test_score = summary.get("test_hard", "N/A")
                            cfg = summary.get("config", {})
                            env_name = cfg.get("env", "unknown")
                        except Exception:
                            pass
                    elif config_path.exists():
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                cfg = json.load(f)
                            env_name = cfg.get("env", "unknown")
                        except Exception:
                            pass

                    # Extract timestamp from folder name (e.g. skillopt_ceramic_capacitors_gpt-5.4_20260528_213642)
                    parts = name.split('_')
                    if len(parts) >= 2:
                        date_str = parts[-2]
                        time_str = parts[-1]
                        if len(date_str) == 8 and len(time_str) == 6:
                            timestamp = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"

                    results.append({
                        "id": name,
                        "name": name,
                        "env": env_name,
                        "baseline_score": baseline_score,
                        "best_score": best_score,
                        "test_score": test_score,
                        "timestamp": timestamp
                    })
        
        # Sort results by timestamp descending
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error scanning results: {e}"}), 500


@app.route('/api/results/<run_id>', methods=['GET'])
def get_run_details(run_id):
    run_dir = PROJECT_ROOT / "outputs" / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return jsonify({"success": False, "message": f"Run {run_id} not found"}), 404

    try:
        details = {
            "run_id": run_id,
            "summary": {},
            "history": [],
            "best_skill": "",
            "rollout_results": []
        }

        # 1. Read summary.json
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                details["summary"] = json.load(f)

        # 2. Read history.json
        history_path = run_dir / "history.json"
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                details["history"] = json.load(f)

        # 3. Read best_skill.md
        best_skill_path = run_dir / "best_skill.md"
        if best_skill_path.exists():
            with open(best_skill_path, 'r', encoding='utf-8') as f:
                details["best_skill"] = f.read()

        # 4. Load step results.jsonl based on best_step
        best_step = details["summary"].get("best_step", 0)
        
        # Find rollout results files
        # Check standard step directory paths
        step_dir_name = f"step_{best_step:04d}"
        results_file = run_dir / "steps" / step_dir_name / "rollout" / "results.jsonl"
        
        # Fallbacks: check in selection_eval_baseline or scan any step if best_step results not found
        if not results_file.exists():
            results_file = run_dir / "selection_eval_baseline" / "results.jsonl"
        if not results_file.exists():
            # Find any results.jsonl in steps/
            steps_dir = run_dir / "steps"
            if steps_dir.exists():
                for results_path in sorted(steps_dir.glob("**/results.jsonl")):
                    results_file = results_path
                    break

        if results_file.exists():
            rollouts = []
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rollouts.append(json.loads(line))
                        except Exception:
                            pass
            details["rollout_results"] = rollouts

        return jsonify({"success": True, "details": details})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading run details: {e}"}), 500

if __name__ == '__main__':
    # Start on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
