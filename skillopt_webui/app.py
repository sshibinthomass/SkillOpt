"""
SkillOpt WebUI — Configure, launch, and monitor training from your browser.

Usage:
    python -m skillopt_webui.app [--port PORT] [--share]
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Config helpers ──────────────────────────────────────────────────────────

def discover_configs() -> list[str]:
    """Find all YAML configs under configs/."""
    pattern = str(PROJECT_ROOT / "configs" / "**" / "*.yaml")
    paths = sorted(glob.glob(pattern, recursive=True))
    return [os.path.relpath(p, PROJECT_ROOT) for p in paths
            if "_base_" not in p]


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(PROJECT_ROOT / path) as f:
        return yaml.safe_load(f)


def config_to_display(cfg: dict) -> str:
    """Pretty-print config for display."""
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)

def process_uploaded_csv(file, dataset_name, input_col, target_col, split_ratio, progress=gr.Progress()):
    import pandas as pd
    import numpy as np
    import json
    
    if file is None:
        return "❌ Please upload a CSV file.", gr.update()
        
    progress(0.1, desc="Parsing input parameters...")
    
    if not dataset_name or not dataset_name.strip():
        dataset_name = Path(file.name).stem
        
    dataset_name = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in dataset_name.strip()])
    
    progress(0.3, desc="Reading CSV dataset file...")
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Error reading CSV file: {e}", gr.update()
        
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    
    # 1. Parse input columns
    if isinstance(input_col, list):
        input_cols = [str(ic).strip() for ic in input_col if ic]
    else:
        input_cols = [c.strip() for c in str(input_col).split(",") if c.strip()]
        
    if not input_cols:
        return "❌ Please select or enter at least one input column.", gr.update()
        
    for ic in input_cols:
        if ic not in cols:
            return f"❌ Input column '{ic}' not found. Available columns: {cols}", gr.update()
        
    # 2. Parse target columns
    if isinstance(target_col, list):
        target_cols_raw = [str(tc).strip() for tc in target_col if tc]
    else:
        target_cols_raw = [c.strip() for c in str(target_col).split(",") if c.strip()]
        
    if "*" in target_cols_raw or any(tc.lower() in ("all", "all_others", "") for tc in target_cols_raw):
        target_cols = [c for c in cols if c not in input_cols]
    else:
        target_cols = target_cols_raw
        for tc in target_cols:
            if tc not in cols:
                return f"❌ Target column '{tc}' not found. Available columns: {cols}", gr.update()
                
    if not target_cols:
        return "❌ Please select or enter at least one target column.", gr.update()
        
    try:
        parts = [int(p.strip()) for p in split_ratio.split(":")]
        if len(parts) != 3 or any(p <= 0 for p in parts):
            raise ValueError
    except Exception:
        return f"❌ Invalid split ratio '{split_ratio}'. Use e.g. 3:1:1.", gr.update()
        
    items = []
    for idx, row in df.iterrows():
        # Build multi-field input representation
        if len(input_cols) == 1:
            val_input = str(row[input_cols[0]]).strip()
        else:
            val_input = "\n".join(f"{ic}: {str(row[ic]).strip()}" for ic in input_cols if not pd.isna(row[ic]))
        
        ground_truth = {}
        for tc in target_cols:
            val_target = row[tc]
            if pd.isna(val_target):
                val_target_str = ""
            else:
                val_target_str = str(val_target).strip()
            try:
                parsed_val = json.loads(val_target_str)
                if isinstance(parsed_val, (dict, list)):
                    ground_truth[tc] = parsed_val
                else:
                    ground_truth[tc] = val_target_str
            except Exception:
                ground_truth[tc] = val_target_str
                
        items.append({
            "id": f"row_{idx+1:03d}",
            "input": val_input,
            "ground_truth": ground_truth
        })
        
    np.random.seed(42)
    shuffled = list(items)
    np.random.shuffle(shuffled)
    
    total = len(shuffled)
    total_ratio = sum(parts)
    train_n = int(total * parts[0] / total_ratio)
    val_n = int(total * parts[1] / total_ratio)
    
    train_items = shuffled[:train_n]
    val_items = shuffled[train_n:train_n+val_n]
    test_items = shuffled[train_n+val_n:]
    
    if not train_items and shuffled:
        train_items = shuffled
        
    progress(0.6, desc="Splitting dataset deterministically...")
    split_dir = PROJECT_ROOT / "data" / "uploaded_csv" / dataset_name
    progress(0.8, desc="Writing JSON dataset splits to file...")
    for name, split_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        path = split_dir / name
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "items.json", "w", encoding="utf-8") as f:
            json.dump(split_items, f, ensure_ascii=False, indent=2)
            
    progress(0.9, desc="Generating configuration YAML file...")
    config_data = {
        "_base_": "../_base_/default.yaml",
        "env": {
            "name": "generic_csv",
            "skill_init": "skillopt/envs/generic_csv/skills/initial.md",
            "split_mode": "split_dir",
            "split_dir": f"data/uploaded_csv/{dataset_name}"
        }
    }
    
    config_dir = PROJECT_ROOT / "configs" / "uploaded_csv"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
    relative_config_path = os.path.relpath(config_path, PROJECT_ROOT)
    choices = discover_configs()
    
    progress(1.0, desc="Conversion complete!")
    return (
        f"✅ Dataset successfully split into data/uploaded_csv/{dataset_name}/\n✅ Config created at {relative_config_path}\nReady to train! Switch to 'Configure & Launch' tab to begin.",
        gr.update(choices=choices, value=relative_config_path)
    )

def delete_config_file(config_path):
    if not config_path:
        return "❌ No config selected.", gr.update(), ""
        
    abs_path = PROJECT_ROOT / config_path
    if not abs_path.exists():
        return f"❌ Config file {config_path} not found.", gr.update(), ""
        
    try:
        os.remove(abs_path)
        
        # Clean up data splits if it is an uploaded CSV config
        if "uploaded_csv" in config_path:
            dataset_name = abs_path.stem
            split_dir = PROJECT_ROOT / "data" / "uploaded_csv" / dataset_name
            if split_dir.exists() and split_dir.is_dir():
                import shutil
                shutil.rmtree(split_dir)
                
        choices = discover_configs()
        new_val = choices[0] if choices else None
        
        preview = ""
        if new_val:
            try:
                preview = config_to_display(load_config(new_val))
            except Exception:
                pass
                
        return (
            f"🗑️ Successfully deleted config: {config_path}",
            gr.update(choices=choices, value=new_val),
            preview
        )
    except Exception as e:
        return f"❌ Error deleting config: {e}", gr.update(), ""


# ─── Training process management ────────────────────────────────────────────

class TrainingManager:
    """Manages a single training subprocess."""

    def __init__(self):
        self._lock = threading.Lock()
        self.process = None
        self.log_lines: list[str] = []
        self.stage = "Idle"
        self.step = 0
        self.total_steps = 0
        self.epoch = 0
        self.total_epochs = 0
        self.running = False

    def start(self, config_path: str, overrides: dict) -> str:
        with self._lock:
            if self.running:
                return "⚠️ Training already running. Stop it first."

        cmd = [
            sys.executable, "scripts/train.py",
            "--config", config_path,
        ]
        cfg_options = []
        for k, v in overrides.items():
            if v is not None and v != "":
                cfg_options.append(f"{k}={v}")
        if cfg_options:
            cmd.append("--cfg-options")
            cmd.extend(cfg_options)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Auto-load API credentials from .secrets/*.env
        secrets_dir = PROJECT_ROOT / ".secrets"
        if secrets_dir.is_dir():
            for env_file in sorted(secrets_dir.glob("*.env")):
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k] = v
        # Propagate OPTIMIZER_* to base AZURE_OPENAI_* when base is missing,
        # so target/default endpoints inherit from optimizer config.
        _propagate = [
            ("ENDPOINT", ""), ("API_VERSION", ""), ("AUTH_MODE", ""),
            ("MANAGED_IDENTITY_CLIENT_ID", ""), ("AD_SCOPE", ""),
            ("API_KEY", ""),
        ]
        for suffix, _ in _propagate:
            base_key = f"AZURE_OPENAI_{suffix}"
            optimizer_key = f"OPTIMIZER_AZURE_OPENAI_{suffix}"
            if not env.get(base_key) and env.get(optimizer_key):
                env[base_key] = env[optimizer_key]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
                env=env,
                start_new_session=True,  # create process group for clean kill
            )
        except Exception as e:
            return f"❌ Failed to start training: {e}"

        with self._lock:
            self.process = proc
            self.log_lines = [f"$ {' '.join(cmd)}\n"]
            self.stage = "Starting"
            self.step = 0
            self.total_steps = 0
            self.epoch = 0
            self.total_epochs = 0
            self.running = True

        thread = threading.Thread(target=self._read_output, daemon=True)
        thread.start()

        return "✅ Training started!"

    def _read_output(self):
        for line in self.process.stdout:
            with self._lock:
                self.log_lines.append(line)
                self._parse_stage(line)
                if len(self.log_lines) > 5000:
                    self.log_lines = self.log_lines[-4000:]
        self.process.wait()
        with self._lock:
            self.running = False
            self.stage = f"Finished (exit={self.process.returncode})"

    def _parse_stage(self, line: str):
        line_lower = line.lower()
        if "1/6 rollout" in line_lower or ("rollout" in line_lower and "worker" in line_lower):
            self.stage = "🎯 Rollout"
        elif "2/6 reflect" in line_lower or ("reflect" in line_lower and "patch" in line_lower):
            self.stage = "🔍 Reflect"
        elif "3/6 aggregate" in line_lower or "merge" in line_lower:
            self.stage = "🔗 Aggregate"
        elif "4/6 select" in line_lower:
            self.stage = "✂️ Select"
        elif "5/6 update" in line_lower:
            self.stage = "📝 Update"
        elif "6/6" in line_lower or ("gate" in line_lower and "score" in line_lower):
            self.stage = "🚦 Gate"
        elif "slow update" in line_lower:
            self.stage = "🔄 Slow Update"
        elif "meta skill" in line_lower:
            self.stage = "🧠 Meta Skill"
        elif "baseline" in line_lower and "evaluate" in line_lower:
            self.stage = "📊 Baseline"
        if "[step" in line_lower:
            try:
                parts = line.split("[STEP")[1].split("]")[0].split("/")
                self.step = int(parts[0].strip())
                self.total_steps = int(parts[1].strip())
            except (IndexError, ValueError):
                pass
        if "[epoch" in line_lower:
            try:
                parts = line.split("[EPOCH")[1].split("]")[0].split("/")
                self.epoch = int(parts[0].strip())
                self.total_epochs = int(parts[1].strip())
            except (IndexError, ValueError):
                pass

    def stop(self) -> str:
        with self._lock:
            if self.process and self.running:
                try:
                    if hasattr(os, "killpg"):
                        # Kill entire process group (children included)
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    else:
                        self.process.terminate()
                except Exception:
                    try:
                        self.process.terminate()
                    except Exception:
                        pass
                self.process.wait(timeout=5)
                self.running = False
                self.stage = "Stopped"
                return "🛑 Training stopped."
            return "No training running."

    def get_logs(self) -> str:
        with self._lock:
            return "".join(self.log_lines[-500:])

    def get_colored_logs_html(self) -> str:
        """Render last 300 log lines with color-coded stages."""
        import html as html_mod
        with self._lock:
            lines = list(self.log_lines[-300:])
        parts = []
        for line in lines:
            # Rebrand: display "skillopt" instead of "reflact" in logs
            line_display = line.replace("reflact", "skillopt").replace("ReflACT", "SkillOpt").replace("Reflact", "Skillopt").replace("REFLACT", "SKILLOPT")
            escaped = html_mod.escape(line_display.rstrip("\n"))
            low = line.lower()
            if "[epoch" in low:
                color = "#f59e0b"  # amber
                weight = "700"
            elif "[step" in low:
                color = "#8b5cf6"  # purple
                weight = "700"
            elif "rollout]" in low or "1/6" in low:
                color = "#3b82f6"  # blue
            elif "reflect" in low or "2/6" in low:
                color = "#f97316"  # orange
            elif "aggregate" in low or "3/6" in low or "merge" in low:
                color = "#06b6d4"  # cyan
            elif "select" in low or "4/6" in low:
                color = "#ec4899"  # pink
            elif "update" in low or "5/6" in low:
                color = "#10b981"  # green
            elif "gate" in low or "6/6" in low:
                color = "#ef4444"  # red
            elif "slow update" in low:
                color = "#f59e0b"  # amber
                weight = "700"
            elif "meta skill" in low:
                color = "#a855f7"  # violet
                weight = "700"
            elif "baseline" in low:
                color = "#6366f1"  # indigo
                weight = "700"
            elif "[rollout]" in low:
                # per-item rollout progress
                if "hard=1" in line:
                    color = "#22c55e"  # green for correct
                elif "hard=0" in line:
                    color = "#f87171"  # red for wrong
                elif "timeout" in low:
                    color = "#fbbf24"  # yellow for timeout
                else:
                    color = "#94a3b8"  # gray
                weight = "400"
            elif "error" in low or "fail" in low:
                color = "#ef4444"
                weight = "700"
            elif "========" in line:
                color = "#64748b"  # separator
                weight = "400"
            else:
                color = "#e2e8f0"  # default light gray
                weight = "400"
            if "weight" not in dir():
                weight = "400"
            parts.append(f'<span style="color:{color};font-weight:{weight}">{escaped}</span>')
            weight = "400"  # reset

        log_html = "<br>".join(parts) if parts else '<span style="color:#94a3b8">No logs yet. Click Refresh after launching training.</span>'
        return f'''<div id="log-container" style="
            height:500px;overflow-y:auto;background:#0f172a;padding:16px;
            border-radius:10px;font-family:'JetBrains Mono',Consolas,monospace;
            font-size:12.5px;line-height:1.6;border:1px solid #1e293b;
            box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);">{log_html}</div>
            <script>
                var el = document.getElementById("log-container");
                if (el) {{ el.scrollTop = el.scrollHeight; }}
            </script>'''

    def get_progress_html(self) -> str:
        """Render a visual progress bar."""
        s = self.get_status()
        step = s["step"]
        total = s["total_steps"]
        epoch = self.epoch
        total_epochs = self.total_epochs
        pct = s["progress"] * 100

        if not self.running and step == 0:
            return '<div style="color:#94a3b8;text-align:center;padding:12px;">Waiting for training to start...</div>'

        # Color based on progress
        if pct < 25:
            bar_color = "linear-gradient(90deg, #3b82f6, #6366f1)"
        elif pct < 50:
            bar_color = "linear-gradient(90deg, #6366f1, #8b5cf6)"
        elif pct < 75:
            bar_color = "linear-gradient(90deg, #8b5cf6, #a855f7)"
        else:
            bar_color = "linear-gradient(90deg, #a855f7, #22c55e)"

        stage_icon = self.stage if self.stage != "Idle" else "⏳"
        status_dot = "🟢" if self.running else ("✅" if "Finished" in self.stage else "⚪")

        epoch_str = f"Epoch {epoch}/{total_epochs}" if total_epochs > 0 else ""
        step_str = f"Step {step}/{total}" if total > 0 else ""

        return f'''
        <div style="background:#1e293b;border-radius:12px;padding:16px;border:1px solid #334155;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="color:#e2e8f0;font-weight:700;font-size:1rem;">{status_dot} {stage_icon}</span>
            <span style="color:#94a3b8;font-size:0.9rem;">{epoch_str} &nbsp; {step_str}</span>
            <span style="color:#e2e8f0;font-weight:700;font-size:1rem;">{pct:.1f}%</span>
          </div>
          <div style="background:#0f172a;border-radius:8px;height:20px;overflow:hidden;border:1px solid #334155;">
            <div style="height:100%;width:{pct}%;background:{bar_color};
                        border-radius:8px;transition:width 0.5s ease;
                        box-shadow:0 0 12px rgba(99,102,241,0.4);"></div>
          </div>
        </div>'''

    def get_status(self) -> dict:
        with self._lock:
            progress = 0
            if self.total_steps > 0:
                progress = self.step / self.total_steps
            return {
                "running": self.running,
                "stage": self.stage,
                "step": self.step,
                "total_steps": self.total_steps,
                "progress": progress,
            }


manager = TrainingManager()


# ─── Pipeline Stage HTML ────────────────────────────────────────────────────

STAGES = ["Rollout", "Reflect", "Aggregate", "Select", "Update", "Gate"]
STAGE_ICONS = ["🎯", "🔍", "🔗", "✂️", "📝", "🚦"]


def render_pipeline_html(active_stage: str = "") -> str:
    """Render animated pipeline HTML."""
    html = '<div style="display:flex;align-items:center;justify-content:center;gap:4px;padding:20px;flex-wrap:wrap;">'
    for i, (name, icon) in enumerate(zip(STAGES, STAGE_ICONS)):
        is_active = name.lower() in active_stage.lower() if active_stage else False
        bg = "#6366f1" if is_active else "#f3f4f6"
        color = "white" if is_active else "#374151"
        border = "3px solid #4f46e5" if is_active else "2px solid #d1d5db"
        shadow = "0 0 20px rgba(99,102,241,0.4)" if is_active else "none"
        pulse = "animation: pulse 1.5s ease-in-out infinite;" if is_active else ""
        html += f'''
        <div style="display:flex;flex-direction:column;align-items:center;padding:12px 16px;
                    border-radius:12px;background:{bg};color:{color};border:{border};
                    min-width:80px;box-shadow:{shadow};transition:all 0.3s;{pulse}">
          <span style="font-size:1.5rem">{icon}</span>
          <span style="font-weight:700;font-size:0.85rem;margin-top:4px">{name}</span>
        </div>'''
        if i < len(STAGES) - 1:
            arrow_color = "#6366f1" if is_active else "#d1d5db"
            html += f'<div style="font-size:1.2rem;color:{arrow_color}">→</div>'
    html += '</div>'
    html += '<style>@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.05)}}</style>'
    return html


# ─── Gradio UI ──────────────────────────────────────────────────────────────

def build_ui():
    configs = discover_configs()

    with gr.Blocks(
        title="SkillOpt WebUI",
    ) as app:
        gr.Markdown("# 🧠 SkillOpt Training Dashboard")
        gr.Markdown("*SKILLOPT: Executive Strategy for Self-Evolving Agent Skills — Configure, launch, and monitor training.*")

        with gr.Tabs():
            # ── Tab 1: Configure & Launch ────────────────────────────
            with gr.Tab("⚙️ Configure & Launch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            config_dropdown = gr.Dropdown(
                                choices=configs,
                                label="Config File",
                                value=configs[0] if configs else None,
                                scale=3
                            )
                            delete_config_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                        config_preview = gr.Code(
                            label="Config Preview",
                            language="yaml",
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Hyperparameters (DL Analogy)")
                        lr = gr.Slider(1, 32, value=4, step=1,
                                       label="Learning Rate (max edits/step)")
                        scheduler = gr.Dropdown(
                            ["cosine", "linear", "constant", "autonomous"],
                            value="cosine",
                            label="LR Scheduler",
                        )
                        num_epochs = gr.Slider(1, 8, value=4, step=1,
                                               label="Epochs")
                        batch_size = gr.Slider(10, 100, value=40, step=5,
                                               label="Batch Size (tasks per step)")
                        analyst_workers = gr.Slider(1, 32, value=16, step=1,
                                                    label="Analyst Workers (parallel reflection)")
                        use_slow_update = gr.Checkbox(value=True,
                                                       label="Slow Update (epoch-boundary momentum)")
                        use_meta_skill = gr.Checkbox(value=True,
                                                      label="Meta Skill (cross-epoch optimizer memory)")
                        use_gate = gr.Checkbox(value=True,
                                                label="Gate (validation-based accept/reject)")

                        with gr.Row():
                            launch_btn = gr.Button("🚀 Launch Training",
                                                    variant="primary", size="lg")
                            stop_btn = gr.Button("🛑 Stop", variant="stop")

                        status_text = gr.Textbox(label="Status", interactive=False)

                def on_config_change(path):
                    if path:
                        try:
                            return config_to_display(load_config(path))
                        except Exception as e:
                            return f"Error: {e}"
                    return ""

                config_dropdown.change(on_config_change, config_dropdown, config_preview)
                
                delete_config_btn.click(
                    delete_config_file,
                    inputs=[config_dropdown],
                    outputs=[status_text, config_dropdown, config_preview]
                )

                def on_launch(cfg_path, lr_val, sched, epochs, batch, workers,
                              slow_update, meta_skill, gate):
                    overrides = {
                        "optimizer.learning_rate": lr_val,
                        "optimizer.lr_scheduler": sched,
                        "train.num_epochs": epochs,
                        "train.batch_size": batch,
                        "gradient.analyst_workers": workers,
                        "optimizer.use_slow_update": slow_update,
                        "optimizer.use_meta_skill": meta_skill,
                        "evaluation.use_gate": gate,
                    }
                    return manager.start(cfg_path, overrides)

                launch_btn.click(
                    on_launch,
                    [config_dropdown, lr, scheduler, num_epochs, batch_size,
                     analyst_workers, use_slow_update, use_meta_skill, use_gate],
                    status_text,
                )
                stop_btn.click(lambda: manager.stop(), outputs=status_text)

            # ── Tab: CSV Upload & Training ───────────────────────────
            with gr.Tab("📁 CSV Upload & Training"):
                gr.Markdown("### Upload and Convert CSV Dataset")
                gr.Markdown("Upload a `.csv` file. It will be parsed and split into `train/`, `val/`, and `test/` splits automatically.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        csv_file = gr.File(label="Upload CSV File", file_types=[".csv"])
                        csv_dataset_name = gr.Textbox(label="Dataset Name", placeholder="e.g. my_custom_dataset (auto-derived if blank)")
                    
                    with gr.Column(scale=1):
                        csv_input_col = gr.Dropdown(label="Input Column Name(s)", choices=["input"], value=["input"], multiselect=True, allow_custom_value=True)
                        csv_target_col = gr.Dropdown(label="Target/Ground Truth Column Name(s) (use '*' for all other columns)", choices=["*"], value=["*"], multiselect=True, allow_custom_value=True)
                        csv_split_ratio = gr.Textbox(label="Split Ratio (Train:Val:Test)", value="3:1:1")
                
                csv_convert_btn = gr.Button("🚀 Convert & Prepare Config", variant="primary")
                csv_status = gr.Textbox(label="Conversion Status", interactive=False)
                
                def on_csv_upload(file):
                    if file is None:
                        return "", gr.update(choices=[]), gr.update(choices=[])
                    import pandas as pd
                    try:
                        df = pd.read_csv(file.name, nrows=0)
                        cols = list(df.columns)
                    except Exception:
                        cols = []
                    
                    dataset_name = Path(file.name).stem
                    
                    default_input = ["input"]
                    if cols:
                        lower_cols = [c.lower() for c in cols]
                        if "input" in lower_cols:
                            default_input = [cols[lower_cols.index("input")]]
                        elif "description" in lower_cols:
                            default_input = [cols[lower_cols.index("description")]]
                        elif "question" in lower_cols:
                            default_input = [cols[lower_cols.index("question")]]
                        else:
                            default_input = [cols[0]]
                            
                    target_choices = ["*"] + cols
                    
                    return (
                        dataset_name,
                        gr.update(choices=cols, value=default_input),
                        gr.update(choices=target_choices, value=["*"])
                    )
                
                csv_file.change(on_csv_upload, csv_file, [csv_dataset_name, csv_input_col, csv_target_col])
                
                csv_convert_btn.click(
                    process_uploaded_csv,
                    inputs=[csv_file, csv_dataset_name, csv_input_col, csv_target_col, csv_split_ratio],
                    outputs=[csv_status, config_dropdown]
                )

            # ── Tab 2: Monitor ───────────────────────────────────────
            with gr.Tab("📊 Monitor"):
                pipeline_html = gr.HTML(
                    value=render_pipeline_html(),
                    label="Pipeline Stage",
                )

                progress_html = gr.HTML(
                    value=manager.get_progress_html(),
                    label="Progress",
                )

                log_html = gr.HTML(
                    value=manager.get_colored_logs_html(),
                    label="Training Logs",
                )

                refresh_btn = gr.Button("🔄 Refresh Logs", variant="primary", size="lg")

                def on_refresh():
                    s = manager.get_status()
                    pipeline = render_pipeline_html(s["stage"])
                    progress = manager.get_progress_html()
                    logs = manager.get_colored_logs_html()
                    return pipeline, progress, logs

                refresh_btn.click(
                    on_refresh,
                    outputs=[pipeline_html, progress_html, log_html],
                )
                
                app.load(
                    on_refresh,
                    inputs=None,
                    outputs=[pipeline_html, progress_html, log_html],
                )
                timer = gr.Timer(2.0)
                timer.tick(
                    on_refresh,
                    outputs=[pipeline_html, progress_html, log_html],
                )

            # ── Tab 3: Results ───────────────────────────────────────
            with gr.Tab("📈 Results"):
                gr.Markdown("### Output Explorer")
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="outputs/",
                    interactive=True,
                )
                scan_btn = gr.Button("🔍 Scan Results")
                results_table = gr.Dataframe(
                    headers=["Experiment", "Benchmark", "Best Score", "Steps"],
                    label="Experiments",
                )

                def scan_outputs(out_dir):
                    rows = []
                    base = PROJECT_ROOT / out_dir
                    if not base.exists():
                        return rows
                    for bench_dir in sorted(base.iterdir()):
                        if not bench_dir.is_dir():
                            continue
                        for run_dir in sorted(bench_dir.iterdir()):
                            if not run_dir.is_dir():
                                continue
                            cfg_file = run_dir / "config.yaml"
                            score = "—"
                            steps = "—"
                            if cfg_file.exists():
                                try:
                                    c = yaml.safe_load(cfg_file.read_text())
                                    steps = str(c.get("train", {}).get("num_steps", "—"))
                                except Exception:
                                    pass
                            # Try to find best score from logs
                            for log_f in run_dir.glob("**/*.jsonl"):
                                try:
                                    with open(log_f) as f:
                                        for line in f:
                                            d = json.loads(line)
                                            if "score" in d:
                                                score = f"{d['score']:.4f}"
                                except Exception:
                                    pass
                            rows.append([
                                run_dir.name,
                                bench_dir.name,
                                score,
                                steps,
                            ])
                    return rows

                scan_btn.click(scan_outputs, output_dir, results_table)

    return app


def main():
    parser = argparse.ArgumentParser(description="SkillOpt WebUI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host. Use 0.0.0.0 for public access.")
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )


if __name__ == "__main__":
    main()
