"""Ceramic Capacitors environment adapter located in train/."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from train.dataloader import CeramicCapacitorsDataLoader
from skillopt.gradient.reflect import run_minibatch_reflect
from skillopt.model import chat_target

class CeramicCapacitorsAdapter(EnvAdapter):
    """Ceramic Capacitors parameter extraction adapter."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 120,
        workers: int = 16,
        analyst_workers: int = 8,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 3,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 4096,
    ) -> None:
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = CeramicCapacitorsDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def evaluate_response(self, response_text: str, ground_truth: dict) -> tuple[float, float, str, dict]:
        """Parse structured answer in <answer> tags and score against ground_truth."""
        try:
            start_tag = "<answer>"
            end_tag = "</answer>"
            if start_tag in response_text and end_tag in response_text:
                json_str = response_text.split(start_tag)[1].split(end_tag)[0].strip()
            else:
                json_str = response_text.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif json_str.startswith("```"):
                    json_str = json_str.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(json_str)
        except Exception as e:
            return 0.0, 0.0, f"JSON parse error: {e}", {}

        import re
        def normalize(val) -> str:
            if val is None:
                return ""
            s = str(val).strip().lower()
            s = s.replace("μ", "u").replace("µ", "u")
            s = "".join(s.split())
            if s.isdigit():
                s = str(int(s))
            try:
                f = float(s)
                if f == int(f):
                    return str(int(f))
                return f"{f:.4f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
            match = re.match(r"^([0-9.]+)([a-z°%]+)$", s)
            if match:
                num_part, unit_part = match.groups()
                try:
                    f = float(num_part)
                    if f == int(f):
                        num_str = str(int(f))
                    else:
                        num_str = f"{f:.4f}".rstrip("0").rstrip(".")
                    return num_str + unit_part
                except ValueError:
                    pass
            return s

        correct = 0
        total = 0
        details = {}
        for k, gold_val in ground_truth.items():
            pred_val = parsed.get(k, "")
            gold_norm = normalize(gold_val)
            pred_norm = normalize(pred_val)
            
            total += 1
            if gold_norm == pred_norm:
                correct += 1
                details[k] = "correct"
            else:
                details[k] = f"expected '{gold_val}', got '{pred_val}'"

        soft_score = correct / total if total > 0 else 0.0
        hard_score = 1.0 if soft_score == 1.0 else 0.0
        
        fail_summary = ""
        if soft_score < 1.0:
            mismatches = [f"{k}: {msg}" for k, msg in details.items() if msg != "correct"]
            fail_summary = "; ".join(mismatches[:3])

        # Strict filter parsed output to ONLY contain the expected gold keys
        if isinstance(parsed, dict):
            filtered_parsed = {}
            for k in ground_truth.keys():
                filtered_parsed[k] = parsed.get(k, "")
            parsed = filtered_parsed

        return hard_score, soft_score, fail_summary, parsed

    def process_one_item(self, item: dict, skill_content: str, out_dir: str) -> dict:
        """Run single-item rollout with target LLM and evaluate it."""
        item_id = str(item["id"])
        description = item["input"]
        ground_truth = item.get("ground_truth", {})

        # Load rollout_system prompt directly from this directory!
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "rollout_system.md")
        with open(prompt_path, encoding="utf-8") as f:
            system = f.read()

        system = system.replace("{skill_section}", f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else "")
        user = f"## Capacitor Description\n{description}"

        result = {
            "id": item_id,
            "description": description,
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": "",
            "gold_answers": [ground_truth],
            "response": "",
            "fail_reason": "",
            "agent_ok": False,
            "n_turns": 1,
        }

        try:
            resp_text, _ = chat_target(
                system=system,
                user=user,
                max_completion_tokens=self.max_completion_tokens,
                retries=3,
                stage="rollout",
                timeout=self.exec_timeout,
            )
            result["response"] = resp_text
            result["agent_ok"] = True
            
            hard, soft, fail_reason, parsed = self.evaluate_response(resp_text, ground_truth)
            result["hard"] = int(hard)
            result["soft"] = soft
            result["predicted_answer"] = json.dumps(parsed, ensure_ascii=False)
            result["fail_reason"] = fail_reason
            
            # Write prediction artifacts for reflection analysis
            pred_dir = os.path.join(out_dir, "predictions", item_id)
            os.makedirs(pred_dir, exist_ok=True)
            with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(system)
            with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(user)
            conversation = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp_text}
            ]
            with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            result["fail_reason"] = f"Runtime error: {e}"

        return result

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        items: list[dict] = env_manager
        results_path = os.path.join(out_dir, "results.jsonl")
        os.makedirs(out_dir, exist_ok=True)

        done_ids: set[str] = set()
        existing: list[dict] = []
        if os.path.exists(results_path):
            with open(results_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done_ids.add(str(r["id"]))
                        existing.append(r)
                    except Exception:
                        pass

        pending = [it for it in items if str(it["id"]) not in done_ids]
        if not pending:
            return existing

        print(f"    [rollout] executing {len(pending)} tasks...", flush=True)
        results = list(existing)

        with open(results_path, "a", encoding="utf-8") as outf:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.process_one_item, it, skill_content, out_dir): it for it in pending}
                for fut in futures:
                    res = fut.result()
                    results.append(res)
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()

        completed = len(results)
        acc = sum(1 for r in results if r.get("hard", 0)) / completed if completed else 0
        soft_avg = sum(r.get("soft", 0.0) for r in results) / completed if completed else 0
        print(f"    [rollout] completed {completed} tasks. Avg Hard Acc: {acc:.3f}, Avg Soft Acc: {soft_avg:.3f}", flush=True)
        return results

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        prediction_dir = kwargs.get("prediction_dir", os.path.join(out_dir, "predictions"))
        patches_dir = kwargs.get("patches_dir", os.path.join(out_dir, "patches"))
        random_seed = kwargs.get("random_seed")
        step_buffer_context = kwargs.get("step_buffer_context", "")
        meta_skill_context = kwargs.get("meta_skill_context", "")

        return run_minibatch_reflect(
            results=results,
            skill_content=skill_content,
            prediction_dir=prediction_dir,
            patches_dir=patches_dir,
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=random_seed,
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def get_task_types(self) -> list[str]:
        return ["ceramic_capacitors"]
