"""Generic CSV environment adapter for ReflACT."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.generic_csv.dataloader import GenericCSVDataLoader
from skillopt.gradient.reflect import run_minibatch_reflect
from skillopt.model import chat_target


def parse_engineering_value(val_str: str) -> tuple[float, str] | None:
    """
    Parses a string like '2.2 kOhm', '100 mA', '10uF' into (base_value, canonical_unit).
    Returns None if it doesn't match a numeric value followed by an engineering unit.
    """
    # Clean up string
    s = val_str.strip()
    # Normalize symbols case-insensitively
    s = s.replace('Ω', 'ohm').replace('ω', 'ohm').replace('μ', 'u')
    s = s.lower()
    s = s.replace('ohms', 'ohm')
    
    # Check for Excel time formatting corruption (e.g. "1:00 AM" -> "1 A", "3:00 AM" -> "3 A")
    time_match = re.match(r'^(\d+):(\d+)(?::(\d+))?\s*(am|pm)?$', s)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        am_pm = time_match.group(4)
        val = float(hour) + float(minute) / 60.0
        if am_pm == 'pm' and hour != 12:
            val += 12.0
        return val, "a"
    
    # Handle resistance/capacitance where 'k', 'u', etc. are inside the number, e.g. '2k2' -> '2.2 k'
    match_internal = re.match(r'^(\d+)([pnuμmkM])(\d+)?\s*(ohm|Ω|ohms|f|v|a|w|hz)?$', s)
    if match_internal:
        whole = match_internal.group(1)
        prefix = match_internal.group(2)
        fraction = match_internal.group(3) or '0'
        unit = match_internal.group(4) or ''
        num_val = float(f"{whole}.{fraction}")
        s = f"{num_val} {prefix}{unit}"

    # Match standard format: (number) (prefix + unit)
    num_re = r'^([+-]?\d*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z%°c\u2126\u03a9]*)$'
    match = re.match(num_re, s)
    if not match:
        return None
        
    num_part = match.group(1)
    unit_part = match.group(2).strip()
    
    if not num_part:
        return None
        
    try:
        val = float(num_part)
    except ValueError:
        return None
        
    # Suffix multiplier scaling
    multiplier = 1.0
    base_unit = unit_part
    
    if not unit_part:
        return val, ""
        
    if unit_part.startswith('meg'):
        multiplier = 1e6
        base_unit = unit_part[3:]
    elif unit_part.startswith('k') or unit_part.startswith('ko'):
        multiplier = 1e3
        base_unit = unit_part[1:]
    elif unit_part.startswith('m') and not unit_part.startswith('micro') and not unit_part.startswith('mega'):
        if 'M' in val_str or 'meg' in s:
            multiplier = 1e6
        else:
            multiplier = 1e-3
        base_unit = unit_part[1:]
    elif unit_part.startswith('u') or unit_part.startswith('micro'):
        multiplier = 1e-6
        if unit_part.startswith('micro'):
            base_unit = unit_part[5:]
        else:
            base_unit = unit_part[1:]
    elif unit_part.startswith('n') or unit_part.startswith('nano'):
        multiplier = 1e-9
        if unit_part.startswith('nano'):
            base_unit = unit_part[4:]
        else:
            base_unit = unit_part[1:]
    elif unit_part.startswith('p') or unit_part.startswith('pico'):
        multiplier = 1e-12
        if unit_part.startswith('pico'):
            base_unit = unit_part[4:]
        else:
            base_unit = unit_part[1:]
            
    base_unit = base_unit.strip()
    if base_unit in ['volt', 'volts', 'v']:
        base_unit = 'v'
    elif base_unit in ['amp', 'amps', 'ampere', 'amperes', 'a']:
        base_unit = 'a'
    elif base_unit in ['farad', 'farads', 'f']:
        base_unit = 'f'
    elif base_unit in ['watt', 'watts', 'w']:
        base_unit = 'w'
    elif base_unit in ['hz', 'hertz']:
        base_unit = 'hz'
    elif base_unit in ['ohm', 'ohms', 'o']:
        base_unit = 'ohm'
    elif base_unit in ['c', '°c', 'degc', 'celsius']:
        base_unit = 'c'
        
    return val * multiplier, base_unit


def values_are_equal(val1: Any, val2: Any) -> bool:
    if isinstance(val1, (dict, list)) or isinstance(val2, (dict, list)):
        try:
            return json.dumps(val1, sort_keys=True) == json.dumps(val2, sort_keys=True)
        except Exception:
            return False
            
    def is_bool_like(v) -> bool:
        if isinstance(v, bool):
            return True
        s = str(v).lower().strip()
        return s in ('true', 'false', 'yes', 'no', '1', '0')

    def to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).lower().strip()
        return s in ('true', '1', 'yes')

    if isinstance(val1, bool) or isinstance(val2, bool) or (is_bool_like(val1) and is_bool_like(val2)):
        return to_bool(val1) == to_bool(val2)

    parsed1 = parse_engineering_value(str(val1))
    parsed2 = parse_engineering_value(str(val2))
    
    if parsed1 is not None and parsed2 is not None:
        num1, unit1 = parsed1
        num2, unit2 = parsed2
        if unit1 == unit2 or not unit1 or not unit2:
            if num1 == 0 and num2 == 0:
                return True
            if num1 == 0 or num2 == 0:
                return False
            return abs(num1 - num2) / max(abs(num1), abs(num2)) < 0.01
            
    def to_clean_str(v) -> str:
        if v is None:
            return ""
        s = str(v).strip().lower()
        if s in ("none", "null", "nan", "undefined", '""', "''", "[]", "{}"):
            return ""
        return s.replace(' ', '').replace('μ', 'u').replace('Ω', 'ohm').replace('ohms', 'ohm')

    s1 = to_clean_str(val1)
    s2 = to_clean_str(val2)
    return s1 == s2


def find_matching_key(gold_key: str, parsed_dict: dict) -> str | None:
    if not isinstance(parsed_dict, dict):
        return None
    if gold_key in parsed_dict:
        return gold_key
        
    gold_lower = gold_key.lower()
    for k in parsed_dict:
        if k.lower() == gold_lower:
            return k
            
    def clean_key(s):
        return s.lower().replace('_', '').replace(' ', '').replace('-', '')
        
    gold_clean = clean_key(gold_key)
    for k in parsed_dict:
        if clean_key(k) == gold_clean:
            return k
            
    if len(gold_clean) > 2:
        for k in parsed_dict:
            k_clean = clean_key(k)
            if len(k_clean) > 2:
                if gold_clean in k_clean or k_clean in gold_clean:
                    return k
            
    return None



class GenericCSVAdapter(EnvAdapter):
    """Generic CSV environment adapter."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "3:1:1",
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
        self.dataloader = GenericCSVDataLoader(
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

    def evaluate_response(self, response_text: str, ground_truth: Any) -> tuple[float, float, str, dict]:
        """Robust parser to evaluate structured dictionary or direct string predictions."""
        parsed = {}
        parsed_ok = False
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
            parsed_ok = True
        except Exception:
            pass

        # Treat non-dictionary ground truths as {"answer": str(ground_truth)}
        if not isinstance(ground_truth, dict):
            gold_dict = {"answer": str(ground_truth)}
        else:
            gold_dict = ground_truth

        correct = 0
        total = 0
        details = {}

        for k, gold_val in gold_dict.items():
            total += 1
            
            matched_key = None
            if parsed_ok and isinstance(parsed, dict):
                matched_key = find_matching_key(k, parsed)
            
            # 1. Match against parsed JSON key
            if matched_key is not None:
                pred_val = parsed[matched_key]
                if values_are_equal(pred_val, gold_val):
                    correct += 1
                    details[k] = "correct"
                    continue
            
            # 2. Match against parsed JSON answer directly if single-key fallback
            if parsed_ok and isinstance(parsed, dict) and len(parsed) == 1:
                single_val = next(iter(parsed.values()))
                if values_are_equal(single_val, gold_val):
                    correct += 1
                    details[k] = "correct"
                    continue

            # 3. Fallback: Robust substring match in the response
            gold_str = str(gold_val).strip()
            gold_str_lower = gold_str.lower()
            if gold_str_lower and gold_str_lower in response_text.strip().lower():
                correct += 1
                details[k] = "correct"
            else:
                # Try engineering normalization comparison for substring fallback
                parsed_gold = parse_engineering_value(gold_str)
                matched_sub = False
                if parsed_gold is not None:
                    words = re.findall(r'[+-]?\d*(?:\.\d+)?(?:[eE][+-]?\d+)?\s*[a-zA-Z%°c\u2126\u03a9]*', response_text)
                    for word in words:
                        if word.strip():
                            if values_are_equal(word, gold_str):
                                matched_sub = True
                                break
                if matched_sub:
                    correct += 1
                    details[k] = "correct"
                else:
                    details[k] = f"expected '{gold_val}'"

        soft_score = correct / total if total > 0 else 0.0
        hard_score = 1.0 if soft_score == 1.0 else 0.0
        
        fail_summary = ""
        if soft_score < 1.0:
            mismatches = [f"{k}: {msg}" for k, msg in details.items() if msg != "correct"]
            fail_summary = "; ".join(mismatches[:3])

        # Strict filter parsed output to ONLY contain the expected gold keys
        if parsed_ok and isinstance(parsed, dict):
            filtered_parsed = {}
            for k in gold_dict.keys():
                matched_key = find_matching_key(k, parsed)
                if matched_key is not None:
                    filtered_parsed[k] = parsed[matched_key]
                else:
                    filtered_parsed[k] = ""
            parsed = filtered_parsed

        return hard_score, soft_score, fail_summary, parsed if parsed_ok else {"response": response_text}

    def process_one_item(self, item: dict, skill_content: str, out_dir: str) -> dict:
        """Run single task rollout with the target model and evaluate."""
        item_id = str(item["id"])
        description = item["input"]
        ground_truth = item.get("ground_truth", {})

        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "rollout_system.md")
        if os.path.exists(prompt_path):
            with open(prompt_path, encoding="utf-8") as f:
                system = f.read()
        else:
            system = "You are a highly capable AI assistant executing a task.\n\n{skill_section}"

        system = system.replace("{skill_section}", f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else "")

        # Inject expected structured output keys dynamically based on target columns
        if isinstance(ground_truth, dict) and ground_truth:
            keys_list = "\n".join(f"- {k}" for k in ground_truth.keys())
            format_instruction = (
                f"\n\n## Expected JSON Keys\n"
                f"Your JSON object output inside the `<answer>...</answer>` tags MUST contain exactly the following keys:\n"
                f"{keys_list}\n"
                f"Do NOT output any other keys, nesting structures, or extra parameters. Include ONLY the keys listed above.\n"
            )
            system = system + format_instruction

        user = f"## Task Input\n{description}"

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
        return ["generic_csv"]
