#!/usr/bin/env python3
"""
Targeted Heterogeneous 2-agent Think-Then-Speak debate evaluation with an Observer.

Identical to run_eval_tts_qwen_llama_observer.py EXCEPT the evaluation set is
restricted to questions that were:
  1. In the DCR pool in the baseline no-memories run (agents initially disagreed
     AND at least one had the correct answer at round 0), AND
  2. Failed to collapse to the correct answer (majority rule: > 50%) by the
     final round of that baseline run.

Baseline CSV:
    evaluation/metrics_log_tts_qwen7B_llama8B_44599920_no_memories.csv

Models:
  Agent_Qwen:  Qwen/Qwen2.5-7B-Instruct         — GPU 0
  Agent_Llama: meta-llama/Llama-3.1-8B-Instruct  — GPU 1
  Observer:    Qwen/Qwen2.5-14B-Instruct          — GPU 2

NOTE: Requires 3 GPUs.
"""

import os
import json
import csv
import logging


from architectures.debate import DebateArena
from agents.tts_debater import ThinkThenSpeakDebater
from agents.observer import ObserverAgent
from data.loader import get_failed_dcr_ids, load_mmlu_pro_by_ids
from data.metrics import Evaluator
from vllm import SamplingParams, LLM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline CSV to derive the targeted question set from
# ---------------------------------------------------------------------------
BASELINE_CSV = "evaluation/metrics_log_tts_qwen7B_llama8B_44599920_no_memories.csv"
BASELINE_AGENT_NAMES = ["Agent_Qwen", "Agent_Llama"]
BASELINE_NUM_ROUNDS  = 5


def truncate_slurm_err_log():
    """Truncate the SLURM .err file in-place to reclaim disk space."""
    job_id   = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME", "tts_qwen_llama_obs_targeted")
    if not job_id:
        return
    err_file = f"evaluation/slurm_{job_name}_{job_id}.err"
    try:
        if os.path.exists(err_file):
            size_mb = os.path.getsize(err_file) / (1024 * 1024)
            with open(err_file, "w") as f:
                f.truncate(0)
            logger.info("Truncated SLURM .err log (%s, was %.1f MB)", err_file, size_mb)
    except OSError as e:
        logger.warning("Could not truncate .err log: %s", e)


def main():
    os.makedirs("evaluation", exist_ok=True)
    job_id    = os.environ.get("SLURM_JOB_ID", "local")
    base_name = f"tts_qwen7B_llama8B_obs14B_{job_id}_targeted_dcr_failures"
    csv_file  = f"evaluation/metrics_log_{base_name}.csv"

    # -----------------------------------------------------------------------
    # Data — load only DCR-eligible questions that failed the baseline run
    # -----------------------------------------------------------------------
    logger.info("Parsing baseline CSV to find DCR-eligible failures: %s", BASELINE_CSV)
    failed_ids = get_failed_dcr_ids(
        csv_path    = BASELINE_CSV,
        agent_names = BASELINE_AGENT_NAMES,
        num_rounds  = BASELINE_NUM_ROUNDS,
    )
    logger.info("Found %d question IDs that failed to collapse in the baseline run.", len(failed_ids))

    logger.info("Loading targeted questions from MMLU Pro...")
    questions = load_mmlu_pro_by_ids(failed_ids)
    logger.info("Loaded %d questions for targeted evaluation.", len(questions))

    if not questions:
        logger.error("No questions matched the failed DCR IDs. Check the baseline CSV path.")
        return

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    qwen_model  = "Qwen/Qwen2.5-7B-Instruct"
    llama_model = "meta-llama/Llama-3.1-8B-Instruct"
    obs_model   = "Qwen/Qwen2.5-14B-Instruct"

    agent_names = ["Agent_Qwen", "Agent_Llama"]
    agent_ids   = ["agent_0", "agent_1"]

    logger.info("Initializing TTS DebateArena with Observer (targeted run, 3-GPU mode):")
    logger.info("  Agent_Qwen:  %s  ->  GPU 0", qwen_model)
    logger.info("  Agent_Llama: %s  ->  GPU 1", llama_model)
    logger.info("  Observer:    %s  ->  GPU 2", obs_model)

    vllm_kwargs = {
        qwen_model: {
            "gpu_memory_utilization": 0.85,   # 0.85 * 48 GB = ~40.8 GB — 14 GB weights -> ~27 GB KV cache
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "0",
        },
        llama_model: {
            "gpu_memory_utilization": 0.85,   # 0.85 * 48 GB = ~40.8 GB — 16 GB weights -> ~25 GB KV cache
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "1",
        },
    }

    # Initialize Observer LLM
    logger.info("Loading Observer LLM '%s' on GPU 2 ...", obs_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    observer_llm = LLM(
        model=obs_model,
        gpu_memory_utilization=0.90,   # 0.90 × 48 GB (L40) = 43.2 GB; 14B bf16 ~28 GB → ~15 GB for KV cache
        max_model_len=4096,            # halved to reduce KV cache footprint; observer only needs short context
        tensor_parallel_size=1
    )
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    observer_agent = ObserverAgent(
        llm=observer_llm,
        sampling_params=SamplingParams(temperature=0.3, max_tokens=256)
    )

    sampling_params = SamplingParams(temperature=0.3, max_tokens=2048)

    arena = DebateArena(
        agent_classes=[ThinkThenSpeakDebater, ThinkThenSpeakDebater],
        memory_classes=[None, None],
        model_names=[qwen_model, llama_model],
        num_rounds=5,
        agent_names=agent_names,
        model_kwargs=vllm_kwargs,
        sampling_params=[sampling_params, sampling_params],
        observer=observer_agent,
    )

    # -----------------------------------------------------------------------
    # Evaluation loop  (identical to run_eval_tts_qwen_llama_observer.py)
    # -----------------------------------------------------------------------
    evaluator  = Evaluator()
    num_rounds = arena.num_rounds

    csv_headers = [
        "question_idx", "question_id", "category",
        "correct_answer", "final_answer", "is_correct"
    ]
    for r in range(num_rounds):
        for name in agent_names:
            csv_headers.append(f"round_{r}_{name}")

    for name in agent_names:
        csv_headers.extend([f"{name}_susceptibility", f"{name}_correction"])

    csv_headers.extend([
        "system_accuracy", "resolved_accuracy",
        "dcr", "nar", "dcr_nar_pool_size",
        "total_sycophancy_flags", "total_repetition_flags", "total_inconsistency_flags"
    ])
    for name in agent_names:
        csv_headers.append(f"{name}_observer_flags")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    for idx, question_data in enumerate(questions, start=1):
        logger.info("--- Processing Question %d / %d ---", idx, len(questions))

        result = arena.run(question_data)
        evaluator.add(result)
        summary = evaluator.summary()

        rounds_data = result.get("rounds", [])
        row = [
            idx,
            result.get("question_id", "unknown"),
            result.get("category", "unknown"),
            result.get("correct_answer", ""),
            result.get("final_answer", ""),
            result.get("final_answer", "") == result.get("correct_answer", "")
            and result.get("final_answer", "") != "unresolved"
        ]

        for r in range(num_rounds):
            if r < len(rounds_data):
                for agent_id in agent_ids:
                    row.append(rounds_data[r].get(agent_id, "?"))
            else:
                for _ in agent_names:
                    row.append("")

        agent_suscep      = summary.get("agent_susceptibility_rate", {})
        agent_correct_rate = summary.get("agent_correction_rate", {})
        for agent_id in agent_ids:
            row.extend([
                agent_suscep.get(agent_id, 0.0),
                agent_correct_rate.get(agent_id, 0.0)
            ])

        row.extend([
            summary["system_accuracy"],
            summary["resolved_accuracy"],
            summary["dcr"],
            summary["nar"],
            summary["dcr_nar_pool_size"],
            result.get("total_sycophancy_flags", 0),
            result.get("total_repetition_flags", 0),
            result.get("total_inconsistency_flags", 0),
        ])

        obs_flags = result.get("observer_flags", [])
        for agent_id in agent_ids:
            count = sum(1 for f in obs_flags if f.get("agent_id") == agent_id)
            row.append(count)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logger.info(
            "Progress Q%d: Acc=%.2f, Resolved=%.2f, DCR=%.2f, NAR=%.2f (pool=%d)",
            idx,
            summary["system_accuracy"],
            summary["resolved_accuracy"],
            summary["dcr"],
            summary["nar"],
            summary["dcr_nar_pool_size"],
        )

        if idx % 3 == 0:
            truncate_slurm_err_log()

    logger.info("Evaluation complete! Final summary:")
    final_summary = evaluator.summary()
    logger.info(json.dumps(final_summary, indent=2))

    summary_file = f"evaluation/final_summary_{base_name}.json"
    with open(summary_file, "w") as f:
        json.dump(final_summary, f, indent=4)
    logger.info("Saved complete final summary to %s", summary_file)


if __name__ == "__main__":
    main()
