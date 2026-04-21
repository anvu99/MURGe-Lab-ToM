#!/usr/bin/env python3
"""
3-Agent Think-Then-Speak debate evaluation (no persistent memories).

Runs a 3-agent debate using ThinkThenSpeakDebater across three domain-specialized
models:
  Agent_Coder (Math/CS): Qwen/Qwen2.5-Coder-7B-Instruct     — GPU 0
  Agent_Math (Math):     mistralai/Mathstral-7B-v0.1         — GPU 1
  Agent_Bio (Biology):   ContactDoctor/Bio-Medical-Llama-3-8B — GPU 2

Each agent reasons privately (full chain-of-thought), then distills its reasoning
into a ≤200-word public message that peers see in subsequent rounds.

NOTE: Requires 3 GPUs.
"""

import os
import json
import csv
import logging
from typing import Dict

from architectures.debate import DebateArena
from agents.tts_debater import ThinkThenSpeakDebater
from data.loader import load_mmlu_pro
from data.metrics import Evaluator
from vllm import SamplingParams

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def truncate_slurm_err_log():
    """Truncate the SLURM .err file in-place to reclaim disk space."""
    job_id = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME", "eval_3agent_tts")
    if not job_id:
        return
    err_file = f"evaluation/slurm_{job_name}_{job_id}.err"
    try:
        if os.path.exists(err_file):
            size_mb = os.path.getsize(err_file) / (1024 * 1024)
            with open(err_file, "w") as f:
                f.truncate(0)
            logger.info(
                "Truncated SLURM .err log (%s, was %.1f MB)",
                err_file, size_mb,
            )
    except OSError as e:
        logger.warning("Could not truncate .err log: %s", e)


def main():
    os.makedirs("evaluation", exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    base_name = f"3agent_tts_{job_id}_no_memories"
    csv_file = f"evaluation/metrics_log_{base_name}.csv"

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    categories = ["math", "computer science", "biology"]
    logger.info("Loading MMLU Pro dataset for categories: %s...", categories)
    questions = load_mmlu_pro(split="test", num_samples=300, categories=categories)
    logger.info("Loaded %d questions.", len(questions))

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    coder_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    math_model  = "mistralai/Mathstral-7B-v0.1"
    bio_model   = "ContactDoctor/Bio-Medical-Llama-3-8B"

    agent_names = ["Agent_Coder", "Agent_Math", "Agent_Bio"]
    agent_ids   = ["agent_0", "agent_1", "agent_2"]

    logger.info("Initializing 3-agent TTS DebateArena:")
    logger.info("  Agent_Coder: %s  ->  GPU 0", coder_model)
    logger.info("  Agent_Math:  %s  ->  GPU 1", math_model)
    logger.info("  Agent_Bio:   %s  ->  GPU 2", bio_model)

    vllm_kwargs = {
        coder_model: {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "0",
        },
        math_model: {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "1",
        },
        bio_model: {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "2",
        },
    }

    # Stage 1 (private reasoning): generous token budget for deep CoT.
    # Stage 2 (public speak): capped inside ThinkThenSpeakDebater._speak_params.
    sampling_params = SamplingParams(temperature=0.3, max_tokens=2048)

    arena = DebateArena(
        agent_classes=[ThinkThenSpeakDebater, ThinkThenSpeakDebater, ThinkThenSpeakDebater],
        memory_classes=[None, None, None],
        model_names=[coder_model, math_model, bio_model],
        num_rounds=5,
        agent_names=agent_names,
        model_kwargs=vllm_kwargs,
        sampling_params=[sampling_params, sampling_params, sampling_params],
    )

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    evaluator = Evaluator()
    num_rounds = arena.num_rounds

    # Build CSV headers dynamically based on number of agents and rounds
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
        "dcr", "nar", "dcr_nar_pool_size"
    ])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    for idx, question_data in enumerate(questions, start=1):
        logger.info("--- Processing Question %d / %d ---", idx, len(questions))

        # 1. Run debate
        result = arena.run(question_data)

        # 2. Accumulate metrics
        evaluator.add(result)
        summary = evaluator.summary()

        # 3. Append per-question data to CSV
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

        agent_suscep = summary.get("agent_susceptibility_rate", {})
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
        ])

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
