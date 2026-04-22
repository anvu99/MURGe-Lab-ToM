#!/usr/bin/env python3
"""
Heterogeneous 2-agent Think-Then-Speak debate evaluation with an Observer.

Runs a 2-agent debate using ThinkThenSpeakDebater:
  Agent_Qwen:  Qwen/Qwen2.5-7B-Instruct         — GPU 0  (~14 GB bf16)
  Agent_Llama: meta-llama/Llama-3.1-8B-Instruct  — GPU 1  (~16 GB bf16)
  Observer:    Qwen/Qwen2.5-14B-Instruct        — GPU 2  (~28 GB bf16)

Each agent reasons privately (full chain-of-thought), then distills its reasoning
into a ≤200-word public message that peers see in subsequent rounds.
The Observer monitors the private reasoning before consensus to correct sycophancy or repetition.

NOTE: Requires 3 GPUs.
"""

import os
import json
import csv
import logging

from architectures.debate import DebateArena
from agents.tts_debater import ThinkThenSpeakDebater
from agents.observer import ObserverAgent
from data.loader import load_mmlu_pro
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


def truncate_slurm_err_log():
    """Truncate the SLURM .err file in-place to reclaim disk space."""
    job_id = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME", "tts_qwen_llama_obs")
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
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    base_name = f"tts_qwen7B_llama8B_obs14B_{job_id}_no_memories"
    csv_file = f"evaluation/metrics_log_{base_name}.csv"

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    logger.info("Loading MMLU Pro dataset...")
    questions = load_mmlu_pro(split="test", num_samples=400)
    logger.info("Loaded %d questions.", len(questions))

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    qwen_model  = "Qwen/Qwen2.5-7B-Instruct"
    llama_model = "meta-llama/Llama-3.1-8B-Instruct"
    obs_model   = "Qwen/Qwen2.5-14B-Instruct"

    agent_names = ["Agent_Qwen", "Agent_Llama"]
    agent_ids   = ["agent_0", "agent_1"]

    logger.info("Initializing TTS DebateArena with Observer:")
    logger.info("  Agent_Qwen:  %s  ->  GPU 0", qwen_model)
    logger.info("  Agent_Llama: %s  ->  GPU 1", llama_model)
    logger.info("  Observer:    %s  ->  GPU 2", obs_model)

    vllm_kwargs = {
        qwen_model: {
            "gpu_memory_utilization": 0.85,   # 0.85 × 48 GB (L40) = 40.8 GB; Qwen-7B bf16 ~14 GB → ~27 GB for KV cache
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "0",
        },
        llama_model: {
            "gpu_memory_utilization": 0.85,   # 0.85 × 48 GB (L40) = 40.8 GB; Llama-8B bf16 ~16 GB → ~25 GB for KV cache
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "1",
        },
    }

    # Initialize Observer LLM
    logger.info("Loading Observer LLM '%s' on GPU(s) 2 ...", obs_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    observer_llm = LLM(
        model=obs_model,
        gpu_memory_utilization=0.90,   # 0.90 × 48 GB (L40) = 43.2 GB; 14B bf16 weights ~28 GB → ~15 GB free for KV cache
        max_model_len=4096,            # halved to reduce KV cache footprint; observer only needs short context
        tensor_parallel_size=1
    )
    # Clear visible devices for the other models to let DebateArena handle setting it
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    observer_agent = ObserverAgent(
        llm=observer_llm,
        sampling_params=SamplingParams(temperature=0.3, max_tokens=256)
    )

    # Stage 1 (private reasoning): generous token budget for deep CoT.
    # Stage 2 (public speak): capped inside ThinkThenSpeakDebater._speak_params.
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
    # Evaluation loop
    # -----------------------------------------------------------------------
    evaluator = Evaluator()
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
        "total_sycophancy_flags", "total_repetition_flags"
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
            result.get("total_sycophancy_flags", 0),
            result.get("total_repetition_flags", 0),
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
