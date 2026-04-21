#!/usr/bin/env python3
"""
Heterogeneous multi-agent debate evaluation.

Runs a 2-agent debate between two different model families:
  Agent_A (Agent_Qwen):  Qwen/Qwen2.5-14B-Instruct       — GPU 0  (~28 GB bf16)
  Agent_B (Agent_Llama): meta-llama/Llama-3.1-8B-Instruct — GPU 1  (~16 GB bf16)

GPU pinning is handled by DebateArena via the 'visible_devices' key in
model_kwargs, which sets CUDA_VISIBLE_DEVICES before each LLM() call.

SLURM submission (2 GPUs, 128 GB RAM):
    sbatch evaluation/submit_eval_hetero.sl

Or interactive:
    srun --mem=128G --time=03:00:00 -p a100-gpu --gres=gpu:2 --qos=gpu_access --pty /bin/bash
    apptainer shell --nv --writable-tmpfs tom_mas.sif
    export PYTHONPATH=/nas/longleaf/home/anvu/ToM:$PYTHONPATH
    python -m evaluation.run_eval_hetero

Outputs (separate from run_eval.py to avoid collision):
  evaluation/metrics_log_hetero.csv
  evaluation/tom_memories_hetero_step_N.json
"""

import os
import json
import csv
import logging
from typing import Dict

from architectures.debate import DebateArena
from agents.debater import DebaterAgent
from memory.tom import ToMMemory
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
    """Truncate the SLURM .err file in-place to reclaim disk space.

    SLURM holds the file descriptor open, so we truncate rather than delete.
    This is safe because the file descriptor remains valid; SLURM will
    continue writing at the new (shorter) position.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME", "eval_deepseek_gemma")
    if not job_id:
        return  # not running under SLURM
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
    csv_file = f"evaluation/metrics_log_hetero_deepseek7B_gemma9B_{job_id}.csv"
    memory_log_file = f"evaluation/tom_memories_log_hetero_deepseek7B_gemma9B_{job_id}.json"

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    logger.info("Loading MMLU Pro dataset...")
    questions = load_mmlu_pro(split="test", num_samples=21)   # TODO: increase to 100 for full run
    logger.info("Loaded %d questions.", len(questions))

    # -----------------------------------------------------------------------
    # Models
    #
    # GPU memory breakdown (2 x A100-40GB):
    #   GPU 0: Qwen-14B bf16   ~28 GB weights + ~3 GB KV cache = ~31 GB
    #   GPU 1: Llama-3.1-8B bf16  ~16 GB weights + ~3 GB KV cache = ~19 GB
    #
    # 'visible_devices' in model_kwargs is consumed by DebateArena.__init__
    # to set CUDA_VISIBLE_DEVICES before each LLM() initialisation, pinning
    # each model to the correct physical GPU. The key is stripped before
    # being forwarded to vLLM's LLM() constructor.
    # -----------------------------------------------------------------------
    deepseek_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    gemma_model = "google/gemma-2-9b-it"

    logger.info("Initializing heterogeneous DebateArena:")
    logger.info("  Agent_Deepseek: %s  ->  GPU 0", deepseek_model)
    logger.info("  Agent_Gemma:    %s  ->  GPU 1", gemma_model)

    vllm_kwargs = {
        deepseek_model: {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "0",
        },
        gemma_model: {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "visible_devices": "1",
        },
    }

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------
    sampling_params = SamplingParams(temperature=0.3, max_tokens=2048)

    arena = DebateArena(
        agent_classes=[DebaterAgent, DebaterAgent],
        memory_classes=[ToMMemory, ToMMemory],
        model_names=[deepseek_model, gemma_model],
        num_rounds=5,
        agent_names=["Agent_Deepseek", "Agent_Gemma"],
        model_kwargs=vllm_kwargs,
        sampling_params=[sampling_params, sampling_params],
    )

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    evaluator = Evaluator()
    num_rounds = arena.num_rounds
    agent_names = ["Agent_Deepseek", "Agent_Gemma"]

    csv_headers = [
        "question_idx", "question_id", "category", 
        "correct_answer", "final_answer", "is_correct"
    ]
    for r in range(num_rounds):
        for agent in agent_names:
            csv_headers.append(f"round_{r}_{agent}")
            
    for agent in agent_names:
        csv_headers.extend([f"{agent}_susceptibility", f"{agent}_correction"])
    
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
            result.get("final_answer", "") == result.get("correct_answer", "") and result.get("final_answer", "") != "unresolved"
        ]
        
        agent_ids = ["agent_0", "agent_1"]
        for r in range(num_rounds):
            if r < len(rounds_data):
                for agent_id in agent_ids:
                    row.append(rounds_data[r].get(agent_id, "?"))
            else:
                for _ in agent_names:
                    row.append("")
        
        # Add flip metrics
        agent_suscep = summary.get("agent_susceptibility_rate", {})
        agent_correct = summary.get("agent_correction_rate", {})
        for agent_id in agent_ids:
            row.extend([
                agent_suscep.get(agent_id, 0.0),
                agent_correct.get(agent_id, 0.0)
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

        # Truncate SLURM .err log every 3 questions to reclaim disk space
        if idx % 3 == 0:
            truncate_slurm_err_log()

        # 4. Snapshot ToM memory every 10 questions into a single accumulating file
        if idx % 10 == 0:
            step_key = f"step_{idx}"
            memories: Dict[str, dict] = {}
            for agent in arena.agents:
                if agent.memory is not None:
                    memories[agent.name] = agent.memory.beliefs

            # Load existing log, append new step, write back
            if os.path.exists(memory_log_file):
                with open(memory_log_file, "r") as f:
                    memory_log = json.load(f)
            else:
                memory_log = {}
            memory_log[step_key] = memories
            with open(memory_log_file, "w") as f:
                json.dump(memory_log, f, indent=4)
            logger.info("Updated ToM memory log at %s (step %d)", memory_log_file, idx)

    logger.info("Evaluation complete! Final summary:")
    final_summary = evaluator.summary()
    logger.info(json.dumps(final_summary, indent=2))
    
    summary_file = f"evaluation/final_summary_hetero_deepseek7B_gemma9B_{job_id}.json"
    with open(summary_file, "w") as f:
        json.dump(final_summary, f, indent=4)
    logger.info("Saved complete final summary to %s", summary_file)


if __name__ == "__main__":
    main()
