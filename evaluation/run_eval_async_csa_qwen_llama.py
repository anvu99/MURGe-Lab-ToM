#!/usr/bin/env python3
"""
Heterogeneous 2-agent CSA debate evaluation.

Uses CSADebater (Communication Strategy Aware) with two memory types:
  - ReasoningMemory: how to better consume and integrate peer arguments
  - CommunicationStrategyMemory: how to package arguments for peer engagement

Agents:
  Agent_Qwen:  Qwen/Qwen2.5-7B-Instruct         — GPU 0  (~14 GB bf16)
  Agent_Llama: meta-llama/Llama-3.1-8B-Instruct  — GPU 1  (~16 GB bf16)

NOTE: Requires 2 GPUs.
"""

import os
import json
import csv
import logging
from typing import Dict

from architectures.per_turn_async_debate import PerTurnAsyncDebateArena
from agents.csa_debater import CSADebater
from memory.reasoning import ReasoningMemory
from memory.communication import CommunicationStrategyMemory
from data.loader import load_mmlu_pro
from data.metrics import Evaluator
from vllm import LLM, SamplingParams

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
    job_name = os.environ.get("SLURM_JOB_NAME", "async_csa_qwen_llama")
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
    base_name = f"async_csa_qwen7B_llama8B_{job_id}"
    csv_file = f"evaluation/metrics_log_{base_name}.csv"
    memory_log_file = f"evaluation/csa_memories_log_{base_name}.json"

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    logger.info("Loading MMLU Pro dataset...")
    questions = load_mmlu_pro(split="test", num_samples=400)
    logger.info("Loaded %d questions.", len(questions))

    # -----------------------------------------------------------------------
    # Models — load LLMs manually so we can pass them to memory constructors
    # -----------------------------------------------------------------------
    qwen_model  = "Qwen/Qwen2.5-7B-Instruct"
    llama_model = "meta-llama/Llama-3.1-8B-Instruct"

    agent_names = ["Agent_Qwen", "Agent_Llama"]
    agent_ids   = ["agent_0", "agent_1"]

    logger.info("Loading LLMs...")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    qwen_llm = LLM(
        model=qwen_model,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        tensor_parallel_size=1,
    )
    logger.info("Loaded Qwen on GPU 0.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llama_llm = LLM(
        model=llama_model,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        tensor_parallel_size=1,
    )
    logger.info("Loaded Llama on GPU 1.")

    # CSA Stage 1 needs max_tokens=3500; Stage 2 capped inside CSADebater._speak_params
    # Note: CSADebater constructor enforces _CSA_THINK_MAX_TOKENS (3500) minimum
    think_params = SamplingParams(temperature=0.0, max_tokens=3500)

    # -----------------------------------------------------------------------
    # Memory instances — one reasoning + one comm_strategy per agent
    # -----------------------------------------------------------------------
    qwen_reasoning_mem  = ReasoningMemory(llm=qwen_llm,  owner_name="Agent_Qwen")
    llama_reasoning_mem = ReasoningMemory(llm=llama_llm, owner_name="Agent_Llama")

    qwen_comm_mem  = CommunicationStrategyMemory(
        llm=qwen_llm,  owner_name="Agent_Qwen",  peer_name="Agent_Llama"
    )
    llama_comm_mem = CommunicationStrategyMemory(
        llm=llama_llm, owner_name="Agent_Llama", peer_name="Agent_Qwen"
    )

    # -----------------------------------------------------------------------
    # Agents — built directly (bypass arena constructor for custom memory wiring)
    # -----------------------------------------------------------------------
    agent_qwen = CSADebater(
        agent_id="agent_0",
        name="Agent_Qwen",
        model_name=qwen_model,
        llm=qwen_llm,
        reasoning_memory=qwen_reasoning_mem,
        comm_memory=qwen_comm_mem,
        sampling_params=think_params,
    )
    agent_llama = CSADebater(
        agent_id="agent_1",
        name="Agent_Llama",
        model_name=llama_model,
        llm=llama_llm,
        reasoning_memory=llama_reasoning_mem,
        comm_memory=llama_comm_mem,
        sampling_params=think_params,
    )

    # -----------------------------------------------------------------------
    # Arena — pass pre-built agents directly
    # AsyncDebateArena accepts an `agents` kwarg (or falls back to agent_classes)
    # We use the agent_classes path with empty memory_classes since agents are
    # already constructed. We override by passing agents directly.
    # -----------------------------------------------------------------------
    # Build a minimal arena using the agents we built.
    # AsyncDebateArena is a DebateArena subclass. We bypass its constructor's
    # agent-building logic by using a thin wrapper approach: instantiate with
    # dummy args, then replace self.agents. This is safe because the arena only
    # uses self.agents during _run_round and self.num_rounds.
    arena = PerTurnAsyncDebateArena.__new__(PerTurnAsyncDebateArena)
    arena.num_rounds = 5
    arena.n_agents = 2
    arena.agents = [agent_qwen, agent_llama]
    arena.observer = None
    arena._llm_cache = {qwen_model: qwen_llm, llama_model: llama_llm}
    arena._sampling_params = [think_params, think_params]

    logger.info("CSA DebateArena ready:")
    logger.info("  Agent_Qwen:  %s  ->  GPU 0", qwen_model)
    logger.info("  Agent_Llama: %s  ->  GPU 1", llama_model)

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
        "total_inconsistency_flags",
        "converged_early", "turns_taken",
    ])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    for idx, question_data in enumerate(questions, start=1):
        logger.info("--- Processing Question %d / %d ---", idx, len(questions))

        # arena.run() calls _update_memories which calls agent.update_memory()
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
            (
                result.get("final_answer", "") == result.get("correct_answer", "")
                and result.get("final_answer", "") != "unresolved"
            ),
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
                agent_correct_rate.get(agent_id, 0.0),
            ])

        row.extend([
            summary["system_accuracy"],
            summary["resolved_accuracy"],
            summary["dcr"],
            summary["nar"],
            summary["dcr_nar_pool_size"],
            result.get("total_inconsistency_flags", 0),
            result.get("converged_early", False),
            result.get("turns_taken", ""),
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

        # Snapshot CSA memories every 10 questions
        if idx % 10 == 0:
            step_key = f"step_{idx}"
            memories: Dict[str, dict] = {
                "Agent_Qwen": {
                    "reasoning_directives": qwen_reasoning_mem.directives,
                    "comm_strategy_for_Llama": qwen_comm_mem.strategy,
                },
                "Agent_Llama": {
                    "reasoning_directives": llama_reasoning_mem.directives,
                    "comm_strategy_for_Qwen": llama_comm_mem.strategy,
                },
            }

            if os.path.exists(memory_log_file):
                with open(memory_log_file, "r") as f:
                    memory_log = json.load(f)
            else:
                memory_log = {}
            memory_log[step_key] = memories
            with open(memory_log_file, "w") as f:
                json.dump(memory_log, f, indent=4)
            logger.info("Updated CSA memory log at %s (step %d)", memory_log_file, idx)

    logger.info("Evaluation complete! Final summary:")
    final_summary = evaluator.summary()
    logger.info(json.dumps(final_summary, indent=2))

    summary_file = f"evaluation/final_summary_{base_name}.json"
    with open(summary_file, "w") as f:
        json.dump(final_summary, f, indent=4)
    logger.info("Saved complete final summary to %s", summary_file)


if __name__ == "__main__":
    main()
