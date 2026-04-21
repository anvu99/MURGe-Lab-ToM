#!/usr/bin/env python3
"""
Evaluation script to run single-model DebateArena using Llama-3.1-8B-Instruct.
Useful for testing HuggingFace token and Llama functionality.
"""

import os
import json
import csv
import logging
from typing import List, Dict

# Local imports
from architectures.debate import DebateArena
from agents.debater import DebaterAgent
from memory.tom import ToMMemory
from data.loader import load_mmlu_pro
from data.metrics import Evaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Setup paths
    os.makedirs("evaluation", exist_ok=True)
    csv_file = "evaluation/metrics_log_llama.csv"
    memory_log_file = "evaluation/tom_memories_log_llama.json"
    
    logger.info("Loading MMLU Pro dataset...")
    # Load 5 samples for a quick test
    questions = load_mmlu_pro(split="test", num_samples=5)
    logger.info(f"Loaded {len(questions)} questions.")

    # Initialize Arena with Llama
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    logger.info("Initializing DebateArena with single model: %s", model_name)

    vllm_kwargs = {
        model_name: {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
        },
    }

    arena = DebateArena(
        agent_classes=[DebaterAgent, DebaterAgent],
        memory_classes=[ToMMemory, ToMMemory],
        model_names=[model_name, model_name],
        num_rounds=3,
        agent_names=["Agent_A", "Agent_B"],
        model_kwargs=vllm_kwargs,
    )

    # Evaluator keeps rolling track of performance
    evaluator = Evaluator()

    # Pre-write CSV headers
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_idx", "question_id", "system_accuracy", "resolved_accuracy", "dcr", "nar"])

    # Loop over questions
    for idx, question_data in enumerate(questions, start=1):
        logger.info(f"--- Processing Question {idx} / {len(questions)} ---")
        
        # 1. Run debate
        result = arena.run(question_data)
        
        # 2. Accumulate metrics
        evaluator.add(result)
        
        # 3. Log metrics to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx,
                question_data.get("question_id", "unknown"),
                evaluator.system_accuracy(),
                evaluator.resolved_accuracy(),
                evaluator.disagreement_collapse_rate(),
                evaluator.negative_agreement_rate()
            ])
            
        logger.info(f"Progress Summary Q{idx}: Acc={evaluator.system_accuracy():.2f}, DCR={evaluator.disagreement_collapse_rate():.2f}, NAR={evaluator.negative_agreement_rate():.2f}")

        # 4. Snapshot ToM memory every 10 questions into a single accumulating file
        if idx % 10 == 0:
            step_key = f"step_{idx}"
            memories: Dict[str, dict] = {}
            for agent in arena.agents:
                if agent.memory is not None:
                    memories[agent.name] = agent.memory.beliefs

            if os.path.exists(memory_log_file):
                with open(memory_log_file, "r") as f:
                    memory_log = json.load(f)
            else:
                memory_log = {}
            memory_log[step_key] = memories
            with open(memory_log_file, "w") as f:
                json.dump(memory_log, f, indent=4)
            logger.info(f"Updated ToM memory log at {memory_log_file} (step {idx})")

    logger.info("Evaluation complete! Final summary:")
    final_summary = evaluator.summary()
    logger.info(json.dumps(final_summary, indent=2))
    
    summary_file = "evaluation/final_summary_llama.json"
    with open(summary_file, "w") as f:
        json.dump(final_summary, f, indent=4)
    logger.info("Saved complete final summary to %s", summary_file)

if __name__ == "__main__":
    main()
