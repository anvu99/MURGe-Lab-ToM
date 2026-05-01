import csv
import random
import math
from collections import defaultdict
from datasets import load_dataset
from typing import Optional, List, Dict, Set


def load_mmlu_pro(split: str = "test", num_samples: int = 1, category: Optional[str] = None, categories: Optional[List[str]] = None, seed: int = 42) -> List[Dict]:
    """
    Load questions and answers from MMLU Pro dataset
    
    The dataset contains multiple-choice questions on various professional and academic topics.
    Each question has a list of possible answers, and one correct answer.
    
    Args:
        split: Dataset split to use ('train', 'validation', 'test')
        num_samples: Number of samples to return
        category: Optional category to filter questions by (e.g., 'biology', 'physics')
        categories: Optional list of categories to filter questions by.
    """
    # Load the dataset from HuggingFace
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    questions_data = []
    
    for item in dataset:
        question = item['question']
        
        # Handle options based on the format
        options = {}
        
        # Check if options is in the new format (list)
        if 'options' in item and isinstance(item['options'], list):
            for i, option_text in enumerate(item['options']):
                # Convert index to letter (0->A, 1->B, etc.)
                letter = chr(65 + i)  # 65 is ASCII for 'A'
                options[letter] = option_text
        else:
            # Handle the old format where options are individual fields (A, B, C, D)
            option_keys = ['A', 'B', 'C', 'D']
            for key in option_keys:
                if key in item:
                    options[key] = item[key]
        
        # Get the correct answer
        if 'answer_index' in item:
            # Convert index to letter (0->A, 1->B, etc.)
            correct_letter = chr(65 + item['answer_index'])
        else:
            correct_letter = item['answer']
        
        correct_answer = options.get(correct_letter, "")
        
        questions_data.append({
            'question': question,
            'options': options,
            'answer': correct_letter,
            'answer_content': correct_answer,
            'category': item.get('category', ''),
            'question_id': item.get('question_id', ''),
            'src': item.get('src', '')
        })
    
    # Filter by category if specified
    if category:
        filtered_data = [q for q in questions_data if q['category'].lower() == category.lower()]
        if not filtered_data:
            print(f"Warning: No questions found for category '{category}'. Using all questions.")
        else:
            questions_data = filtered_data
            print(f"Filtered to {len(questions_data)} questions in category '{category}'")
    elif categories:
        lower_cats = [c.lower() for c in categories]
        filtered_data = [q for q in questions_data if q['category'].lower() in lower_cats]
        if not filtered_data:
            print(f"Warning: No questions found for categories {categories}. Using all questions.")
        else:
            questions_data = filtered_data
            print(f"Filtered to {len(questions_data)} questions in categories {categories}")
    
    if num_samples is not None and num_samples < len(questions_data):
        # 1. Group by category
        by_cat = defaultdict(list)
        for q in questions_data:
            by_cat[q['category']].append(q)
            
        total_qs = sum(len(qs) for qs in by_cat.values())
        
        # 2. Calculate quotas using Largest Remainder Method
        quotas = {}
        remainders = []
        for cat, qs in by_cat.items():
            exact = (len(qs) / total_qs) * num_samples
            floor_val = math.floor(exact)
            quotas[cat] = floor_val
            remainders.append((exact - floor_val, cat))
            
        # 3. Distribute remaining
        remaining_samples = num_samples - sum(quotas.values())
        remainders.sort(reverse=True, key=lambda x: x[0])
        
        for i in range(remaining_samples):
            cat = remainders[i][1]
            quotas[cat] += 1
            
        # 4. Reproducible Sampling
        rng = random.Random(seed)
        sampled_data = []
        for cat, quota in quotas.items():
            actual_quota = min(quota, len(by_cat[cat]))
            sampled_data.extend(rng.sample(by_cat[cat], actual_quota))
            
        # 5. Final Shuffle
        rng.shuffle(sampled_data)
        questions_data = sampled_data

    return questions_data


def get_failed_dcr_ids(
    csv_path: str,
    agent_names: List[str],
    num_rounds: int = 5,
) -> Set[str]:
    """
    Read a past metrics CSV and return question IDs that were in the DCR pool
    but failed to collapse to the correct answer (using majority rule: > 50%).

    DCR-eligible conditions (round 0):
      - At least two agents disagreed (not all same answer).
      - At least one agent had the correct answer.

    Failed to collapse (final round, majority rule):
      - The correct answer did NOT receive > 50% of votes in the last round.

    Args:
        csv_path:    Path to the metrics CSV file.
        agent_names: List of agent display names as they appear in the CSV
                     column headers (e.g., ["Agent_Qwen", "Agent_Llama"]).
        num_rounds:  Number of debate rounds used in that evaluation.

    Returns:
        Set of question_id strings.
    """
    failed_ids: Set[str] = set()
    last_round = num_rounds - 1
    r0_cols    = [f"round_0_{name}" for name in agent_names]
    rf_cols    = [f"round_{last_round}_{name}" for name in agent_names]
    n_agents   = len(agent_names)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = row["correct_answer"]

            # Round-0 answers
            r0_answers = [row.get(col, "?") for col in r0_cols]
            initial_disagree = len(set(r0_answers)) > 1
            initial_correct  = correct in r0_answers
            dcr_eligible     = initial_disagree and initial_correct

            if not dcr_eligible:
                continue

            # Final-round answers — majority rule
            rf_answers    = [row.get(col, "?") for col in rf_cols]
            correct_count = sum(1 for a in rf_answers if a == correct)
            collapsed     = correct_count > n_agents / 2

            if not collapsed:
                failed_ids.add(row["question_id"])

    return failed_ids


def load_mmlu_pro_by_ids(question_ids: Set[str], split: str = "test") -> List[Dict]:
    """
    Load specific questions from MMLU Pro by question_id.

    Loads the full dataset and filters to the given IDs. Order matches the
    order questions appear in the dataset (reproducible, no shuffling).

    Args:
        question_ids: Set of question_id strings to retrieve.
        split:        Dataset split to search in (default: "test").

    Returns:
        List of question dicts in the same format as load_mmlu_pro().
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    results: List[Dict] = []

    for item in dataset:
        qid = str(item.get("question_id", ""))
        if qid not in question_ids:
            continue

        options: Dict[str, str] = {}
        if "options" in item and isinstance(item["options"], list):
            for i, text in enumerate(item["options"]):
                options[chr(65 + i)] = text
        else:
            for key in ["A", "B", "C", "D"]:
                if key in item:
                    options[key] = item[key]

        if "answer_index" in item:
            correct_letter = chr(65 + item["answer_index"])
        else:
            correct_letter = item["answer"]

        results.append({
            "question":       item["question"],
            "options":        options,
            "answer":         correct_letter,
            "answer_content": options.get(correct_letter, ""),
            "category":       item.get("category", ""),
            "question_id":    qid,
            "src":            item.get("src", ""),
        })

    return results
