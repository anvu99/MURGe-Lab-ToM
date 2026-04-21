import random
import math
from collections import defaultdict
from datasets import load_dataset
from typing import Optional, List, Dict


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
    
    if num_samples < len(questions_data):
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
