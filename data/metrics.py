"""
Evaluation metrics for multi-agent debate.
Metrics implemented: Success Rate (accuracy), Disagreement Collapse Rate (DCR),
Negative Agreement Rate (NAR).
"""

from typing import List, Dict, Any, Optional


class RoundAnalyzer:
    """
    Analyzes a single debate result.
    Provides methods to extract signals for metrics calculation.

    Expected result format:
        {
            "question_id": str,
            "correct_answer": str,        # e.g., "A"
            "category": str,              # e.g., "business"
            "rounds": [
                {                         # round 0
                    "agent_0": "A",
                    "agent_1": "B",
                },
                ...
            ],
            "final_answer": str,          # system's final answer, or "unresolved"
        }
    """

    def __init__(self, result: Dict[str, Any]):
        self.result = result
        self.correct_answer = result.get("correct_answer", "")
        self.final_answer = result.get("final_answer", "")
        self.rounds = result.get("rounds", [])
        self.category = result.get("category", "unknown")

        self.agent_names = []
        if self.rounds:
            self.agent_names = sorted(self.rounds[0].keys())

    def is_resolved(self) -> bool:
        return self.final_answer != "unresolved"

    def is_correct(self) -> bool:
        return self.is_resolved() and self.final_answer == self.correct_answer

    def has_initial_disagreement(self) -> bool:
        if not self.rounds:
            return False
        round_0_answers = list(self.rounds[0].values())
        return len(set(round_0_answers)) > 1

    def has_initial_correct(self) -> bool:
        if not self.rounds:
            return False
        round_0_answers = list(self.rounds[0].values())
        return self.correct_answer in round_0_answers

    def is_dcr_nar_eligible(self) -> bool:
        """
        Eligible if there was initial disagreement AND
        at least one agent had the correct answer.
        """
        return self.has_initial_disagreement() and self.has_initial_correct()

    def has_collapsed(self) -> bool:
        """Did a strict majority of agents converge to the CORRECT answer by the final round?"""
        if not self.rounds:
            return False
        final_round_answers = list(self.rounds[-1].values())
        
        correct_count = sum(1 for ans in final_round_answers if ans == self.correct_answer)
        # Strict majority: > 50%
        return correct_count > len(final_round_answers) / 2

    def is_negative_agreement(self) -> bool:
        """Did a strict majority of agents converge to the SAME, WRONG answer?"""
        if not self.rounds:
            return False
        final_round_answers = list(self.rounds[-1].values())
        
        from collections import Counter
        counts = Counter(final_round_answers)
        majority_threshold = len(final_round_answers) / 2
        
        for ans, count in counts.items():
            if count > majority_threshold:
                return ans != self.correct_answer
                
        return False

    def get_agent_answers_by_round(self) -> Dict[str, List[str]]:
        """Return dict mapping agent_name to a list of their answers across rounds."""
        history = {agent: [] for agent in self.agent_names}
        for r in self.rounds:
            for agent in self.agent_names:
                history[agent].append(r.get(agent, "?"))
        return history


class Evaluator:
    """
    Accumulates signals from multiple debate results to compute overall metrics.
    
    Usage:
        evaluator = Evaluator()
        for question in dataset:
            result = arena.run(question)
            evaluator.add(result)
        summary = evaluator.summary()
    """

    def __init__(self):
        self.total_questions = 0
        self.resolved_questions = 0
        self.correct_questions = 0

        self.dcr_eligible_pool = 0
        self.collapsed_count = 0
        self.negative_agreement_count = 0

        self.category_counts: Dict[str, Dict[str, int]] = {}

        # Agent accuracy tracking: {agent_id: {"total": [0,0,..], "correct": [0,0,..]}}
        self.agent_stats_by_round: Dict[str, Dict[str, List[int]]] = {}

        # Round 0 (solo baseline) accuracy per agent per category.
        # Maps agent_id -> category -> {"total": int, "correct": int}.
        # Lets you cross-reference ToM domain observations against actual performance.
        self.agent_round0_category_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Agent flip rates: Susceptibility (correct -> incorrect) and Correction (incorrect -> correct)
        # Maps agent_id -> {"susceptibility_ops": 0, "susceptibility_events": 0, "correction_ops": 0, "correction_events": 0}
        self.agent_flip_stats: Dict[str, Dict[str, int]] = {}

    def add(self, result: Dict[str, Any]):
        """Process a single debate result and update running counters."""
        self.total_questions += 1
        analyzer = RoundAnalyzer(result)

        # 1. System Accuracy
        if analyzer.is_resolved():
            self.resolved_questions += 1
            if analyzer.is_correct():
                self.correct_questions += 1

        # 2. Category Accuracy
        cat = analyzer.category
        if cat not in self.category_counts:
            self.category_counts[cat] = {"total": 0, "correct": 0}
        self.category_counts[cat]["total"] += 1
        if analyzer.is_correct():
            self.category_counts[cat]["correct"] += 1

        # 3. DCR and NAR
        if analyzer.is_dcr_nar_eligible():
            self.dcr_eligible_pool += 1
            if analyzer.has_collapsed():
                self.collapsed_count += 1
            if analyzer.is_negative_agreement():
                self.negative_agreement_count += 1

        # 4. Agent Accuracy by Round
        agent_answers = analyzer.get_agent_answers_by_round()
        correct_ans = analyzer.correct_answer

        for agent, answers in agent_answers.items():
            if agent not in self.agent_stats_by_round:
                self.agent_stats_by_round[agent] = {"total": [], "correct": []}

            stats = self.agent_stats_by_round[agent]
            # Ensure lists are long enough
            while len(stats["total"]) < len(answers):
                stats["total"].append(0)
                stats["correct"].append(0)

            for round_idx, ans in enumerate(answers):
                stats["total"][round_idx] += 1
                if ans == correct_ans:
                    stats["correct"][round_idx] += 1

        # 5. Agent Round 0 accuracy by category (solo baseline before debate)
        if analyzer.rounds:
            round_0_answers = analyzer.rounds[0]  # {agent_id: answer_letter}
            cat = analyzer.category
            for agent_id, answer in round_0_answers.items():
                if agent_id not in self.agent_round0_category_stats:
                    self.agent_round0_category_stats[agent_id] = {}
                if cat not in self.agent_round0_category_stats[agent_id]:
                    self.agent_round0_category_stats[agent_id][cat] = {"total": 0, "correct": 0}
                self.agent_round0_category_stats[agent_id][cat]["total"] += 1
                if answer == correct_ans:
                    self.agent_round0_category_stats[agent_id][cat]["correct"] += 1

        # 6. Agent Flip Rates (Susceptibility and Correction)
        for agent, answers in agent_answers.items():
            if agent not in self.agent_flip_stats:
                self.agent_flip_stats[agent] = {
                    "susceptibility_ops": 0, "susceptibility_events": 0,
                    "correction_ops": 0, "correction_events": 0
                }
            
            stats = self.agent_flip_stats[agent]
            # Need at least 2 rounds to check transitions
            for i in range(len(answers) - 1):
                prev_ans = answers[i]
                next_ans = answers[i+1]
                
                prev_correct = (prev_ans == correct_ans)
                next_correct = (next_ans == correct_ans)
                
                if prev_correct:
                    stats["susceptibility_ops"] += 1
                    if not next_correct:
                        stats["susceptibility_events"] += 1
                else:
                    stats["correction_ops"] += 1
                    if next_correct:
                        stats["correction_events"] += 1

    def system_accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.correct_questions / self.total_questions

    def resolved_accuracy(self) -> float:
        if self.resolved_questions == 0:
            return 0.0
        return self.correct_questions / self.resolved_questions

    def disagreement_collapse_rate(self) -> float:
        if self.dcr_eligible_pool == 0:
            return 0.0
        return self.collapsed_count / self.dcr_eligible_pool

    def negative_agreement_rate(self) -> float:
        if self.dcr_eligible_pool == 0:
            return 0.0
        return self.negative_agreement_count / self.dcr_eligible_pool

    def accuracy_by_category(self) -> Dict[str, float]:
        return {
            cat: counts["correct"] / counts["total"]
            for cat, counts in sorted(self.category_counts.items())
            if counts["total"] > 0
        }

    def agent_accuracy_by_round(self) -> Dict[str, List[float]]:
        result = {}
        for agent, stats in self.agent_stats_by_round.items():
            result[agent] = []
            for c, t in zip(stats["correct"], stats["total"]):
                result[agent].append(c / t if t > 0 else 0.0)
        return result

    def agent_round0_accuracy_by_category(self) -> Dict[str, Dict[str, float]]:
        """
        Per-agent accuracy at Round 0 (solo baseline, no peer context) broken
        down by question category.

        Use this to cross-validate ToM memory: if ToM says Agent_A is weak in
        biology, this metric should show a lower biology accuracy for agent_0.

        Returns:
            {agent_id: {category: accuracy_float}}, categories sorted alphabetically.
        """
        result = {}
        for agent_id, cat_stats in self.agent_round0_category_stats.items():
            result[agent_id] = {
                cat: counts["correct"] / counts["total"]
                for cat, counts in sorted(cat_stats.items())
                if counts["total"] > 0
            }
        return result

    def agent_susceptibility_rate(self) -> Dict[str, float]:
        """Rate of flipping from correct to incorrect answer."""
        result = {}
        for agent, stats in self.agent_flip_stats.items():
            ops = stats["susceptibility_ops"]
            events = stats["susceptibility_events"]
            result[agent] = events / ops if ops > 0 else 0.0
        return result

    def agent_correction_rate(self) -> Dict[str, float]:
        """Rate of flipping from incorrect to correct answer."""
        result = {}
        for agent, stats in self.agent_flip_stats.items():
            ops = stats["correction_ops"]
            events = stats["correction_events"]
            result[agent] = events / ops if ops > 0 else 0.0
        return result

    def summary(self) -> Dict[str, Any]:
        """Generate a full evaluation summary."""
        return {
            "system_accuracy": self.system_accuracy(),
            "resolved_accuracy": self.resolved_accuracy(),
            "num_questions": self.total_questions,
            "num_resolved": self.resolved_questions,
            "num_unresolved": self.total_questions - self.resolved_questions,
            "agent_accuracy_by_round": self.agent_accuracy_by_round(),
            "accuracy_by_category": self.accuracy_by_category(),
            "agent_round0_accuracy_by_category": self.agent_round0_accuracy_by_category(),
            "agent_susceptibility_rate": self.agent_susceptibility_rate(),
            "agent_correction_rate": self.agent_correction_rate(),
            "dcr": self.disagreement_collapse_rate(),
            "nar": self.negative_agreement_rate(),
            "dcr_nar_pool_size": self.dcr_eligible_pool,
        }
