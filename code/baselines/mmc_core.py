#!/usr/bin/env python3
"""
MMC: Metacognitive Memory Calibration
Core module — dual memory architecture + retrieval + reflection

Two memory types:
  1. Performance Memory: stores (question_summary, answer, correct, domain, confidence)
  2. Metacognitive Memory: stores self-generated error patterns (model discovers its own weaknesses)

Three phases:
  Phase 1 (Calibration): answer questions, receive feedback, build both memories
  Phase 2 (Test): retrieve relevant cases + error patterns, calibrate confidence
  Phase 3 (Online update): continuously update memories
"""
import numpy as np
import json, re, time, requests
from collections import defaultdict
from datetime import datetime


class PerformanceMemory:
    """Stores individual question-answer episodes with outcomes."""
    
    def __init__(self, max_per_domain=50):
        self.episodes = []  # list of dicts
        self.by_domain = defaultdict(list)  # domain -> [episode_idx]
        self.max_per_domain = max_per_domain
    
    def add(self, question_summary, domain, answer, correct, confidence, reasoning=""):
        ep = {
            "id": len(self.episodes),
            "question": question_summary[:200],
            "domain": domain,
            "answer": answer,
            "correct": correct,
            "confidence": confidence,
            "reasoning": reasoning[:200],
            "timestamp": len(self.episodes),
        }
        self.episodes.append(ep)
        self.by_domain[domain].append(ep["id"])
        
        # Evict oldest if over limit
        if len(self.by_domain[domain]) > self.max_per_domain:
            self.by_domain[domain] = self.by_domain[domain][-self.max_per_domain:]
    
    def retrieve_domain(self, domain, k=6):
        """Retrieve k most recent episodes from a domain (balanced correct/incorrect)."""
        indices = self.by_domain.get(domain, [])
        if not indices:
            return []
        
        recent = [self.episodes[i] for i in indices[-20:]]  # last 20
        correct_eps = [e for e in recent if e["correct"]]
        wrong_eps = [e for e in recent if not e["correct"]]
        
        # Balance: take up to k//2 correct and k//2 wrong
        half = k // 2
        selected = wrong_eps[-half:] + correct_eps[-half:]
        return selected[-k:]
    
    def get_domain_stats(self, domain):
        """Get accuracy stats for a domain."""
        indices = self.by_domain.get(domain, [])
        if not indices:
            return {"accuracy": 0.5, "total": 0, "correct": 0}
        eps = [self.episodes[i] for i in indices]
        correct = sum(1 for e in eps if e["correct"])
        return {"accuracy": correct / len(eps), "total": len(eps), "correct": correct}


class MetacognitiveMemory:
    """Stores self-generated error pattern analyses."""
    
    def __init__(self, max_patterns=20):
        self.patterns = []  # list of dicts {pattern, domain, evidence_count, timestamp}
        self.max_patterns = max_patterns
    
    def add(self, pattern_text, source_domain, evidence_count=1):
        """Add a new error pattern or reinforce existing one."""
        # Check for similar existing pattern (simple keyword overlap)
        for p in self.patterns:
            if self._similarity(p["pattern"], pattern_text) > 0.5:
                p["evidence_count"] += evidence_count
                p["timestamp"] = len(self.patterns)
                return
        
        self.patterns.append({
            "pattern": pattern_text[:300],
            "source_domain": source_domain,
            "evidence_count": evidence_count,
            "timestamp": len(self.patterns),
        })
        
        # Evict lowest evidence if over limit
        if len(self.patterns) > self.max_patterns:
            self.patterns.sort(key=lambda p: p["evidence_count"])
            self.patterns = self.patterns[1:]
    
    def retrieve(self, domain=None, k=5):
        """Retrieve top-k patterns, optionally filtered by domain relevance."""
        if not self.patterns:
            return []
        
        # Sort by evidence count (most confirmed patterns first)
        sorted_p = sorted(self.patterns, key=lambda p: -p["evidence_count"])
        
        # Prioritize same-domain but include cross-domain patterns
        if domain:
            same = [p for p in sorted_p if p["source_domain"] == domain]
            cross = [p for p in sorted_p if p["source_domain"] != domain]
            result = (same + cross)[:k]
        else:
            result = sorted_p[:k]
        
        return result
    
    def _similarity(self, text1, text2):
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)


def build_reflection_prompt(wrong_episodes, domain):
    """Build prompt for metacognitive reflection on errors."""
    examples = "\n".join([
        f"- Q: {e['question'][:100]}... → Your answer: {e['answer']} (WRONG)"
        for e in wrong_episodes[:5]
    ])
    
    return f"""You recently got several questions wrong in the domain "{domain}".

Here are your mistakes:
{examples}

Analyze your errors and identify 1-2 specific patterns. Focus on:
- What TYPE of questions you struggle with (not specific facts)
- What REASONING errors you make (e.g., confusing similar concepts, multi-step logic failures)
- Patterns that might apply ACROSS domains (e.g., "I struggle with questions requiring precise numerical reasoning")

Format each pattern as a single sentence starting with "I tend to..."
Patterns:"""


def build_test_prompt(question, choices, perf_episodes, meta_patterns, domain_stats):
    """Build the test-time prompt with retrieved memories."""
    
    # Format performance memory examples
    if perf_episodes:
        perf_section = "[YOUR PAST PERFORMANCE ON SIMILAR QUESTIONS]\n"
        for e in perf_episodes:
            mark = "✓ CORRECT" if e["correct"] else "✗ WRONG"
            perf_section += f"  {mark}: {e['question'][:80]}...\n"
        perf_section += f"\n  Summary: {domain_stats['correct']}/{domain_stats['total']} correct ({domain_stats['accuracy']:.0%}) in this domain\n"
    else:
        perf_section = "[No prior experience in this domain]\n"
    
    # Format metacognitive patterns
    if meta_patterns:
        meta_section = "[YOUR KNOWN WEAKNESSES]\n"
        for p in meta_patterns:
            meta_section += f"  - {p['pattern']}\n"
    else:
        meta_section = ""
    
    return f"""{perf_section}
{meta_section}
[CURRENT QUESTION]
Domain: {domain_stats.get('domain_name', 'Unknown')}
Question: {question}
{choices}

[INSTRUCTIONS]
1. Look at your past performance above. Which of your past correct/wrong answers is most similar to this question?
2. Check if any of your known weaknesses apply to this question.
3. Based on this self-assessment, answer and provide a calibrated confidence.

Answer: [A/B/C/D]
Similarity judgment: This question is most similar to [a correct/wrong past case] because [reason].
Weakness check: [relevant weakness or "none apply"]
Confidence: [number]%"""


def parse_mmc_output(text, domain_acc=0.5):
    """Parse answer and confidence from MMC prompt output."""
    answer = None
    for c in ['A', 'B', 'C', 'D']:
        if f"Answer: {c}" in text or f"answer: {c}" in text.lower():
            answer = c
            break
    if not answer:
        for c in ['A', 'B', 'C', 'D']:
            if text.strip().startswith(c):
                answer = c
                break
    
    conf_match = re.search(r'[Cc]onfidence:?\s*(\d+)', text)
    if conf_match:
        conf = int(conf_match.group(1)) / 100.0
    else:
        conf = domain_acc  # fallback to domain accuracy
    
    return answer, np.clip(conf, 0.01, 0.99)


def parse_reflection_output(text):
    """Parse error patterns from reflection output."""
    patterns = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("I tend to") or line.startswith("- I tend to"):
            patterns.append(line.lstrip("- ").strip())
        elif line.startswith("I struggle") or line.startswith("- I struggle"):
            patterns.append(line.lstrip("- ").strip())
        elif line.startswith("I often") or line.startswith("- I often"):
            patterns.append(line.lstrip("- ").strip())
    
    # Fallback: take any line starting with dash that mentions error/wrong/confuse
    if not patterns:
        for line in text.split("\n"):
            line = line.strip().lstrip("- ").lstrip("1234567890.)")
            if any(kw in line.lower() for kw in ["confus", "struggle", "fail", "error", "wrong", "mistake", "difficult"]):
                if len(line) > 20:
                    patterns.append(line[:300])
    
    return patterns[:3]  # max 3 patterns per reflection


if __name__ == "__main__":
    print("MMC Core module loaded.")
    print("Classes: PerformanceMemory, MetacognitiveMemory")
    print("Functions: build_reflection_prompt, build_test_prompt, parse_mmc_output, parse_reflection_output")
    
    # Quick test
    pm = PerformanceMemory()
    pm.add("What is mitosis?", "biology", "B", True, 0.9)
    pm.add("BRCA1 mutation mechanism?", "biology", "A", False, 0.8)
    pm.add("Cell membrane structure?", "biology", "C", True, 0.85)
    print(f"\nPerformance Memory test: {pm.get_domain_stats('biology')}")
    print(f"Retrieved: {len(pm.retrieve_domain('biology'))} episodes")
    
    mm = MetacognitiveMemory()
    mm.add("I tend to confuse similar molecular pathways", "biology")
    mm.add("I struggle with multi-step causal reasoning", "physics")
    print(f"\nMetacognitive Memory test: {len(mm.retrieve())} patterns")
    print(f"Cross-domain retrieval for chemistry: {len(mm.retrieve('chemistry'))} patterns")
