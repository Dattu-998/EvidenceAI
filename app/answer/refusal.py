from typing import List, Dict, Any

def evaluate_evidence_sufficiency(query: str, evidence: List[Dict[str, Any]]) -> bool:
    if not evidence:
        return False

    combined_evidence = " ".join([e["text"] for e in evidence])
    if len(combined_evidence.strip()) < 50:
        return False

    return True
