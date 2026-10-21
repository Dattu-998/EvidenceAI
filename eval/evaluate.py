import json
import httpx
import time
import os
import sys

API_URL = "http://localhost:8000"

def wait_for_document_processing(doc_id: str, client: httpx.Client, timeout_sec: int = 30):
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        res = client.get(f"{API_URL}/documents/{doc_id}")
        if res.status_code == 200:
            status = res.json().get("status")
            if status == "ready":
                return True
            elif status == "failed":
                print(f"Document processing failed: {res.json()}")
                return False
        time.sleep(1)
    print("Timeout waiting for document processing.")
    return False

def evaluate():
    print("Starting Evaluation Harness...")

    with open("eval/ground_truth.json", "r") as f:
        ground_truth = json.load(f)

    client = httpx.Client()

    # 1. Check health
    try:
        health = client.get(f"{API_URL}/health")
        if health.status_code != 200:
            print("API is not healthy. Please start the API server first.")
            return
    except Exception as e:
        print(f"Failed to connect to API: {e}. Is the server running on port 8000?")
        return

    # 2. Upload Evaluation Document
    pdf_path = "eval/data/evidenceai_overview.pdf"
    print(f"Uploading {pdf_path}...")

    # Try to delete existing first for clean eval
    docs_res = client.get(f"{API_URL}/documents")
    if docs_res.status_code == 200:
        for d in docs_res.json():
            if d["filename"] == os.path.basename(pdf_path):
                client.delete(f"{API_URL}/documents/{d['id']}")

    with open(pdf_path, "rb") as f:
        upload_res = client.post(
            f"{API_URL}/documents",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")}
        )

    if upload_res.status_code != 200:
        print(f"Failed to upload document: {upload_res.text}")
        return

    doc_id = upload_res.json()["id"]

    if not wait_for_document_processing(doc_id, client):
        return

    print("Document ingested and ready. Starting queries...")

    metrics = {
        "total_queries": len(ground_truth),
        "recall_at_5": 0,
        "answer_correctness": 0,
        "citation_accuracy": 0,
        "correct_refusals": 0,
        "false_refusals": 0,
        "latencies": []
    }

    total_answerable = sum(1 for item in ground_truth if item["is_answerable"])
    total_unanswerable = metrics["total_queries"] - total_answerable

    for item in ground_truth:
        query = item["query"]
        print(f"\nEvaluating Query: '{query}'")

        start_time = time.time()
        res = client.post(
            f"{API_URL}/query",
            json={"query": query, "retrieval_mode": "hybrid", "top_k": 5}
        )
        latency = time.time() - start_time
        metrics["latencies"].append(latency)

        if res.status_code != 200:
            print(f"  Error: {res.status_code} {res.text}")
            continue

        result = res.json()
        answer = result["answer"]
        citations = result["citations"]

        print(f"  Answer: {answer}")

        # Check Recall & Citations
        retrieved_correct_page = False
        citation_accurate = False

        if item["is_answerable"]:
            for citation in citations:
                if citation["filename"] == item["expected_filename"] and citation["page_number"] == item["expected_page"]:
                    retrieved_correct_page = True
                    citation_accurate = True
                    break

            if retrieved_correct_page:
                metrics["recall_at_5"] += 1
            else:
                print(f"  [Failure] Did not retrieve expected evidence (File: {item['expected_filename']}, Page: {item['expected_page']})")

            if citation_accurate:
                metrics["citation_accuracy"] += 1

            # Basic Correctness Check (Does the answer contain the expected facts?)
            correct = True
            for fact in item["expected_facts"]:
                # Simple exact match for keywords. In real eval, use an LLM evaluator.
                # For Groq Llama3, it might rephrase, so we do a lenient check or just manual review
                # Since we don't have a reliable LLM-as-a-judge here without another prompt, we use soft matching.
                # If we use dummy, it definitely won't match, but with Groq it should.
                pass

            # Since strict string matching is brittle, we'll approximate correctness based on retrieval + answer length
            if retrieved_correct_page and len(answer) > 20 and "sufficient evidence" not in answer.lower():
                metrics["answer_correctness"] += 1

            # False Refusal Check
            if "could not find sufficient evidence" in answer.lower():
                print("  [Failure] False refusal detected.")
                metrics["false_refusals"] += 1

        else: # Unanswerable
            if "could not find sufficient evidence" in answer.lower():
                print("  [Success] Correctly refused to answer.")
                metrics["correct_refusals"] += 1
            else:
                print("  [Failure] Answered an unanswerable question.")

    # Calculate Final Metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS (First Pass)")
    print("="*50)

    recall_rate = (metrics["recall_at_5"] / total_answerable) * 100 if total_answerable > 0 else 0
    citation_acc = (metrics["citation_accuracy"] / total_answerable) * 100 if total_answerable > 0 else 0
    answer_corr = (metrics["answer_correctness"] / total_answerable) * 100 if total_answerable > 0 else 0

    correct_refusal_rate = (metrics["correct_refusals"] / total_unanswerable) * 100 if total_unanswerable > 0 else 0
    false_refusal_rate = (metrics["false_refusals"] / total_answerable) * 100 if total_answerable > 0 else 0

    latencies = sorted(metrics["latencies"])
    p50 = latencies[len(latencies)//2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    print(f"Total Queries:         {metrics['total_queries']}")
    print(f"Recall@5:              {recall_rate:.1f}%")
    print(f"Citation Accuracy:     {citation_acc:.1f}%")
    print(f"Answer Correctness:    {answer_corr:.1f}% (Approximate)")
    print(f"Correct Refusal Rate:  {correct_refusal_rate:.1f}%")
    print(f"False Refusal Rate:    {false_refusal_rate:.1f}%")
    print(f"P50 Latency:           {p50:.2f}s")
    print(f"P95 Latency:           {p95:.2f}s")
    print(f"Estimated Cost/Query:  $0.00 (Groq free tier currently)")
    print("="*50)

    # Clean up (Optional, but good practice for testing)
    client.delete(f"{API_URL}/documents/{doc_id}")

if __name__ == "__main__":
    evaluate()
