import sys
import json
import csv
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import run_agent
from src.semantic_retrieval import SemanticRetriever


GOLDEN_FILE = Path("data/golden/golden_set.csv")

# Full evaluation output.
OUTPUT_FILE = Path("data/golden/agent_eval_250.jsonl")

# Evaluate all labeled golden examples.
NUM_EXAMPLES = 250

# Free Gemini tier: keep requests spaced out.
SLEEP_SECONDS = 2

# Retry failed API requests.
MAX_RETRIES = 3

INTENTS = [
    "account_access",
    "delivery_issue",
    "device_technical",
    "missing_package",
    "order_issue",
    "other",
    "payment_billing",
    "return_refund",
    "subscription_service",
]


def load_golden():
    rows = []

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Only evaluate labeled examples.
            if row.get("intent", "").strip():
                rows.append(row)

    return rows


def save_results(results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )


def load_existing_results():
    """
    Resume an interrupted evaluation.

    If agent_eval_250.jsonl already exists, previously completed
    conversation IDs are loaded so we don't call Gemini again.
    """

    if not OUTPUT_FILE.exists():
        return []

    results = []

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                result = json.loads(line)

                # Don't resume failed API calls.
                if "error" not in result:
                    results.append(result)

            except json.JSONDecodeError:
                continue

    return results


def run_with_retry(message, evidence):
    """
    Call Gemini with retries for temporary API failures.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            return run_agent(
                message,
                evidence
            )

        except Exception as e:

            last_error = e

            print(
                f"  API attempt {attempt}/{MAX_RETRIES} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < MAX_RETRIES:
                wait_time = attempt * 5

                print(
                    f"  Waiting {wait_time}s before retry..."
                )

                time.sleep(wait_time)

    raise last_error


def calculate_metrics(results):

    successful = [
        r for r in results
        if "error" not in r
    ]

    if not successful:
        return None

    y_true_intent = [
        r["human_intent"]
        for r in successful
    ]

    y_pred_intent = [
        r["predicted_intent"]
        for r in successful
    ]

    y_true_escalation = [
        r["human_escalate"]
        for r in successful
    ]

    y_pred_escalation = [
        r["predicted_escalate"]
        for r in successful
    ]

    intent_accuracy = accuracy_score(
        y_true_intent,
        y_pred_intent
    )

    intent_macro_f1 = f1_score(
        y_true_intent,
        y_pred_intent,
        labels=INTENTS,
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        y_true_intent,
        y_pred_intent,
        labels=INTENTS,
        average="weighted",
        zero_division=0,
    )

    escalation_accuracy = accuracy_score(
        y_true_escalation,
        y_pred_escalation
    )

    escalation_precision, escalation_recall, escalation_f1, _ = (
        precision_recall_fscore_support(
            y_true_escalation,
            y_pred_escalation,
            average="binary",
            zero_division=0,
        )
    )

    # Per-intent metrics.
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true_intent,
            y_pred_intent,
            labels=INTENTS,
            zero_division=0,
        )
    )

    per_intent = {}

    for i, intent in enumerate(INTENTS):

        per_intent[intent] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Confusion matrix.
    cm = confusion_matrix(
        y_true_intent,
        y_pred_intent,
        labels=INTENTS,
    )

    confusion = {
        "labels": INTENTS,
        "matrix": cm.tolist(),
    }

    # Important safety/business metrics.
    false_auto_handle = sum(
        1
        for r in successful
        if r["human_escalate"] is True
        and r["predicted_escalate"] is False
    )

    false_escalation = sum(
        1
        for r in successful
        if r["human_escalate"] is False
        and r["predicted_escalate"] is True
    )

    return {
        "n_total": len(results),
        "n_successful": len(successful),

        "intent_accuracy": float(intent_accuracy),
        "intent_macro_f1": float(intent_macro_f1),
        "intent_weighted_f1": float(intent_weighted_f1),

        "escalation_accuracy": float(escalation_accuracy),
        "escalation_precision": float(escalation_precision),
        "escalation_recall": float(escalation_recall),
        "escalation_f1": float(escalation_f1),

        "false_auto_handle": false_auto_handle,
        "false_escalation": false_escalation,

        "per_intent": per_intent,
        "confusion_matrix": confusion,
    }


def print_metrics(results):

    metrics = calculate_metrics(results)

    if metrics is None:
        print("\nNo successful examples to evaluate.")
        return

    print()
    print("=" * 70)
    print("FINAL EVALUATION METRICS")
    print("=" * 70)

    print(
        f"Successful examples: "
        f"{metrics['n_successful']}/{metrics['n_total']}"
    )

    print()
    print("INTENT CLASSIFICATION")
    print(
        f"Accuracy:    {metrics['intent_accuracy']:.3f}"
    )
    print(
        f"Macro-F1:    {metrics['intent_macro_f1']:.3f}"
    )
    print(
        f"Weighted-F1: {metrics['intent_weighted_f1']:.3f}"
    )

    print()
    print("ESCALATION")
    print(
        f"Accuracy:    {metrics['escalation_accuracy']:.3f}"
    )
    print(
        f"Precision:   {metrics['escalation_precision']:.3f}"
    )
    print(
        f"Recall:      {metrics['escalation_recall']:.3f}"
    )
    print(
        f"F1:          {metrics['escalation_f1']:.3f}"
    )

    print()
    print("ESCALATION ERRORS")
    print(
        f"False auto-handle: "
        f"{metrics['false_auto_handle']}"
    )
    print(
        f"False escalation:  "
        f"{metrics['false_escalation']}"
    )

    print()
    print("PER-INTENT METRICS")
    print(
        f"{'Intent':<22}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Support':>10}"
    )

    print("-" * 62)

    for intent in INTENTS:

        m = metrics["per_intent"][intent]

        print(
            f"{intent:<22}"
            f"{m['precision']:>10.3f}"
            f"{m['recall']:>10.3f}"
            f"{m['f1']:>10.3f}"
            f"{m['support']:>10}"
        )


def main():

    print("=" * 70)
    print("AGENT EVALUATION — FULL 250 GOLDEN EXAMPLES")
    print("=" * 70)

    golden = load_golden()

    print(f"Golden examples available: {len(golden)}")

    if len(golden) < NUM_EXAMPLES:
        print(
            f"WARNING: Expected {NUM_EXAMPLES} examples, "
            f"but only found {len(golden)} labeled examples."
        )

    golden = golden[:NUM_EXAMPLES]

    # Load previous successful results for resume support.
    existing_results = load_existing_results()

    completed_ids = {
        str(r["conversation_id"])
        for r in existing_results
    }

    results = existing_results.copy()

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Remaining: "
        f"{len(golden) - len(completed_ids)}"
    )

    retriever = SemanticRetriever()

    for i, row in enumerate(golden, 1):

        conversation_id = str(
            row["conversation_id"]
        )

        if conversation_id in completed_ids:
            print(
                f"\n[{i}/{len(golden)}] "
                f"Skipping {conversation_id} "
                f"(already completed)"
            )
            continue

        message = row["customer_messages"].strip()

        human_intent = row["intent"].strip()

        human_escalate = (
            row["escalate"].strip().lower()
            == "yes"
        )

        print()
        print("=" * 70)
        print(
            f"EXAMPLE {i}/{len(golden)}"
        )
        print(
            f"Conversation ID: {conversation_id}"
        )
        print(
            f"Human intent: {human_intent}"
        )
        print(
            f"Human escalation: {human_escalate}"
        )
        print(
            f"Customer: {message[:500]}"
        )

        try:

            # Retrieve historical evidence.
            evidence = retriever.search(
                message,
                top_k=5
            )

            # Run Gemini with retries.
            prediction = run_with_retry(
                message,
                evidence
            )

            predicted_intent = (
                prediction.get(
                    "intent",
                    ""
                )
                .strip()
            )

            predicted_action = (
                prediction.get(
                    "action",
                    ""
                )
                .strip()
                .lower()
            )

            predicted_escalate = (
                predicted_action == "escalate"
            )

            intent_correct = (
                predicted_intent
                == human_intent
            )

            escalation_correct = (
                predicted_escalate
                == human_escalate
            )

            result = {
                "conversation_id": conversation_id,
                "customer_message": message,

                "human_intent": human_intent,
                "predicted_intent": predicted_intent,
                "intent_correct": intent_correct,

                "human_escalate": human_escalate,
                "predicted_action": predicted_action,
                "predicted_escalate": predicted_escalate,
                "escalation_correct": escalation_correct,

                "confidence": prediction.get(
                    "confidence"
                ),

                "reason": prediction.get(
                    "reason"
                ),

                "reply": prediction.get(
                    "reply"
                ),

                "evidence": evidence,
            }

            results.append(result)
            completed_ids.add(conversation_id)

            print()
            print("PREDICTION")
            print(
                f"Intent: "
                f"{predicted_intent}"
            )
            print(
                f"Confidence: "
                f"{prediction.get('confidence')}"
            )
            print(
                f"Action: "
                f"{predicted_action}"
            )
            print(
                f"Intent correct: "
                f"{intent_correct}"
            )
            print(
                f"Escalation correct: "
                f"{escalation_correct}"
            )

            print()
            print("Reply:")
            print(
                prediction.get("reply")
            )

        except Exception as e:

            print()
            print("ERROR:")
            print(
                type(e).__name__,
                str(e)
            )

            result = {
                "conversation_id": conversation_id,
                "customer_message": message,

                "human_intent": human_intent,
                "predicted_intent": None,
                "intent_correct": False,

                "human_escalate": human_escalate,
                "predicted_action": None,
                "predicted_escalate": None,
                "escalation_correct": False,

                "error": str(e),
            }

            results.append(result)

        # Save after EVERY example.
        save_results(results)

        # Rate-limit protection.
        if i < len(golden):
            time.sleep(SLEEP_SECONDS)

    # Save one final time.
    save_results(results)

    print_metrics(results)

    print()
    print("=" * 70)
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()