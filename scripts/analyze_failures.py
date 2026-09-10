import json
from collections import Counter
from pathlib import Path

INPUT = Path("data/golden/agent_eval_250.jsonl")

rows = []

with INPUT.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"Loaded {len(rows)} evaluation examples")


# =========================================================
# INTENT FAILURES
# =========================================================

intent_failures = [
    r for r in rows
    if not r.get("intent_correct", False)
]

print("\n" + "=" * 70)
print("INTENT FAILURE SUMMARY")
print("=" * 70)

print(f"Intent failures: {len(intent_failures)}/{len(rows)}")

confusions = Counter(
    (
        r.get("human_intent"),
        r.get("predicted_intent")
    )
    for r in intent_failures
)

print("\nMost common intent confusions:")

for (gold, pred), count in confusions.most_common(10):
    print(f"{gold} -> {pred}: {count}")


# =========================================================
# ESCALATION FAILURES
# =========================================================

escalation_failures = [
    r for r in rows
    if not r.get("escalation_correct", False)
]

print("\n" + "=" * 70)
print("ESCALATION FAILURE SUMMARY")
print("=" * 70)

print(
    f"Escalation failures: "
    f"{len(escalation_failures)}/{len(rows)}"
)

false_auto = [
    r for r in escalation_failures
    if r.get("human_escalate") is True
    and r.get("predicted_escalate") is False
]

false_escalation = [
    r for r in escalation_failures
    if r.get("human_escalate") is False
    and r.get("predicted_escalate") is True
]

print(f"False auto-handle: {len(false_auto)}")
print(f"False escalation: {len(false_escalation)}")


# =========================================================
# FAILURE PATTERNS
# =========================================================

patterns = Counter()

for r in rows:

    gold_intent = r.get("human_intent")
    pred_intent = r.get("predicted_intent")

    gold_esc = r.get("human_escalate")
    pred_esc = r.get("predicted_escalate")

    if gold_intent != pred_intent:
        patterns[
            f"Intent: {gold_intent} -> {pred_intent}"
        ] += 1

    if gold_esc != pred_esc:

        if gold_esc is True and pred_esc is False:
            patterns["Escalation: false auto-handle"] += 1

        elif gold_esc is False and pred_esc is True:
            patterns["Escalation: false escalation"] += 1


print("\n" + "=" * 70)
print("TOP FAILURE PATTERNS")
print("=" * 70)

for pattern, count in patterns.most_common(10):
    print(f"{count} cases — {pattern}")


# =========================================================
# REAL EXAMPLES
# =========================================================

print("\n" + "=" * 70)
print("REAL FAILURE EXAMPLES")
print("=" * 70)

shown = set()

for r in rows:

    gold_intent = r.get("human_intent")
    pred_intent = r.get("predicted_intent")

    gold_esc = r.get("human_escalate")
    pred_esc = r.get("predicted_escalate")

    intent_failed = gold_intent != pred_intent
    escalation_failed = gold_esc != pred_esc

    if not intent_failed and not escalation_failed:
        continue

    # Prefer showing intent failures first
    if intent_failed:
        pattern = f"Intent: {gold_intent} -> {pred_intent}"
    elif gold_esc and not pred_esc:
        pattern = "Escalation: false auto-handle"
    else:
        pattern = "Escalation: false escalation"

    if pattern in shown:
        continue

    shown.add(pattern)

    print("\n" + "-" * 70)
    print(pattern)

    print(f"Conversation ID: {r.get('conversation_id')}")
    print(f"Gold intent: {gold_intent}")
    print(f"Predicted intent: {pred_intent}")
    print(f"Gold escalation: {gold_esc}")
    print(f"Predicted escalation: {pred_esc}")
    print(f"Confidence: {r.get('confidence')}")

    print("\nCustomer:")
    print(str(r.get("customer_message", ""))[:1500])

    print("\nAI reply:")
    print(str(r.get("reply", ""))[:1500])

    print("\nHistorical evidence:")
    print(str(r.get("evidence", ""))[:1500])

    print("\nEscalation reason:")
    print(str(r.get("reason", ""))[:1000])

    if len(shown) >= 10:
        break