# Golden set

Create 150–250 hand-labelled examples **after** selecting the brand and taxonomy.

Recommended JSONL schema:
```json
{"id":1,"customer_message":"...","intent":"refund_pending","should_escalate":false,"difficulty":"medium","notes":"..."}
```

Keep this set out of retrieval/training data. Record sampling methodology and label definitions in the report.
