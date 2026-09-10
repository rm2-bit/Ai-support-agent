# SDE AI Support Agent

An evidence-first customer-support agent built using the **Customer Support on Twitter** dataset.

I built this project to see whether an AI agent can handle common customer-support conversations without blindly trusting the model. For each incoming customer message, the system:

1. Identifies the customer's intent.

2. Retrieves similar historical AmazonHelp conversations.

3. Uses those examples as evidence when drafting a response.

4. Decides whether the conversation can be auto-handled or should be sent to a human, with a reason.

The main focus of the project is not just getting a high accuracy number. I also wanted to understand **when the system makes mistakes and whether those mistakes are safe enough for automation**.

**## Quick Start

1. Create a virtual environment

python -m venv .venv
source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Configure the Gemini API

Copy the example environment file:

cp .env.example .env

Open .env and add your Gemini API key:

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite

Do not commit .env or expose the API key.

4. Run the project

The repository contains the processed AmazonHelp data and evaluation artifacts, so the headline evaluation does not require processing the complete 3M+ tweet dataset again.

If the processed files and semantic index are already present, run the final agent evaluation directly:

python scripts/evaluate_agent.py

The evaluation uses the fixed 250-example golden set.

Full pipeline from a fresh checkout

If starting from the raw dataset, run:

python scripts/download_data.py
python scripts/profile_dataset.py
python src/data/create_train_split.py
python src/data/weak_label.py
python src/data/build_semantic_index.py
python scripts/evaluate_baselines.py
python scripts/evaluate_agent.py

The semantic retrieval index is built from training data only. The 250 golden examples are kept out of both training and retrieval.

Useful commands

Run a manual agent test:

python scripts/test_agent.py

Run the baseline evaluation:

python scripts/evaluate_baselines.py

Run the final 250-example evaluation:

python scripts/evaluate_agent.py

The agent evaluation is checkpointed/resumable, so if an API call fails or the process is interrupted, running the same command again continues from the saved evaluation results.

Dataset**

The project uses the **Customer Support on Twitter** dataset by Thought Vector.

The dataset contains more than 2.8M tweets and includes reply relationships between tweets. I used these relationships to reconstruct multi-turn support conversations instead of treating every tweet as an isolated classification example.

For this project I selected **AmazonHelp**.

Some of the useful dataset statistics were:

| Statistic                        |         Count |

| -------------------------------- | ------------: |

| Total tweets                     |     2,811,774 |

| AmazonHelp outbound tweets       |       169,840 |

| Reconstructed AmazonHelp threads |        81,467 |

| Average thread length            | 3.56 messages |

| Golden examples                  |           250 |

The data is noisy and contains non-support conversations as well. I kept that noise rather than assuming every message represented a real support request.

## Why AmazonHelp?

I chose AmazonHelp because it had a large amount of usable customer-support activity in the dataset.

The goal was to have enough historical conversations for retrieval while still working with one specific brand and its support style.

## How the system works

The overall flow is:

```text

Customer message

   |

   v

Intent classification

   |

   v

Historical conversation retrieval

   |

   v

Evidence filtering

   |

   +--------------------+

   \|                    |

   v                    v

Auto-handle          Escalate

   \|                    |

   v                    v

Grounded reply        Human + reason

```

### 1. Intent classification

I defined a small taxonomy from the AmazonHelp conversations rather than using a large generic intent dataset.

The final taxonomy contains 9 intents:

* `delivery_issue`

* `missing_package`

* `order_issue`

* `return_refund`

* `payment_billing`

* `account_access`

* `subscription_service`

* `device_technical`

* `other`

Some classes are deliberately separated even though they look similar.

For example:

```text

"Where is my package? It was supposed to arrive yesterday."

→ delivery_issue

"It says delivered, but I never received it."

→ missing_package

"I received the wrong color."

→ order_issue

"The item is defective and I want my money back."

→ return_refund

```

I also kept an `other` class because the original dataset contains general conversations, unclear messages, and noise.

### 2. Historical retrieval

For reply generation, I built a semantic index from the **training data only**.

The retrieval model is:

```text

paraphrase-multilingual-MiniLM-L12-v2

```

For an incoming customer message, the system retrieves similar historical customer-support examples and uses the associated AmazonHelp replies as evidence.

This is important because I don't want the LLM to invent a completely new support policy. The historical conversations give it examples of how similar cases were handled.

I also remove case-specific URLs, handles, and similar details from generated responses so that information from an old case is not accidentally presented as if it belongs to the current customer.

### 3. Escalation

The system makes an explicit decision:

```text

AUTO_HANDLE

```

or

```text

ESCALATE

```

with a reason.

I use more conservative handling for cases such as:

* account-access problems

* payment/billing issues

* order-specific investigations

* missing packages

* refunds/returns requiring account information

* persistent technical problems

At the same time, I don't automatically escalate a customer just because they are angry.

For example, a message containing strong negative language but no specific support problem can still be auto-handled.

## Golden evaluation set

I created a fixed set of **250 hand-labelled examples**.

Each example was labelled for:

* intent

* escalation decision

* escalation reason

* notes

The golden examples were removed from both the training split and the retrieval index.

This was important to avoid a situation where the agent could retrieve the exact evaluation conversation and appear better than it really is.

### Golden-set distribution

| Intent                 | Count | Percentage |

| ---------------------- | ----: | ---------: |

| `delivery_issue`       |    69 |      27.6% |

| `other`                |    51 |      20.4% |

| `payment_billing`      |    26 |      10.4% |

| `missing_package`      |    22 |       8.8% |

| `return_refund`        |    20 |       8.0% |

| `order_issue`          |    17 |       6.8% |

| `subscription_service` |    17 |       6.8% |

| `device_technical`     |    16 |       6.4% |

| `account_access`       |    12 |       4.8% |

Escalation labels:

* Human escalation: **115/250 = 46.0%**

* Auto-handle: **135/250 = 54.0%**

## Evaluation

I used two baselines before evaluating the AI agent.

### Baseline 1 — Majority class

The majority baseline always predicts `delivery_issue`.

Result:

**69/250 = 27.6% accuracy**

This gives a simple lower bound.

### Baseline 2 — TF-IDF + Logistic Regression

The second baseline is a traditional text classifier using TF-IDF features and Logistic Regression.

Results:

| Metric      |    Result |

| ----------- | --------: |

| Accuracy    | **49.2%** |

| Macro-F1    | **39.5%** |

| Weighted-F1 | **46.5%** |

This baseline worked reasonably well for common intents but struggled with less frequent and closely related classes.

For example, for `missing_package`, it correctly classified only **1 of 22 examples (4.5% recall)**.

### Final AI agent

The final agent achieved:

**81.6% intent accuracy — 204/250 correct**

Compared with:

```text

Majority baseline       27.6%

TF-IDF + Logistic Reg.  49.2%

AI agent                81.6%

```

## Escalation results

The escalation classifier achieved:

| Metric            |              Result |

| ----------------- | ------------------: |

| Accuracy          | **76.8% (192/250)** |

| Precision         | **69.7% (101/145)** |

| Recall            | **87.8% (101/115)** |

| F1                |           **77.0%** |

| False auto-handle |   **5.6% (14/250)** |

| False escalation  |  **17.6% (44/250)** |

The false auto-handle rate is particularly important to me because these are cases where the system decided not to involve a human when the human label said it should.

## What went wrong?

The agent still makes mistakes, and the evaluation helped identify the main ones.

### 1. Too many unnecessary escalations

There were **44 false escalations (17.6%)**.

The system sometimes treats repeated complaints or strong dissatisfaction as a reason to escalate even when the issue could be handled automatically.

### 2. Persistent account issues

One customer had been unable to log in for about a month. The model correctly identified `account_access`, but still decided to auto-handle it.

The problem was not the intent classification. It was the escalation decision.

### 3. Delivery vs. missing package

The model sometimes confused:

```text

package is late

```

with:

```text

package says delivered but was not received

```

These sound similar but require different support handling.

### 4. Order vs. delivery

Some order-specific problems were classified as delivery issues because the customer was waiting for an item.

For example, one customer had an order that had not been dispatched and also had problems with the status link. The human label was `order_issue`, but the agent predicted `delivery_issue`.

### 5. Payment vs. refund

Words such as "money", "charge", "refund", and "return" frequently appear together.

One customer complained about a delivery problem, money being taken, and wanting the money back. The model predicted `payment_billing`, while the human label was `return_refund`.

These failures suggest that the next improvement should focus on understanding the **reason behind the request**, rather than simply matching important keywords.

## LLM-as-judge

I also built an LLM-based judge for generated replies.

It evaluates:

* groundedness

* helpfulness

* safety

on a 1–5 scale.

I did not assume that an LLM judge is automatically reliable. I compared the judge against human ratings on **30 examples**.

Results:

| Dimension    | Within ±1 | Correlation |

| ------------ | --------: | ----------: |

| Groundedness |     76.7% |      -0.188 |

| Helpfulness  |     76.7% |      -0.257 |

| Safety       |     56.7% |       0.095 |

The results were not strong enough to claim that the judge is validated.

So I treat the LLM judge as **secondary evidence**, not the main evaluation metric.

## What is misleading about my headline number?

The headline result is **81.6% intent accuracy**.

It is a useful number, but it should not be interpreted as "the agent is 81.6% ready for production."

There are several reasons.

First, the evaluation set contains only **250 examples**.

Second, the classes are not balanced. For example, `delivery_issue` represents **27.6%** of the golden set, while `account_access` represents only **4.8%**.

Third, intent accuracy does not tell us whether the generated reply is useful or safe.

Finally, an incorrect escalation decision can be more important than an incorrect intent label. A system could identify an account problem correctly but still make the wrong decision to auto-handle it.

That is why I report the intent result together with escalation metrics, judge calibration, and real failure examples.

## Key decisions

1. **AmazonHelp** was selected because it had enough support activity for meaningful retrieval.

2. I reconstructed **multi-turn threads** rather than treating tweets independently.

3. I defined **9 intents** from the actual dataset.

4. I kept `other` because the source data contains noise.

5. I labelled **250 golden examples** manually.

6. Golden examples were removed from training and retrieval.

7. I used a **majority baseline** to establish a simple lower bound.

8. I used **TF-IDF + Logistic Regression** as the simple non-LLM baseline.

9. I used Macro-F1 for the baseline because the intent classes are imbalanced.

10. I added semantic retrieval to ground generated replies in historical cases.

11. I used multilingual embeddings because the source contains multiple languages.

12. I added deterministic escalation guardrails after observing inconsistent LLM decisions.

13. I removed URLs and handles from generated replies to reduce case-specific leakage.

14. I made evaluation checkpointed/resumable because API calls can fail or hit rate limits.

15. I compared the LLM judge with human ratings instead of assuming it was reliable.

## What I would do with one more week

If I had another week, I would focus on the errors found during evaluation.

### Better retrieval

Combine semantic retrieval with BM25/keyword retrieval and rerank the results.

### Conversation-aware retrieval

Use the recent conversation context instead of only the latest customer message.

### Better escalation calibration

Build a dedicated escalation classifier or calibrated threshold so that false auto-handles are reduced without creating too many unnecessary escalations.

### More difficult evaluation examples

Expand the golden set with more examples around:

* delivery vs. missing package

* payment vs. refund

* order vs. delivery

* multilingual conversations

* persistent technical problems

* angry but straightforward customers

### Production measurements

I would also measure:

* latency

* API cost

* retrieval quality

* escalation rate

* false auto-handle rate

* human override rate

That would tell me whether the system is useful in an actual support workflow.

## Project structure

```text

.

├── data/

│   ├── raw/

│   ├── processed/

│   └── golden/

│

├── src/

│   ├── agent.py

│   ├── semantic_retrieval.py

│   └── data/

│

├── scripts/

│   ├── evaluate_agent.py

│   ├── llm_judge.py

│   ├── create_judge_calibration.py

│   └── rate_calibration.py

│

├── requirements.txt

├── .env.example

└── README.md

```

## Notes

The project uses the Customer Support on Twitter dataset as its primary source.

The system is intended as a prototype for support decision-making. It does **not** perform authenticated Twitter actions, issue refunds, access customer accounts, or make transactions.

The evaluation results are based on the fixed 250-example golden set described above. The purpose of the project is to demonstrate the complete workflow—from messy support data to an evaluated AI system—and to be transparent about where the system still needs improvement.