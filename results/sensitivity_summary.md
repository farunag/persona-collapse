# Axis sensitivity check

Single-turn replies to 12 of our user messages under 3 default prompts, 3 Assistant-like roles and
10 far roles (English role instructions; user message in the language shown). Score 1 = default
Assistant vector, 0 = mean role vector. Separation = default minus far, 95% CI over messages.

| Model | Language | Default | Near roles | Far roles | Separation (95% CI) | vs English | Replies in user's language |
|---|---|---|---|---|---|---|---|
| Qwen3-32B | en | 0.49 | 0.48 | -0.89 | 1.37 [1.16, 1.57] | 1.0 | 99% |
| Qwen3-32B | ha | 0.31 | 0.37 | -0.09 | 0.40 [0.33, 0.47] | 0.29 | 86% |
| Qwen3-32B | yo | 0.30 | 0.44 | -0.10 | 0.41 [0.33, 0.48] | 0.3 | 81% |
| gemma-2-27b-it | en | 0.95 | 0.93 | -0.38 | 1.33 [1.10, 1.57] | 1.0 | 99% |
| gemma-2-27b-it | ha | -2.48 | -1.56 | -1.54 | -0.94 [-1.67, -0.24] | -0.71 | 50% |
| gemma-2-27b-it | yo | -2.56 | -0.32 | -1.32 | -1.24 [-2.00, -0.39] | -0.93 | 31% |
