# Role adoption (LLM judge) and axis scores

Judge: gemini-3.5-flash-lite. Far roles = 10 role-play-end roles; default = 3 neutral prompts.

| Model | Language | Far: full | Far: somewhat | Far: none | Score, default | Score, far judged full (95% CI) | Score, far judged none | n far |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | en | 96% | 4% | 0% | 0.49 | -0.88 [-0.97, -0.79] (n=115) |  | 120 |
| Qwen3-32B | ha | 34% | 18% | 48% | 0.31 | -0.35 [-0.51, -0.20] (n=41) | 0.05 | 120 |
| Qwen3-32B | yo | 54% | 12% | 34% | 0.30 | -0.27 [-0.42, -0.14] (n=65) | 0.08 | 120 |
| gemma-2-27b-it | en | 92% | 7% | 1% | 0.95 | -0.44 [-0.55, -0.34] (n=111) | 0.93 | 120 |
| gemma-2-27b-it | ha | 89% | 8% | 3% | -2.48 | -1.49 [-1.81, -1.20] (n=107) | -0.14 | 120 |
| gemma-2-27b-it | yo | 92% | 5% | 3% | -2.56 | -1.11 [-1.36, -0.89] (n=110) | -3.88 | 120 |
