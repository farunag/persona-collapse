Read CLAUDE.md, then read /workspace/assistant-axis/notebooks/project_transcript.ipynb,
the assistant_axis library source, and /workspace/assistant-axis/transcripts/README.md.

Goal: pass Gate A — confirm our setup reproduces the paper's persona-drift measurements.

1. Write scripts/gate_a.py that, for Qwen/Qwen3-32B (target layer 32, published axis
   from lu-christina/assistant-axis-vectors, qwen-3-32b folder):
   a. Loads each Qwen 3 32B transcript in /workspace/assistant-axis/transcripts/
      (the four domain files coding/writing/therapy/philosophy, plus the case studies),
   b. Recomputes the per-turn projection onto the axis using the same method as the
      notebook (mean over each assistant turn's response tokens),
   c. Compares our projections with the "projections" field stored in each file
      (report max absolute difference and correlation per file),
   d. Saves results/gate_a_qwen.json and a plot results/gate_a_qwen.png with one line
      per domain (projection vs turn), coding/writing vs therapy/philosophy clearly visible.
2. Test on ONE transcript first and show me the numbers before running all of them.
3. Then repeat for google/gemma-2-27b-it (layer 22) on any Gemma transcripts that exist.
   If there are none, run the four domain conversations' user turns through Gemma
   (replay: fixed user turns, fresh Gemma replies), project, and plot — note in the
   output that these are new replies, not the paper's.
4. Commit and push. Then tell me plainly: does our recomputation match the stored
   projections, and do therapy/philosophy sit lower than coding/writing?

Gate A passes if (Qwen) our projections closely match the stored ones and the domain
ordering matches the paper's Figure 7 in shape.
