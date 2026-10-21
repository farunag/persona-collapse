#!/bin/bash
# Logs start/end times of block 3 runs (a) and (b). Read-only: only watches logs and processes.
T=/workspace/outputs/logs/block3_times.log
UL=/workspace/outputs/logs/full_uncapped.log
CL=/workspace/outputs/logs/full_capped.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "(a) uncapped vLLM start: $(date -u -d "$(ps -o lstart= -p 8572)" +%Y-%m-%dT%H:%M:%SZ) (pid 8572)" >> $T
while kill -0 8572 2>/dev/null; do sleep 30; done
echo "(a) uncapped vLLM end:   $(ts); $(grep -E '^Done:' $UL || echo 'no Done line: FAILED, see full_uncapped.log')" >> $T
for i in $(seq 1 20); do [ -s "$CL" ] && break; sleep 3; done
if [ -s "$CL" ]; then
  echo "(b) capped HF start:     $(ts)" >> $T
  until grep -q "block3 exit" $UL; do sleep 30; done
  echo "(b) capped HF end:       $(ts); $(grep -E '^Done:' $CL || echo 'no Done line: FAILED, see full_capped.log'); $(grep 'block3 exit' $UL)" >> $T
else
  echo "(b) did not start (run (a) failed): $(grep 'block3 exit' $UL)" >> $T
fi
