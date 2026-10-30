"""Run the whole project from scratch:  python run_all.py   (about 10 minutes on 2 cores).

1. unit + randomised equivalence tests
2. dataset for every rule budget 1..8 (portfolio, labels, exhaustive oracle)
3. model comparison with cross-validation for every budget
4. evaluation on the held-out test programs, charts, tables
"""
import subprocess
import sys

STEPS = [[sys.executable, "-m", "unittest", "discover", "-s", "tests"]]
for b in range(1, 9):
    STEPS.append([sys.executable, "dataset.py", "--budget", str(b)])
for b in range(1, 9):
    STEPS.append([sys.executable, "train.py", "--budget", str(b)])
STEPS.append([sys.executable, "evaluate.py"])

for step in STEPS:
    print(">>", " ".join(step[1:]), flush=True)
    subprocess.run(step, check=True)
print("finished - see results/")
