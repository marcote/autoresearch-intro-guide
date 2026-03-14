"""
run.py — The AutoResearch Orchestrator

This is the loop that drives autonomous experimentation:
1. Read program.md (research instructions) and train.py (current code)
2. Ask the LLM to propose a modification to train.py
3. Apply the modification and commit
4. Run the experiment (python train.py)
5. Parse results (val_profit)
6. If improved: keep the commit. If not: git reset.
7. Log results to results.tsv
8. Repeat

Usage:
    python run.py                    # run the loop
    python run.py --max-steps 5      # limit number of experiments
    python run.py --dry-run          # show what would happen without running
"""

import subprocess
import re
import os
import sys
import argparse
import datetime

# ---- Configuration ----
RESULTS_FILE = "results.tsv"
TRAIN_FILE = "train.py"
PROGRAM_FILE = "program.md"
PREPARE_FILE = "prepare.py"

def read_file(path):
    with open(path) as f:
        return f.read()

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def git(*args):
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
    )
    return result.stdout.strip(), result.returncode

def get_commit_hash():
    out, _ = git("rev-parse", "--short", "HEAD")
    return out[:7]

def run_experiment():
    """Run train.py and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_FILE],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(__file__) or "."
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT: experiment exceeded 30s", 1

def parse_val_profit(output):
    """Extract val_profit from experiment output."""
    match = re.search(r"val_profit=([-\d.]+)", output)
    if match:
        return float(match.group(1))
    return None

def init_results():
    """Initialize results.tsv if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("step\tcommit\tval_profit\tstatus\tdescription\ttimestamp\n")

def log_result(step, commit, val_profit, status, description):
    """Append a result to results.tsv."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(RESULTS_FILE, "a") as f:
        profit_str = f"{val_profit:.2f}" if val_profit is not None else "N/A"
        f.write(f"{step}\t{commit}\t{profit_str}\t{status}\t{description}\t{timestamp}\n")

def call_llm(messages):
    """
    Call the LLM to get a code modification.

    Uses the Anthropic API. Set ANTHROPIC_API_KEY env var.
    Falls back to a simple heuristic if no API key is available.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [!] No ANTHROPIC_API_KEY found. Using demo mode (random perturbation).")
        return demo_mode_modify()

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=messages,
        )
        return response.content[0].text
    except ImportError:
        print("  [!] anthropic package not installed. Using demo mode.")
        return demo_mode_modify()
    except Exception as e:
        print(f"  [!] LLM call failed: {e}. Using demo mode.")
        return demo_mode_modify()

def demo_mode_modify():
    """
    Fallback when no LLM is available.
    Makes random but sensible modifications to train.py for demonstration.
    """
    import random
    current = read_file(TRAIN_FILE)

    modifications = [
        # Try different hidden sizes
        (r"HIDDEN_SIZE = \d+", f"HIDDEN_SIZE = {random.choice([16, 32, 48, 64, 96, 128])}"),
        # Try different layer counts
        (r"NUM_LAYERS = \d+", f"NUM_LAYERS = {random.choice([1, 2, 3, 4])}"),
        # Try different learning rates
        (r"LEARNING_RATE = [\d.]+", f"LEARNING_RATE = {random.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])}"),
        # Try different batch sizes
        (r"BATCH_SIZE = \d+", f"BATCH_SIZE = {random.choice([8, 16, 32, 64, 128])}"),
        # Try different activations
        (r'ACTIVATION = "\w+"', f'ACTIVATION = "{random.choice(["tanh", "relu", "sigmoid"])}"'),
    ]

    # Pick a random modification
    pattern, replacement = random.choice(modifications)
    old_match = re.search(pattern, current)
    old_value = old_match.group(0) if old_match else "unknown"
    new_code = re.sub(pattern, replacement, current)

    description = f"Changed {old_value} -> {replacement}"

    return f"DESCRIPTION: {description}\n```python\n{new_code}\n```"

def extract_code_and_description(llm_response):
    """Parse the LLM response to get new train.py code and description."""
    # Extract description
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", llm_response)
    description = desc_match.group(1).strip() if desc_match else "LLM modification"

    # Extract code block
    code_match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip(), description

    # If no code block, try to find the full file content
    if "import numpy" in llm_response and "prepare.evaluate" in llm_response:
        return llm_response.strip(), description

    return None, description

def build_prompt(train_code, program_md, prepare_code, results_history, best_profit):
    """Build the prompt for the LLM agent."""
    system = """You are an autonomous ML research agent. Your job is to modify train.py
to maximize val_profit (percentage return from trading SPY based on model predictions).

You must respond with:
1. A line starting with DESCRIPTION: explaining your change in <10 words
2. The complete modified train.py wrapped in ```python ... ```

Rules:
- Output the COMPLETE file (not a diff)
- Only change training logic, hyperparameters, or architecture
- Keep the if __name__ == "__main__" block and prepare.evaluate() call
- Only use numpy (no other packages)
- Be creative but make targeted changes
- The model receives 10 features and outputs a probability (0-1) via sigmoid
- Higher val_profit = better (we want to MAXIMIZE profit)"""

    results_context = ""
    if results_history:
        results_context = f"\n\nPrevious experiment results:\n{results_history}\n\nCurrent best val_profit: {best_profit:.2f}%"

    user_msg = f"""## Research Program
{program_md}

## Current train.py
```python
{train_code}
```

## prepare.py (READ ONLY — do not modify)
```python
{prepare_code}
```
{results_context}

Now propose your next modification to train.py. Remember:
- DESCRIPTION: <your change in <10 words>
- Then the complete modified train.py in a python code block"""

    return [
        {"role": "user", "content": f"{system}\n\n{user_msg}"}
    ]

# ---- Main Loop ----
def main():
    parser = argparse.ArgumentParser(description="AutoResearch Orchestrator")
    parser.add_argument("--max-steps", type=int, default=10, help="Max experiments to run")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    print("=" * 60)
    print("  AutoResearch — Autonomous Experiment Loop")
    print("=" * 60)
    print(f"  Max steps: {args.max_steps}")
    print(f"  Train file: {TRAIN_FILE}")
    print(f"  Results: {RESULTS_FILE}")
    print()

    # ---- Clean Slate ----
    baseline_file = "train_baseline.py"
    if os.path.exists(baseline_file):
        print("  Restoring train.py to baseline...")
        write_file(TRAIN_FILE, read_file(baseline_file))

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    init_results()

    # Check git
    _, rc = git("status")
    if rc != 0:
        print("Initializing git repository...")
        git("init")
    git("add", ".")
    git("commit", "-m", "clean slate: reset for new experiment run")

    # Run baseline
    print("[Step 0] Running baseline...")
    output, rc = run_experiment()
    if rc != 0:
        print(f"  Baseline failed! Output:\n{output}")
        return

    best_profit = parse_val_profit(output)
    if best_profit is None:
        print(f"  Could not parse val_profit from output:\n{output}")
        return

    print(f"  Baseline val_profit = {best_profit:.2f}%")
    log_result(0, get_commit_hash(), best_profit, "baseline", "Initial baseline")
    print()

    if args.dry_run:
        print("[Dry run] Would start experiment loop here.")
        return

    # Read fixed files
    program_md = read_file(PROGRAM_FILE)
    prepare_code = read_file(PREPARE_FILE)

    # Experiment loop
    for step in range(1, args.max_steps + 1):
        print(f"[Step {step}/{args.max_steps}] " + "-" * 40)

        # Read current state
        train_code = read_file(TRAIN_FILE)
        results_history = read_file(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else ""

        # Ask LLM for modification
        print("  Asking LLM for next experiment...")
        messages = build_prompt(train_code, program_md, prepare_code, results_history, best_profit)
        llm_response = call_llm(messages)

        # Parse response
        new_code, description = extract_code_and_description(llm_response)
        if new_code is None:
            print("  Could not extract code from LLM response. Skipping.")
            log_result(step, "N/A", None, "skip", "Failed to parse LLM response")
            continue

        print(f"  Hypothesis: {description}")

        # Save original for rollback
        original_code = train_code

        # Apply modification
        write_file(TRAIN_FILE, new_code)
        git("add", TRAIN_FILE)
        git("commit", "-m", f"experiment {step}: {description}")
        commit_hash = get_commit_hash()

        # Run experiment
        print("  Running experiment...")
        output, rc = run_experiment()

        if rc != 0:
            # Experiment crashed
            print(f"  CRASHED! Rolling back.")
            print(f"  Error: {output[-200:]}")
            write_file(TRAIN_FILE, original_code)
            git("add", TRAIN_FILE)
            git("commit", "-m", f"revert experiment {step}: crashed")
            log_result(step, commit_hash, None, "crash", description)

        else:
            val_profit = parse_val_profit(output)
            if val_profit is None:
                print("  Could not parse val_profit. Rolling back.")
                write_file(TRAIN_FILE, original_code)
                git("add", TRAIN_FILE)
                git("commit", "-m", f"revert experiment {step}: no metric")
                log_result(step, commit_hash, None, "crash", description)

            elif val_profit > best_profit:
                # Improvement! Keep it. (higher profit is better)
                improvement = val_profit - best_profit
                print(f"  val_profit = {val_profit:.2f}% (improved by +{improvement:.2f}%) ✓ KEEP")
                best_profit = val_profit
                log_result(step, commit_hash, val_profit, "keep", description)

            else:
                # No improvement. Revert.
                print(f"  val_profit = {val_profit:.2f}% (best: {best_profit:.2f}%) ✗ DISCARD")
                write_file(TRAIN_FILE, original_code)
                git("add", TRAIN_FILE)
                git("commit", "-m", f"revert experiment {step}: no improvement")
                log_result(step, commit_hash, val_profit, "discard", description)

        print()

    # Summary
    print("=" * 60)
    print("  Experiment Summary")
    print("=" * 60)
    print(f"  Best val_profit: {best_profit:.2f}%")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"  Final train.py reflects the best configuration found.")
    print()

if __name__ == "__main__":
    main()
