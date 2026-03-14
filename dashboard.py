"""
dashboard.py — Visual TUI for AutoResearch (SPY Trading Edition)

Shows a live dashboard with:
- ASCII price chart with buy/sell signals
- Profit history chart
- Color-coded experiment log
- What's being sent to the LLM (prompt summary)
- What the LLM changed (code diff)
- LLM reasoning

Usage:
    uv run python dashboard.py                  # run with LLM (needs ANTHROPIC_API_KEY)
    uv run python dashboard.py --max-steps 10   # limit experiments
    uv run python dashboard.py --demo           # demo mode (no API key needed)
"""

import subprocess
import re
import os
import sys
import argparse
import datetime
import time
import difflib
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich import box

console = Console()

# ---- Configuration ----
RESULTS_FILE = "results.tsv"
TRAIN_FILE = "train.py"
BASELINE_FILE = "train_baseline.py"
PROGRAM_FILE = "program.md"
PREPARE_FILE = "prepare.py"

# ---- State ----
class ExperimentState:
    def __init__(self):
        self.results = []
        self.best_profit = None
        self.current_step = 0
        self.max_steps = 0
        self.status = "Initializing..."
        self.current_hypothesis = ""
        self.trading_data = None    # (prices_val, signals, accuracy, capital)
        self.profit_history = []    # list of (step, profit, status)
        # LLM transparency
        self.prompt_summary = ""
        self.llm_reasoning = ""
        self.code_diff = ""
        self.phase = ""
        self.using_llm = False
        # Extra trading info
        self.accuracy = 0.0
        self.trades = 0
        self.final_capital = 0.0

state = ExperimentState()

# ---- ASCII Price Chart with Buy/Sell Signals ----
def make_price_chart(prices_val, signals, width=58, height=15):
    """Create an ASCII chart of SPY prices with buy/sell markers."""
    if prices_val is None or len(prices_val) < 2:
        return Text("  Waiting for data...", style="dim")

    pf = prices_val.flatten() if hasattr(prices_val, 'flatten') else np.array(prices_val)
    sf = signals.flatten() if hasattr(signals, 'flatten') else np.array(signals)

    p_min, p_max = float(pf.min()) * 0.998, float(pf.max()) * 1.002
    n = len(pf)

    canvas = [[" " for _ in range(width)] for _ in range(height)]

    def to_canvas(x_idx, price):
        cx = int(x_idx / max(n - 1, 1) * (width - 1))
        cy = int((1 - (price - p_min) / max(p_max - p_min, 0.01)) * (height - 1))
        return max(0, min(width - 1, cx)), max(0, min(height - 1, cy))

    # Draw price line
    step = max(1, n // width)
    prev_cx, prev_cy = to_canvas(0, float(pf[0]))
    for i in range(0, n, step):
        cx, cy = to_canvas(i, float(pf[i]))
        canvas[cy][cx] = "─"

    # Draw buy/sell signals (sample to fit width)
    signal_step = max(1, n // (width // 2))
    for i in range(0, min(n, len(sf)), signal_step):
        cx, cy = to_canvas(i, float(pf[i]))
        if sf[i] > 0.5:
            canvas[cy][cx] = "▲"  # buy
        else:
            canvas[cy][cx] = "▼"  # sell/out

    result = Text()
    for row_idx, row in enumerate(canvas):
        if row_idx == 0:
            label = f"${p_max:>6.0f}│"
        elif row_idx == height - 1:
            label = f"${p_min:>6.0f}│"
        elif row_idx == height // 2:
            mid = (p_max + p_min) / 2
            label = f"${mid:>6.0f}│"
        else:
            label = "       │"
        result.append(label, style="dim")
        for ch in row:
            if ch == "▲":
                result.append(ch, style="bold green")
            elif ch == "▼":
                result.append(ch, style="bold red")
            elif ch == "─":
                result.append(ch, style="bold white")
            else:
                result.append(ch)
        result.append("\n")

    result.append(f"       └{'─' * width}\n", style="dim")
    result.append(f"        Day 1", style="dim")
    result.append(" " * (width - 16))
    result.append(f"Day {n}\n", style="dim")
    result.append("       ", style="dim")
    result.append("─ ", style="bold white")
    result.append("SPY price  ", style="dim")
    result.append("▲ ", style="bold green")
    result.append("BUY signal  ", style="dim")
    result.append("▼ ", style="bold red")
    result.append("OUT/SELL", style="dim")

    return result

# ---- Profit History Chart ----
def make_profit_chart(profit_history, width=58):
    """Create a bar chart of profit over experiments."""
    if not profit_history:
        return Text("  No experiments yet...", style="dim")

    result = Text()
    valid = [(s, p, st) for s, p, st in profit_history if p is not None]
    if not valid:
        return Text("  No valid results yet...", style="dim")

    profit_values = [p for _, p, _ in valid]
    max_abs = max(abs(p) for p in profit_values) if profit_values else 1.0
    if max_abs == 0:
        max_abs = 1.0

    entries = profit_history[-15:]

    for step, profit, status in entries:
        if profit is None:
            result.append(f"  {step:>3} │ ", style="dim")
            result.append("  CRASH\n", style="bold red")
            continue

        bar_len = max(1, int(abs(profit) / max_abs * (width - 25)))
        is_positive = profit >= 0

        if status == "keep":
            style = "bold green"
            marker = "█"
        elif status == "baseline":
            style = "bold cyan"
            marker = "█"
        elif status == "discard":
            style = "yellow"
            marker = "░"
        else:
            style = "red"
            marker = "░"

        sign = "+" if is_positive else "-"
        result.append(f"  {step:>3} │ ", style="dim")
        if not is_positive:
            result.append(marker * bar_len, style="bold red")
            result.append(f" {sign}{abs(profit):.1f}%", style="bold red")
        else:
            result.append(marker * bar_len, style=style)
            result.append(f" {sign}{abs(profit):.1f}%", style=style)
        if status == "keep":
            result.append(" ✓", style="bold green")
        elif status == "baseline":
            result.append(" ◆", style="bold cyan")
        result.append("\n")

    result.append("\n  ", style="dim")
    result.append("█", style="bold cyan")
    result.append(" base  ", style="dim")
    result.append("█", style="bold green")
    result.append(" keep  ", style="dim")
    result.append("░", style="yellow")
    result.append(" discard  ", style="dim")
    result.append("CRASH", style="bold red")

    return result

# ---- Experiment Table ----
def make_results_table(results, max_rows=10):
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold",
                  expand=True, padding=(0, 1))
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Profit", width=10, justify="right")
    table.add_column("Status", width=9)
    table.add_column("Description", ratio=1)

    for r in results[-max_rows:]:
        if r["profit"] is not None:
            sign = "+" if r["profit"] >= 0 else ""
            profit_str = f"{sign}{r['profit']:.1f}%"
        else:
            profit_str = "N/A"
        status_map = {
            "keep": ("bold green", "✓ KEEP"),
            "baseline": ("bold cyan", "◆ BASE"),
            "discard": ("yellow", "✗ DISC"),
            "crash": ("bold red", "⚠ CRASH"),
        }
        style, status_display = status_map.get(r["status"], ("dim", r["status"]))
        table.add_row(str(r["step"]), profit_str, status_display, r["description"], style=style)

    return table

# ---- Code Diff Display ----
def make_diff_display(diff_text, max_lines=20):
    if not diff_text:
        return Text("  No changes yet...", style="dim")
    result = Text()
    lines = diff_text.split("\n")[:max_lines]
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            result.append(f"  {line}\n", style="bold green")
        elif line.startswith("-") and not line.startswith("---"):
            result.append(f"  {line}\n", style="bold red")
        elif line.startswith("@@"):
            result.append(f"  {line}\n", style="cyan")
        else:
            result.append(f"  {line}\n", style="dim")
    if len(diff_text.split("\n")) > max_lines:
        result.append(f"  ... ({len(diff_text.split(chr(10))) - max_lines} more lines)\n", style="dim")
    return result

# ---- LLM Context / Reasoning Display ----
def make_prompt_summary():
    if not state.prompt_summary:
        return Text("  Waiting for first experiment...\n", style="dim")
    result = Text()
    result.append(state.prompt_summary)
    return result

def make_reasoning_display():
    if not state.llm_reasoning:
        return Text("  Waiting for LLM response...", style="dim")
    result = Text()
    lines = state.llm_reasoning.split("\n")[:12]
    for line in lines:
        result.append(f"  {line}\n", style="italic")
    if len(state.llm_reasoning.split("\n")) > 12:
        result.append("  ...\n", style="dim")
    return result

# ---- Dashboard Layout ----
def make_dashboard():
    # Header
    header_text = Text()
    header_text.append("  AutoResearch SPY Trader", style="bold white")
    header_text.append(f"  │  Step {state.current_step}/{state.max_steps}", style="dim")
    if state.best_profit is not None:
        sign = "+" if state.best_profit >= 0 else ""
        color = "bold green" if state.best_profit >= 0 else "bold red"
        header_text.append(f"  │  Best: {sign}{state.best_profit:.1f}%", style=color)
    if state.accuracy:
        header_text.append(f"  │  Acc: {state.accuracy:.0f}%", style="bold yellow")
    if state.final_capital:
        header_text.append(f"  │  ${state.final_capital:,.0f}", style="bold white")
    mode = "LLM" if state.using_llm else "DEMO"
    header_text.append(f"  │  {mode}", style="bold magenta")
    header_text.append(f"  │  {state.status}", style="bold cyan")
    header = Panel(header_text, style="blue", height=3)

    # Phase indicator
    phase_text = Text()
    phases = ["READ", "PROMPT", "LLM", "APPLY", "RUN", "EVALUATE", "DECIDE"]
    for p in phases:
        if p == state.phase:
            phase_text.append(f" ▶ {p} ", style="bold white on blue")
        else:
            phase_text.append(f"   {p} ", style="dim")
        if p != phases[-1]:
            phase_text.append("→", style="dim")
    phase_bar = Panel(phase_text, style="blue", height=3, title="[bold]Loop Phase[/bold]")

    # Price chart
    if state.trading_data is not None:
        prices_val, signals, _, _ = state.trading_data
        chart_content = make_price_chart(prices_val, signals)
    else:
        chart_content = Text("  Waiting for first experiment...", style="dim")
    chart_panel = Panel(chart_content,
                        title="[bold]SPY Validation Period — Buy/Sell Signals[/bold]",
                        border_style="green")

    # Profit chart
    profit_panel = Panel(make_profit_chart(state.profit_history),
                         title="[bold]Profit History (%)[/bold]",
                         border_style="yellow")

    # Results table
    table_content = make_results_table(state.results) if state.results else Text("  No experiments yet...")
    table_panel = Panel(table_content, title="[bold]Experiment Log[/bold]", border_style="cyan")

    # Prompt summary
    prompt_panel = Panel(make_prompt_summary(),
                         title="[bold]What We Sent to the LLM[/bold]",
                         border_style="blue")

    # LLM reasoning
    reasoning_panel = Panel(make_reasoning_display(),
                            title="[bold]LLM Reasoning[/bold]",
                            border_style="magenta")

    # Code diff
    diff_panel = Panel(make_diff_display(state.code_diff),
                       title="[bold]Code Changes (diff)[/bold]",
                       border_style="red")

    # Layout
    layout = Layout()
    layout.split_column(
        Layout(header, name="header", size=3),
        Layout(phase_bar, name="phase", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(chart_panel, name="chart"),
        Layout(profit_panel, name="profit"),
    )
    layout["right"].split_column(
        Layout(prompt_panel, name="prompt", ratio=1),
        Layout(reasoning_panel, name="reasoning", ratio=1),
        Layout(diff_panel, name="diff", ratio=1),
        Layout(table_panel, name="table", ratio=1),
    )
    return layout

# ---- File/Git Helpers ----
def read_file(path):
    with open(path) as f:
        return f.read()

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def git(*args):
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
    )
    return result.stdout.strip(), result.returncode

def get_commit_hash():
    out, _ = git("rev-parse", "--short", "HEAD")
    return out[:7]

def run_experiment():
    try:
        result = subprocess.run(
            ["uv", "run", "python", TRAIN_FILE],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(__file__) or "."
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 1

def parse_val_profit(output):
    """Parse val_profit from experiment output."""
    match = re.search(r"val_profit=([-\d.]+)", output)
    return float(match.group(1)) if match else None

def parse_val_accuracy(output):
    match = re.search(r"val_accuracy=([\d.]+)", output)
    return float(match.group(1)) if match else 0.0

def parse_val_trades(output):
    match = re.search(r"val_trades=(\d+)", output)
    return int(match.group(1)) if match else 0

def parse_val_capital(output):
    match = re.search(r"val_final_capital=\$([\d.]+)", output)
    return float(match.group(1)) if match else 0.0

def get_trading_signals():
    """Run model and get trading signals for visualization."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", """
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
import importlib.util, prepare as p
p.TIME_BUDGET = 5
spec = importlib.util.spec_from_file_location("train", "train.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
model = train_mod.train()
_, _, x_val, y_val, _, prices_val = p.get_data()
preds = model(x_val)
signals = (preds > 0.5).astype(float)
accuracy = float((signals.flatten() == y_val.flatten()).mean() * 100)
print("SIGNALS:" + json.dumps({
    'prices': prices_val.tolist(),
    'signals': signals.flatten().tolist(),
    'accuracy': accuracy
}))
"""],
            capture_output=True, text=True, timeout=15,
            cwd=os.path.dirname(__file__) or "."
        )
        match = re.search(r"SIGNALS:(.+)", result.stdout)
        if match:
            import json
            d = json.loads(match.group(1))
            return (np.array(d['prices']),
                    np.array(d['signals']),
                    d['accuracy'], None)
    except Exception:
        pass
    return None

def init_results():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("step\tcommit\tval_profit\tstatus\tdescription\n")

def log_result(step, commit, val_profit, status, description):
    with open(RESULTS_FILE, "a") as f:
        p_str = f"{val_profit:.2f}" if val_profit is not None else "N/A"
        f.write(f"{step}\t{commit}\t{p_str}\t{status}\t{description}\n")

def compute_diff(old_code, new_code):
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines,
                                 fromfile="train.py (before)",
                                 tofile="train.py (after)", n=1)
    return "".join(diff)

def build_prompt_summary(train_code, results_history, best_profit):
    result = Text()
    result.append("  Context sent to LLM:\n\n", style="bold")
    result.append("  Files: ", style="dim")
    result.append("program.md", style="bold cyan")
    result.append(" + ", style="dim")
    result.append("train.py", style="bold green")
    result.append(" + ", style="dim")
    result.append("prepare.py", style="bold yellow")
    result.append("\n")

    result.append("\n  Current hyperparams:\n", style="bold")
    for pattern, label in [
        (r"HIDDEN_SIZE = (\d+)", "Hidden size"),
        (r"NUM_LAYERS = (\d+)", "Layers"),
        (r"LEARNING_RATE = ([\d.]+)", "Learning rate"),
        (r"BATCH_SIZE = (\d+)", "Batch size"),
        (r'ACTIVATION = "(\w+)"', "Activation"),
    ]:
        match = re.search(pattern, train_code)
        if match:
            result.append(f"    {label}: ", style="dim")
            result.append(f"{match.group(1)}\n", style="bold white")

    if best_profit is not None:
        sign = "+" if best_profit >= 0 else ""
        result.append(f"\n  Best profit so far: ", style="dim")
        result.append(f"{sign}{best_profit:.1f}%\n", style="bold green")

    if results_history:
        lines = results_history.strip().split("\n")
        n_total = len(lines) - 1
        result.append(f"  Past experiments: ", style="dim")
        result.append(f"{n_total}\n", style="bold white")

    result.append("\n  Goal: ", style="dim")
    result.append('"Maximize val_profit (% return on SPY trades)"\n', style="italic cyan")
    return result

def extract_reasoning(llm_response):
    reasoning = re.sub(r"```python.*?```", "", llm_response, flags=re.DOTALL)
    reasoning = re.sub(r"DESCRIPTION:.*?\n", "", reasoning)
    reasoning = reasoning.strip()
    lines = reasoning.split("\n")
    if len(lines) > 15:
        reasoning = "\n".join(lines[:15]) + "\n..."
    return reasoning if reasoning else "(No additional reasoning provided)"

# ---- LLM ----
def call_llm(messages):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        state.using_llm = False
        return demo_mode_modify()
    try:
        import anthropic
        state.using_llm = True
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=messages,
        )
        return response.content[0].text
    except ImportError:
        state.using_llm = False
        return demo_mode_modify()
    except Exception as e:
        state.using_llm = False
        state.llm_reasoning = f"LLM error: {e}\nFalling back to demo mode."
        return demo_mode_modify()

def demo_mode_modify():
    import random
    current = read_file(TRAIN_FILE)
    modifications = [
        (r"HIDDEN_SIZE = \d+", f"HIDDEN_SIZE = {random.choice([8, 16, 32, 48, 64])}"),
        (r"NUM_LAYERS = \d+", f"NUM_LAYERS = {random.choice([1, 2, 3])}"),
        (r"LEARNING_RATE = [\d.]+", f"LEARNING_RATE = {random.choice([0.001, 0.005, 0.01, 0.02, 0.05])}"),
        (r"BATCH_SIZE = \d+", f"BATCH_SIZE = {random.choice([8, 16, 32, 64])}"),
        (r'ACTIVATION = "\w+"', f'ACTIVATION = "{random.choice(["tanh", "relu", "sigmoid"])}"'),
    ]
    pattern, replacement = random.choice(modifications)
    old_match = re.search(pattern, current)
    old_value = old_match.group(0) if old_match else "unknown"
    new_code = re.sub(pattern, replacement, current)
    description = f"Changed {old_value} -> {replacement}"
    reasoning = f"Demo mode: randomly trying {replacement}.\nNo LLM reasoning — random perturbation to demonstrate the loop."
    return f"DESCRIPTION: {description}\n\n{reasoning}\n\n```python\n{new_code}\n```"

def extract_code_and_description(llm_response):
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", llm_response)
    description = desc_match.group(1).strip() if desc_match else "LLM modification"
    code_match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip(), description
    if "import numpy" in llm_response and "prepare.evaluate" in llm_response:
        return llm_response.strip(), description
    return None, description

def build_prompt(train_code, program_md, prepare_code, results_history, best_profit):
    system = """You are an autonomous ML research agent. Your job is to modify train.py
to maximize val_profit (percentage return from trading SPY based on model predictions).

You must respond with:
1. A line starting with DESCRIPTION: explaining your change in <10 words
2. A brief explanation of WHY you're making this change (2-3 sentences)
3. The complete modified train.py wrapped in ```python ... ```

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

## prepare.py (READ ONLY)
```python
{prepare_code}
```
{results_context}

Now propose your next modification to train.py. Remember:
- DESCRIPTION: <your change in <10 words>
- Brief explanation of your reasoning
- Then the complete modified train.py in a python code block"""

    return [{"role": "user", "content": f"{system}\n\n{user_msg}"}]

# ---- Main Loop ----
def main():
    parser = argparse.ArgumentParser(description="AutoResearch SPY Trading Dashboard")
    parser.add_argument("--max-steps", type=int, default=10, help="Max experiments")
    parser.add_argument("--demo", action="store_true", help="Force demo mode (no LLM)")
    args = parser.parse_args()

    if args.demo and "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    state.max_steps = args.max_steps
    state.using_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ---- Clean Slate ----
    # Restore train.py to baseline
    if os.path.exists(BASELINE_FILE):
        baseline = read_file(BASELINE_FILE)
        write_file(TRAIN_FILE, baseline)

    # Remove previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    init_results()

    # Reset git to clean state
    _, rc = git("status")
    if rc != 0:
        git("init")
    git("add", ".")
    git("commit", "-m", "clean slate: reset for new experiment run")

    with Live(make_dashboard(), console=console, refresh_per_second=2, screen=True) as live:

        # ---- Baseline ----
        state.phase = "RUN"
        state.status = "Running baseline..."
        live.update(make_dashboard())

        output, rc = run_experiment()
        if rc != 0:
            state.status = f"Baseline FAILED! {output[-200:]}"
            live.update(make_dashboard())
            time.sleep(5)
            return

        best_profit = parse_val_profit(output)
        if best_profit is None:
            state.status = f"Could not parse val_profit! Output: {output[:200]}"
            live.update(make_dashboard())
            time.sleep(5)
            return

        state.best_profit = best_profit
        state.accuracy = parse_val_accuracy(output)
        state.trades = parse_val_trades(output)
        state.final_capital = parse_val_capital(output)
        state.phase = "EVALUATE"
        state.results.append({"step": 0, "profit": best_profit, "status": "baseline",
                              "description": f"Initial baseline ({state.accuracy:.0f}% acc)"})
        state.profit_history.append((0, best_profit, "baseline"))
        log_result(0, get_commit_hash(), best_profit, "baseline", "Initial baseline")

        # Get trading signals for plot
        state.status = "Getting trading signals for chart..."
        live.update(make_dashboard())
        signals = get_trading_signals()
        if signals:
            state.trading_data = signals

        train_code = read_file(TRAIN_FILE)
        state.prompt_summary = build_prompt_summary(train_code, "", best_profit)
        live.update(make_dashboard())

        program_md = read_file(PROGRAM_FILE)
        prepare_code = read_file(PREPARE_FILE)

        # ---- Experiment Loop ----
        for step in range(1, args.max_steps + 1):
            state.current_step = step

            state.phase = "READ"
            state.status = "Reading current code..."
            live.update(make_dashboard())
            train_code = read_file(TRAIN_FILE)
            results_history = read_file(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else ""
            time.sleep(0.3)

            state.phase = "PROMPT"
            state.status = "Building prompt for LLM..."
            state.prompt_summary = build_prompt_summary(train_code, results_history, best_profit)
            live.update(make_dashboard())
            messages = build_prompt(train_code, program_md, prepare_code, results_history, best_profit)
            time.sleep(0.3)

            state.phase = "LLM"
            state.status = "Waiting for LLM response..."
            state.llm_reasoning = "Thinking..."
            state.code_diff = ""
            live.update(make_dashboard())

            llm_response = call_llm(messages)
            state.llm_reasoning = extract_reasoning(llm_response)
            live.update(make_dashboard())

            new_code, description = extract_code_and_description(llm_response)
            if new_code is None:
                state.current_hypothesis = "Failed to parse LLM response"
                state.llm_reasoning = f"Could not extract code.\n\nPreview:\n{llm_response[:200]}..."
                state.results.append({"step": step, "profit": None, "status": "crash", "description": "Parse error"})
                state.profit_history.append((step, None, "crash"))
                live.update(make_dashboard())
                time.sleep(1)
                continue

            state.current_hypothesis = description

            state.phase = "APPLY"
            state.status = f"Applying: {description}"
            state.code_diff = compute_diff(train_code, new_code)
            live.update(make_dashboard())
            time.sleep(0.5)

            original_code = train_code
            write_file(TRAIN_FILE, new_code)
            git("add", TRAIN_FILE)
            git("commit", "-m", f"experiment {step}: {description}")
            commit_hash = get_commit_hash()

            state.phase = "RUN"
            state.status = f"Running experiment {step}..."
            live.update(make_dashboard())

            output, rc = run_experiment()
            state.phase = "EVALUATE"

            if rc != 0:
                state.status = "CRASHED!"
                state.results.append({"step": step, "profit": None, "status": "crash", "description": description})
                state.profit_history.append((step, None, "crash"))
                live.update(make_dashboard())
                time.sleep(0.5)
                state.phase = "DECIDE"
                state.status = "Reverting crashed experiment"
                write_file(TRAIN_FILE, original_code)
                git("add", TRAIN_FILE)
                git("commit", "-m", f"revert experiment {step}: crashed")
                log_result(step, commit_hash, None, "crash", description)
            else:
                val_profit = parse_val_profit(output)
                if val_profit is None:
                    state.phase = "DECIDE"
                    state.status = "No metric — reverting"
                    state.results.append({"step": step, "profit": None, "status": "crash", "description": description})
                    state.profit_history.append((step, None, "crash"))
                    write_file(TRAIN_FILE, original_code)
                    git("add", TRAIN_FILE)
                    git("commit", "-m", f"revert experiment {step}: no metric")
                    log_result(step, commit_hash, None, "crash", description)

                elif val_profit > best_profit:
                    # KEEP — profit improved! (higher is better)
                    state.phase = "DECIDE"
                    improvement = val_profit - best_profit
                    state.status = f"KEEP! Profit up +{improvement:.1f}%"
                    best_profit = val_profit
                    state.best_profit = best_profit
                    state.accuracy = parse_val_accuracy(output)
                    state.trades = parse_val_trades(output)
                    state.final_capital = parse_val_capital(output)
                    acc = state.accuracy
                    state.results.append({"step": step, "profit": val_profit, "status": "keep",
                                          "description": f"{description} ({acc:.0f}% acc)"})
                    state.profit_history.append((step, val_profit, "keep"))
                    log_result(step, commit_hash, val_profit, "keep", description)

                    signals = get_trading_signals()
                    if signals:
                        state.trading_data = signals
                else:
                    # DISCARD — no improvement
                    state.phase = "DECIDE"
                    sign = "+" if val_profit >= 0 else ""
                    state.status = f"DISCARD ({sign}{val_profit:.1f}% <= +{best_profit:.1f}%)"
                    acc = parse_val_accuracy(output)
                    state.results.append({"step": step, "profit": val_profit, "status": "discard",
                                          "description": f"{description} ({acc:.0f}% acc)"})
                    state.profit_history.append((step, val_profit, "discard"))
                    write_file(TRAIN_FILE, original_code)
                    git("add", TRAIN_FILE)
                    git("commit", "-m", f"revert experiment {step}: no improvement")
                    log_result(step, commit_hash, val_profit, "discard", description)

            live.update(make_dashboard())
            time.sleep(1)

        state.status = "COMPLETE"
        state.phase = ""
        live.update(make_dashboard())
        time.sleep(5)

    console.print()
    sign = "+" if state.best_profit >= 0 else ""
    console.print(Panel(
        f"[bold green]Best profit: {sign}{state.best_profit:.1f}%[/bold green]\n"
        f"Final capital: ${state.final_capital:,.2f}\n"
        f"Accuracy: {state.accuracy:.0f}%\n"
        f"Results saved to: {RESULTS_FILE}\n"
        f"Experiments run: {state.max_steps}",
        title="[bold]Trading Experiment Complete[/bold]",
        border_style="green",
    ))

if __name__ == "__main__":
    main()
