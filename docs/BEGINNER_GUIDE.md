# Beginner guide: reproduce the study step by step (Ubuntu 22.04)

This guide assumes **no prior terminal experience**. It reproduces the study on a laptop running Ubuntu 22.04
(tested on an AMD Ryzen 7 6800H laptop with 16 GB RAM). Ubuntu is recommended because Python is preinstalled, the CPU has active
cooling (fanless laptops slow down during multi-hour runs, which spoils timing measurements), and the solver installs with one command.

**How to use it:** do the steps in order. Each grey box is ONE thing to copy, paste into the terminal and run with Enter. Wait until the terminal
shows the prompt again (a line ending in `$`) before the next step. In Ubuntu's terminal, paste with **Ctrl+Shift+V**. When asked for a password,
type it (nothing appears on screen) and press Enter.

---

## Part A: One-time setup (about 15 minutes)

**A1.** Plug in the charger and connect to the internet. Keep the lid open during long runs.

**A2.** Open a terminal with **Ctrl+Alt+T**, then install the basic tools:
```
sudo apt update && sudo apt install -y python3-venv python3-pip unzip git
```

**A3.** Stop the laptop from sleeping while plugged in:
```
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
```

**A4.** Get the code (either download the repository as a ZIP and unzip it in your home folder, or clone it) and enter the folder:
```
cd ~/fragile-proofs
```

**A5.** Create a private Python environment and install the requirements (a few minutes):
```
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
The prompt now starts with `(.venv)`.

> **Every time you open a NEW terminal**, first run this one line:
> ```
> cd ~/fragile-proofs && source .venv/bin/activate
> ```

**A6.** Check that everything works. Every line must start with `[ OK ]`:
```
python -m fragile check
```
If the solver line says `[FAIL]`, run `sudo apt install -y z3` and repeat A6.

**A7.** Run the automatic self-tests. The last line must say `OK`:
```
python -m unittest discover -s tests
```

---

## Part B: Get the data (about 5-20 minutes, about 300 MB)

**B1.** Download only the benchmark queries from the Mariposa data repository:
```
cd ~ && git clone --depth 1 --filter=blob:none --sparse https://github.com/secure-foundations/mariposa-data.git && cd mariposa-data && git sparse-checkout set benchmarks && cd ~/fragile-proofs
```

**B2.** Build the list of queries. You should see 5 project groups and 935 queries in total:
```
python -m fragile prepare-mariposa ~/mariposa-data/benchmarks --out data_mariposa
```

---

## Part C: Measure (the long part)

**C1.** Small test run (roughly 15 minutes), then a health check. The health check must show no `WARNING ... error` line:
```
python -m fragile collect --queries ~/mariposa-data/benchmarks --manifest data_mariposa/manifest.csv --out runs/main --per-project 5 --workers 4 --timeout 10 --mem-mb 3000
```
```
python -m fragile status runs/main
```

**C2.** Main run (about 2.5 hours on the reference machine; the cap stops it after 12 hours). Leave the laptop alone, plugged in, lid open:
```
python -m fragile collect --queries ~/mariposa-data/benchmarks --manifest data_mariposa/manifest.csv --out runs/main --per-project 60 --workers 4 --timeout 10 --mem-mb 3000 --max-hours 12
```
Stop it any time with **Ctrl+C**. To continue, open a terminal, run the "every time" line from A5, and paste the same command again.

---

## Part D: Results

**D1.** Compare the strategies (a few minutes):
```
python -m fragile evaluate runs/main --ablations
```
Outputs are written to `runs/main/results/` (`summary.md`, `eval.json`, figures).

**D2.** Re-run the cheapest 30 witness pairs five times each, to separate real input sensitivity from timing noise (about an hour):
```
python -m fragile confirm runs/main --repeats 5 --max 30
```

**D3.** Copy the shareable outputs into the repository's `results/` folder (host names and home paths are removed):
```
scripts/export_results.sh runs/main
```

---

## If something goes wrong
* "command not found" or "No module named fragile": you are in a new terminal. Run the "every time" line from A5.
* The run was interrupted (power cut, closed terminal): run the same `collect` command again; it resumes.
* Anything else: open a GitHub issue and paste the last 30 lines of the terminal.
