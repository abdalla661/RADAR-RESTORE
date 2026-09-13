# RADAR-RESTORE

Research implementation of the **Radiotherapy Analysis of Disruptions and Recovery (RADAR)** framework and the proposed **RESTORE** recovery strategy.

This repository accompanies the Master's thesis:

**Intelligent Radiotherapy Operations Platform for Efficient Patient Management**  
Abdallah Abou Ziki  
Research Master in Data Science  
Lebanese University – Faculty of Sciences  
September 2026

---

## Overview

Radiotherapy treatment schedules may be disrupted by unexpected reductions in machine or workforce availability.

This repository contains the main implementation developed to study how an established radiotherapy schedule can be recovered after such disruptions.

The workflow consists of:

1. Converting a benchmark problem instance into structured scheduling data.
2. Generating a baseline radiotherapy schedule.
3. Simulating resource disruptions.
4. Recovering the disrupted schedule using three different strategies:
   - Patient-Local Repair
   - Full Reoptimization
   - RESTORE

RESTORE stands for:

**RE**covery with **ST**ability-**O**ptimality trade-offs in **R**adiotherapy under disruptive **E**vents.

RESTORE uses global recovery flexibility while explicitly penalizing unnecessary deviations from the established baseline schedule.

---

## Repository Structure

```text
RADAR-RESTORE/
│
├── 00_model_instance.py
├── 01_baseline_scheduler.py
├── 02_disruption_simulation.py
├── 03_patient_local_repair.py
├── 04_full_reoptimization.py
├── 05_restore.py
│
├── data_example/
│   └── demo_instance.csv
│
├── digital_twin_visualizer/
│   ├── app.py
│   ├── config.py
│   ├── convert_schedules.py
│   ├── chevron-left.png
│   ├── right-chevron.png
│   │
│   ├── templates/
│   │   └── <INSTANCE_ID> base.xlsx
│   │
│   └── data/
│       └── <INSTANCE_ID>/
│           ├── baseline/
│           ├── disruptions/
│           └── recovery/
│
├── requirements.txt
├── .gitignore
└── README.md
```

The Python files `00` to `05` are numbered according to their position in the experimental pipeline.  
The `digital_twin_visualizer/` directory contains the Streamlit interface used to inspect baseline, disrupted, and recovered schedules.

---

# Framework Pipeline

The scripts are intended to be executed in the following order:

```text
data_example/demo_instance.csv
            │
            ▼
00_model_instance.py
            │
            ▼
modeled_instance/
            │
            ▼
01_baseline_scheduler.py
            │
            ▼
baseline_results/
            │
            ▼
02_disruption_simulation.py
            │
            ▼
disruption_results/
            │
     ┌──────┼──────────┐
     │      │          │
     ▼      ▼          ▼
03_patient_ 04_full_   05_restore.py
local_      reoptimization.py
repair.py
```

The three recovery strategies are independent once the baseline and disruption files have been generated. Therefore, scripts `03`, `04`, and `05` do not need to be run in a particular order relative to one another.

---

# Requirements

The framework was developed using:

- Python 3.13.5
- Google OR-Tools CP-SAT
- pandas
- Streamlit
- openpyxl

The required Python packages are listed in `requirements.txt`:

```text
pandas
ortools==9.15.6755
streamlit
openpyxl
```

Python standard-library packages such as `os`, `time`, `ast`, `re`, `math`, `random`, and `collections` do not need to be installed separately.

---

# Installation

## 1. Open a terminal in the repository

On Windows:

```bash
cd C:\path\to\RADAR-RESTORE
```

For example:

```bash
cd C:\Users\YOUR_NAME\PycharmProjects\RADAR-RESTORE
```

All scripts should be executed from the repository root.

---

## 2. Create a virtual environment

```bash
py -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

The terminal should then show `(.venv)` before the command prompt.

The `.venv` directory is local and is excluded from GitHub through `.gitignore`.

---

## 3. Install the requirements

```bash
py -m pip install -r requirements.txt
```

If `python` is configured directly on the system, the equivalent command is:

```bash
python -m pip install -r requirements.txt
```

---

# Configuration

Each script contains a `CONFIGURATION` section near the beginning of the file.

For the simple repository structure used here, the folder configuration should follow this pattern.

### `00_model_instance.py`

```python
INSTANCE_PATH = "data_example/demo_instance.csv"
OUTPUT_FOLDER = "modeled_instance"
```

### `01_baseline_scheduler.py`

```python
INPUT_FOLDER = "modeled_instance"
OUTPUT_FOLDER = "baseline_results"
```

### `02_disruption_simulation.py`

```python
BASELINE_FOLDER = "baseline_results"
OUTPUT_FOLDER = "disruption_results"
```

### `03_patient_local_repair.py`

```python
INPUT_FOLDER = "modeled_instance"
BASELINE_FOLDER = "baseline_results"
DISRUPTION_FOLDER = "disruption_results"
OUTPUT_FOLDER = "patient_local_repair_results"
```

### `04_full_reoptimization.py`

```python
INPUT_FOLDER = "modeled_instance"
BASELINE_FOLDER = "baseline_results"
DISRUPTION_FOLDER = "disruption_results"
OUTPUT_FOLDER = "full_reoptimization_results"
```

### `05_restore.py`

```python
INPUT_FOLDER = "modeled_instance"
BASELINE_FOLDER = "baseline_results"
DISRUPTION_FOLDER = "disruption_results"
OUTPUT_FOLDER = "restore_results"
```

These output directories are generated locally by the scripts and do not need to be stored in the GitHub repository.

---

# Step 1 — Model the Benchmark Instance

Run:

```bash
py 00_model_instance.py
```

## Purpose

`00_model_instance.py` reads the raw radiotherapy scheduling benchmark instance and converts it into structured CSV files used by the optimization models.

The benchmark file is treated as a structured problem-instance definition rather than as a conventional row-based CSV table.

The script extracts information about:

- patients,
- treatment protocols,
- prescribed fractions,
- fraction durations,
- machine compatibility,
- preferred machines,
- treatment days,
- treatment windows,
- machine capacity.

It also introduces the RTT workforce capacity representation used in this research.

## RTT modelling assumption

The original benchmark does not explicitly model RTT teams.

RTT capacity is therefore introduced as a modelling extension in this research. One RTT capacity unit represents one treatment-delivery team, and the baseline RTT capacity is derived from the number of machines available in each day/window combination.

## Output

Running the script creates:

```text
modeled_instance/
├── patients.csv
├── fractions.csv
├── machine_capacity.csv
├── rtt_capacity.csv
└── instance_summary.csv
```

### `patients.csv`

Contains patient-level scheduling information such as:

- patient identifier,
- priority,
- earliest treatment start,
- target treatment day,
- number of prescribed fractions,
- compatible machines,
- preferred machines.

### `fractions.csv`

Contains fraction-level information such as:

- patient,
- fraction number,
- treatment duration,
- machine compatibility,
- priority,
- scheduling bounds.

### `machine_capacity.csv`

Contains the available machine capacity for every:

```text
machine × day × treatment window
```

combination.

### `rtt_capacity.csv`

Contains the RTT-team capacity available for each:

```text
day × treatment window
```

combination.

### `instance_summary.csv`

Provides a compact summary of the modeled benchmark instance.

---

# Step 2 — Generate the Baseline Schedule

After Step 1 completes successfully, run:

```bash
py 01_baseline_scheduler.py
```

## Purpose

`01_baseline_scheduler.py` creates the baseline radiotherapy treatment schedule using Google OR-Tools CP-SAT.

Each treatment fraction is assigned to:

- a treatment day,
- a treatment window,
- a compatible LINAC.

The model does **not** schedule exact clock appointment times.

The baseline optimization considers:

- treatment assignment and eligibility,
- LINAC capacity,
- RTT capacity,
- fraction ordering,
- treatment continuity,
- patient waiting time,
- patient priority,
- machine preference,
- machine consistency.

The baseline objective penalizes:

- patient waiting time,
- excess gaps between consecutive fractions,
- non-preferred machine assignments,
- machine switches between consecutive fractions.

## Output

The script creates:

```text
baseline_results/
├── baseline_schedule.csv
├── baseline_summary.csv
├── machine_utilization.csv
├── rtt_utilization.csv
└── patient_schedule_summary.csv
```

### `baseline_schedule.csv`

The main baseline treatment schedule.

Each row corresponds to one treatment fraction and includes its assigned:

```text
patient
fraction
machine
day
window
duration
```

### `baseline_summary.csv`

Contains solver and baseline-performance information.

### `machine_utilization.csv`

Reports treatment-machine utilization by machine, day, and treatment window.

### `rtt_utilization.csv`

Reports RTT workforce utilization by day and treatment window.

### `patient_schedule_summary.csv`

Contains patient-level schedule information such as treatment start and waiting time.

---

# Step 3 — Simulate Disruptions

After the baseline schedule has been generated, run:

```bash
py 02_disruption_simulation.py
```

## Purpose

`02_disruption_simulation.py` introduces unexpected resource disruptions into the established baseline schedule.

The framework supports five disruption scenarios.

### D1 — Short-duration LINAC failure

One LINAC becomes unavailable for one treatment window.

### D2 — Extended LINAC failure

A LINAC becomes unavailable starting from a selected treatment window and remains unavailable through the end of the same treatment day.


### D3 — Temporary power outage

All LINACs become unavailable during one treatment window.

### D4 — RTT workforce capacity reduction

RTT capacity is reduced using the following disruption levels:

```text
25%
50%
75%
```

### D5 — Combined disruption

A LINAC failure is combined with a 50% RTT workforce reduction.

The disruption generator identifies the treatment fractions affected by each scenario.

## Output

The script creates files inside:

```text
disruption_results/
```

For each scenario, the main outputs include:

```text
affected_fractions_<scenario>.csv
no_recovery_summary_<scenario>.csv
```

For workforce disruptions such as D4 and D5, scenario-specific RTT capacity files are also generated:

```text
disrupted_rtt_capacity_<scenario>.csv
```

These files are used by the recovery strategies.

---

# Step 4 — Patient-Local Repair

Run:

```bash
py 03_patient_local_repair.py
```

## Purpose

Patient-Local Repair represents a restricted recovery strategy.

The strategy attempts to recover treatment fractions belonging to patients directly affected by the disruption while preserving treatment assignments belonging to other patients.

Its recovery process attempts progressively broader adjustments within the affected patient's treatment course while keeping the repair local.

Other patients are not globally rescheduled.

This strategy represents the low-flexibility recovery approach used in the thesis.

## Output

For each selected disruption scenario, a separate result directory is created containing files such as:

```text
patient_local_repair_schedule.csv
patient_local_repair_recovered_fractions.csv
patient_local_repair_unrecovered_fractions.csv
patient_local_repair_shifted_course_fractions.csv
patient_local_repair_summary.csv
```

The summary reports the recovery outcome and modifications introduced by the local repair process.

---

# Step 5 — Full Reoptimization

Run:

```bash
py 04_full_reoptimization.py
```

## Purpose

Full Reoptimization provides a global recovery strategy.

Fractions that have already been delivered remain fixed.

The recoverable part of the future schedule is reconsidered globally under the post-disruption resource availability.

Full Reoptimization retains the operational objective used for baseline scheduling but does **not** explicitly penalize changes relative to the original baseline schedule.

It therefore represents a high-flexibility, operationally oriented recovery strategy.

## Output

Each scenario produces a directory containing:

```text
pure_full_reoptimization_schedule.csv
pure_full_reoptimization_summary.csv
```

The recovered schedule contains the complete post-disruption treatment schedule.

The summary includes information about:

- affected fractions,
- recovered fractions,
- recovery rate,
- solver performance,
- modified fractions,
- machine changes,
- treatment-window changes,
- day changes,
- treatment-course consistency.

---

# Step 6 — RESTORE

Run:

```bash
py 05_restore.py
```

## Purpose

RESTORE is the proposed stability-aware global recovery strategy.

Like Full Reoptimization, RESTORE retains sufficient flexibility to reconsider future treatment assignments after a disruption.

However, RESTORE additionally introduces an explicit schedule-stability cost.

The stability component penalizes:

- modification of a baseline fraction assignment,
- change of treatment day,
- change of LINAC,
- change of treatment window,
- magnitude of treatment-day displacement.

The RESTORE objective combines:

```text
operational cost + schedule-stability cost
```

through the trade-off parameter `lambda`.

The default experimental configuration uses:

```python
LAMBDA_NUM = 5
LAMBDA_DEN = 10
```

corresponding to:

```text
lambda = 0.5
```

The stability weights used in the implementation are:

```text
Modified fraction       10
Day change              15
Machine change           8
Treatment-window change  2
Day displacement         4
```

## Output

For every selected disruption scenario, RESTORE generates:

```text
restore_schedule.csv
restore_summary.csv
restore_build_timing_report.csv
```

### `restore_schedule.csv`

Contains the complete recovered schedule.

### `restore_summary.csv`

Contains recovery, operational, stability, and solver metrics.

### `restore_build_timing_report.csv`

Contains implementation timing information for different stages of model construction and solution.

---


# Digital Twin Visualizer

The repository also includes a Streamlit-based Digital Twin Visualizer under:

```text
digital_twin_visualizer/
```

The interface is used to visually inspect and compare:

- the baseline schedule,
- the disrupted schedule,
- Patient-Local Repair,
- Full Reoptimization,
- RESTORE.

The interface reads the schedule CSV files generated by the framework and displays treatment assignments by:

```text
day × LINAC × treatment window
```

It also highlights affected fractions and the changes introduced during schedule recovery.

## Interface files

The main files are:

### `app.py`

The Streamlit application.

It provides:

- instance selection,
- disruption selection,
- schedule-view selection,
- day-by-day treatment-horizon navigation,
- visualization of treatment fractions by LINAC and treatment window,
- highlighting of affected fractions,
- comparison of recovered schedules against the baseline,
- recovery indicators such as recovered fractions, unrecovered fractions, modified fractions, day changes, machine changes, and treatment-window changes.

### `config.py`

Defines the Digital Twin Visualizer data structure and the filenames expected by the interface.

It maps:

- available benchmark instances,
- disruption scenarios,
- recovery strategies,
- baseline files,
- disruption files,
- recovery files.

### `convert_schedules.py`

Builds the formatted baseline Excel workbook used by the Baseline and Disrupted views.

It reads `baseline_schedule.csv` and populates the corresponding Excel template.

### Navigation images

The interface uses:

```text
chevron-left.png
right-chevron.png
```

These files must remain directly inside `digital_twin_visualizer/` because the current application searches for them beside `app.py`.

---

## Digital Twin Visualizer data structure

The interface automatically discovers an instance when the following file exists:

```text
digital_twin_visualizer/data/<INSTANCE_ID>/baseline/baseline_schedule.csv
```

A complete demonstration instance can therefore follow this structure:

```text
digital_twin_visualizer/
│
├── app.py
├── config.py
├── convert_schedules.py
├── chevron-left.png
├── right-chevron.png
│
├── templates/
│   └── <INSTANCE_ID> base.xlsx
│
└── data/
    └── <INSTANCE_ID>/
        │
        ├── baseline/
        │   └── baseline_schedule.csv
        │
        ├── disruptions/
        │   ├── D1/
        │   │   └── affected_fractions_D1_machine_window.csv
        │   ├── D2/
        │   │   └── affected_fractions_D2_machine_full_day.csv
        │   ├── D3/
        │   │   └── affected_fractions_D3_power_outage_window.csv
        │   └── D4/
        │       └── affected_fractions_D4_rtt_shortage_50pct.csv
        │
        └── recovery/
            ├── D1/
            │   ├── local repair/
            │   │   └── patient_local_repair_schedule.csv
            │   ├── full reoptimization/
            │   │   └── pure_full_reoptimization_schedule.csv
            │   └── restore/
            │       └── restore_schedule.csv
            │
            ├── D2/
            │   └── ...
            ├── D3/
            │   └── ...
            └── D4/
                └── ...
```

The same recovery structure is used for D2, D3, and D4.

Only one complete demonstration instance is required for the GitHub repository.

---

## Excel template

The Baseline and Disrupted views use a formatted Excel workbook generated from the baseline schedule.

For an instance named:

```text
<INSTANCE_ID>
```

the expected template name is:

```text
<INSTANCE_ID> base.xlsx
```

and the recommended location is:

```text
digital_twin_visualizer/templates/<INSTANCE_ID> base.xlsx
```

The instance folder name and the template filename must use the same instance identifier.

---

## Files required by the interface

For the current implementation, the interface directly requires:

### Baseline

```text
baseline/baseline_schedule.csv
```

The schedule must contain:

```text
patient
fraction
machine
day
window
duration_minutes
priority
is_preferred_machine
```

### Disruption

For each displayed disruption, the interface requires the corresponding:

```text
affected_fractions_<scenario>.csv
```

with at least:

```text
patient
fraction
machine
day
window
```

### Recovery

For every displayed recovery strategy, the interface requires the recovered schedule:

```text
patient_local_repair_schedule.csv
pure_full_reoptimization_schedule.csv
restore_schedule.csv
```

The interface calculates its displayed recovery indicators by comparing these recovered schedules with the baseline and affected-fraction files.

The additional solver summaries and utilization files are useful for the experimental analysis but are not required simply to display the current Digital Twin Visualizer.

---

## Run the Digital Twin Visualizer

From the repository root:

```bash
py -m streamlit run digital_twin_visualizer/app.py
```

Alternatively, enter the Digital Twin directory:

```bash
cd digital_twin_visualizer
```

and run:

```bash
py -m streamlit run app.py
```

Streamlit will start a local web server and open the interface in the browser.

---

## Digital Twin Visualizer generated files

The interface may create a local directory:

```text
digital_twin_visualizer/generated/
```

for generated Excel workbooks.

This directory does not need to be committed to GitHub because it can be recreated locally.

---


# Selecting Recovery Scenarios

The recovery scripts contain a `SCENARIOS` list near the beginning of each file.

For example:

```python
SCENARIOS = [
    "D1_machine_window",
    "D2_machine_full_day",
    "D3_power_outage_window",
    "D4_rtt_shortage_50pct",
]
```

Additional disruption conditions can be enabled when their files have been generated by `02_disruption_simulation.py`, for example:

```python
"D4_rtt_shortage_25pct",
"D4_rtt_shortage_75pct",
"D5_combined_machine_day_rtt",
```

The scenario identifier in a recovery script must correspond to a disruption generated by `02_disruption_simulation.py`.

---

# Complete Execution Example

Starting from a fresh repository:

```bash
cd C:\path\to\RADAR-RESTORE
```

Create the environment:

```bash
py -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

Then run the framework in order:

```bash
py 00_model_instance.py
py 01_baseline_scheduler.py
py 02_disruption_simulation.py
py 03_patient_local_repair.py
py 04_full_reoptimization.py
py 05_restore.py
```

The first three commands must be executed sequentially because each stage depends on files produced by the previous stage.

Once disruption files have been created, the three recovery strategies can be executed independently.

To launch the Digital Twin Visualizer from the repository root:

```bash
py -m streamlit run digital_twin_visualizer/app.py
```

---

# Generated Files

Running the complete framework will create directories similar to:

```text
RADAR-RESTORE/
│
├── modeled_instance/
├── baseline_results/
├── disruption_results/
├── patient_local_repair_results/
├── full_reoptimization_results/
└── restore_results/
```

These directories contain generated experimental outputs.

The Digital Twin Visualizer Visualizer may additionally create:

```text
digital_twin_visualizer/generated/
```

These generated directories are not required to be committed to the source-code repository and can be excluded using `.gitignore`.

---

# Important Modelling Notes

## Treatment-time representation

The framework does not assign exact clock times to treatments.

Each fraction is assigned to:

```text
treatment day
treatment window
LINAC
```

A treatment window represents a portion of the treatment day rather than an exact appointment time.

## RTT capacity

RTT workforce capacity is a modelling extension introduced in this research.

The original benchmark does not explicitly represent RTT teams.

The RTT capacity representation is therefore constructed by the framework to support experiments involving workforce-capacity disruptions.

## Data

The repository contains a demonstration benchmark instance under:

```text
data_example/demo_instance.csv
```

The framework was developed and evaluated using publicly available radiotherapy scheduling benchmark instances.

The demonstration data are intended to illustrate the expected input format.

---

# Output Directories and Git

Generated experiment files, Python cache files, virtual environments, and IDE configuration should not be committed.

The `.gitignore` file should therefore include entries such as:

```text
.venv/
.idea/
__pycache__/
*.pyc

modeled_instance/
baseline_results/
disruption_results/
patient_local_repair_results/
full_reoptimization_results/
restore_results/

digital_twin_visualizer/generated/
```

---

# Research Scope

This repository is provided as a research artifact accompanying the Master's thesis.

It is intended to document the implementation of the RADAR framework and the RESTORE recovery strategy.

The software is an experimental research implementation and is **not a production clinical scheduling system**.

---

# Framework Summary

The role of each script can be summarized as follows:

| Script | Purpose |
|---|---|
| `00_model_instance.py` | Convert a raw benchmark instance into structured scheduling data |
| `01_baseline_scheduler.py` | Generate the baseline radiotherapy schedule using CP-SAT |
| `02_disruption_simulation.py` | Simulate machine, power, workforce, and combined disruptions |
| `03_patient_local_repair.py` | Recover affected treatment courses using restricted local repair |
| `04_full_reoptimization.py` | Globally reoptimize the recoverable schedule using the operational objective |
| `05_restore.py` | Globally recover the schedule while balancing operational performance and schedule stability |
| `digital_twin_visualizer/app.py` | Visualize and compare baseline, disrupted, and recovered treatment schedules |

---

# Author

**Abdallah Abou Ziki**

Research Master in Data Science  
Lebanese University – Faculty of Sciences  
2026
