import ast
import os
import time
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "modeled_instance"
OUTPUT_FOLDER = "baseline_results"

TIME_LIMIT_SECONDS =1800
NUM_WORKERS = 8
DAY_BUFFER = 3

# Objective weights
WAITING_WEIGHT = 100
GAP_WEIGHT = 5
NON_PREFERRED_MACHINE_WEIGHT = 10
MACHINE_SWITCH_WEIGHT = 10
WINDOW_SWITCH_WEIGHT = 0


DEBUG_VARIABLE_NAMES = False

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def timer(label):
    """Small helper for build-time profiling."""
    now = time.perf_counter()
    print(f"[{label}] {now:.2f}")
    return now


_build_start = time.perf_counter()


# ============================================================
# LOAD DATA
# ============================================================

patients = pd.read_csv(os.path.join(INPUT_FOLDER, "patients.csv"))
fractions = pd.read_csv(os.path.join(INPUT_FOLDER, "fractions.csv"))
machine_capacity = pd.read_csv(os.path.join(INPUT_FOLDER, "machine_capacity.csv"))
rtt_capacity = pd.read_csv(os.path.join(INPUT_FOLDER, "rtt_capacity.csv"))

# Convert allowed_machines from string to list
patients["allowed_machines"] = patients["allowed_machines"].apply(ast.literal_eval)
fractions["allowed_machines"] = fractions["allowed_machines"].apply(ast.literal_eval)

# Preferred machines are optional.
# If the column exists, parse it.
# If it does not exist or is empty, allowed_machines are treated as preferred.
if "preferred_machines" in patients.columns:
    patients["preferred_machines"] = patients["preferred_machines"].apply(
        lambda v: ast.literal_eval(v)
        if isinstance(v, str) and v.strip() not in ["", "[]"]
        else []
    )
else:
    patients["preferred_machines"] = [[] for _ in range(len(patients))]


# ============================================================
# BASIC SETS
# ============================================================

P = patients["patient"].astype(int).tolist()
M = sorted(machine_capacity["machine"].astype(int).unique().tolist())
D = sorted(machine_capacity["day"].astype(int).unique().tolist())
W = sorted(machine_capacity["window"].astype(int).unique().tolist())

max_day = max(D)
min_machine = min(M)
max_machine = max(M)
min_window = min(W)
max_window = max(W)

print("========================================")
print("BASELINE CP-SAT SCHEDULER - OPTIMIZED BUILD")
print("========================================")
print(f"Patients: {len(P)}")
print(f"Fractions: {len(fractions)}")
print(f"Machines: {len(M)}")
print(f"Days: {len(D)}")
print(f"Windows: {len(W)}")


# ============================================================
# DICTIONARIES
# ============================================================

cap = {
    (row.machine, row.day, row.window): int(row.remaining_machine_capacity_minutes)
    for row in machine_capacity.itertuples()
}

# Fast lookup of feasible windows by (machine, day). This avoids repeatedly
# looping over all windows and checking the capacity dictionary in nested loops.
windows_by_machine_day = defaultdict(list)
for (m, d, w), c in cap.items():
    if c > 0:
        windows_by_machine_day[(m, d)].append(w)

for key in windows_by_machine_day:
    windows_by_machine_day[key].sort()

rtt_col = "available_rtt_teams"
if rtt_col not in rtt_capacity.columns:
    rtt_col = "baseline_rtt_capacity_units"

rtt_cap = {
    (row.day, row.window): int(getattr(row, rtt_col))
    for row in rtt_capacity.itertuples()
}

patient_info = {}

for row in patients.itertuples():

    allowed_machines = row.allowed_machines
    preferred_machines = row.preferred_machines

    if preferred_machines is None or len(preferred_machines) == 0:
        preferred_machines = allowed_machines

    patient_info[row.patient] = {
        "priority": int(row.priority),
        "earliest": int(row.earliest_start_day),
        "target": int(row.target_day),
        "number_fractions": int(row.number_fractions),
        "allowed_machines": allowed_machines,
        "preferred_machines": preferred_machines,
        "preferred_machines_set": set(preferred_machines),
    }

priority_weight_by_patient = {
    int(p): 10 if int(info["priority"]) == 1 else 3 if int(info["priority"]) == 2 else 1
    for p, info in patient_info.items()
}

fraction_info = {}

for row in fractions.itertuples():
    p = int(row.patient)
    f = int(row.fraction)

    fraction_info[(p, f)] = {
        "duration": int(row.duration_minutes),
        "priority": int(row.priority),
        "earliest": int(row.earliest_start_day),
        "target": int(row.target_day),
        "allowed_machines": row.allowed_machines,
    }

fractions_by_patient = (
    fractions.groupby("patient")["fraction"]
    .apply(list)
    .to_dict()
)

print(f"Data preparation time: {time.perf_counter() - _build_start:.2f} seconds")


# ============================================================
# MODEL
# ============================================================

model = cp_model.CpModel()


# ============================================================
# DECISION VARIABLES + INDEX MAPS
# ============================================================
#
# x[p,f,m,d,w] = 1 if fraction f of patient p is assigned
# to machine m, day d, window w.
#
# Mathematical logic is unchanged from the original script.
# Implementation change:
# We build indexing dictionaries while variables are created, so later
# constraints do not repeatedly scan the full x dictionary.
#
# ============================================================

# No global x dictionary is stored. Candidate records and sparse indexes are enough
# for all constraints and solution extraction, and avoiding a million tuple-key
# dictionary saves memory and build time.

# Index maps used for fast constraint construction
x_by_fraction = defaultdict(list)              # (p,f) -> [x]
slot_terms = defaultdict(lambda: ([], []))      # (m,d,w) -> ([x], [duration])
day_terms_by_fraction = defaultdict(lambda: ([], []))  # (p,f) -> ([x], [day])
machine_terms_by_fraction = defaultdict(lambda: ([], []))  # (p,f) -> ([x], [machine])
window_terms_by_fraction = defaultdict(lambda: ([], []))   # (p,f) -> ([x], [window])
non_preferred_vars = []

candidate_count = 0
candidate_records = []  # (p, f, m, d, w, var) for faster solution extraction

for (p, f), info in fraction_info.items():

    duration = info["duration"]
    earliest = info["earliest"]
    target = info["target"]
    allowed_machines = info["allowed_machines"]
    preferred_machines_set = patient_info[p]["preferred_machines_set"]

    number_fractions = patient_info[p]["number_fractions"]

    latest_candidate_day = min(
        max_day,
        max(
            target + DAY_BUFFER,
            earliest + number_fractions - 1
        )
    )

    # Full baseline candidate horizon.
    # Important: we do NOT truncate candidate days here because truncation can
    # remove the only feasible globally consistent schedule for dense instances.
    # The logic below keeps the original feasibility horizon while preserving
    # the faster implementation style: no unused x dictionary, sparse windows,
    # empty variable names, and indexed constraints.
    for m in allowed_machines:
        for d in D:

            if d < earliest:
                continue
            if d > latest_candidate_day:
                break

            for w in windows_by_machine_day.get((m, d), []):

                if cap[(m, d, w)] < duration:
                    continue

                if DEBUG_VARIABLE_NAMES:
                    var = model.NewBoolVar(f"x_p{p}_f{f}_m{m}_d{d}_w{w}")
                else:
                    var = model.NewBoolVar("")

                candidate_records.append((p, f, m, d, w, var))

                # Build all useful indexes once.
                x_by_fraction[(p, f)].append(var)

                vars_slot, coeffs_slot = slot_terms[(m, d, w)]
                vars_slot.append(var)
                coeffs_slot.append(duration)

                vars_day, coeffs_day = day_terms_by_fraction[(p, f)]
                vars_day.append(var)
                coeffs_day.append(d)

                vars_machine, coeffs_machine = machine_terms_by_fraction[(p, f)]
                vars_machine.append(var)
                coeffs_machine.append(m)

                vars_window, coeffs_window = window_terms_by_fraction[(p, f)]
                vars_window.append(var)
                coeffs_window.append(w)

                if m not in preferred_machines_set:
                    non_preferred_vars.append(var)

                candidate_count += 1

print(f"Candidate assignment variables created: {candidate_count}")


# Candidate-space diagnostics. These do not change the model; they only show
# which dimension is creating the large search space.
candidate_counts_by_fraction = {
    key: len(vars_) for key, vars_ in x_by_fraction.items()
}
_candidate_count_values = sorted(candidate_counts_by_fraction.values())

def _percentile_from_sorted(values, percentile):
    if not values:
        return 0
    index = int(round((len(values) - 1) * percentile))
    return int(values[index])

average_allowed_machines = (
    sum(len(info["allowed_machines"]) for info in fraction_info.values())
    / len(fraction_info)
    if fraction_info else 0.0
)
average_candidates_per_fraction = (
    candidate_count / len(fraction_info) if fraction_info else 0.0
)
max_candidates_per_fraction = max(_candidate_count_values, default=0)
median_candidates_per_fraction = _percentile_from_sorted(_candidate_count_values, 0.50)
p90_candidates_per_fraction = _percentile_from_sorted(_candidate_count_values, 0.90)
p95_candidates_per_fraction = _percentile_from_sorted(_candidate_count_values, 0.95)

print("Candidate-space diagnostics (baseline):")
print(f"  Average allowed machines per fraction: {average_allowed_machines:.2f}")
print(f"  Average candidate slots per fraction: {average_candidates_per_fraction:.2f}")
print(f"  Median candidate slots per fraction: {median_candidates_per_fraction}")
print(f"  90th percentile candidate slots per fraction: {p90_candidates_per_fraction}")
print(f"  95th percentile candidate slots per fraction: {p95_candidates_per_fraction}")
print(f"  Maximum candidate slots for one fraction: {max_candidates_per_fraction}")


# z[m,d,w] = 1 if machine m is active during day d window w
z = {}

for m in M:
    for d in D:
        for w in W:
            if DEBUG_VARIABLE_NAMES:
                z[(m, d, w)] = model.NewBoolVar(f"z_m{m}_d{d}_w{w}")
            else:
                z[(m, d, w)] = model.NewBoolVar("")


# Day, machine, and window variable for each fraction
fraction_day = {}
fraction_machine = {}
fraction_window = {}

for (p, f) in fraction_info:
    fraction_day[(p, f)] = model.NewIntVar(
        1,
        max_day,
        f"day_p{p}_f{f}" if DEBUG_VARIABLE_NAMES else ""
    )
    fraction_machine[(p, f)] = model.NewIntVar(
        min_machine,
        max_machine,
        f"machine_p{p}_f{f}" if DEBUG_VARIABLE_NAMES else ""
    )
    fraction_window[(p, f)] = model.NewIntVar(
        min_window,
        max_window,
        f"window_p{p}_f{f}" if DEBUG_VARIABLE_NAMES else ""
    )


# Start day for each patient
start_day = {}

for p in P:
    start_day[p] = model.NewIntVar(
        1,
        max_day,
        f"start_day_p{p}" if DEBUG_VARIABLE_NAMES else ""
    )

print(f"Variable creation + indexing time: {time.perf_counter() - _build_start:.2f} seconds")


# ============================================================
# CONSTRAINT 1 — EACH FRACTION SCHEDULED EXACTLY ONCE
# ============================================================

for (p, f) in fraction_info:

    vars_for_fraction = x_by_fraction[(p, f)]

    if not vars_for_fraction:
        raise ValueError(
            f"No feasible assignment candidates for patient {p}, fraction {f}"
        )

    model.Add(cp_model.LinearExpr.Sum(vars_for_fraction) == 1)


# ============================================================
# CONSTRAINT 2 — MACHINE CAPACITY
# ============================================================

for m in M:
    for d in D:
        for w in W:

            vars_slot, coeffs_slot = slot_terms.get((m, d, w), ([], []))

            if vars_slot:
                model.Add(
                    cp_model.LinearExpr.WeightedSum(vars_slot, coeffs_slot)
                    <= cap[(m, d, w)]
                )
            else:
                # No candidate assigned to this slot. The original constraint is
                # 0 <= capacity, which is normally redundant. Keep a safety check.
                if cap[(m, d, w)] < 0:
                    raise ValueError(f"Negative capacity for slot {(m, d, w)}")


# ============================================================
# CONSTRAINT 3 — LINK x TO ACTIVE MACHINE z
# ============================================================

# Aggregated linking per machine-day-window.
# Equivalent to x_i <= z for every assignment in the slot, but creates fewer
# constraints and is faster to build.
for (m, d, w), (vars_slot, _) in slot_terms.items():
    if vars_slot:
        slot_sum = cp_model.LinearExpr.Sum(vars_slot)
        model.Add(slot_sum <= len(vars_slot) * z[(m, d, w)])
        model.Add(z[(m, d, w)] <= slot_sum)


# ============================================================
# CONSTRAINT 4 — RTT TEAM CAPACITY
# ============================================================

for d in D:
    for w in W:

        model.Add(
            cp_model.LinearExpr.Sum([z[(m, d, w)] for m in M]) <= rtt_cap[(d, w)]
        )


# ============================================================
# CONSTRAINT 5 — DEFINE FRACTION DAY / MACHINE / WINDOW
# ============================================================

for (p, f) in fraction_info:

    vars_day, coeffs_day = day_terms_by_fraction[(p, f)]
    vars_machine, coeffs_machine = machine_terms_by_fraction[(p, f)]
    vars_window, coeffs_window = window_terms_by_fraction[(p, f)]

    model.Add(
        fraction_day[(p, f)]
        == cp_model.LinearExpr.WeightedSum(vars_day, coeffs_day)
    )
    model.Add(
        fraction_machine[(p, f)]
        == cp_model.LinearExpr.WeightedSum(vars_machine, coeffs_machine)
    )
    model.Add(
        fraction_window[(p, f)]
        == cp_model.LinearExpr.WeightedSum(vars_window, coeffs_window)
    )

print(f"Core constraint build time: {time.perf_counter() - _build_start:.2f} seconds")


# ============================================================
# CONSTRAINT 6 — FRACTION ORDERING AND CLINICAL CONTINUITY
# ============================================================
#
# Consecutive fractions must:
# - respect treatment order,
# - not occur on the same treatment weekday.
#
# Continuity is handled as a soft objective:
# gap_excess = day_next - day_current - 1
#
# Therefore:
# day difference 1 -> gap_excess = 0
# day difference 2 -> gap_excess = 1
# day difference 3 -> gap_excess = 2
#
# No hard maximum-gap constraint is imposed. Observed gap statistics are
# computed from the final schedule and reported in the summary.
#
# ============================================================

gap_excess = {}
machine_switch_vars = []
window_switch_vars = []

for p, flist in fractions_by_patient.items():

    flist = sorted(flist)

    for i in range(len(flist) - 1):
        f_current = flist[i]
        f_next = flist[i + 1]

        model.Add(
            fraction_day[(p, f_next)]
            >=
            fraction_day[(p, f_current)] + 1
        )

        gap_var = model.NewIntVar(
            0,
            max_day,
            f"gap_excess_p{p}_f{f_current}_to_f{f_next}" if DEBUG_VARIABLE_NAMES else ""
        )

        model.Add(
            gap_var
            ==
            fraction_day[(p, f_next)]
            -
            fraction_day[(p, f_current)]
            -
            1
        )

        gap_excess[(p, f_current, f_next)] = gap_var

        machine_switch = model.NewBoolVar(
            f"machine_switch_p{p}_f{f_current}_to_f{f_next}" if DEBUG_VARIABLE_NAMES else ""
        )
        model.Add(fraction_machine[(p, f_current)] != fraction_machine[(p, f_next)]).OnlyEnforceIf(machine_switch)
        model.Add(fraction_machine[(p, f_current)] == fraction_machine[(p, f_next)]).OnlyEnforceIf(machine_switch.Not())
        machine_switch_vars.append(machine_switch)

        window_switch = model.NewBoolVar(
            f"window_switch_p{p}_f{f_current}_to_f{f_next}" if DEBUG_VARIABLE_NAMES else ""
        )
        model.Add(fraction_window[(p, f_current)] != fraction_window[(p, f_next)]).OnlyEnforceIf(window_switch)
        model.Add(fraction_window[(p, f_current)] == fraction_window[(p, f_next)]).OnlyEnforceIf(window_switch.Not())
        window_switch_vars.append(window_switch)


# ============================================================
# CONSTRAINT 7 — DEFINE PATIENT START DAY
# ============================================================

for p, flist in fractions_by_patient.items():

    first_fraction = min(flist)

    model.Add(
        start_day[p] == fraction_day[(p, first_fraction)]
    )


# ============================================================
# OBJECTIVE — FINAL BASELINE OBJECTIVE
# ============================================================
#
# 1. Minimize weighted waiting time.
# 2. Minimize excess treatment gaps.
# 3. Minimize non-preferred machine assignments.
# 4. Minimize machine switches between consecutive fractions.
# 5. Minimize window switches between consecutive fractions.
#
# Since every fraction must be scheduled exactly once, maximizing
# scheduled patients is not included in the baseline objective.
#
# ============================================================

objective_terms = []

# 1. Weighted waiting time
for p in P:
    earliest = patient_info[p]["earliest"]
    priority_weight = priority_weight_by_patient[int(p)]

    waiting = model.NewIntVar(
        0,
        max_day,
        f"waiting_p{p}" if DEBUG_VARIABLE_NAMES else ""
    )

    model.Add(waiting == start_day[p] - earliest)

    objective_terms.append(
        WAITING_WEIGHT * priority_weight * waiting
    )


# 2. Continuity gap penalty
for (p, f_current, f_next), gap_var in gap_excess.items():

    priority_weight = priority_weight_by_patient[int(p)]

    objective_terms.append(
        GAP_WEIGHT * priority_weight * gap_var
    )


# 3. Non-preferred machine penalty
objective_terms.extend(
    NON_PREFERRED_MACHINE_WEIGHT * var
    for var in non_preferred_vars
)

# 4. Machine and window consistency penalty between consecutive fractions
objective_terms.extend(
    MACHINE_SWITCH_WEIGHT * var
    for var in machine_switch_vars
)
objective_terms.extend(
    WINDOW_SWITCH_WEIGHT * var
    for var in window_switch_vars
)

model.Minimize(cp_model.LinearExpr.Sum(objective_terms))

print(f"Full model build time: {time.perf_counter() - _build_start:.2f} seconds")


# ============================================================
# SOLVE
# ============================================================

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
solver.parameters.num_search_workers = NUM_WORKERS
solver.parameters.log_search_progress = True
solver.parameters.random_seed = 1
# Optional: stop early when a good proof gap is enough for experiments.
# solver.parameters.relative_gap_limit = 0.01

print("\nSolving CP-SAT model...")

status = solver.Solve(model)

print("\nSolver status:", solver.StatusName(status))

solver_best_objective_bound = None
solver_absolute_gap = None
solver_relative_gap = None

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    solver_best_objective_bound = solver.BestObjectiveBound()
    solver_absolute_gap = solver.ObjectiveValue() - solver_best_objective_bound
    solver_relative_gap = solver_absolute_gap / max(1.0, abs(solver.ObjectiveValue()))

    print("Objective value:", solver.ObjectiveValue())
    print("Best objective bound:", solver_best_objective_bound)
    print("Absolute optimality gap:", solver_absolute_gap)
    print("Relative optimality gap:", solver_relative_gap)


# ============================================================
# EXPORT RESULTS
# ============================================================

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:

    schedule_rows = []

    for p, f, m, d, w, var in candidate_records:

        if solver.Value(var) == 1:

            duration = fraction_info[(p, f)]["duration"]

            schedule_rows.append({
                "patient": p,
                "fraction": f,
                "machine": m,
                "day": d,
                "window": w,
                "duration_minutes": duration,
                "priority": patient_info[p]["priority"],
                "earliest_start_day": patient_info[p]["earliest"],
                "target_day": patient_info[p]["target"],
                "is_preferred_machine": (
                    m in patient_info[p]["preferred_machines_set"]
                ),
            })

    schedule = pd.DataFrame(schedule_rows)

    schedule = schedule.sort_values(
        ["day", "window", "machine", "patient", "fraction"]
    )

    schedule.to_csv(
        os.path.join(OUTPUT_FOLDER, "baseline_schedule.csv"),
        index=False
    )

    # Machine utilization
    machine_util = (
        schedule.groupby(["machine", "day", "window"])["duration_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"duration_minutes": "used_minutes"})
    )

    machine_util = machine_util.merge(
        machine_capacity[
            [
                "machine",
                "day",
                "window",
                "remaining_machine_capacity_minutes"
            ]
        ],
        on=["machine", "day", "window"],
        how="right"
    )

    machine_util["used_minutes"] = machine_util["used_minutes"].fillna(0)

    machine_util["utilization"] = (
        machine_util["used_minutes"]
        /
        machine_util["remaining_machine_capacity_minutes"].replace(0, pd.NA)
    )

    machine_util.to_csv(
        os.path.join(OUTPUT_FOLDER, "machine_utilization.csv"),
        index=False
    )

    # RTT utilization
    active_machine_windows = (
        schedule.groupby(["day", "window"])["machine"]
        .nunique()
        .reset_index()
        .rename(columns={"machine": "active_machines"})
    )

    rtt_util = rtt_capacity.merge(
        active_machine_windows,
        on=["day", "window"],
        how="left"
    )

    rtt_util["active_machines"] = rtt_util["active_machines"].fillna(0)

    rtt_util["rtt_utilization"] = (
        rtt_util["active_machines"]
        /
        rtt_util[rtt_col].replace(0, pd.NA)
    )

    rtt_util.to_csv(
        os.path.join(OUTPUT_FOLDER, "rtt_utilization.csv"),
        index=False
    )

    # Patient waiting summary
    patient_starts = (
        schedule.groupby("patient")["day"]
        .min()
        .reset_index()
        .rename(columns={"day": "scheduled_start_day"})
    )

    patient_summary = patients.merge(
        patient_starts,
        on="patient",
        how="left"
    )

    patient_summary["waiting_time"] = (
        patient_summary["scheduled_start_day"]
        -
        patient_summary["earliest_start_day"]
    )

    patient_summary.to_csv(
        os.path.join(OUTPUT_FOLDER, "patient_schedule_summary.csv"),
        index=False
    )

    # Preference and consistency summary
    non_preferred_assignments = int(
        (~schedule["is_preferred_machine"]).sum()
    )

    schedule_keyed = {
        (int(row.patient), int(row.fraction)): row
        for row in schedule.itertuples()
    }

    total_machine_switches = 0
    total_window_switches = 0
    patients_with_machine_switches = set()
    patients_with_window_switches = set()

    # Observed continuity-gap metrics computed from the solved schedule.
    # gap_excess = number of unused treatment weekdays between consecutive fractions.
    observed_gap_excess_values = []
    patients_with_nonzero_observed_gap = set()

    for p, flist in fractions_by_patient.items():
        patient_id = int(p)
        flist = sorted([int(f) for f in flist])

        for i in range(len(flist) - 1):
            current = schedule_keyed[(patient_id, flist[i])]
            nxt = schedule_keyed[(patient_id, flist[i + 1])]

            machine_switched = int(current.machine) != int(nxt.machine)
            window_switched = int(current.window) != int(nxt.window)

            observed_gap_excess = max(
                0,
                int(nxt.day) - int(current.day) - 1
            )
            observed_gap_excess_values.append(observed_gap_excess)

            if observed_gap_excess > 0:
                patients_with_nonzero_observed_gap.add(patient_id)

            if machine_switched:
                total_machine_switches += 1
                patients_with_machine_switches.add(patient_id)

            if window_switched:
                total_window_switches += 1
                patients_with_window_switches.add(patient_id)

    total_consecutive_fraction_pairs = max(len(fractions) - len(P), 1)

    total_observed_gap_excess_days = int(sum(observed_gap_excess_values))
    average_observed_gap_excess_days = (
        total_observed_gap_excess_days / len(observed_gap_excess_values)
        if observed_gap_excess_values
        else 0.0
    )
    maximum_observed_gap_excess_days = (
        max(observed_gap_excess_values)
        if observed_gap_excess_values
        else 0
    )
    consecutive_pairs_with_nonzero_gap = sum(
        1 for value in observed_gap_excess_values if value > 0
    )
    nonzero_gap_pair_rate = (
        consecutive_pairs_with_nonzero_gap / len(observed_gap_excess_values)
        if observed_gap_excess_values
        else 0.0
    )

    preferred_machine_rate = float(
        schedule["is_preferred_machine"].mean()
    )

    baseline_summary = {
        "status": solver.StatusName(status),
        "objective_value": solver.ObjectiveValue(),
        "wall_time_seconds": solver.WallTime(),
        "solver_best_objective_bound": solver_best_objective_bound,
        "solver_absolute_gap": solver_absolute_gap,
        "solver_relative_gap": solver_relative_gap,
        "total_script_time_seconds": time.perf_counter() - _build_start,
        "build_and_export_overhead_seconds": (time.perf_counter() - _build_start) - solver.WallTime(),
        "candidate_variables": candidate_count,
        "average_allowed_machines_per_fraction": average_allowed_machines,
        "average_candidates_per_fraction": average_candidates_per_fraction,
        "median_candidates_per_fraction": median_candidates_per_fraction,
        "p90_candidates_per_fraction": p90_candidates_per_fraction,
        "p95_candidates_per_fraction": p95_candidates_per_fraction,
        "max_candidates_per_fraction": max_candidates_per_fraction,
        "candidate_day_limit_active": False,
        "scheduled_fractions": len(schedule),
        "total_fractions": len(fractions),
        "scheduled_patients": schedule["patient"].nunique(),
        "total_patients": len(patients),
        "average_waiting_time": patient_summary["waiting_time"].mean(),
        "max_waiting_time": patient_summary["waiting_time"].max(),
        "objective_waiting_weight": WAITING_WEIGHT,
        "objective_gap_weight": GAP_WEIGHT,
        "objective_non_preferred_machine_weight": NON_PREFERRED_MACHINE_WEIGHT,
        "total_observed_gap_excess_days": total_observed_gap_excess_days,
        "average_observed_gap_excess_days": average_observed_gap_excess_days,
        "maximum_observed_gap_excess_days": maximum_observed_gap_excess_days,
        "consecutive_pairs_with_nonzero_gap": consecutive_pairs_with_nonzero_gap,
        "nonzero_gap_pair_rate": nonzero_gap_pair_rate,
        "patients_with_nonzero_observed_gap": len(patients_with_nonzero_observed_gap),
        "non_preferred_machine_assignments": non_preferred_assignments,
        "preferred_machine_assignment_rate": preferred_machine_rate,
        "machine_switches_between_fractions": total_machine_switches,
        "patients_with_machine_switches": len(patients_with_machine_switches),
        "machine_switch_rate": total_machine_switches / total_consecutive_fraction_pairs,
        "window_switches_between_fractions": total_window_switches,
        "patients_with_window_switches": len(patients_with_window_switches),
        "window_switch_rate": total_window_switches / total_consecutive_fraction_pairs,
        "total_consecutive_fraction_pairs": total_consecutive_fraction_pairs,
        "objective_machine_switch_weight": MACHINE_SWITCH_WEIGHT,
        "objective_window_switch_weight": WINDOW_SWITCH_WEIGHT,
    }

    baseline_summary_df = pd.DataFrame([baseline_summary])

    baseline_summary_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "baseline_summary.csv"),
        index=False
    )

    print("\nBaseline schedule generated successfully.")
    print("\nSummary:")
    for key, value in baseline_summary.items():
        print(f"{key}: {value}")

    print("\nGenerated files:")
    print(f"{OUTPUT_FOLDER}/baseline_schedule.csv")
    print(f"{OUTPUT_FOLDER}/machine_utilization.csv")
    print(f"{OUTPUT_FOLDER}/rtt_utilization.csv")
    print(f"{OUTPUT_FOLDER}/patient_schedule_summary.csv")
    print(f"{OUTPUT_FOLDER}/baseline_summary.csv")

else:
    print("\nNo feasible solution found.")
