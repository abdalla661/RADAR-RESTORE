import ast
import os
import time
from collections import defaultdict
import pandas as pd
from ortools.sat.python import cp_model
BUILD_ID = "RESTORE_CONSISTENT_CAPACITY_BASE_WEIGHTS_2026_07_26"

# ============================================================
# RESTORE RECOVERY MODEL — OPTIMIZED IMPLEMENTATION
# ============================================================
#
# RESTORE = REcovery with STability-Optimality trade-offs
#           in Radiotherapy under disruptive Events
# Mathematical logic is unchanged from the original RESTORE model:
# - delivered fractions before the disruption instant are fixed, including earlier same-day windows,
# - unaffected fractions in the disruption-start window remain fixed,
# - affected fractions in the disruption-start window and later fractions are reoptimized,
# - every future fraction is scheduled exactly once,
# - machine capacity, RTT capacity, ordering, continuity, and
#   one-fraction-per-patient-per-day constraints are enforced,
# - objective balances baseline optimality and schedule stability.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "modeled_instance"
BASELINE_FOLDER = "baseline_results"
DISRUPTION_FOLDER = "disruption_results"
OUTPUT_FOLDER = "restore_results"

SCENARIOS = [
    "D1_machine_window",
    "D2_machine_full_day",
    "D3_power_outage_window",
    "D4_rtt_shortage_50pct",
    # "D4_rtt_shortage_25pct",
    # "D4_rtt_shortage_75pct",
    # "D5_combined_machine_day_rtt",
]

TIME_LIMIT_SECONDS = 1800
NUM_WORKERS = 8
# The recovery horizon remains the original planning horizon. Candidate
# generation includes every treatment-compatible machine and removes only
# assignments that are provably infeasible.
RECOVERY_HORIZON_EXTENSION_DAYS = 0


COURSE_SLACK_DAYS = 14

# Lambda values are represented as integers to avoid floating point objective.
LAMBDA_NUM = 5
LAMBDA_DEN = 10

# Baseline optimality weights and terms — exactly identical to baseline_scheduler.py
WAITING_WEIGHT = 100
GAP_WEIGHT = 5
NON_PREFERRED_MACHINE_WEIGHT = 10
MACHINE_SWITCH_WEIGHT = 10
WINDOW_SWITCH_WEIGHT = 0


# Stability weights
# Stability is modeled as a layered cost:
# - MOVED: fixed administrative/operational cost of altering an appointment at all
# - MACHINE/WINDOW/DAY CHANGE: extra cost for the type of alteration
# - DAY SHIFT: extra cost proportional to displacement magnitude
MOVED_FRACTION_WEIGHT = 10
MACHINE_CHANGE_WEIGHT = 8
WINDOW_CHANGE_WEIGHT = 2
DAY_CHANGE_WEIGHT = 15
DAY_SHIFT_WEIGHT = 4

# Set to True only for debugging. Empty names are much faster for large models.
DEBUG_VAR_NAMES = False

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# SIMPLE TIMER
# ============================================================

class Timer:
    def __init__(self):
        self.start = time.time()
        self.last = self.start
        self.rows = []

    def mark(self, label):
        now = time.time()
        delta = now - self.last
        cumulative = now - self.start
        self.rows.append({
            "section": label,
            "seconds": delta,
            "cumulative_seconds": cumulative,
        })
        print(f"[TIMER] {label}: {delta:.2f}s | cumulative: {cumulative:.2f}s")
        self.last = now


# ============================================================
# LOAD DATA
# ============================================================

patients = pd.read_csv(os.path.join(INPUT_FOLDER, "patients.csv"))
fractions = pd.read_csv(os.path.join(INPUT_FOLDER, "fractions.csv"))
machine_capacity = pd.read_csv(os.path.join(INPUT_FOLDER, "machine_capacity.csv"))
rtt_capacity = pd.read_csv(os.path.join(INPUT_FOLDER, "rtt_capacity.csv"))
baseline_schedule = pd.read_csv(os.path.join(BASELINE_FOLDER, "baseline_schedule.csv"))

patients["allowed_machines"] = patients["allowed_machines"].apply(ast.literal_eval)
fractions["allowed_machines"] = fractions["allowed_machines"].apply(ast.literal_eval)

if "preferred_machines" in patients.columns:
    patients["preferred_machines"] = patients["preferred_machines"].apply(
        lambda v: ast.literal_eval(v)
        if isinstance(v, str) and v.strip() not in ["", "[]"]
        else []
    )
else:
    patients["preferred_machines"] = [[] for _ in range(len(patients))]

for df in [machine_capacity, rtt_capacity, baseline_schedule]:
    for col in ["machine", "day", "window"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

baseline_schedule["patient"] = baseline_schedule["patient"].astype(int)
baseline_schedule["fraction"] = baseline_schedule["fraction"].astype(int)
baseline_schedule["duration_minutes"] = baseline_schedule["duration_minutes"].astype(int)

# ============================================================
# LIMITED RECOVERY-HORIZON EXTENSION
# ============================================================
#
# A severe disruption removes real capacity from the original horizon.
# Global reoptimization can redistribute work, but it cannot create capacity.
# We therefore append a limited number of future treatment days.
#
# New-day machine capacity is based on the median TOTAL capacity observed for
# each machine/window in the original horizon:
#     total capacity = baseline residual capacity + baseline scheduled minutes.
#
# New-day RTT capacity is based on the median available RTT teams per window.
# These new days contain no baseline appointments, so all reconstructed
# capacity is available to recovery.
# ============================================================

if RECOVERY_HORIZON_EXTENSION_DAYS > 0:
    original_max_day = int(machine_capacity["day"].max())

    baseline_used_by_slot = (
        baseline_schedule.groupby(["machine", "day", "window"])["duration_minutes"]
        .sum()
        .to_dict()
    )

    machine_capacity = machine_capacity.copy()
    machine_capacity["_total_capacity_minutes"] = machine_capacity.apply(
        lambda row: float(row["remaining_machine_capacity_minutes"])
        + float(
            baseline_used_by_slot.get(
                (int(row["machine"]), int(row["day"]), int(row["window"])),
                0,
            )
        ),
        axis=1,
    )

    typical_machine_capacity = (
        machine_capacity.groupby(["machine", "window"])["_total_capacity_minutes"]
        .median()
        .to_dict()
    )

    if "available_rtt_teams" in rtt_capacity.columns:
        _extension_rtt_col = "available_rtt_teams"
    elif "baseline_rtt_capacity_units" in rtt_capacity.columns:
        _extension_rtt_col = "baseline_rtt_capacity_units"
    else:
        raise ValueError("No RTT capacity column available for horizon extension.")

    typical_rtt_capacity = (
        rtt_capacity.groupby("window")[_extension_rtt_col]
        .median()
        .round()
        .astype(int)
        .to_dict()
    )

    machines_for_extension = sorted(
        machine_capacity["machine"].astype(int).unique().tolist()
    )
    windows_for_extension = sorted(
        machine_capacity["window"].astype(int).unique().tolist()
    )

    new_machine_rows = []
    new_rtt_rows = []

    for offset in range(1, RECOVERY_HORIZON_EXTENSION_DAYS + 1):
        new_day = original_max_day + offset

        for machine in machines_for_extension:
            for window in windows_for_extension:
                total_capacity = int(
                    round(
                        typical_machine_capacity.get(
                            (int(machine), int(window)),
                            0,
                        )
                    )
                )
                new_machine_rows.append({
                    "machine": int(machine),
                    "day": int(new_day),
                    "window": int(window),
                    "remaining_machine_capacity_minutes": total_capacity,
                    "_total_capacity_minutes": total_capacity,
                })

        for window in windows_for_extension:
            new_rtt_rows.append({
                "day": int(new_day),
                "window": int(window),
                _extension_rtt_col: int(
                    typical_rtt_capacity.get(int(window), 0)
                ),
            })

    machine_capacity = pd.concat(
        [machine_capacity, pd.DataFrame(new_machine_rows)],
        ignore_index=True,
        sort=False,
    )
    rtt_capacity = pd.concat(
        [rtt_capacity, pd.DataFrame(new_rtt_rows)],
        ignore_index=True,
        sort=False,
    )

    machine_capacity.drop(
        columns=["_total_capacity_minutes"],
        inplace=True,
        errors="ignore",
    )


# ============================================================
# SETS AND DICTIONARIES
# ============================================================

P = patients["patient"].astype(int).tolist()
M = sorted(machine_capacity["machine"].astype(int).unique().tolist())
D = sorted(machine_capacity["day"].astype(int).unique().tolist())
W = sorted(machine_capacity["window"].astype(int).unique().tolist())
MAX_DAY = max(D)

cap_base = {
    (int(row.machine), int(row.day), int(row.window)):
        int(row.remaining_machine_capacity_minutes)
    for row in machine_capacity.itertuples()
}

rtt_col = "available_rtt_teams"
if rtt_col not in rtt_capacity.columns:
    rtt_col = "baseline_rtt_capacity_units"

patient_info = {}

for row in patients.itertuples():
    allowed = [int(m) for m in row.allowed_machines]
    preferred = row.preferred_machines

    if preferred is None or len(preferred) == 0:
        preferred = allowed
    else:
        preferred = [int(m) for m in preferred]

    patient_info[int(row.patient)] = {
        "priority": int(row.priority),
        "earliest": int(row.earliest_start_day),
        "target": int(row.target_day),
        "number_fractions": int(row.number_fractions),
        "allowed_machines": allowed,
        "preferred_machines": preferred,
        "preferred_set": set(preferred),
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
        "allowed_machines": [int(m) for m in row.allowed_machines],
    }

fractions_by_patient = (
    fractions.groupby("patient")["fraction"]
    .apply(list)
    .to_dict()
)

fractions_by_patient = {
    int(p): sorted([int(f) for f in flist])
    for p, flist in fractions_by_patient.items()
}

baseline_map = {}

for row in baseline_schedule.itertuples():
    baseline_map[(int(row.patient), int(row.fraction))] = {
        "machine": int(row.machine),
        "day": int(row.day),
        "window": int(row.window),
        "duration_minutes": int(row.duration_minutes),
        "priority": int(row.priority),
        "earliest_start_day": int(row.earliest_start_day),
        "target_day": int(row.target_day),
        "is_preferred_machine": bool(row.is_preferred_machine),
    }

# Cache priority weights.
priority_weight_by_patient = {}
for p in P:
    priority = int(patient_info[p]["priority"])
    if priority == 1:
        priority_weight_by_patient[p] = 10
    elif priority == 2:
        priority_weight_by_patient[p] = 3
    else:
        priority_weight_by_patient[p] = 1


# ============================================================
# HELPERS
# ============================================================

def scenario_paths(scenario_id):
    return (
        os.path.join(DISRUPTION_FOLDER, f"affected_fractions_{scenario_id}.csv"),
        os.path.join(DISRUPTION_FOLDER, f"no_recovery_summary_{scenario_id}.csv"),
    )


def load_disruption_summary(scenario_id):
    _, summary_path = scenario_paths(scenario_id)

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing disruption summary: {summary_path}")

    summary = pd.read_csv(summary_path)

    if summary.empty:
        raise ValueError(f"Empty disruption summary: {summary_path}")

    return summary.iloc[0].to_dict()


def infer_disruption_day(summary):
    for key in ["failed_day", "affected_day"]:
        if key in summary and pd.notna(summary[key]):
            return int(summary[key])
    raise ValueError("Could not infer disruption day.")


def infer_disruption_window(summary):
    """
    Return the disruption-start window.

    Window-specific disruptions use their reported failed window.
    D2 also reports the unexpected failure-start window; the failed machine
    remains unavailable from that window through the end of the day.
    """
    for key in ["failed_window", "affected_window", "disruption_window"]:
        if key in summary and pd.notna(summary[key]):
            return int(summary[key])

    return int(min(W))


def build_forbidden_machine_slots(summary):
    forbidden = set()
    disruption_type = str(summary.get("disruption_type", ""))

    if disruption_type == "machine_failure_window":
        failed_machine = int(summary["failed_machine"])
        failed_day = int(summary["failed_day"])
        failed_window = int(summary["failed_window"])
        forbidden.add((failed_machine, failed_day, failed_window))

    elif disruption_type in [
        "machine_failure_full_day",
        "combined_machine_full_day_rtt_shortage",
    ]:
        failed_machine = int(summary["failed_machine"])
        failed_day = int(summary["failed_day"])
        failed_window = int(summary.get("failed_window", min(W)))

        # Unexpected failure starts in failed_window and persists through
        # the end of the treatment day. Earlier windows are already delivered.
        for w in W:
            if int(w) >= failed_window:
                forbidden.add((failed_machine, failed_day, int(w)))

    elif disruption_type == "power_outage_window":
        failed_day = int(summary["failed_day"])
        failed_window = int(summary["failed_window"])
        for m in M:
            forbidden.add((int(m), failed_day, failed_window))

    return forbidden


def get_rtt_capacity_for_scenario(scenario_id):
    """
    Load scenario-specific RTT capacity and fill any missing extended-horizon
    day/window entries from the extended baseline RTT table.

    Scenario files usually contain only the original planning horizon. Without
    this merge, newly appended recovery days produce missing-key errors.
    """
    possible_path = os.path.join(
        DISRUPTION_FOLDER,
        f"disrupted_rtt_capacity_{scenario_id}.csv"
    )

    if os.path.exists(possible_path):
        scenario_rtt_df = pd.read_csv(possible_path)
    else:
        scenario_rtt_df = rtt_capacity.copy()

    for col in ["day", "window"]:
        scenario_rtt_df[col] = scenario_rtt_df[col].astype(int)

    scenario_map = {
        (int(row.day), int(row.window)): int(getattr(row, rtt_col))
        for row in scenario_rtt_df.itertuples()
    }

    baseline_extended_map = {
        (int(row.day), int(row.window)): int(getattr(row, rtt_col))
        for row in rtt_capacity.itertuples()
    }

    # Preserve disruption-specific values where present; use extended baseline
    # values only for missing day/window combinations.
    return {
        (int(d), int(w)): int(
            scenario_map.get(
                (int(d), int(w)),
                baseline_extended_map[(int(d), int(w))]
            )
        )
        for d in D
        for w in W
    }
def build_available_slots_by_machine_day(cap, forbidden_slots):
    """Index positive-capacity, non-forbidden slots by (machine, day)."""
    slots_by_machine_day = defaultdict(list)
    for (m, d, w), capacity in cap.items():
        if capacity <= 0 or (m, d, w) in forbidden_slots:
            continue
        slots_by_machine_day[(m, d)].append((w, capacity))
    for key in slots_by_machine_day:
        slots_by_machine_day[key].sort(key=lambda item: item[0])
    return slots_by_machine_day


def add_weighted_objective_terms(vars_list, coeffs_list, var, coeff):
    if coeff:
        vars_list.append(var)
        coeffs_list.append(int(coeff))


# ============================================================
# RESTORE MODEL
# ============================================================


def exact_chain_filter(candidate_specs_by_fraction, reopt_keys):
    """Apply exact iterative forward and backward chain-support filtering."""
    filtered = {key: list(specs) for key, specs in candidate_specs_by_fraction.items()}
    reopt_keys = set(reopt_keys)
    removed_forward = 0
    removed_backward = 0
    changed = True
    while changed:
        changed = False
        for p, flist in fractions_by_patient.items():
            flist = sorted(flist)
            for index in range(1, len(flist)):
                previous_key = (p, flist[index - 1])
                current_key = (p, flist[index])
                if current_key not in reopt_keys:
                    continue
                if previous_key in reopt_keys:
                    previous_specs = filtered.get(previous_key, [])
                    if not previous_specs:
                        if filtered.get(current_key):
                            removed_forward += len(filtered[current_key])
                            filtered[current_key] = []
                            changed = True
                        continue
                    minimum_previous_day = min(spec[1] for spec in previous_specs)
                else:
                    minimum_previous_day = int(baseline_map[previous_key]["day"])
                before = filtered.get(current_key, [])
                after = [spec for spec in before if spec[1] > minimum_previous_day]
                if len(after) != len(before):
                    removed_forward += len(before) - len(after)
                    filtered[current_key] = after
                    changed = True

            for index in range(len(flist) - 2, -1, -1):
                current_key = (p, flist[index])
                next_key = (p, flist[index + 1])
                if current_key not in reopt_keys:
                    continue
                if next_key in reopt_keys:
                    next_specs = filtered.get(next_key, [])
                    if not next_specs:
                        if filtered.get(current_key):
                            removed_backward += len(filtered[current_key])
                            filtered[current_key] = []
                            changed = True
                        continue
                    maximum_next_day = max(spec[1] for spec in next_specs)
                else:
                    maximum_next_day = int(baseline_map[next_key]["day"])
                before = filtered.get(current_key, [])
                after = [spec for spec in before if spec[1] < maximum_next_day]
                if len(after) != len(before):
                    removed_backward += len(before) - len(after)
                    filtered[current_key] = after
                    changed = True
    empty = [key for key in sorted(reopt_keys) if not filtered.get(key)]
    if empty:
        first = empty[0]
        raise ValueError(
            f"Exact chain filtering removed every candidate for patient "
            f"{first[0]}, fraction {first[1]}. The recovery problem is "
            "infeasible for that course under the current disruption rules."
        )
    return filtered, removed_forward, removed_backward

def run_restore(scenario_id):
    timer = Timer()
    run_start = timer.start

    affected_path, _ = scenario_paths(scenario_id)
    affected = pd.read_csv(affected_path)

    affected_keys = set(
        zip(
            affected["patient"].astype(int),
            affected["fraction"].astype(int)
        )
    )

    summary = load_disruption_summary(scenario_id)
    disruption_day = infer_disruption_day(summary)
    disruption_window = infer_disruption_window(summary)
    forbidden_slots = build_forbidden_machine_slots(summary)

    # machine_capacity.csv stores capacity remaining after external/pre-existing
    # allocations but before the generated baseline schedule is assigned.
    # Start from that pre-baseline capacity. Fixed delivered appointments are
    # subtracted below; future appointments are not subtracted, so their baseline
    # capacity is automatically released for reoptimization.
    cap = dict(cap_base)
    rtt_cap = get_rtt_capacity_for_scenario(scenario_id)
    timer.mark("load disruption and scenario data")

    fixed_keys = set()
    reopt_keys = set()

    # Recovery-scope policy aligned with Full Reoptimization:
    # - all appointments before the disruption day are already delivered and fixed;
    # - earlier windows on the disruption day are already delivered and fixed;
    # - in the disruption-start window, only affected appointments are released;
    #   unaffected appointments remain fixed;
    # - later windows on the disruption day and all later days are reoptimized.
    for key, base in baseline_map.items():
        base_day = int(base["day"])
        base_window = int(base["window"])

        if base_day < disruption_day:
            fixed_keys.add(key)

        elif (
            base_day == disruption_day
            and base_window < disruption_window
        ):
            fixed_keys.add(key)

        elif (
            base_day == disruption_day
            and base_window == disruption_window
            and key not in affected_keys
        ):
            fixed_keys.add(key)

        else:
            reopt_keys.add(key)

    released_capacity_minutes = 0  # future baseline appointments were never subtracted

    # Build a patient-centred bounded recovery horizon. The first undelivered
    # fraction starts no earlier than the disruption, its own clinical lower
    # bound, and the day after the last delivered fraction. Fraction i then
    # receives COURSE_SLACK_DAYS extra candidate days beyond its earliest
    # chain-consistent position. This is a heuristic search neighbourhood,
    # applied identically to every compatible machine.
    course_bounds = {}
    patient_course_summary = {}

    for p, full_fraction_list in fractions_by_patient.items():
        full_fraction_list = sorted(int(f) for f in full_fraction_list)
        remaining_fractions = [
            f for f in full_fraction_list if (p, f) in reopt_keys
        ]
        if not remaining_fractions:
            continue

        delivered_days = [
            int(baseline_map[(p, f)]["day"])
            for f in full_fraction_list
            if (p, f) in fixed_keys
        ]
        day_after_last_delivered = (
            max(delivered_days) + 1 if delivered_days else disruption_day
        )

        first_key = (p, remaining_fractions[0])
        first_recovery_day = max(
            int(disruption_day),
            int(day_after_last_delivered),
            int(fraction_info[first_key]["earliest"]),
        )
        course_end_day = min(
            int(MAX_DAY),
            int(first_recovery_day)
            + len(remaining_fractions)
            - 1
            + int(COURSE_SLACK_DAYS),
        )

        patient_course_summary[p] = {
            "first_recovery_day": first_recovery_day,
            "course_end_day": course_end_day,
            "remaining_fractions": len(remaining_fractions),
        }

        for remaining_position, f in enumerate(remaining_fractions):
            remaining_after_current = (
                len(remaining_fractions) - remaining_position - 1
            )
            key = (p, f)
            earliest_candidate_day = max(
                int(fraction_info[key]["earliest"]),
                int(disruption_day),
                int(first_recovery_day + remaining_position),
            )
            latest_candidate_day = min(
                int(MAX_DAY - remaining_after_current),
                int(course_end_day - remaining_after_current),
            )

            if latest_candidate_day < earliest_candidate_day:
                raise ValueError(
                    f"Bounded course horizon leaves no day for patient {p}, "
                    f"fraction {f}: earliest={earliest_candidate_day}, "
                    f"latest={latest_candidate_day}, "
                    f"slack={COURSE_SLACK_DAYS}. Increase COURSE_SLACK_DAYS."
                )

            course_bounds[key] = (
                earliest_candidate_day,
                latest_candidate_day,
            )

    fixed_rows = []

    for (p, f) in fixed_keys:
        base = baseline_map[(p, f)]

        row = {
            "patient": p,
            "fraction": f,
            "machine": base["machine"],
            "day": base["day"],
            "window": base["window"],
            "duration_minutes": base["duration_minutes"],
            "priority": base["priority"],
            "earliest_start_day": base["earliest_start_day"],
            "target_day": base["target_day"],
            "is_preferred_machine": base["is_preferred_machine"],
            "restore_status": "fixed_delivered",
        }

        fixed_rows.append(row)

    # Reserve machine minutes already consumed by delivered appointments.
    # Because future baseline appointments are not subtracted from cap, their
    # original capacity is available to the recovery model without an add-back.
    fixed_capacity_minutes = 0
    for row in fixed_rows:
        slot = (
            int(row["machine"]),
            int(row["day"]),
            int(row["window"]),
        )

        if slot not in cap:
            raise KeyError(
                f"Fixed delivered appointment uses missing machine slot {slot}."
            )

        duration = int(row["duration_minutes"])
        cap[slot] -= duration
        fixed_capacity_minutes += duration

        if cap[slot] < 0:
            raise ValueError(
                "Fixed delivered appointments exceed pre-baseline available "
                f"capacity in slot {slot}: residual={cap[slot]} minutes."
            )

    fixed_active = {(d, w): set() for d in D for w in W}
    fixed_patient_day_count = {(p, d): 0 for p in P for d in D}

    for row in fixed_rows:
        fixed_active[(row["day"], row["window"])].add(row["machine"])
        fixed_patient_day_count[(row["patient"], row["day"])] += 1

    fixed_active_count = {
        (d, w): len(fixed_active[(d, w)])
        for d in D
        for w in W
    }

    slots_by_machine_day = build_available_slots_by_machine_day(cap, forbidden_slots)
    timer.mark("fixed delivered processing, remaining capacity update, and slot indexing")

    model = cp_model.CpModel()
    timer.mark("initialize CP-SAT model")

    # --------------------------------------------------------
    # Decision variables and sparse indexes
    # --------------------------------------------------------

    candidate_records = []  # (p, f, m, d, w, var)
    x_by_fraction = defaultdict(list)
    x_by_slot_vars = defaultdict(list)
    x_by_slot_durations = defaultdict(list)
    x_by_patient_day = defaultdict(list)
    day_terms_by_fraction_vars = defaultdict(list)
    day_terms_by_fraction_coeffs = defaultdict(list)
    machine_terms_by_fraction_vars = defaultdict(list)
    machine_terms_by_fraction_coeffs = defaultdict(list)
    window_terms_by_fraction_vars = defaultdict(list)
    window_terms_by_fraction_coeffs = defaultdict(list)
    candidate_count = 0

    # Objective terms represented as parallel arrays for fast WeightedSum.
    optimality_vars = []
    optimality_coeffs = []
    stability_vars = []
    stability_coeffs = []

    raw_candidate_specs = defaultdict(list)

    for (p, f) in sorted(reopt_keys):
        info = fraction_info[(p, f)]
        duration = info["duration"]
        earliest, latest_candidate_day = course_bounds[(p, f)]
        allowed_machines = set(info["allowed_machines"])
        base = baseline_map[(p, f)]
        base_machine = int(base["machine"])
        base_day = int(base["day"])
        base_window = int(base["window"])

        # Full compatible-machine domain. No baseline- or preference-based
        # restriction is imposed; infeasible slots are removed below.
        candidate_machines = sorted(allowed_machines)

        for m in candidate_machines:
            for d in D:
                if d < earliest:
                    continue
                if d > latest_candidate_day:
                    break
                if fixed_patient_day_count.get((p, d), 0) >= 1:
                    continue
                for w, capacity in slots_by_machine_day.get((m, d), []):
                    if capacity < duration:
                        continue
                    if d == disruption_day and w < disruption_window:
                        continue
                    if (
                        (p, f) in affected_keys
                        and d == base_day
                        and w <= base_window
                    ):
                        continue
                    if (
                        m not in fixed_active[(d, w)]
                        and fixed_active_count[(d, w)] >= rtt_cap[(d, w)]
                    ):
                        continue
                    raw_candidate_specs[(p, f)].append((m, d, w))

    raw_candidate_count = sum(len(v) for v in raw_candidate_specs.values())
    filtered_specs, removed_forward, removed_backward = exact_chain_filter(
        raw_candidate_specs, reopt_keys
    )

    for (p, f) in sorted(reopt_keys):
        duration = fraction_info[(p, f)]["duration"]
        preferred_set = patient_info[p]["preferred_set"]
        base = baseline_map[(p, f)]
        base_machine = int(base["machine"])
        base_day = int(base["day"])
        base_window = int(base["window"])

        for m, d, w in filtered_specs[(p, f)]:
            machine_changed = m != base_machine
            non_preferred = m not in preferred_set
            day_changed = d != base_day
            window_changed = w != base_window
            day_shift = abs(d - base_day)
            moved = machine_changed or day_changed or window_changed

            name = f"x_p{p}_f{f}_m{m}_d{d}_w{w}" if DEBUG_VAR_NAMES else ""
            var = model.NewBoolVar(name)
            candidate_records.append((p, f, m, d, w, var))
            candidate_count += 1

            slot = (m, d, w)
            x_by_fraction[(p, f)].append(var)
            x_by_slot_vars[slot].append(var)
            x_by_slot_durations[slot].append(duration)
            x_by_patient_day[(p, d)].append(var)
            day_terms_by_fraction_vars[(p, f)].append(var)
            day_terms_by_fraction_coeffs[(p, f)].append(d)
            machine_terms_by_fraction_vars[(p, f)].append(var)
            machine_terms_by_fraction_coeffs[(p, f)].append(m)
            window_terms_by_fraction_vars[(p, f)].append(var)
            window_terms_by_fraction_coeffs[(p, f)].append(w)

            if non_preferred:
                add_weighted_objective_terms(
                    optimality_vars, optimality_coeffs, var,
                    NON_PREFERRED_MACHINE_WEIGHT,
                )

            stability_coeff = 0
            if moved:
                stability_coeff += MOVED_FRACTION_WEIGHT
            if machine_changed:
                stability_coeff += MACHINE_CHANGE_WEIGHT
            if window_changed:
                stability_coeff += WINDOW_CHANGE_WEIGHT
            if day_changed:
                stability_coeff += DAY_CHANGE_WEIGHT
            if day_shift > 0:
                stability_coeff += DAY_SHIFT_WEIGHT * day_shift
            if stability_coeff:
                add_weighted_objective_terms(
                    stability_vars, stability_coeffs, var, stability_coeff,
                )

    # Initial hints are built in patient-fraction order. Baseline assignments
    # are preferred when legal; otherwise choose the nearest candidate strictly
    # after the previous hinted fraction day.
    candidates_by_fraction = defaultdict(list)
    for p, f, m, d, w, var in candidate_records:
        candidates_by_fraction[(p, f)].append((m, d, w, var))

    hinted_fraction_count = 0
    for p, flist in fractions_by_patient.items():
        previous_hinted_day = None
        for f in flist:
            key_pf = (p, f)
            if key_pf in fixed_keys:
                previous_hinted_day = int(baseline_map[key_pf]["day"])
                continue
            if key_pf not in reopt_keys:
                continue

            base = baseline_map[key_pf]
            candidates = [
                item for item in candidates_by_fraction.get(key_pf, [])
                if previous_hinted_day is None or item[1] > previous_hinted_day
            ]
            if not candidates:
                candidates = candidates_by_fraction.get(key_pf, [])
            if not candidates:
                continue

            baseline_candidate = next(
                (
                    item for item in candidates_by_fraction.get(key_pf, [])
                    if item[0] == int(base["machine"])
                    and item[1] == int(base["day"])
                    and item[2] == int(base["window"])
                ),
                None,
            )
            if baseline_candidate is not None and (
                previous_hinted_day is None or int(base["day"]) > previous_hinted_day
            ):
                chosen = baseline_candidate
            else:
                chosen = min(
                    candidates,
                    key=lambda item: (
                        abs(item[1] - int(base["day"])),
                        int(item[0] != int(base["machine"])),
                        abs(item[2] - int(base["window"])),
                        item[1], item[2], item[0],
                    ),
                )
            model.AddHint(chosen[3], 1)
            previous_hinted_day = chosen[1]
            hinted_fraction_count += 1

    print()
    print("========================================")
    print(f"RESTORE - {scenario_id}")
    print("========================================")
    print(f"Lambda: {LAMBDA_NUM}/{LAMBDA_DEN}")
    print(f"Disruption day: {disruption_day}")
    print(f"Disruption window: {disruption_window}")
    print(f"Original maximum treatment day: {MAX_DAY}")
    print("Candidate policy: all allowed machines within a patient-specific bounded course horizon")
    print(f"Course horizon slack days: {COURSE_SLACK_DAYS}")
    print(f"Raw locally feasible candidates: {raw_candidate_count}")
    print(f"Removed by forward chain filtering: {removed_forward}")
    print(f"Removed by backward chain filtering: {removed_backward}")
    print(f"Disruption window: {disruption_window}")
    fixed_nonaffected_disruption_window = sum(
        1
        for key in fixed_keys
        if int(baseline_map[key]["day"]) == disruption_day
        and int(baseline_map[key]["window"]) == disruption_window
        and key not in affected_keys
    )
    affected_reoptimized_disruption_window = sum(
        1
        for key in reopt_keys
        if int(baseline_map[key]["day"]) == disruption_day
        and int(baseline_map[key]["window"]) == disruption_window
        and key in affected_keys
    )
    print(f"Fixed delivered/non-movable fractions: {len(fixed_keys)}")
    print(f"Reoptimized future fractions: {len(reopt_keys)}")
    print(
        "Fixed non-affected fractions in disruption window: "
        f"{fixed_nonaffected_disruption_window}"
    )
    print(
        "Affected fractions released from disruption window: "
        f"{affected_reoptimized_disruption_window}"
    )
    print(f"Candidate variables: {candidate_count}")
    print(f"Initially hinted fractions: {hinted_fraction_count}")
    print(f"Forbidden machine slots: {len(forbidden_slots)}")

    # Create activity variables only for slots that can actually receive a
    # reoptimized assignment. This avoids dense machine-day-window allocation.
    z = {}
    for m, d, w in x_by_slot_vars:
        name = f"z_m{m}_d{d}_w{w}" if DEBUG_VAR_NAMES else ""
        z[(m, d, w)] = model.NewBoolVar(name)

    fraction_day = {}
    fraction_machine = {}
    fraction_window = {}

    for key in sorted(reopt_keys):
        p, f = key
        candidate_days = sorted({d for _, d, _ in filtered_specs[key]})
        candidate_machines = sorted({m for m, _, _ in filtered_specs[key]})
        candidate_windows = sorted({w for _, _, w in filtered_specs[key]})
        name = f"day_p{p}_f{f}" if DEBUG_VAR_NAMES else ""
        fraction_day[key] = model.NewIntVar(candidate_days[0], candidate_days[-1], name)
        name_m = f"machine_p{p}_f{f}" if DEBUG_VAR_NAMES else ""
        fraction_machine[key] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(candidate_machines), name_m
        )
        name_w = f"window_p{p}_f{f}" if DEBUG_VAR_NAMES else ""
        fraction_window[key] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(candidate_windows), name_w
        )

    def day_expr(key):
        if key in fixed_keys:
            return int(baseline_map[key]["day"])
        return fraction_day[key]

    def machine_expr(key):
        if key in fixed_keys:
            return int(baseline_map[key]["machine"])
        return fraction_machine[key]

    def window_expr(key):
        if key in fixed_keys:
            return int(baseline_map[key]["window"])
        return fraction_window[key]

    start_day = {}

    for p in P:
        name = f"start_day_p{p}" if DEBUG_VAR_NAMES else ""
        start_day[p] = model.NewIntVar(1, MAX_DAY, name)

    timer.mark("auxiliary variable creation z, fraction_day, start_day")

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    timer.mark("constraint fixed delivered fractions")

    for (p, f) in reopt_keys:
        vars_for_fraction = x_by_fraction.get((p, f), [])

        if not vars_for_fraction:
            raise ValueError(
                f"No feasible RESTORE candidates for patient {p}, fraction {f}, scenario {scenario_id}"
            )

        model.Add(cp_model.LinearExpr.Sum(vars_for_fraction) == 1)

    timer.mark("constraint each future fraction exactly once")

    for slot, vars_ in x_by_slot_vars.items():
        coeffs = x_by_slot_durations[slot]
        model.Add(
            cp_model.LinearExpr.WeightedSum(vars_, coeffs)
            <= cap.get(slot, 0)
        )

    timer.mark("constraint machine capacity")

    # Aggregate x -> z linking per machine-day-window.
    # This is equivalent to individual constraints x_i <= z for every candidate:
    # if z = 0 then all x_i must be 0; if z = 1 the constraint is nonbinding.
    for slot, vars_ in x_by_slot_vars.items():
        slot_sum = cp_model.LinearExpr.Sum(vars_)
        model.Add(slot_sum <= len(vars_) * z[slot])
        model.Add(z[slot] <= slot_sum)

    timer.mark("constraint link assignments to active machines")

    relevant_rtt_pairs = (
        {(d, w) for _, d, w in z}
        | {(d, w) for (d, w), machines in fixed_active.items() if machines}
    )
    for d, w in sorted(relevant_rtt_pairs):
        newly_activated_machine_vars = [
            z[(m, d, w)]
            for m in M
            if m not in fixed_active[(d, w)] and (m, d, w) in z
        ]
        model.Add(
            fixed_active_count[(d, w)]
            + cp_model.LinearExpr.Sum(newly_activated_machine_vars)
            <= rtt_cap[(d, w)]
        )

    timer.mark("constraint RTT capacity")

    for (p, f) in reopt_keys:
        vars_ = day_terms_by_fraction_vars[(p, f)]
        coeffs = day_terms_by_fraction_coeffs[(p, f)]
        model.Add(
            fraction_day[(p, f)]
            == cp_model.LinearExpr.WeightedSum(vars_, coeffs)
        )

        vars_m = machine_terms_by_fraction_vars[(p, f)]
        coeffs_m = machine_terms_by_fraction_coeffs[(p, f)]
        model.Add(
            fraction_machine[(p, f)]
            == cp_model.LinearExpr.WeightedSum(vars_m, coeffs_m)
        )

        vars_w = window_terms_by_fraction_vars[(p, f)]
        coeffs_w = window_terms_by_fraction_coeffs[(p, f)]
        model.Add(
            fraction_window[(p, f)]
            == cp_model.LinearExpr.WeightedSum(vars_w, coeffs_w)
        )

    timer.mark("constraint define future fraction day")

    for (p, d), terms in x_by_patient_day.items():
        model.Add(
            fixed_patient_day_count[(p, d)]
            + cp_model.LinearExpr.Sum(terms)
            <= 1
        )

    # Preserve infeasibility detection for impossible fixed schedules.
    for (p, d), fixed_count in fixed_patient_day_count.items():
        if fixed_count > 1 and (p, d) not in x_by_patient_day:
            model.Add(fixed_count <= 1)

    timer.mark("constraint one fraction per patient per day")

    gap_excess = {}
    machine_switch_vars = []
    window_switch_vars = []

    for p, flist in fractions_by_patient.items():
        for i in range(len(flist) - 1):
            f_current = flist[i]
            f_next = flist[i + 1]

            model.Add(
                day_expr((p, f_next))
                >= day_expr((p, f_current)) + 1
            )

            name = f"gap_excess_p{p}_f{f_current}_to_f{f_next}" if DEBUG_VAR_NAMES else ""
            gap_var = model.NewIntVar(0, MAX_DAY, name)

            model.Add(
                gap_var
                == day_expr((p, f_next))
                - day_expr((p, f_current))
                - 1
            )

            gap_excess[(p, f_current, f_next)] = gap_var

            name_ms = f"machine_switch_p{p}_f{f_current}_to_f{f_next}" if DEBUG_VAR_NAMES else ""
            machine_switch = model.NewBoolVar(name_ms)
            model.Add(machine_expr((p, f_current)) != machine_expr((p, f_next))).OnlyEnforceIf(machine_switch)
            model.Add(machine_expr((p, f_current)) == machine_expr((p, f_next))).OnlyEnforceIf(machine_switch.Not())
            machine_switch_vars.append(machine_switch)

            name_ws = f"window_switch_p{p}_f{f_current}_to_f{f_next}" if DEBUG_VAR_NAMES else ""
            window_switch = model.NewBoolVar(name_ws)
            model.Add(window_expr((p, f_current)) != window_expr((p, f_next))).OnlyEnforceIf(window_switch)
            model.Add(window_expr((p, f_current)) == window_expr((p, f_next))).OnlyEnforceIf(window_switch.Not())
            window_switch_vars.append(window_switch)

    timer.mark("constraint fraction ordering and continuity")

    for p, flist in fractions_by_patient.items():
        first_fraction = min(flist)
        model.Add(start_day[p] == day_expr((p, first_fraction)))

    timer.mark("constraint patient start day")

    # --------------------------------------------------------
    # Objective: baseline optimality + RESTORE stability
    # --------------------------------------------------------

    for p in P:
        earliest = patient_info[p]["earliest"]
        pw = priority_weight_by_patient[p]

        name = f"waiting_p{p}" if DEBUG_VAR_NAMES else ""
        waiting = model.NewIntVar(0, MAX_DAY, name)
        model.Add(waiting == start_day[p] - earliest)

        add_weighted_objective_terms(
            optimality_vars,
            optimality_coeffs,
            waiting,
            WAITING_WEIGHT * pw,
        )

    for (p, f_current, f_next), gap_var in gap_excess.items():
        pw = priority_weight_by_patient[p]
        add_weighted_objective_terms(
            optimality_vars,
            optimality_coeffs,
            gap_var,
            GAP_WEIGHT * pw,
        )

    for var in machine_switch_vars:
        add_weighted_objective_terms(
            optimality_vars,
            optimality_coeffs,
            var,
            MACHINE_SWITCH_WEIGHT,
        )

    for var in window_switch_vars:
        add_weighted_objective_terms(
            optimality_vars,
            optimality_coeffs,
            var,
            WINDOW_SWITCH_WEIGHT,
        )

    optimality_cost = cp_model.LinearExpr.WeightedSum(optimality_vars, optimality_coeffs)
    stability_cost = cp_model.LinearExpr.WeightedSum(stability_vars, stability_coeffs)

    final_objective = (
        LAMBDA_NUM * optimality_cost
        + (LAMBDA_DEN - LAMBDA_NUM) * stability_cost
    )

    # Single-phase RESTORE solve using the final trade-off objective.
    model.Minimize(final_objective)
    timer.mark("final objective construction")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
    solver.parameters.num_search_workers = NUM_WORKERS
    solver.parameters.log_search_progress = False
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 1
    solver.parameters.symmetry_level = 2
    solver.parameters.random_seed = 1
    if hasattr(solver.parameters, "use_optimization_hints"):
        solver.parameters.use_optimization_hints = True

    status = solver.Solve(model)
    timer.mark("CP-SAT solve")
    print(f"Solver status: {solver.StatusName(status)}")
    print(f"Solver wall time: {solver.WallTime():.2f}s")

    scenario_output = os.path.join(
        OUTPUT_FOLDER,
        f"{scenario_id}_lambda_{LAMBDA_NUM}_{LAMBDA_DEN}"
    )
    os.makedirs(scenario_output, exist_ok=True)
    timer.mark("create scenario output folder")

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        pd.DataFrame([{
            "scenario_id": scenario_id,
            "recovery_method": "restore_layered_stability",
            "lambda_num": LAMBDA_NUM,
            "lambda_den": LAMBDA_DEN,
            "lambda_value": LAMBDA_NUM / LAMBDA_DEN,
            "status": solver.StatusName(status),
            "solver_wall_time_seconds": solver.WallTime(),
            "script_runtime_seconds": time.time() - run_start,
            "disruption_day": disruption_day,
            "disruption_window": disruption_window,
            "fixed_delivered_capacity_minutes": fixed_capacity_minutes,
            "fixed_delivered_fractions": len(fixed_keys),
            "reoptimized_future_fractions": len(reopt_keys),
            "candidate_variables": candidate_count,
            "course_slack_days": COURSE_SLACK_DAYS,
            "candidate_policy": "patient_specific_bounded_course_horizon",
            "forbidden_machine_slots": len(forbidden_slots),
        }]).to_csv(
            os.path.join(scenario_output, "restore_summary.csv"),
            index=False
        )
        pd.DataFrame(timer.rows).to_csv(
            os.path.join(scenario_output, "restore_build_timing_report.csv"),
            index=False
        )
        return

    recovered_rows = []
    recovered_rows.extend(fixed_rows)

    # Extract only selected future variables.
    for p, f, m, d, w, var in candidate_records:
        if solver.Value(var) == 1:
            recovered_rows.append({
                "patient": p,
                "fraction": f,
                "machine": m,
                "day": d,
                "window": w,
                "duration_minutes": fraction_info[(p, f)]["duration"],
                "priority": patient_info[p]["priority"],
                "earliest_start_day": patient_info[p]["earliest"],
                "target_day": patient_info[p]["target"],
                "is_preferred_machine": m in patient_info[p]["preferred_set"],
                "restore_status": "reoptimized_future",
            })

    recovered_schedule = pd.DataFrame(recovered_rows)
    recovered_schedule = recovered_schedule.sort_values(
        ["day", "window", "machine", "patient", "fraction"]
    )

    recovered_schedule.to_csv(
        os.path.join(scenario_output, "restore_schedule.csv"),
        index=False
    )
    timer.mark("extract solution and write recovered schedule")

    recovered_keys = set(
        zip(
            recovered_schedule["patient"].astype(int),
            recovered_schedule["fraction"].astype(int)
        )
    )

    affected_total = len(affected_keys)
    affected_recovered = len([key for key in affected_keys if key in recovered_keys])
    affected_unrecovered = affected_total - affected_recovered

    moved_fractions = 0
    machine_changes = 0
    window_changes = 0
    day_changes = 0
    same_day_machine_changes = 0
    same_day_window_changes = 0
    same_day_machine_and_window_changes = 0
    same_machine_same_day_window_changes = 0
    total_abs_time_shift = 0
    max_abs_time_shift = 0

    total_delay = 0
    max_delay = 0
    non_preferred_assignments = 0

    for row in recovered_schedule.itertuples():
        key = (int(row.patient), int(row.fraction))
        base = baseline_map[key]

        machine_changed = int(row.machine) != int(base["machine"])
        day_changed = int(row.day) != int(base["day"])
        window_changed = int(row.window) != int(base["window"])
        moved = machine_changed or day_changed or window_changed

        if moved:
            moved_fractions += 1
        if machine_changed:
            machine_changes += 1
        if window_changed:
            window_changes += 1
        if day_changed:
            day_changes += 1
        if machine_changed and not day_changed:
            same_day_machine_changes += 1
        if window_changed and not day_changed:
            same_day_window_changes += 1
        if machine_changed and window_changed and not day_changed:
            same_day_machine_and_window_changes += 1
        if (not machine_changed) and window_changed and (not day_changed):
            same_machine_same_day_window_changes += 1

        shift = int(row.day) - int(base["day"])
        abs_shift = abs(shift)
        total_abs_time_shift += abs_shift
        max_abs_time_shift = max(max_abs_time_shift, abs_shift)

        delay = max(0, shift)
        total_delay += delay
        max_delay = max(max_delay, delay)

        if not bool(row.is_preferred_machine):
            non_preferred_assignments += 1

    avg_abs_shift = total_abs_time_shift / len(recovered_schedule)
    avg_delay = total_delay / len(recovered_schedule)

    patient_starts = recovered_schedule.groupby("patient")["day"].min().to_dict()

    total_weighted_waiting = 0
    for p in P:
        waiting = int(patient_starts[p]) - int(patient_info[p]["earliest"])
        total_weighted_waiting += WAITING_WEIGHT * priority_weight_by_patient[p] * waiting

    schedule_day_map = {
        (int(row.patient), int(row.fraction)): int(row.day)
        for row in recovered_schedule.itertuples()
    }

    total_weighted_gap = 0
    for p, flist in fractions_by_patient.items():
        pw = priority_weight_by_patient[p]
        for i in range(len(flist) - 1):
            f_current = flist[i]
            f_next = flist[i + 1]
            gap = (
                schedule_day_map[(p, f_next)]
                - schedule_day_map[(p, f_current)]
                - 1
            )
            total_weighted_gap += GAP_WEIGHT * pw * gap

    schedule_machine_map = {
        (int(row.patient), int(row.fraction)): int(row.machine)
        for row in recovered_schedule.itertuples()
    }
    schedule_window_map = {
        (int(row.patient), int(row.fraction)): int(row.window)
        for row in recovered_schedule.itertuples()
    }

    total_consecutive_machine_switches = 0
    total_consecutive_window_switches = 0
    for p, flist in fractions_by_patient.items():
        flist = sorted([int(f) for f in flist])
        for i in range(len(flist) - 1):
            f_current = flist[i]
            f_next = flist[i + 1]
            total_consecutive_machine_switches += int(
                schedule_machine_map[(p, f_current)] != schedule_machine_map[(p, f_next)]
            )
            total_consecutive_window_switches += int(
                schedule_window_map[(p, f_current)] != schedule_window_map[(p, f_next)]
            )

    reported_optimality_cost = (
        total_weighted_waiting
        + total_weighted_gap
        + NON_PREFERRED_MACHINE_WEIGHT * non_preferred_assignments
        + MACHINE_SWITCH_WEIGHT * total_consecutive_machine_switches
        + WINDOW_SWITCH_WEIGHT * total_consecutive_window_switches
    )

    reported_stability_cost = (
        MOVED_FRACTION_WEIGHT * moved_fractions
        + MACHINE_CHANGE_WEIGHT * machine_changes
        + WINDOW_CHANGE_WEIGHT * window_changes
        + DAY_CHANGE_WEIGHT * day_changes
        + DAY_SHIFT_WEIGHT * total_abs_time_shift
    )

    # --------------------------------------------------------
    # Additional interpretability metrics
    # --------------------------------------------------------
    # These metrics mirror the Full Reoptimization summary so RESTORE can be
    # compared directly against it in tables:
    # - stability relative to the original baseline,
    # - clinical consistency inside each patient's recovered course,
    # - permanent future machine/window relocations,
    # - objective decomposition.
    # --------------------------------------------------------

    recovered_keyed = {
        (int(row.patient), int(row.fraction)): row
        for row in recovered_schedule.itertuples()
    }

    affected_moved_fractions = 0
    nonaffected_moved_fractions = 0

    patients_with_any_moved_fraction = set()
    patients_with_machine_change_from_baseline = set()
    patients_with_window_change_from_baseline = set()
    patients_with_day_shift_from_baseline = set()

    affected_patients_with_moved_fraction = set()
    nonaffected_patients_with_moved_fraction = set()

    for row in recovered_schedule.itertuples():
        key = (int(row.patient), int(row.fraction))
        base = baseline_map[key]

        machine_changed = int(row.machine) != int(base["machine"])
        day_changed = int(row.day) != int(base["day"])
        window_changed = int(row.window) != int(base["window"])
        moved = machine_changed or day_changed or window_changed

        if moved:
            patients_with_any_moved_fraction.add(int(row.patient))

            if key in affected_keys:
                affected_moved_fractions += 1
                affected_patients_with_moved_fraction.add(int(row.patient))
            else:
                nonaffected_moved_fractions += 1
                nonaffected_patients_with_moved_fraction.add(int(row.patient))

        if machine_changed:
            patients_with_machine_change_from_baseline.add(int(row.patient))

        if window_changed:
            patients_with_window_change_from_baseline.add(int(row.patient))

        if day_changed:
            patients_with_day_shift_from_baseline.add(int(row.patient))

    recovered_machine_switches_between_fractions = 0
    recovered_window_switches_between_fractions = 0
    baseline_machine_switches_between_fractions = 0
    baseline_window_switches_between_fractions = 0

    # Actual continuity gaps in the RESTORE schedule.
    recovered_observed_gap_excess_values = []

    patients_with_recovered_machine_switches = set()
    patients_with_recovered_window_switches = set()
    patients_with_baseline_machine_switches = set()
    patients_with_baseline_window_switches = set()

    total_consecutive_fraction_pairs = 0

    patients_with_future_machine_relocation = set()
    patients_with_future_window_relocation = set()

    for p, flist in fractions_by_patient.items():
        p = int(p)
        flist = sorted([int(f) for f in flist])
        total_consecutive_fraction_pairs += max(len(flist) - 1, 0)

        for i in range(len(flist) - 1):
            f_current = flist[i]
            f_next = flist[i + 1]

            curr_rec = recovered_keyed[(p, f_current)]
            next_rec = recovered_keyed[(p, f_next)]

            curr_base = baseline_map[(p, f_current)]
            next_base = baseline_map[(p, f_next)]

            recovered_gap_excess = max(
                0,
                int(next_rec.day) - int(curr_rec.day) - 1
            )
            recovered_observed_gap_excess_values.append(recovered_gap_excess)

            if int(curr_rec.machine) != int(next_rec.machine):
                recovered_machine_switches_between_fractions += 1
                patients_with_recovered_machine_switches.add(p)

            if int(curr_rec.window) != int(next_rec.window):
                recovered_window_switches_between_fractions += 1
                patients_with_recovered_window_switches.add(p)

            if int(curr_base["machine"]) != int(next_base["machine"]):
                baseline_machine_switches_between_fractions += 1
                patients_with_baseline_machine_switches.add(p)

            if int(curr_base["window"]) != int(next_base["window"]):
                baseline_window_switches_between_fractions += 1
                patients_with_baseline_window_switches.add(p)

        future_fractions = [
            f for f in flist
            if (p, f) in recovered_keyed
            and getattr(recovered_keyed[(p, f)], "restore_status") == "reoptimized_future"
        ]

        if future_fractions:
            rec_future_machines = {
                int(recovered_keyed[(p, f)].machine)
                for f in future_fractions
            }
            base_future_machines = {
                int(baseline_map[(p, f)]["machine"])
                for f in future_fractions
            }

            rec_future_windows = {
                int(recovered_keyed[(p, f)].window)
                for f in future_fractions
            }
            base_future_windows = {
                int(baseline_map[(p, f)]["window"])
                for f in future_fractions
            }

            if len(rec_future_machines) == 1 and rec_future_machines != base_future_machines:
                patients_with_future_machine_relocation.add(p)

            if len(rec_future_windows) == 1 and rec_future_windows != base_future_windows:
                patients_with_future_window_relocation.add(p)

    unchanged_fractions_total = len(recovered_schedule) - moved_fractions

    moved_fraction_rate = (
        moved_fractions / len(recovered_schedule)
        if len(recovered_schedule)
        else 0.0
    )
    machine_change_rate = (
        machine_changes / len(recovered_schedule)
        if len(recovered_schedule)
        else 0.0
    )
    window_change_rate = (
        window_changes / len(recovered_schedule)
        if len(recovered_schedule)
        else 0.0
    )
    day_shift_fraction_rate = (
        day_changes / len(recovered_schedule)
        if len(recovered_schedule)
        else 0.0
    )

    total_recovered_observed_gap_excess_days = int(
        sum(recovered_observed_gap_excess_values)
    )
    average_recovered_observed_gap_excess_days = (
        total_recovered_observed_gap_excess_days
        / len(recovered_observed_gap_excess_values)
        if recovered_observed_gap_excess_values
        else 0.0
    )
    maximum_recovered_observed_gap_excess_days = (
        max(recovered_observed_gap_excess_values)
        if recovered_observed_gap_excess_values
        else 0
    )

    recovered_machine_switch_rate = (
        recovered_machine_switches_between_fractions / total_consecutive_fraction_pairs
        if total_consecutive_fraction_pairs
        else 0.0
    )
    recovered_window_switch_rate = (
        recovered_window_switches_between_fractions / total_consecutive_fraction_pairs
        if total_consecutive_fraction_pairs
        else 0.0
    )
    baseline_machine_switch_rate = (
        baseline_machine_switches_between_fractions / total_consecutive_fraction_pairs
        if total_consecutive_fraction_pairs
        else 0.0
    )
    baseline_window_switch_rate = (
        baseline_window_switches_between_fractions / total_consecutive_fraction_pairs
        if total_consecutive_fraction_pairs
        else 0.0
    )

    weighted_optimality_component = LAMBDA_NUM * reported_optimality_cost
    weighted_stability_component = (LAMBDA_DEN - LAMBDA_NUM) * reported_stability_cost

    timer.mark("compute recovery and stability metrics")

    summary_row = {
        "scenario_id": scenario_id,
        "recovery_method": "restore_layered_stability",
        "lambda_num": LAMBDA_NUM,
        "lambda_den": LAMBDA_DEN,
        "lambda_value": LAMBDA_NUM / LAMBDA_DEN,
        "status": solver.StatusName(status),
        "objective_value": solver.ObjectiveValue(),
        "reported_optimality_cost": reported_optimality_cost,
        "reported_stability_cost": reported_stability_cost,
        "weighted_optimality_component": weighted_optimality_component,
        "weighted_stability_component": weighted_stability_component,
        "total_consecutive_machine_switches": total_consecutive_machine_switches,
        "total_consecutive_window_switches": total_consecutive_window_switches,
        "objective_waiting_weight": WAITING_WEIGHT,
        "objective_gap_weight": GAP_WEIGHT,
        "objective_non_preferred_machine_weight": NON_PREFERRED_MACHINE_WEIGHT,
        "objective_machine_switch_weight": MACHINE_SWITCH_WEIGHT,
        "objective_window_switch_weight": WINDOW_SWITCH_WEIGHT,
        "solver_wall_time_seconds": solver.WallTime(),
        "script_runtime_seconds": time.time() - run_start,
        "disruption_day": disruption_day,
        "disruption_window": disruption_window,
        "fixed_delivered_capacity_minutes": fixed_capacity_minutes,
        "fixed_delivered_fractions": len(fixed_keys),
        "reoptimized_future_fractions": len(reopt_keys),
        "affected_fractions": affected_total,
        "affected_recovered": affected_recovered,
        "affected_unrecovered": affected_unrecovered,
        "recovery_rate": affected_recovered / affected_total if affected_total else 1.0,
        "total_scheduled_fractions_after_restore": len(recovered_schedule),
        "total_patients_after_restore": recovered_schedule["patient"].nunique(),
        "moved_fractions_total": moved_fractions,
        "unchanged_fractions_total": unchanged_fractions_total,
        "moved_fraction_rate": moved_fraction_rate,
        "affected_moved_fractions": affected_moved_fractions,
        "nonaffected_moved_fractions": nonaffected_moved_fractions,
        "patients_with_any_moved_fraction": len(patients_with_any_moved_fraction),
        "patients_with_machine_change_from_baseline": len(patients_with_machine_change_from_baseline),
        "patients_with_window_change_from_baseline": len(patients_with_window_change_from_baseline),
        "patients_with_day_shift_from_baseline": len(patients_with_day_shift_from_baseline),
        "affected_patients_with_moved_fraction": len(affected_patients_with_moved_fraction),
        "nonaffected_patients_with_moved_fraction": len(nonaffected_patients_with_moved_fraction),
        "machine_changes_total": machine_changes,
        "machine_change_rate": machine_change_rate,
        "window_changes_total": window_changes,
        "window_change_rate": window_change_rate,
        "day_changes": day_changes,
        "same_day_machine_changes": same_day_machine_changes,
        "same_day_window_changes": same_day_window_changes,
        "same_day_machine_and_window_changes": same_day_machine_and_window_changes,
        "same_machine_same_day_window_changes": same_machine_same_day_window_changes,
        "total_absolute_time_shift": total_abs_time_shift,
        "average_absolute_time_shift": avg_abs_shift,
        "max_absolute_time_shift": max_abs_time_shift,
        "total_delay_days": total_delay,
        "average_delay_days": avg_delay,
        "max_delay_days": max_delay,
        "day_shift_fraction_rate": day_shift_fraction_rate,

        # Clinical course consistency after RESTORE
        "total_consecutive_fraction_pairs": total_consecutive_fraction_pairs,
        "total_recovered_observed_gap_excess_days": total_recovered_observed_gap_excess_days,
        "average_recovered_observed_gap_excess_days": average_recovered_observed_gap_excess_days,
        "maximum_recovered_observed_gap_excess_days": maximum_recovered_observed_gap_excess_days,
        "recovered_machine_switches_between_fractions": recovered_machine_switches_between_fractions,
        "patients_with_recovered_machine_switches": len(patients_with_recovered_machine_switches),
        "recovered_machine_switch_rate": recovered_machine_switch_rate,
        "recovered_window_switches_between_fractions": recovered_window_switches_between_fractions,
        "patients_with_recovered_window_switches": len(patients_with_recovered_window_switches),
        "recovered_window_switch_rate": recovered_window_switch_rate,

        # Baseline course consistency for comparison
        "baseline_machine_switches_between_fractions": baseline_machine_switches_between_fractions,
        "patients_with_baseline_machine_switches": len(patients_with_baseline_machine_switches),
        "baseline_machine_switch_rate": baseline_machine_switch_rate,
        "baseline_window_switches_between_fractions": baseline_window_switches_between_fractions,
        "patients_with_baseline_window_switches": len(patients_with_baseline_window_switches),
        "baseline_window_switch_rate": baseline_window_switch_rate,

        # Permanent future relocation indicators
        "patients_with_future_machine_relocation": len(patients_with_future_machine_relocation),
        "patients_with_future_window_relocation": len(patients_with_future_window_relocation),

        "non_preferred_assignments": non_preferred_assignments,
        "candidate_variables": candidate_count,
        "course_slack_days": COURSE_SLACK_DAYS,
        "candidate_policy": "patient_specific_bounded_course_horizon",
        "forbidden_machine_slots": len(forbidden_slots),
        "waiting_weight": WAITING_WEIGHT,
        "gap_weight": GAP_WEIGHT,
        "non_preferred_machine_weight": NON_PREFERRED_MACHINE_WEIGHT,
        "moved_fraction_weight": MOVED_FRACTION_WEIGHT,
        "machine_change_weight": MACHINE_CHANGE_WEIGHT,
        "window_change_weight": WINDOW_CHANGE_WEIGHT,
        "day_change_weight": DAY_CHANGE_WEIGHT,
        "day_shift_weight": DAY_SHIFT_WEIGHT,
    }

    pd.DataFrame([summary_row]).to_csv(
        os.path.join(scenario_output, "restore_summary.csv"),
        index=False
    )

    pd.DataFrame(timer.rows).to_csv(
        os.path.join(scenario_output, "restore_build_timing_report.csv"),
        index=False
    )
    timer.mark("write restore summary and timing report")

    print("Summary:")
    for key, value in summary_row.items():
        print(f"{key}: {value}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"BUILD ID: {BUILD_ID}")

    print("========================================")
    print("RESTORE RECOVERY")
    print("========================================")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Baseline folder: {BASELINE_FOLDER}")
    print(f"Disruption folder: {DISRUPTION_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Lambda: {LAMBDA_NUM}/{LAMBDA_DEN}")

    for scenario in SCENARIOS:
        run_restore(scenario)

    print()
    print("========================================")
    print("RESTORE COMPLETE")
    print("========================================")