import os
import time
from collections import defaultdict
import pandas as pd

BUILD_ID = "PATIENT_LOCAL_REPAIR_UPDATED_2026_07_26"


# ============================================================
# PATIENT-LOCAL REPAIR RECOVERY
# ============================================================
#
# Purpose:
# Repair disrupted radiotherapy schedules locally around each affected patient.
# Other patients remain fixed throughout the repair.
#
# Recovery principle:
# - All appointments belonging to other patients remain fixed.
# - All already delivered fractions remain fixed.
# - Affected appointments are removed from the disrupted slot.
# - First, try a strictly later same-day window on the original machine.
# - Then try a strictly later same-day window on another compatible machine.
# - Then try an available day before the next fixed fraction.
# - If direct insertion is impossible, shift the affected fraction and the
#   remaining fractions of the same patient together.
# - No appointment belonging to another patient may move.
# - Fraction order, machine capacity, RTT capacity, and compatibility are enforced.
#
# Feasibility checks:
# - machine compatibility approximated using machines observed for the patient
#   in the baseline schedule, plus the original affected machine
# - remaining machine capacity
# - RTT capacity
# - no same-day duplicate fraction for the same patient
# - chronological order of fraction labels is preserved
# #
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "modeled_instance"
BASELINE_FOLDER = "baseline_results"
DISRUPTION_FOLDER = "disruption_results"
OUTPUT_FOLDER = "patient_local_repair_results"

SCENARIOS = [
    "D1_machine_window",
    "D2_machine_full_day",
    "D3_power_outage_window",
    # "D4_rtt_shortage_25pct",
    "D4_rtt_shortage_50pct",
    # "D4_rtt_shortage_75pct",
    # "D5_combined_machine_day_rtt",
]

MAX_PATIENT_COURSE_SHIFT_DAYS = 1
MAX_ALLOWED_GAP_EXCESS_DAYS = 1
MAX_CASCADE_FRACTIONS = 3

WEIGHT_DAY_SHIFT = 100
WEIGHT_MACHINE_CHANGE = 10
WEIGHT_WINDOW_SHIFT = 1
WEIGHT_PRIORITY = {
    1: 10,
    2: 3,
    3: 1,
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# LOAD BASELINE DATA
# ============================================================

baseline_schedule = pd.read_csv(
    os.path.join(BASELINE_FOLDER, "baseline_schedule.csv")
)
# Optional true compatibility source
FRACTIONS_FILE = os.path.join(INPUT_FOLDER, "fractions.csv")
PATIENTS_FILE = os.path.join(INPUT_FOLDER, "patients.csv")

fractions_data = pd.read_csv(FRACTIONS_FILE) if os.path.exists(FRACTIONS_FILE) else pd.DataFrame()
patients_data = pd.read_csv(PATIENTS_FILE) if os.path.exists(PATIENTS_FILE) else pd.DataFrame()


machine_capacity = pd.read_csv(
    os.path.join(BASELINE_FOLDER, "machine_utilization.csv")
)

rtt_capacity = pd.read_csv(
    os.path.join(BASELINE_FOLDER, "rtt_utilization.csv")
)

if "available_rtt_teams" in rtt_capacity.columns:
    rtt_col = "available_rtt_teams"
elif "baseline_rtt_capacity_units" in rtt_capacity.columns:
    rtt_col = "baseline_rtt_capacity_units"
else:
    raise ValueError("No RTT capacity column found in rtt_utilization.csv")

for df in [baseline_schedule, machine_capacity, rtt_capacity]:
    for col in ["machine", "day", "window"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

baseline_schedule["patient"] = baseline_schedule["patient"].astype(int)
baseline_schedule["fraction"] = baseline_schedule["fraction"].astype(int)
baseline_schedule["duration_minutes"] = baseline_schedule["duration_minutes"].astype(int)

# Full baseline lookup is also used as an order blocker for affected fractions
# that have not yet been repaired. This prevents artificial gaps caused by
# removing all affected fractions from the capacity state at once.
baseline_fraction_map = {
    (int(row.patient), int(row.fraction)): {
        "day": int(row.day),
        "machine": int(row.machine),
        "window": int(row.window),
    }
    for row in baseline_schedule.itertuples()
}

DAYS = sorted(machine_capacity["day"].unique().tolist())
WINDOWS = sorted(machine_capacity["window"].unique().tolist())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_priority_weight(priority):
    return WEIGHT_PRIORITY.get(int(priority), 1)


def scenario_paths(scenario_id):
    affected_path = os.path.join(
        DISRUPTION_FOLDER,
        f"affected_fractions_{scenario_id}.csv"
    )

    summary_path = os.path.join(
        DISRUPTION_FOLDER,
        f"no_recovery_summary_{scenario_id}.csv"
    )

    return affected_path, summary_path


def load_disruption_summary(scenario_id):
    _, summary_path = scenario_paths(scenario_id)

    if not os.path.exists(summary_path):
        return {}

    summary = pd.read_csv(summary_path)

    if summary.empty:
        return {}

    return summary.iloc[0].to_dict()


def infer_disruption_window(disruption_summary):
    """Return the first unavailable/passed window for the disruption."""
    for key in ["failed_window", "affected_window", "disruption_window"]:
        if key in disruption_summary and pd.notna(disruption_summary[key]):
            return int(disruption_summary[key])
    return int(min(WINDOWS))


def build_forbidden_machine_slots(disruption_summary):
    """
    Returns machine/day/window tuples that cannot be used during repair.

    D1: failed machine, failed day, failed window
    D2: failed machine, failed day, all windows
    D5: failed machine, failed day, all windows
    D4 RTT shortage: no specific machine slots are forbidden here;
       feasibility is controlled through disrupted RTT capacity.
    """

    forbidden = set()

    disruption_type = str(disruption_summary.get("disruption_type", ""))

    if disruption_type == "machine_failure_window":
        failed_machine = int(disruption_summary["failed_machine"])
        failed_day = int(disruption_summary["failed_day"])
        failed_window = int(disruption_summary["failed_window"])

        forbidden.add((failed_machine, failed_day, failed_window))

    elif disruption_type in [
        "machine_failure_full_day",
        "combined_machine_full_day_rtt_shortage",
    ]:

        failed_machine = int(disruption_summary["failed_machine"])
        failed_day = int(disruption_summary["failed_day"])

        failed_window = int(
            disruption_summary.get(
                "failed_window",
                min(WINDOWS)
            )
        )

        for w in WINDOWS:
            if int(w) >= failed_window:
                forbidden.add((failed_machine, failed_day, int(w)))

    elif disruption_type == "power_outage_window":
        failed_day = int(disruption_summary["failed_day"])
        failed_window = int(disruption_summary["failed_window"])

        for m in baseline_schedule["machine"].unique():
            forbidden.add((int(m), failed_day, failed_window))

    return forbidden


def get_disrupted_rtt_capacity_if_available(scenario_id):
    possible_path = os.path.join(
        DISRUPTION_FOLDER,
        f"disrupted_rtt_capacity_{scenario_id}.csv"
    )

    if os.path.exists(possible_path):
        disrupted = pd.read_csv(possible_path)
        for col in ["day", "window"]:
            disrupted[col] = disrupted[col].astype(int)
        return disrupted

    return rtt_capacity.copy()


def build_capacity_state(fixed_schedule, rtt_cap_df):
    """
    Build the exact residual capacity available to Local Repair.

    machine_capacity.csv is the capacity remaining after pre-existing external
    allocations, but before the generated baseline schedule is placed. Hence:

    - unaffected/fixed baseline appointments must be subtracted;
    - affected appointments are omitted from fixed_schedule, so their baseline
      capacity is automatically released for repair;
    - no separate add-back operation is required.
    """
    remaining_machine_capacity = {}

    for row in machine_capacity.itertuples():
        key = (int(row.machine), int(row.day), int(row.window))
        remaining_machine_capacity[key] = float(
            row.remaining_machine_capacity_minutes
        )

    for row in fixed_schedule.itertuples():
        key = (int(row.machine), int(row.day), int(row.window))

        if key not in remaining_machine_capacity:
            raise KeyError(
                f"Fixed baseline appointment uses missing machine slot {key}."
            )

        remaining_machine_capacity[key] -= float(row.duration_minutes)

        if remaining_machine_capacity[key] < -1e-9:
            raise ValueError(
                "Fixed baseline appointments exceed the pre-baseline available "
                f"capacity in slot {key}: residual="
                f"{remaining_machine_capacity[key]:.2f} minutes."
            )

    rtt_cap = {}

    for row in rtt_cap_df.itertuples():
        key = (int(row.day), int(row.window))
        rtt_cap[key] = int(getattr(row, rtt_col))

    active_machines = {}

    for d in DAYS:
        for w in WINDOWS:
            active_machines[(d, w)] = set()

    for row in fixed_schedule.itertuples():
        key = (int(row.day), int(row.window))
        active_machines[key].add(int(row.machine))

    return remaining_machine_capacity, rtt_cap, active_machines


def build_patient_fraction_map(schedule_df):
    mapping = {}

    has_pref = "is_preferred_machine" in schedule_df.columns

    for row in schedule_df.itertuples():
        mapping[(int(row.patient), int(row.fraction))] = {
            "patient": int(row.patient),
            "fraction": int(row.fraction),
            "machine": int(row.machine),
            "day": int(row.day),
            "window": int(row.window),
            "duration_minutes": int(row.duration_minutes),
            "priority": int(row.priority),
            "earliest_start_day": int(row.earliest_start_day),
            "target_day": int(row.target_day),
            "is_preferred_machine": bool(row.is_preferred_machine) if has_pref else True,
        }

    return mapping


def parse_machine_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, (int, float)):
        return [int(value)]

    text = str(value)
    for ch in ["[", "]", "(", ")", "{", "}", "'", '"']:
        text = text.replace(ch, "")

    machines = []
    for part in text.replace(";", ",").replace("|", ",").split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            machines.append(int(float(part)))
        except ValueError:
            pass

    return sorted(set(machines))


def patient_allowed_machines(patient_id, fraction_id, original_machine):
    """Return the recovery machine set for one fraction.

    Candidate machines are the fraction's baseline/original machine plus all
    patient-preferred machines. The set is intersected with the true allowed
    machines whenever compatibility data are available. If preferred machines
    are missing or empty, the allowed machines are treated as preferred, which
    matches the baseline scheduler.
    """

    p = int(patient_id)
    f = int(fraction_id)
    original_machine = int(original_machine)

    true_allowed = []

    # 1) Fraction-level allowed machines, when available.
    if not fractions_data.empty:
        fd = fractions_data.copy()
        fd.columns = [c.lower() for c in fd.columns]
        patient_col = "patient" if "patient" in fd.columns else "patient_id"
        fraction_col = "fraction" if "fraction" in fd.columns else "fraction_id"

        for col in [
            "allowed_machines",
            "compatible_machines",
            "machine_compatibility",
            "machines",
        ]:
            if col in fd.columns:
                rows = fd[
                    (fd[patient_col].astype(int) == p)
                    & (fd[fraction_col].astype(int) == f)
                ]
                if not rows.empty:
                    true_allowed = parse_machine_list(rows.iloc[0][col])
                    if true_allowed:
                        break

    # 2) Patient-level allowed machines as fallback.
    patient_row = None
    if not patients_data.empty:
        pdx = patients_data.copy()
        pdx.columns = [c.lower() for c in pdx.columns]
        patient_col = "patient" if "patient" in pdx.columns else "patient_id"
        rows = pdx[pdx[patient_col].astype(int) == p]
        if not rows.empty:
            patient_row = rows.iloc[0]

            if not true_allowed:
                for col in [
                    "allowed_machines",
                    "compatible_machines",
                    "machine_compatibility",
                    "machines",
                ]:
                    if col in pdx.columns:
                        true_allowed = parse_machine_list(patient_row[col])
                        if true_allowed:
                            break

    # 3) Last-resort compatibility fallback.
    if not true_allowed:
        used = baseline_schedule[
            baseline_schedule["patient"] == p
        ]["machine"].unique().tolist()
        true_allowed = sorted(
            set([int(m) for m in used] + [original_machine])
        )

    # Preferred machines are patient-level. Empty/missing means all allowed.
    preferred = []
    if patient_row is not None and "preferred_machines" in patient_row.index:
        preferred = parse_machine_list(patient_row["preferred_machines"])
    if not preferred:
        preferred = list(true_allowed)

    selected = (set(preferred) | {original_machine}) & set(true_allowed)

    # Preserve the actual baseline machine even if inconsistent source data fail
    # to list it as allowed; the existing baseline proves it was used.
    selected.add(original_machine)

    return sorted(int(m) for m in selected)

def is_clinically_feasible(candidate, current_map):
    """
    Check patient ordering and one-fraction-per-day feasibility.

    Fractions already fixed or repaired use their current location. Affected
    fractions that have not yet been processed temporarily retain their
    baseline day as an order blocker. This prevents artificial gaps caused by
    removing all affected fractions from current_map at the start.
    """

    p = int(candidate["patient"])
    f = int(candidate["fraction"])
    d = int(candidate["day"])

    patient_fraction_ids = sorted(
        int(ff)
        for pp, ff in baseline_fraction_map
        if int(pp) == p
    )

    for other_fraction in patient_fraction_ids:
        if other_fraction == f:
            continue

        key = (p, other_fraction)
        if key in current_map:
            other_day = int(current_map[key]["day"])
        else:
            other_day = int(baseline_fraction_map[key]["day"])

        if other_day == d:
            return False

        if other_fraction < f and other_day >= d:
            return False

        if other_fraction > f and other_day <= d:
            return False

    return True

def generate_ordered_candidates(
    original_day,
    original_window,
    original_machine,
    allowed_machines,
    latest_repair_day,
    disruption_window,
    disruption_type,
):
    """
    Strict clinically ordered Local Repair neighborhood.

    For each affected fraction, its original appointment window and all
    earlier same-day windows are unavailable. This represents the time needed
    for plan transfer, physics verification, and treatment preparation.

    Search hierarchy:
    1. Same day, strictly later window, same machine if still available.
    2. Same day, strictly later window, compatible machine.
    3. Future day within an existing chronological gap.
    4. Patient-course cascade is attempted elsewhere if all direct options fail.
    """

    candidates = []

    # Consistent with Full Reoptimization and RESTORE:
    # an affected fraction cannot reuse its own original day-window, even on
    # another compatible machine, because plan transfer, physics verification,
    # and machine preparation require time not represented inside the coarse
    # window-based model.
    #
    # Same-day repair therefore starts strictly after this fraction's own
    # original appointment window.
    later_windows = sorted(
        int(w) for w in WINDOWS
        if int(w) > int(original_window)
    )

    for w in later_windows:
        candidates.append(
            (int(original_day), int(w), int(original_machine), "same_day_later_same_machine")
        )

    compatible_other_machines = sorted(
        int(m) for m in allowed_machines
        if int(m) != int(original_machine)
    )

    for w in later_windows:
        for m in compatible_other_machines:
            candidates.append(
                (int(original_day), int(w), int(m), "same_day_later_compatible_machine")
            )

    max_day = min(max(DAYS), int(latest_repair_day))
    future_days = sorted(
        int(d) for d in DAYS
        if int(d) > int(original_day) and int(d) <= max_day
    )

    ordered_machines = [int(original_machine)] + compatible_other_machines

    for d in future_days:
        ordered_windows = sorted(
            [int(w) for w in WINDOWS],
            key=lambda w: abs(int(w) - int(original_window))
        )
        for w in ordered_windows:
            for m in ordered_machines:
                candidates.append((d, w, m, "future_gap"))

    seen = set()
    unique = []

    for d, w, m, phase in candidates:
        key = (int(d), int(w), int(m))
        if key not in seen:
            seen.add(key)
            unique.append((int(d), int(w), int(m), phase))

    return unique


def find_best_slot(affected_row, current_map, remaining_machine_capacity, rtt_cap, active_machines, forbidden_machine_slots, disruption_window, disruption_type):
    p = int(affected_row["patient"])
    f = int(affected_row["fraction"])
    original_machine = int(affected_row["machine"])
    original_day = int(affected_row["day"])
    original_window = int(affected_row["window"])
    duration = int(affected_row["duration_minutes"])
    priority = int(affected_row["priority"])

    allowed_machines = patient_allowed_machines(p, f, original_machine)

    best_candidate = None
    best_cost = None
    best_phase = None

    # The local search is capped at one treatment day after the original day.
    # If a later fraction is already scheduled sooner, the repaired fraction
    # must be inserted strictly before it.
    next_fraction_days = [
        int(value["day"])
        for (patient_id, fraction_id), value in current_map.items()
        if int(patient_id) == p and int(fraction_id) > f
    ]

    latest_repair_day = min(
        int(original_day) + MAX_PATIENT_COURSE_SHIFT_DAYS,
        max(DAYS),
    )

    if next_fraction_days:
        latest_repair_day = min(
            latest_repair_day,
            min(next_fraction_days) - 1,
        )

    ordered_candidates = generate_ordered_candidates(
        original_day=original_day,
        original_window=original_window,
        original_machine=original_machine,
        allowed_machines=allowed_machines,
        latest_repair_day=latest_repair_day,
        disruption_window=disruption_window,
        disruption_type=disruption_type,
    )

    for d, w, m, phase in ordered_candidates:

        key_m = (m, d, w)
        key_r = (d, w)

        if key_m in forbidden_machine_slots:
            continue

        if key_m not in remaining_machine_capacity:
            continue

        if remaining_machine_capacity[key_m] < duration:
            continue

        active_set = active_machines.get(key_r, set())
        additional_machine_activation = 0 if m in active_set else 1

        if len(active_set) + additional_machine_activation > rtt_cap.get(key_r, 0):
            continue

        candidate = {
            "patient": p,
            "fraction": f,
            "machine": m,
            "day": d,
            "window": w,
            "duration_minutes": duration,
            "priority": priority,
            "earliest_start_day": int(affected_row["earliest_start_day"]),
            "target_day": int(affected_row["target_day"]),
            "is_preferred_machine": bool(
                affected_row["is_preferred_machine"]
            ) if "is_preferred_machine" in affected_row else True,
        }

        if not is_clinically_feasible(candidate, current_map):
            continue

        delay = max(0, d - original_day)
        machine_change = 0 if m == original_machine else 1
        window_shift = abs(w - original_window)

        phase_penalty = {
            "same_day_later_same_machine": 0,
            "same_day_later_compatible_machine": 1000,
            "future_gap": 10000,
        }[phase]

        cost = (
            phase_penalty
            + WEIGHT_DAY_SHIFT * get_priority_weight(priority) * delay
            + WEIGHT_MACHINE_CHANGE * machine_change
            + WEIGHT_WINDOW_SHIFT * window_shift
        )

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_candidate = candidate
            best_phase = phase

    if best_candidate is not None:
        best_candidate["repair_phase"] = best_phase

    return best_candidate, best_cost

def apply_assignment(candidate, current_map, remaining_machine_capacity, active_machines):
    key_pf = (int(candidate["patient"]), int(candidate["fraction"]))
    current_map[key_pf] = candidate

    key_m = (
        int(candidate["machine"]),
        int(candidate["day"]),
        int(candidate["window"]),
    )

    remaining_machine_capacity[key_m] -= float(candidate["duration_minutes"])

    key_r = (int(candidate["day"]), int(candidate["window"]))
    active_machines[key_r].add(int(candidate["machine"]))


def rebuild_state_from_map(current_map, rtt_cap_df):
    """
    Rebuild remaining machine capacity and active-machine sets from a schedule map.
    Used by the patient-course cascade so temporary releases and shifts remain exact.
    """
    schedule_df = pd.DataFrame(list(current_map.values()))
    return build_capacity_state(schedule_df, rtt_cap_df)


def try_patient_course_cascade(
    affected_row,
    current_map,
    rtt_cap_df,
    forbidden_machine_slots,
):
    """
    Minimal patient-local suffix cascade.

    If direct insertion fails, move the affected fraction by at most one
    treatment day. Shift only the minimum number of consecutive later fractions
    needed to preserve strict fraction order, but never more than
    MAX_CASCADE_FRACTIONS fractions in total including the affected fraction.
    Stop immediately when an existing baseline gap absorbs the displacement.

    Example:
        F6 day 21 affected
        F7 day 22
        F8 day 23
        F9 day 25

    Minimal repair:
        F6 -> day 22
        F7 -> day 23
        F8 -> day 24
        F9 remains day 25

    Rules:
    - no appointment belonging to another patient may move;
    - later shifted fractions keep their current machine and window;
    - the affected fraction may use any compatible machine/window;
    - maximum individual day shift is MAX_PATIENT_COURSE_SHIFT_DAYS;
    - maximum observed gap excess is MAX_ALLOWED_GAP_EXCESS_DAYS;
    - maximum cascade size is MAX_CASCADE_FRACTIONS fractions total;
    - the next untouched fraction must remain strictly after the shifted suffix;
    - if no existing gap absorbs the shift, the affected fraction is unrecovered;
    - machine capacity, RTT capacity, forbidden slots, and order are enforced.
    """

    p = int(affected_row["patient"])
    f = int(affected_row["fraction"])
    original_machine = int(affected_row["machine"])
    original_day = int(affected_row["day"])
    original_window = int(affected_row["window"])

    allowed_machines = patient_allowed_machines(
        p, f, original_machine
    )
    allowed_machines = sorted(
        [int(m) for m in allowed_machines],
        key=lambda m: (m != original_machine, m)
    )

    later_items = []
    for (pp, ff), value in current_map.items():
        if int(pp) == p and int(ff) > f:
            later_items.append((int(ff), dict(value)))
    later_items.sort(key=lambda item: item[0])

    affected_template = {
        "patient": p,
        "fraction": f,
        "machine": original_machine,
        "day": original_day,
        "window": original_window,
        "duration_minutes": int(affected_row["duration_minutes"]),
        "priority": int(affected_row["priority"]),
        "earliest_start_day": int(affected_row["earliest_start_day"]),
        "target_day": int(affected_row["target_day"]),
        "is_preferred_machine": bool(
            affected_row.get("is_preferred_machine", True)
        ),
    }

    for shift in range(1, MAX_PATIENT_COURSE_SHIFT_DAYS + 1):
        affected_new_day = original_day + shift

        if affected_new_day not in DAYS:
            continue

        # Determine the minimum suffix that must move.
        # A later fraction moves only if it would collide with, or precede,
        # the previously shifted fraction.
        #
        # Locality bound:
        # MAX_CASCADE_FRACTIONS counts the affected fraction itself.
        # Therefore, at most MAX_CASCADE_FRACTIONS - 1 downstream fractions
        # may be shifted.
        suffix_items = []
        previous_shifted_day = affected_new_day
        max_downstream_to_shift = max(0, MAX_CASCADE_FRACTIONS - 1)

        for fraction_id, source in later_items:
            source_day = int(source["day"])

            if source_day <= previous_shifted_day:
                if len(suffix_items) >= max_downstream_to_shift:
                    # The required cascade would exceed the allowed local
                    # neighborhood, so reject this repair attempt.
                    suffix_items = None
                    break

                shifted_day = source_day + shift
                suffix_items.append((fraction_id, source, shifted_day))
                previous_shifted_day = shifted_day
            else:
                # Existing gap absorbs the displacement.
                break

        if suffix_items is None:
            continue

        # Strict locality boundary:
        # After shifting the affected fraction and the bounded downstream suffix,
        # the immediately following untouched fraction must remain strictly later.
        # Missing/unprocessed affected fractions are treated as occupying their
        # baseline day for order checking, so they cannot create artificial gaps.
        moved_fraction_ids = [int(f)] + [
            int(fid) for fid, _, _ in suffix_items
        ]
        last_moved_fraction = max(moved_fraction_ids)
        last_moved_day = (
            int(suffix_items[-1][2])
            if suffix_items
            else int(affected_new_day)
        )

        all_patient_fraction_ids = sorted(
            int(ff)
            for (pp, ff) in baseline_fraction_map
            if int(pp) == p
        )

        following_fraction_ids = [
            ff for ff in all_patient_fraction_ids
            if ff > last_moved_fraction
        ]

        if following_fraction_ids:
            next_fraction_id = min(following_fraction_ids)
            next_key = (p, next_fraction_id)

            if next_key in current_map:
                next_untouched_day = int(current_map[next_key]["day"])
            else:
                # The next fraction may also be affected and not yet repaired.
                # Its baseline day still blocks the cascade boundary.
                next_untouched_day = int(
                    baseline_fraction_map[next_key]["day"]
                )

            if next_untouched_day <= last_moved_day:
                # No pre-existing gap exists to absorb the bounded cascade.
                # Reject the repair and leave the affected fraction unrecovered.
                continue

        # Release only the fractions in the minimal suffix.
        suffix_keys = {(p, int(fid)) for fid, _, _ in suffix_items}

        base_map = {
            key: dict(value)
            for key, value in current_map.items()
            if key not in suffix_keys
        }

        # Candidate choices for the affected fraction on the shifted day.
        affected_slot_candidates = []
        for m in allowed_machines:
            ordered_windows = sorted(
                [int(w) for w in WINDOWS],
                key=lambda w: (
                    m != original_machine,
                    w != original_window,
                    abs(w - original_window),
                    w,
                )
            )
            for w in ordered_windows:
                affected_slot_candidates.append((m, w))

        for affected_machine, affected_window in affected_slot_candidates:
            trial_map = {
                key: dict(value)
                for key, value in base_map.items()
            }

            remaining_capacity, rtt_cap, active_machines = rebuild_state_from_map(
                trial_map,
                rtt_cap_df,
            )

            shifted_rows = []
            feasible = True

            affected_new = dict(affected_template)
            affected_new["machine"] = int(affected_machine)
            affected_new["day"] = int(affected_new_day)
            affected_new["window"] = int(affected_window)
            affected_new["repair_phase"] = "minimal_patient_suffix_cascade"
            affected_new["patient_local_status"] = "recovered"

            course_rows = [(f, affected_new)]

            for fraction_id, source, shifted_day in suffix_items:
                shifted = dict(source)
                shifted["patient"] = p
                shifted["fraction"] = int(fraction_id)
                shifted["day"] = int(shifted_day)
                shifted["machine"] = int(source["machine"])
                shifted["window"] = int(source["window"])
                shifted["repair_phase"] = "minimal_suffix_shifted"
                shifted["patient_local_status"] = "shifted_with_patient_course"
                course_rows.append((int(fraction_id), shifted))

            for fraction_id, new_row in course_rows:
                slot = (
                    int(new_row["machine"]),
                    int(new_row["day"]),
                    int(new_row["window"]),
                )
                day_window = (
                    int(new_row["day"]),
                    int(new_row["window"]),
                )

                if int(new_row["day"]) not in DAYS:
                    feasible = False
                    break

                if slot in forbidden_machine_slots:
                    feasible = False
                    break

                duration = int(new_row["duration_minutes"])
                if remaining_capacity.get(slot, 0) < duration:
                    feasible = False
                    break

                active_set = active_machines.get(day_window, set())
                extra_activation = (
                    0 if int(new_row["machine"]) in active_set else 1
                )

                if (
                    len(active_set) + extra_activation
                    > rtt_cap.get(day_window, 0)
                ):
                    feasible = False
                    break

                duplicate_day = any(
                    int(pp) == p
                    and int(ff) != int(fraction_id)
                    and int(value["day"]) == int(new_row["day"])
                    for (pp, ff), value in trial_map.items()
                )
                if duplicate_day:
                    feasible = False
                    break

                trial_map[(p, int(fraction_id))] = new_row
                remaining_capacity[slot] -= duration
                active_machines[day_window].add(int(new_row["machine"]))
                shifted_rows.append(new_row)

            if not feasible:
                continue

            # Final patient-course validation.
            patient_rows = sorted(
                [
                    value
                    for (pp, _), value in trial_map.items()
                    if int(pp) == p
                ],
                key=lambda value: int(value["fraction"]),
            )

            patient_days = [int(value["day"]) for value in patient_rows]

            order_ok = all(
                patient_days[i] < patient_days[i + 1]
                for i in range(len(patient_days) - 1)
            )

            gap_ok = all(
                patient_days[i + 1] - patient_days[i] - 1
                <= MAX_ALLOWED_GAP_EXCESS_DAYS
                for i in range(len(patient_days) - 1)
            )

            if order_ok and gap_ok:
                return (
                    trial_map,
                    remaining_capacity,
                    active_machines,
                    shifted_rows,
                    shift,
                )

    return None, None, None, [], None

def compute_recovery_metrics(
    scenario_id,
    repaired_schedule,
    affected_original,
    unrecovered_rows,
    runtime,
    patients_with_course_cascade,
):
    """
    Compute metrics from the final repaired schedule.

    All schedule-change metrics compare the final Local Repair schedule with
    the original baseline, so patient-course cascade shifts are included.
    """

    affected_keys = set(
        zip(
            affected_original["patient"].astype(int),
            affected_original["fraction"].astype(int),
        )
    )

    baseline_keyed = {
        (int(row.patient), int(row.fraction)): row
        for row in baseline_schedule.itertuples()
    }
    repaired_keyed = {
        (int(row.patient), int(row.fraction)): row
        for row in repaired_schedule.itertuples()
    }

    recovered_count = sum(
        1 for key in affected_keys if key in repaired_keyed
    )
    affected_count = len(affected_keys)
    unrecovered_count = affected_count - recovered_count
    recovery_rate = (
        recovered_count / affected_count if affected_count else 1.0
    )

    moved_fractions = 0
    machine_changes = 0
    window_changes = 0
    nonaffected_moved_fractions = 0
    day_changes = 0
    total_time_shift = 0
    max_time_shift = 0
    total_delay = 0
    max_delay = 0
    total_delay_among_moved_fractions = 0

    for key, repaired in repaired_keyed.items():
        baseline = baseline_keyed[key]

        machine_changed = int(repaired.machine) != int(baseline.machine)
        window_changed = int(repaired.window) != int(baseline.window)
        day_shift = int(repaired.day) - int(baseline.day)
        day_changed = day_shift != 0

        moved = int(
            machine_changed or window_changed or day_changed
        )

        moved_fractions += moved

        if moved and key not in affected_keys:
            nonaffected_moved_fractions += 1

        day_changes += int(day_changed)
        machine_changes += int(machine_changed)
        window_changes += int(window_changed)

        total_time_shift += abs(day_shift)
        max_time_shift = max(max_time_shift, abs(day_shift))

        delay = max(0, day_shift)
        total_delay += delay

        if moved:
            total_delay_among_moved_fractions += delay

        max_delay = max(max_delay, delay)

    average_time_shift = (
        total_time_shift / len(repaired_keyed)
        if repaired_keyed else 0.0
    )
    average_delay = (
        total_delay / len(repaired_keyed)
        if repaired_keyed else 0.0
    )
    average_delay_among_moved_fractions = (
        total_delay_among_moved_fractions / moved_fractions
        if moved_fractions
        else 0.0
    )

    # Consecutive-pair metrics must use the original fraction sequence,
    # not only the surviving repaired schedule. Otherwise, if an affected
    # fraction is unrecovered, the code would incorrectly connect nonconsecutive
    # labels (for example F5 directly to F7) and create artificial gaps/switches.
    baseline_fractions_by_patient = (
        baseline_schedule.groupby("patient")["fraction"]
        .apply(lambda values: sorted(int(v) for v in values.tolist()))
        .to_dict()
    )

    total_consecutive_fraction_pairs = 0
    consecutive_machine_switches = 0
    consecutive_window_switches = 0
    patients_with_machine_switching = set()
    patients_with_window_switching = set()
    observed_gap_excess_values = []

    for p, fraction_list in baseline_fractions_by_patient.items():
        p = int(p)
        fraction_list = sorted(int(f) for f in fraction_list)

        for i in range(len(fraction_list) - 1):
            f_current = int(fraction_list[i])
            f_next = int(fraction_list[i + 1])

            current_key = (p, f_current)
            next_key = (p, f_next)

            # Only evaluate a true consecutive pair when both fractions are
            # actually present in the repaired schedule.
            if current_key not in repaired_keyed:
                continue
            if next_key not in repaired_keyed:
                continue

            current = repaired_keyed[current_key]
            nxt = repaired_keyed[next_key]

            total_consecutive_fraction_pairs += 1

            if int(current.machine) != int(nxt.machine):
                consecutive_machine_switches += 1
                patients_with_machine_switching.add(p)

            if int(current.window) != int(nxt.window):
                consecutive_window_switches += 1
                patients_with_window_switching.add(p)

            observed_gap_excess_values.append(
                max(0, int(nxt.day) - int(current.day) - 1)
            )

    total_observed_gap_excess_days = int(
        sum(observed_gap_excess_values)
    )
    average_observed_gap_excess_days = (
        total_observed_gap_excess_days
        / len(observed_gap_excess_values)
        if observed_gap_excess_values else 0.0
    )
    maximum_observed_gap_excess_days = (
        max(observed_gap_excess_values)
        if observed_gap_excess_values else 0
    )

    final_fractions_by_patient = (
        repaired_schedule.groupby("patient")["fraction"]
        .apply(lambda values: sorted(int(v) for v in values.tolist()))
        .to_dict()
    )

    patients_with_future_machine_relocation = set()
    affected_fractions_by_patient = defaultdict(list)
    for p, f in affected_keys:
        affected_fractions_by_patient[int(p)].append(int(f))

    for p, affected_fractions in affected_fractions_by_patient.items():
        if p not in final_fractions_by_patient:
            continue

        first_affected_fraction = min(affected_fractions)
        future_fractions = [
            int(fraction)
            for fraction in final_fractions_by_patient[p]
            if int(fraction) >= first_affected_fraction
        ]

        recovered_future_machines = {
            int(repaired_keyed[(p, f)].machine)
            for f in future_fractions
            if (p, f) in repaired_keyed
        }
        baseline_future_machines = {
            int(baseline_keyed[(p, f)].machine)
            for f in future_fractions
            if (p, f) in baseline_keyed
        }

        if (
            len(recovered_future_machines) == 1
            and recovered_future_machines != baseline_future_machines
        ):
            patients_with_future_machine_relocation.add(p)

    return {
        "scenario_id": scenario_id,
        "recovery_method": "patient_local_repair_original_plus_preferred",
        "affected_fractions": affected_count,
        "recovered_fractions": recovered_count,
        "unrecovered_fractions": unrecovered_count,
        "recovery_rate": recovery_rate,
        "moved_fractions": moved_fractions,
        "patients_with_course_cascade": int(patients_with_course_cascade),
        "machine_changes": machine_changes,
        "window_changes": window_changes,
        "nonaffected_moved_fractions": nonaffected_moved_fractions,
        "day_changes": day_changes,

        "patients_with_future_machine_relocation": len(
            patients_with_future_machine_relocation
        ),
        "consecutive_machine_switches": consecutive_machine_switches,
        "patients_with_machine_switching": len(
            patients_with_machine_switching
        ),
        "consecutive_window_switches": consecutive_window_switches,
        "patients_with_window_switching": len(
            patients_with_window_switching
        ),
        "total_consecutive_fraction_pairs": total_consecutive_fraction_pairs,
        "total_observed_gap_excess_days": total_observed_gap_excess_days,
        "average_observed_gap_excess_days": average_observed_gap_excess_days,
        "maximum_observed_gap_excess_days": maximum_observed_gap_excess_days,
        "total_absolute_time_shift": total_time_shift,
        "average_absolute_time_shift": average_time_shift,
        "max_absolute_time_shift": max_time_shift,
        "total_delay_days": total_delay,
        "average_delay_days": average_delay,
        "average_delay_among_moved_fractions": average_delay_among_moved_fractions,
        "max_delay_days": max_delay,
        "runtime_seconds": runtime,
        "total_scheduled_fractions_after_repair": len(repaired_schedule),
        "total_patients_after_repair": repaired_schedule["patient"].nunique(),
    }


def run_local_repair_for_scenario(scenario_id):
    start_time = time.time()

    affected_path, summary_path = scenario_paths(scenario_id)

    if not os.path.exists(affected_path):
        raise FileNotFoundError(f"Missing affected file: {affected_path}")

    affected = pd.read_csv(affected_path)

    for col in ["patient", "fraction", "machine", "day", "window"]:
        if col in affected.columns:
            affected[col] = affected[col].astype(int)

    if len(affected) == 0:
        print(f"{scenario_id}: no affected fractions.")
        return

    disruption_summary = load_disruption_summary(scenario_id)
    disruption_window = infer_disruption_window(disruption_summary)
    disruption_type = str(disruption_summary.get("disruption_type", ""))
    forbidden_machine_slots = build_forbidden_machine_slots(disruption_summary)

    rtt_cap_df = get_disrupted_rtt_capacity_if_available(scenario_id)

    affected_keys = set(
        zip(
            affected["patient"].astype(int),
            affected["fraction"].astype(int)
        )
    )

    baseline_keys = list(
        zip(
            baseline_schedule["patient"].astype(int),
            baseline_schedule["fraction"].astype(int)
        )
    )

    fixed_mask = [key not in affected_keys for key in baseline_keys]

    fixed_schedule = baseline_schedule[fixed_mask].copy()

    remaining_machine_capacity, rtt_cap, active_machines = build_capacity_state(
        fixed_schedule,
        rtt_cap_df
    )

    current_map = build_patient_fraction_map(fixed_schedule)

    affected_sorted = affected.sort_values(
        ["priority", "day", "window", "patient", "fraction"],
        ascending=[True, True, True, True, True]
    )

    repaired_rows = []
    cascade_shifted_rows = []
    patients_with_course_cascade = set()
    unrecovered_rows = []

    for row in affected_sorted.to_dict("records"):

        candidate, cost = find_best_slot(
            row,
            current_map,
            remaining_machine_capacity,
            rtt_cap,
            active_machines,
            forbidden_machine_slots,
            disruption_window,
            disruption_type,
        )

        if candidate is not None:
            candidate["patient_local_status"] = "recovered"
            candidate["repair_phase"] = candidate.get(
                "repair_phase", "unknown"
            )
            candidate["patient_local_cost"] = cost
            candidate["original_machine"] = int(row["machine"])
            candidate["original_day"] = int(row["day"])
            candidate["original_window"] = int(row["window"])

            repaired_rows.append(candidate)

            apply_assignment(
                candidate,
                current_map,
                remaining_machine_capacity,
                active_machines
            )
            continue

        (
            cascade_map,
            cascade_capacity,
            cascade_active_machines,
            shifted_rows,
            cascade_shift,
        ) = try_patient_course_cascade(
            row,
            current_map,
            rtt_cap_df,
            forbidden_machine_slots,
        )

        if cascade_map is not None:
            patients_with_course_cascade.add(int(row["patient"]))
            current_map = cascade_map
            remaining_machine_capacity = cascade_capacity
            active_machines = cascade_active_machines

            for shifted in shifted_rows:
                shifted["cascade_shift_days"] = cascade_shift
                shifted["original_machine"] = int(
                    baseline_schedule.loc[
                        (baseline_schedule["patient"] == int(shifted["patient"]))
                        & (baseline_schedule["fraction"] == int(shifted["fraction"])),
                        "machine",
                    ].iloc[0]
                )
                shifted["original_day"] = int(
                    baseline_schedule.loc[
                        (baseline_schedule["patient"] == int(shifted["patient"]))
                        & (baseline_schedule["fraction"] == int(shifted["fraction"])),
                        "day",
                    ].iloc[0]
                )
                shifted["original_window"] = int(
                    baseline_schedule.loc[
                        (baseline_schedule["patient"] == int(shifted["patient"]))
                        & (baseline_schedule["fraction"] == int(shifted["fraction"])),
                        "window",
                    ].iloc[0]
                )

            repaired_rows.append(shifted_rows[0])
            cascade_shifted_rows.extend(shifted_rows[1:])
            continue

        unrecovered = row.copy()
        unrecovered["patient_local_status"] = "unrecovered"
        unrecovered_rows.append(unrecovered)

    repaired_schedule = pd.DataFrame(list(current_map.values()))

    repaired_schedule = repaired_schedule.sort_values(
        ["day", "window", "machine", "patient", "fraction"]
    )

    runtime = time.time() - start_time

    metrics = compute_recovery_metrics(
        scenario_id,
        repaired_schedule,
        affected,
        unrecovered_rows,
        runtime,
        len(patients_with_course_cascade),
    )

    metrics["patient_course_shifted_fractions"] = len(
        cascade_shifted_rows
    )
    metrics["downstream_shifted_fractions"] = len(
        cascade_shifted_rows
    )

    metrics["forbidden_machine_slots"] = len(forbidden_machine_slots)

    scenario_output = os.path.join(OUTPUT_FOLDER, scenario_id)
    os.makedirs(scenario_output, exist_ok=True)

    repaired_schedule.to_csv(
        os.path.join(scenario_output, "patient_local_repair_schedule.csv"),
        index=False
    )

    pd.DataFrame(repaired_rows).to_csv(
        os.path.join(scenario_output, "patient_local_repair_recovered_fractions.csv"),
        index=False
    )

    pd.DataFrame(unrecovered_rows).to_csv(
        os.path.join(scenario_output, "patient_local_repair_unrecovered_fractions.csv"),
        index=False
    )

    pd.DataFrame(cascade_shifted_rows).to_csv(
        os.path.join(
            scenario_output,
            "patient_local_repair_shifted_course_fractions.csv"
        ),
        index=False
    )

    pd.DataFrame([metrics]).to_csv(
        os.path.join(scenario_output, "patient_local_repair_summary.csv"),
        index=False
    )

    print()
    print("========================================")
    print(f"PATIENT-LOCAL REPAIR SUMMARY - {scenario_id}")
    print("========================================")
    for key, value in metrics.items():
        print(f"{key}: {value}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"BUILD ID: {BUILD_ID}")
    print("========================================")
    print("PATIENT-LOCAL REPAIR RECOVERY")
    print("========================================")
    print(f"Baseline folder: {BASELINE_FOLDER}")
    print(f"Disruption folder: {DISRUPTION_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")

    for scenario in SCENARIOS:
        run_local_repair_for_scenario(scenario)

    print()
    print("========================================")
    print("PATIENT-LOCAL REPAIR COMPLETE")
    print("========================================")
