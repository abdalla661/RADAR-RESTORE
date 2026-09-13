import os
import math
import random
import pandas as pd


# ============================================================
# DISRUPTION SCENARIO GENERATOR
# ============================================================
#
# Supported scenarios:
# D1: Machine failure - highest-workload machine/window
# D2: Unexpected machine failure starting in one window and persisting for the rest of the day
# D3: Power outage - all machines unavailable for one highly utilized window
# D4: RTT shortage - reduce RTT capacity by rho in the highest-workload system window
# D5: Combined stress test - full-day machine failure + RTT shortage
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_FOLDER = "baseline_results"
OUTPUT_FOLDER = "disruption_results"

RUN_D1_MACHINE_WINDOW = True
RUN_D2_MACHINE_FULL_DAY = True
RUN_D3_POWER_OUTAGE_WINDOW = True
RUN_D4_RTT_SHORTAGE = True
RUN_D5_COMBINED_MACHINE_DAY_RTT = True

RTT_SHORTAGE_LEVELS = [0.25, 0.50, 0.75]
COMBINED_RTT_SHORTAGE = 0.50
RANDOM_SEED = 42  # used only when candidates tie on treatment minutes and fractions

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# LOAD BASELINE FILES
# ============================================================

schedule = pd.read_csv(os.path.join(BASELINE_FOLDER, "baseline_schedule.csv"))
machine_util = pd.read_csv(os.path.join(BASELINE_FOLDER, "machine_utilization.csv"))
rtt_util = pd.read_csv(os.path.join(BASELINE_FOLDER, "rtt_utilization.csv"))
baseline_summary = pd.read_csv(os.path.join(BASELINE_FOLDER, "baseline_summary.csv"))

if "available_rtt_teams" in rtt_util.columns:
    rtt_col = "available_rtt_teams"
elif "baseline_rtt_capacity_units" in rtt_util.columns:
    rtt_col = "baseline_rtt_capacity_units"
else:
    raise ValueError("No RTT capacity column found in rtt_utilization.csv")

for col in ["machine", "day", "window"]:
    if col in schedule.columns:
        schedule[col] = schedule[col].astype(int)
    if col in machine_util.columns:
        machine_util[col] = machine_util[col].astype(int)
    if col in rtt_util.columns:
        rtt_util[col] = rtt_util[col].astype(int)


# ============================================================
# HELPERS
# ============================================================

def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def choose_by_workload(candidates, minutes_col, fractions_col):
    """
    Common disruption-location rule:
    1) maximum scheduled treatment minutes,
    2) if tied, maximum affected fractions,
    3) if still tied, random selection among exact ties.

    RANDOM_SEED makes the random tie-break reproducible.
    """
    if candidates.empty:
        raise ValueError("No disruption candidates found.")

    max_minutes = candidates[minutes_col].max()
    tied = candidates[candidates[minutes_col] == max_minutes].copy()

    max_fractions = tied[fractions_col].max()
    tied = tied[tied[fractions_col] == max_fractions].copy()

    if len(tied) == 1:
        return tied.iloc[0]

    rng = random.Random(RANDOM_SEED)
    return tied.iloc[rng.randrange(len(tied))]


def select_highest_used_machine_window():
    occupied = machine_util[machine_util["used_minutes"] > 0].copy()
    if occupied.empty:
        raise ValueError("No occupied machine-window slots found.")

    fraction_counts = (
        schedule.groupby(["machine", "day", "window"], as_index=False)
        .agg(affected_fractions=("fraction", "count"))
    )
    occupied = occupied.merge(
        fraction_counts, on=["machine", "day", "window"], how="left"
    )
    occupied["affected_fractions"] = occupied["affected_fractions"].fillna(0).astype(int)

    selected = choose_by_workload(
        occupied,
        minutes_col="used_minutes",
        fractions_col="affected_fractions",
    )

    return {
        "machine": int(selected["machine"]),
        "day": int(selected["day"]),
        "window": int(selected["window"]),
        "used_minutes": float(selected["used_minutes"]),
        "affected_fractions": int(selected["affected_fractions"]),
        "capacity_minutes": float(selected["remaining_machine_capacity_minutes"]),
        "utilization": float(selected["utilization"]),
    }


def select_highest_impact_machine_day_from_window():
    """
    Select one machine/day/start-window outage. The machine becomes unavailable
    from the selected window through the end of that treatment day.
    """
    rows = []
    windows = sorted(schedule["window"].astype(int).unique().tolist())

    starts = (
        schedule[["machine", "day", "window"]]
        .drop_duplicates()
        .sort_values(["machine", "day", "window"])
    )

    for start in starts.itertuples():
        machine = int(start.machine)
        day = int(start.day)
        failed_window = int(start.window)

        affected_suffix = schedule[
            (schedule["machine"] == machine)
            & (schedule["day"] == day)
            & (schedule["window"] >= failed_window)
        ]

        if affected_suffix.empty:
            continue

        suffix_windows = [w for w in windows if w >= failed_window]
        suffix_capacity = machine_util[
            (machine_util["machine"] == machine)
            & (machine_util["day"] == day)
            & (machine_util["window"].isin(suffix_windows))
        ]["remaining_machine_capacity_minutes"].sum()

        used = float(affected_suffix["duration_minutes"].sum())
        capacity = float(suffix_capacity)

        rows.append({
            "machine": machine,
            "day": day,
            "failed_window": failed_window,
            "affected_fractions": int(len(affected_suffix)),
            "affected_patients": int(affected_suffix["patient"].nunique()),
            "used_minutes": used,
            "capacity_minutes": capacity,
            "utilization": used / capacity if capacity > 0 else 0.0,
        })

    if not rows:
        raise ValueError("No occupied machine/day/window outage starts found.")

    selected = choose_by_workload(
        pd.DataFrame(rows),
        minutes_col="used_minutes",
        fractions_col="affected_fractions",
    )

    return {
        "machine": int(selected["machine"]),
        "day": int(selected["day"]),
        "failed_window": int(selected["failed_window"]),
        "affected_fractions": int(selected["affected_fractions"]),
        "affected_patients": int(selected["affected_patients"]),
        "used_minutes": float(selected["used_minutes"]),
        "capacity_minutes": float(selected["capacity_minutes"]),
        "utilization": float(selected["utilization"]),
    }


def select_highest_impact_machine_day():
    """
    Original full-day selector retained for D5.
    """
    impact = (
        schedule.groupby(["machine", "day"], as_index=False)
        .agg(
            affected_fractions=("fraction", "count"),
            affected_patients=("patient", "nunique"),
            used_minutes=("duration_minutes", "sum"),
        )
    )

    day_capacity = (
        machine_util.groupby(["machine", "day"], as_index=False)
        .agg(capacity_minutes=("remaining_machine_capacity_minutes", "sum"))
    )

    impact = impact.merge(day_capacity, on=["machine", "day"], how="left")
    impact["utilization"] = (
        impact["used_minutes"] / impact["capacity_minutes"].replace(0, pd.NA)
    )

    selected = choose_by_workload(
        impact,
        minutes_col="used_minutes",
        fractions_col="affected_fractions",
    )

    return {
        "machine": int(selected["machine"]),
        "day": int(selected["day"]),
        "affected_fractions": int(selected["affected_fractions"]),
        "affected_patients": int(selected["affected_patients"]),
        "used_minutes": float(selected["used_minutes"]),
        "capacity_minutes": float(selected["capacity_minutes"]),
        "utilization": float(selected["utilization"]),
    }


def select_highest_used_system_window():
    sys_util = (
        machine_util
        .groupby(["day", "window"], as_index=False)
        .agg(
            used_minutes=("used_minutes", "sum"),
            capacity_minutes=("remaining_machine_capacity_minutes", "sum")
        )
    )

    fraction_counts = (
        schedule.groupby(["day", "window"], as_index=False)
        .agg(affected_fractions=("fraction", "count"))
    )
    sys_util = sys_util.merge(fraction_counts, on=["day", "window"], how="left")
    sys_util["affected_fractions"] = sys_util["affected_fractions"].fillna(0).astype(int)

    sys_util = sys_util[sys_util["used_minutes"] > 0].copy()
    if sys_util.empty:
        raise ValueError("No occupied system-wide windows found.")

    sys_util["utilization"] = (
        sys_util["used_minutes"] /
        sys_util["capacity_minutes"].replace(0, pd.NA)
    )

    selected = choose_by_workload(
        sys_util,
        minutes_col="used_minutes",
        fractions_col="affected_fractions",
    )

    return {
        "day": int(selected["day"]),
        "window": int(selected["window"]),
        "used_minutes": float(selected["used_minutes"]),
        "affected_fractions": int(selected["affected_fractions"]),
        "capacity_minutes": float(selected["capacity_minutes"]),
        "utilization": float(selected["utilization"]),
    }


def select_d4_workload_window():
    """
    Select the D4 day/window using the same workload rule as the other scenarios:
    maximum scheduled treatment minutes, then maximum fractions, then random tie-break.
    RTT capacity is then read for that selected window.
    """
    selected_window = select_highest_used_system_window()

    rtt_row = rtt_util[
        (rtt_util["day"] == selected_window["day"]) &
        (rtt_util["window"] == selected_window["window"])
    ]

    if rtt_row.empty:
        raise ValueError("No RTT capacity row found for selected D4 window.")

    rtt_row = rtt_row.iloc[0]
    active_machines = int(
        schedule[
            (schedule["day"] == selected_window["day"]) &
            (schedule["window"] == selected_window["window"])
        ]["machine"].nunique()
    )

    return {
        "day": selected_window["day"],
        "window": selected_window["window"],
        "used_minutes": selected_window["used_minutes"],
        "affected_fractions": selected_window["affected_fractions"],
        "active_machines": active_machines,
        "rtt_capacity": float(rtt_row[rtt_col]),
    }


def make_summary(scenario_id, scenario_name, affected, selection_details, rtt_shortage_level=None):
    total_fractions = int(len(schedule))
    total_patients = int(schedule["patient"].nunique())
    affected_fraction_count = int(len(affected))
    affected_patient_count = int(affected["patient"].nunique()) if affected_fraction_count else 0
    lost_treatment_minutes = float(affected["duration_minutes"].sum()) if affected_fraction_count else 0.0

    summary = {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "baseline_folder": BASELINE_FOLDER,
        "total_fractions": total_fractions,
        "total_patients": total_patients,
        "affected_fractions": affected_fraction_count,
        "affected_patients": affected_patient_count,
        "unaffected_fractions": total_fractions - affected_fraction_count,
        "lost_treatment_minutes": lost_treatment_minutes,
        "cancelled_fractions_no_recovery": affected_fraction_count,
        "recovered_fractions_no_recovery": 0,
        "recovery_rate_no_recovery": 0.0,
    }

    if rtt_shortage_level is not None:
        summary["rtt_shortage_level"] = rtt_shortage_level

    for key, value in selection_details.items():
        summary[key] = value

    for col in baseline_summary.columns:
        summary[f"baseline_{col}"] = baseline_summary.iloc[0][col]

    return summary


def export_scenario(scenario_id, scenario_name, affected, summary):
    affected = affected.copy()
    affected["scenario_id"] = scenario_id
    affected["scenario_name"] = scenario_name
    affected["recovery_status"] = "unrecovered_no_recovery"

    affected_path = os.path.join(OUTPUT_FOLDER, f"affected_fractions_{scenario_id}.csv")
    summary_path = os.path.join(OUTPUT_FOLDER, f"no_recovery_summary_{scenario_id}.csv")

    affected.to_csv(affected_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print()
    print("========================================")
    print(f"{scenario_id}: {scenario_name}")
    print("========================================")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Generated files:")
    print(affected_path)
    print(summary_path)


def export_disrupted_rtt_capacity(scenario_id, rho, target_day=None, target_window=None):
    disrupted = rtt_util.copy()

    if target_day is None and target_window is None:
        mask = pd.Series(True, index=disrupted.index)
    else:
        mask = (
            (disrupted["day"] == int(target_day)) &
            (disrupted["window"] == int(target_window))
        )

    disrupted[f"{rtt_col}_before_disruption"] = disrupted[rtt_col]
    disrupted.loc[mask, rtt_col] = disrupted.loc[mask, rtt_col].apply(
        lambda v: max(0, int(math.floor(v * (1 - rho))))
    )
    disrupted["rtt_shortage_level"] = rho

    out_path = os.path.join(OUTPUT_FOLDER, f"disrupted_rtt_capacity_{scenario_id}.csv")
    disrupted.to_csv(out_path, index=False)
    return disrupted, out_path


# ============================================================
# D1
# ============================================================

if RUN_D1_MACHINE_WINDOW:
    s = select_highest_used_machine_window()
    affected = schedule[
        (schedule["machine"] == s["machine"]) &
        (schedule["day"] == s["day"]) &
        (schedule["window"] == s["window"])
    ].copy()

    details = {
        "disruption_type": "machine_failure_window",
        "failed_machine": s["machine"],
        "failed_day": s["day"],
        "failed_window": s["window"],
        "slot_used_minutes_before_disruption": s["used_minutes"],
        "slot_capacity_minutes_before_disruption": s["capacity_minutes"],
        "slot_utilization_before_disruption": s["utilization"],
    }

    summary = make_summary(
        "D1_machine_window",
        "One LINAC unavailable for one highly utilized treatment window",
        affected,
        details
    )
    export_scenario("D1_machine_window", summary["scenario_name"], affected, summary)


# ============================================================
# D2
# ============================================================

if RUN_D2_MACHINE_FULL_DAY:
    s = select_highest_impact_machine_day_from_window()

    affected = schedule[
        (schedule["machine"] == s["machine"])
        & (schedule["day"] == s["day"])
        & (schedule["window"] >= s["failed_window"])
    ].copy()

    affected_windows = sorted(
        int(w)
        for w in schedule["window"].unique()
        if int(w) >= int(s["failed_window"])
    )

    details = {
        "disruption_type": "machine_failure_full_day",
        "failure_timing": "unexpected_during_operations",
        "failed_machine": s["machine"],
        "failed_day": s["day"],
        "failed_window": s["failed_window"],
        "failure_persists_through_end_of_day": True,
        "affected_windows": affected_windows,
        "machine_suffix_used_minutes_before_disruption": s["used_minutes"],
        "machine_suffix_capacity_minutes_before_disruption": s["capacity_minutes"],
        "machine_suffix_utilization_before_disruption": s["utilization"],
        "selected_machine_suffix_affected_fractions": s["affected_fractions"],
        "selected_machine_suffix_affected_patients": s["affected_patients"],
    }

    scenario_name = (
        "Unexpected LINAC failure starting in treatment window "
        f"{s['failed_window']} and persisting through the end of the day"
    )

    summary = make_summary(
        "D2_machine_full_day",
        scenario_name,
        affected,
        details,
    )
    export_scenario(
        "D2_machine_full_day",
        scenario_name,
        affected,
        summary,
    )


# ============================================================
# D3
# ============================================================

if RUN_D3_POWER_OUTAGE_WINDOW:
    s = select_highest_used_system_window()
    affected = schedule[
        (schedule["day"] == s["day"]) &
        (schedule["window"] == s["window"])
    ].copy()

    details = {
        "disruption_type": "power_outage_window",
        "failed_machines": "all",
        "failed_day": s["day"],
        "failed_window": s["window"],
        "system_window_used_minutes_before_disruption": s["used_minutes"],
        "system_window_capacity_minutes_before_disruption": s["capacity_minutes"],
        "system_window_utilization_before_disruption": s["utilization"],
    }

    summary = make_summary(
        "D3_power_outage_window",
        "All LINACs unavailable for one highly utilized treatment window",
        affected,
        details
    )
    export_scenario("D3_power_outage_window", summary["scenario_name"], affected, summary)


# ============================================================
# D4
# ============================================================

if RUN_D4_RTT_SHORTAGE:
    s = select_d4_workload_window()
    exposed = schedule[
        (schedule["day"] == s["day"]) &
        (schedule["window"] == s["window"])
    ].copy()

    for rho in RTT_SHORTAGE_LEVELS:
        scenario_id = f"D4_rtt_shortage_{safe_name(int(rho * 100))}pct"
        disrupted_rtt, rtt_path = export_disrupted_rtt_capacity(
            scenario_id, rho, target_day=s["day"], target_window=s["window"]
        )

        before_capacity = int(s["rtt_capacity"])
        after_capacity = max(0, int(math.floor(before_capacity * (1 - rho))))
        active_machines = int(s["active_machines"])
        excess_active_machines = max(0, active_machines - after_capacity)

        if excess_active_machines > 0:
            # Select only the number of active machines that must stop operating
            # because of the reduced RTT capacity. Machines are ranked by their
            # treatment load in the affected window, so severity increases with rho.
            machine_load = (
                exposed
                .groupby("machine", as_index=False)
                .agg(
                    machine_window_fractions=("fraction", "count"),
                    machine_window_patients=("patient", "nunique"),
                    machine_window_minutes=("duration_minutes", "sum")
                )
                .sort_values(
                    ["machine_window_minutes", "machine_window_fractions"],
                    ascending=[True, True]
                )
            )

            # Stop the lowest-load LINACs first so the largest amount of
            # scheduled treatment activity is preserved under reduced RTT capacity.
            stopped_machines = machine_load.head(excess_active_machines)["machine"].tolist()

            affected = exposed[
                exposed["machine"].isin(stopped_machines)
            ].copy()

            affected["rtt_stopped_machine_due_to_shortage"] = affected["machine"]
        else:
            stopped_machines = []
            affected = exposed.iloc[0:0].copy()

        details = {
            "disruption_type": "rtt_shortage",
            "rtt_shortage_level": rho,
            "affected_day": s["day"],
            "affected_window": s["window"],
            "rtt_capacity_before": before_capacity,
            "rtt_capacity_after": after_capacity,
            "active_machines_before": active_machines,
            "excess_active_machines_after_shortage": excess_active_machines,
            "stopped_machines_due_to_rtt_shortage": stopped_machines,
            "disrupted_rtt_capacity_file": rtt_path,
        }

        summary = make_summary(
            scenario_id,
            f"RTT shortage of {int(rho * 100)}% in the highest-workload treatment window",
            affected,
            details,
            rtt_shortage_level=rho
        )
        export_scenario(scenario_id, summary["scenario_name"], affected, summary)


# ============================================================
# D5
# ============================================================

if RUN_D5_COMBINED_MACHINE_DAY_RTT:
    s_machine = select_highest_impact_machine_day()

    affected_machine = schedule[
        (schedule["machine"] == s_machine["machine"]) &
        (schedule["day"] == s_machine["day"])
    ].copy()

    disrupted = rtt_util.copy()
    disrupted[f"{rtt_col}_before_disruption"] = disrupted[rtt_col]
    day_mask = disrupted["day"] == s_machine["day"]

    disrupted.loc[day_mask, rtt_col] = disrupted.loc[day_mask, rtt_col].apply(
        lambda v: max(0, int(math.floor(v * (1 - COMBINED_RTT_SHORTAGE))))
    )
    disrupted["rtt_shortage_level"] = COMBINED_RTT_SHORTAGE

    rtt_path = os.path.join(
        OUTPUT_FOLDER,
        "disrupted_rtt_capacity_D5_combined_machine_day_rtt.csv"
    )
    disrupted.to_csv(rtt_path, index=False)

    additional_rtt_affected_rows = []

    for row in disrupted[day_mask].itertuples():
        day = int(row.day)
        window = int(row.window)
        rtt_capacity_after = int(getattr(row, rtt_col))

        active_machines_after_machine_failure = schedule[
            (schedule["day"] == day) &
            (schedule["window"] == window) &
            (schedule["machine"] != s_machine["machine"])
        ]["machine"].nunique()

        excess_active_machines = max(
            0, active_machines_after_machine_failure - rtt_capacity_after
        )

        if excess_active_machines > 0:
            # After the selected LINAC has failed, stop only the additional
            # number of LINACs required to satisfy the reduced RTT capacity.
            # Rank the remaining active LINACs by treatment load in this
            # window, consistent with the D4 RTT-shortage logic.
            exposed_remaining = schedule[
                (schedule["day"] == day) &
                (schedule["window"] == window) &
                (schedule["machine"] != s_machine["machine"])
            ].copy()

            machine_load = (
                exposed_remaining
                .groupby("machine", as_index=False)
                .agg(
                    machine_window_fractions=("fraction", "count"),
                    machine_window_patients=("patient", "nunique"),
                    machine_window_minutes=("duration_minutes", "sum"),
                )
                .sort_values(
                    ["machine_window_minutes", "machine_window_fractions"],
                    ascending=[True, True],
                )
            )

            # Preserve the busiest remaining LINACs and stop only the
            # lowest-load LINACs required to satisfy RTT capacity.
            stopped_machines = machine_load.head(
                excess_active_machines
            )["machine"].tolist()

            extra = exposed_remaining[
                exposed_remaining["machine"].isin(stopped_machines)
            ].copy()
            extra["rtt_stopped_machine_due_to_shortage"] = extra["machine"]
            additional_rtt_affected_rows.append(extra)

    if additional_rtt_affected_rows:
        affected_rtt = pd.concat(additional_rtt_affected_rows, ignore_index=True)
    else:
        affected_rtt = schedule.iloc[0:0].copy()

    affected = pd.concat([affected_machine, affected_rtt], ignore_index=True)
    affected = affected.drop_duplicates(
        subset=["patient", "fraction", "machine", "day", "window"]
    )

    details = {
        "disruption_type": "combined_machine_full_day_rtt_shortage",
        "failed_machine": s_machine["machine"],
        "failed_day": s_machine["day"],
        "affected_windows": "all",
        "rtt_shortage_level": COMBINED_RTT_SHORTAGE,
        "machine_day_used_minutes_before_disruption": s_machine["used_minutes"],
        "machine_day_capacity_minutes_before_disruption": s_machine["capacity_minutes"],
        "machine_day_utilization_before_disruption": s_machine["utilization"],
        "selected_machine_day_affected_fractions": s_machine.get("affected_fractions", None),
        "selected_machine_day_affected_patients": s_machine.get("affected_patients", None),
        "disrupted_rtt_capacity_file": rtt_path,
    }

    summary = make_summary(
        "D5_combined_machine_day_rtt",
        "Full-day LINAC failure combined with RTT shortage on the same day",
        affected,
        details,
        rtt_shortage_level=COMBINED_RTT_SHORTAGE
    )
    export_scenario("D5_combined_machine_day_rtt", summary["scenario_name"], affected, summary)


print()
print("========================================")
print("DISRUPTION GENERATION COMPLETE")
print("========================================")
print(f"Output folder: {OUTPUT_FOLDER}")