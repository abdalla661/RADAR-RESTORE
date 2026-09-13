import os
import re
import ast
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INSTANCE_PATH = r"data_example\demo_instance.csv"

OUTPUT_FOLDER = "modeled_instance"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_variable(text, variable_name):
    pattern = rf"{variable_name}\s*=\s*(.*?);"
    match = re.search(pattern, text, flags=re.S)

    if not match:
        raise ValueError(f"Variable not found: {variable_name}")

    return ast.literal_eval(match.group(1))


# ============================================================
# READ INSTANCE
# ============================================================

with open(INSTANCE_PATH, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

number_patients = extract_variable(text, "number_patients")
number_machines = extract_variable(text, "number_machines")
number_weekdays = extract_variable(text, "number_weekdays")
number_windows = extract_variable(text, "number_windows")
number_protocols = extract_variable(text, "number_protocols")
window_length = extract_variable(text, "window_length")

occupied_timeslots_machines = extract_variable(text, "occupied_timeslots_machines")


# ============================================================
# PATIENT PARAMETERS
# ============================================================

protocol_ind_patient = extract_variable(text, "protocol_ind_patient")
target_days_patient = extract_variable(text, "target_days_patient")
day_min_patient = extract_variable(text, "day_min_patient")


# ============================================================
# PROTOCOL PARAMETERS
# ============================================================

number_fractions_protocol = extract_variable(text, "number_fractions_protocol")
durations_first_fraction_protocol = extract_variable(text, "durations_first_fraction_protocol")
durations_remaining_fractions_protocol = extract_variable(text, "durations_remaining_fractions_protocol")
protocol_priority = extract_variable(text, "protocol_priority")
allowed_machines_protocol = extract_variable(text, "allowed_machines_protocol")

# Optional variables
try:
    preferred_machines_protocol = extract_variable(text, "preferred_machines_protocol")
except ValueError:
    preferred_machines_protocol = None


# ============================================================
# BUILD MACHINE CAPACITY TABLE
# ============================================================

machine_capacity_rows = []

for m in range(number_machines):
    for d in range(number_weekdays):
        for w in range(number_windows):

            raw_capacity = window_length[w]
            occupied = occupied_timeslots_machines[m][d][w]
            remaining_capacity = raw_capacity - occupied

            machine_capacity_rows.append({
                "machine": m + 1,
                "day": d + 1,
                "window": w + 1,
                "window_length_minutes": raw_capacity,
                "occupied_minutes": occupied,
                "remaining_machine_capacity_minutes": remaining_capacity
            })

machine_capacity_df = pd.DataFrame(machine_capacity_rows)

machine_capacity_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "machine_capacity.csv"),
    index=False
)


# ============================================================
# BUILD PATIENT TABLE
# ============================================================

patient_rows = []

for p in range(number_patients):

    protocol = protocol_ind_patient[p]

    protocol_index = protocol - 1

    number_fractions = number_fractions_protocol[protocol_index]

    first_duration = durations_first_fraction_protocol[protocol_index]
    remaining_duration = durations_remaining_fractions_protocol[protocol_index]

    total_demand = (
        first_duration
        + (number_fractions - 1) * remaining_duration
    )

    allowed_machines = sorted(list(allowed_machines_protocol[protocol_index]))

    if preferred_machines_protocol is not None:
        preferred_machines = sorted(list(preferred_machines_protocol[protocol_index]))
    else:
        preferred_machines = []

    earliest_start = day_min_patient[p]
    target_day = target_days_patient[p]

    slack = max(target_day - earliest_start + 1, 1)

    time_pressure = number_fractions / slack

    patient_rows.append({
        "patient": p + 1,
        "protocol": protocol,
        "priority": protocol_priority[protocol_index],
        "number_fractions": number_fractions,
        "first_fraction_duration": first_duration,
        "remaining_fraction_duration": remaining_duration,
        "total_demand_minutes": total_demand,
        "earliest_start_day": earliest_start,
        "target_day": target_day,
        "allowed_machines": allowed_machines,
        "preferred_machines": preferred_machines,
        "number_allowed_machines": len(allowed_machines),
        "time_pressure": time_pressure
    })

patients_df = pd.DataFrame(patient_rows)

patients_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "patients.csv"),
    index=False
)


# ============================================================
# BUILD FRACTION TABLE
# ============================================================
#
# Each patient p has fractions f = 1,...,F_p.
# First fraction may have different duration.
#
# This table is very useful later for baseline scheduling.
#
# ============================================================

fraction_rows = []

for _, row in patients_df.iterrows():

    patient = row["patient"]
    number_fractions = int(row["number_fractions"])

    for f in range(1, number_fractions + 1):

        if f == 1:
            duration = row["first_fraction_duration"]
        else:
            duration = row["remaining_fraction_duration"]

        fraction_rows.append({
            "patient": patient,
            "fraction": f,
            "duration_minutes": duration,
            "priority": row["priority"],
            "earliest_start_day": row["earliest_start_day"],
            "target_day": row["target_day"],
            "allowed_machines": row["allowed_machines"]
        })

fractions_df = pd.DataFrame(fraction_rows)

fractions_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "fractions.csv"),
    index=False
)


# ============================================================
# RTT CAPACITY EXTENSION
# ============================================================
#
# The original RTSP benchmark includes machine capacity,
# but does not include RTT staff capacity.
#
# We introduce a synthetic RTT capacity resource.
#
# Baseline assumption:
# Each active machine-window requires one RTT treatment unit.
#
# Therefore:
# RTT capacity in a day-window = number of machines with positive remaining capacity.
#
# ============================================================

rtt_rows = []

for d in range(1, number_weekdays + 1):
    for w in range(1, number_windows + 1):

        subset = machine_capacity_df[
            (machine_capacity_df["day"] == d)
            & (machine_capacity_df["window"] == w)
        ]

        active_machines = (
            subset["remaining_machine_capacity_minutes"] > 0
        ).sum()

        rtt_rows.append({
            "day": d,
            "window": w,
            "available_rtt_teams": int(active_machines),
            "definition": "one_treatment_delivery_team_per_active_machine_window"
        })

rtt_capacity_df = pd.DataFrame(rtt_rows)

rtt_capacity_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "rtt_capacity.csv"),
    index=False
)


# ============================================================
# INSTANCE SUMMARY
# ============================================================

total_raw_capacity = machine_capacity_df["window_length_minutes"].sum()
total_occupied = machine_capacity_df["occupied_minutes"].sum()
total_remaining_capacity = machine_capacity_df["remaining_machine_capacity_minutes"].sum()
total_demand = patients_df["total_demand_minutes"].sum()
total_fractions = len(fractions_df)

summary = {
    "instance": os.path.basename(INSTANCE_PATH),
    "number_patients": number_patients,
    "number_machines": number_machines,
    "number_weekdays": number_weekdays,
    "number_windows": number_windows,
    "number_protocols": number_protocols,
    "total_raw_capacity_minutes": total_raw_capacity,
    "total_occupied_minutes": total_occupied,
    "total_effective_capacity_minutes": total_remaining_capacity,
    "total_demand_minutes": total_demand,
    "total_fractions": total_fractions,
    "demand_ratio": total_demand / total_remaining_capacity,
    "average_allowed_machines": patients_df["number_allowed_machines"].mean(),
    "average_time_pressure": patients_df["time_pressure"].mean(),
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "instance_summary.csv"),
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("RTSP DATA MODEL WITH RTT CAPACITY BUILT")
print("========================================")

print("\nInstance:")
print(os.path.basename(INSTANCE_PATH))

print("\nSets:")
print(f"Patients: {number_patients}")
print(f"Machines: {number_machines}")
print(f"Weekdays: {number_weekdays}")
print(f"Windows: {number_windows}")
print(f"Protocols: {number_protocols}")

print("\nCapacity:")
print(f"Raw capacity: {total_raw_capacity}")
print(f"Occupied capacity: {total_occupied}")
print(f"Effective capacity: {total_remaining_capacity}")

print("\nDemand:")
print(f"Total demand minutes: {total_demand}")
print(f"Total fractions: {total_fractions}")
print(f"Demand ratio: {total_demand / total_remaining_capacity:.4f}")

print("\nGenerated files:")
print(f"{OUTPUT_FOLDER}/patients.csv")
print(f"{OUTPUT_FOLDER}/fractions.csv")
print(f"{OUTPUT_FOLDER}/machine_capacity.csv")
print(f"{OUTPUT_FOLDER}/rtt_capacity.csv")
print(f"{OUTPUT_FOLDER}/instance_summary.csv")