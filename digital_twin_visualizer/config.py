from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
GENERATED_DIR = ROOT_DIR / "generated"
TEMPLATES_DIR = ROOT_DIR / "templates"

DISRUPTION_DEFINITIONS = {
    "D1": {
        "label": "Machine unavailable for one window",
        "affected_filename": "affected_fractions_D1_machine_window.csv",
        "summary_filename": "no_recovery_summary_D1_machine_window.csv",
    },
    "D2": {
        "label": "Machine unavailable for a full day",
        "affected_filename": "affected_fractions_D2_machine_full_day.csv",
        "summary_filename": "no_recovery_summary_D2_machine_full_day.csv",
    },
    "D3": {
        "label": "Power outage during one window",
        "affected_filename": "affected_fractions_D3_power_outage_window.csv",
        "summary_filename": "no_recovery_summary_D3_power_outage_window.csv",
    },
    "D4": {
        "label": "RTT shortage 50%",
        "affected_filename": "affected_fractions_D4_rtt_shortage_50pct.csv",
        "summary_filename": "no_recovery_summary_D4_rtt_shortage_50pct.csv",
        "disrupted_rtt_capacity_filename": "disrupted_rtt_capacity_D4_rtt_shortage_50pct.csv",
    },
}

STRATEGY_FOLDER_NAMES = {
    "Local Repair": "local repair",
    "RESTORE": "restore",
    "Full Reoptimization": "full reoptimization",
}

STRATEGY_FILE_NAMES = {
    "Local Repair": {
        "schedule": "patient_local_repair_schedule.csv",
        "summary": "patient_local_repair_summary.csv",
        "recovered_fractions": "patient_local_repair_recovered_fractions.csv",
        "unrecovered_fractions": "patient_local_repair_unrecovered_fractions.csv",
        "shifted_course_fractions": "patient_local_repair_shifted_course_fractions.csv",
    },
    "RESTORE": {"schedule": "restore_schedule.csv", "summary": "restore_summary.csv"},
    "Full Reoptimization": {
        "schedule": "pure_full_reoptimization_schedule.csv",
        "summary": "pure_full_reoptimization_summary.csv",
    },
}

GENERATED_FILE_NAMES = {
    "Local Repair": "local_repair.xlsx",
    "RESTORE": "restore.xlsx",
    "Full Reoptimization": "full_reoptimization.xlsx",
}


def get_available_instances() -> list[str]:
    """Return instance folder names that contain a baseline schedule CSV."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        path.name
        for path in DATA_DIR.iterdir()
        if path.is_dir() and (path / "baseline" / "baseline_schedule.csv").exists()
    )


def instance_data_dir(instance_id: str) -> Path:
    return DATA_DIR / instance_id


def instance_generated_dir(instance_id: str) -> Path:
    return GENERATED_DIR / instance_id


def baseline_paths(instance_id: str) -> dict[str, Path]:
    data_dir = instance_data_dir(instance_id) / "baseline"
    return {
        "schedule_csv": data_dir / "baseline_schedule.csv",
        "schedule_excel": instance_generated_dir(instance_id) / "baseline.xlsx",
        "summary": data_dir / "baseline_summary.csv",
        "machine_utilization": data_dir / "machine_utilization.csv",
        "patient_summary": data_dir / "patient_schedule_summary.csv",
        "rtt_utilization": data_dir / "rtt_utilization.csv",
    }


def disruptions_for(instance_id: str) -> dict[str, dict[str, Path | str]]:
    disruption_root = instance_data_dir(instance_id) / "disruptions"
    result: dict[str, dict[str, Path | str]] = {}
    for disruption_id, definition in DISRUPTION_DEFINITIONS.items():
        folder = disruption_root / disruption_id
        item: dict[str, Path | str] = {
            "label": definition["label"],
            "affected": folder / definition["affected_filename"],
            "summary": folder / definition["summary_filename"],
        }
        if "disrupted_rtt_capacity_filename" in definition:
            item["disrupted_rtt_capacity"] = folder / definition["disrupted_rtt_capacity_filename"]
        result[disruption_id] = item
    return result


def strategy_paths(instance_id: str, disruption_id: str, strategy: str) -> dict[str, Path]:
    disruptions = disruptions_for(instance_id)
    if disruption_id not in disruptions:
        raise ValueError(f"Unknown disruption '{disruption_id}'. Expected one of: {list(disruptions)}")
    if strategy not in STRATEGY_FOLDER_NAMES:
        raise ValueError(f"Unknown strategy '{strategy}'. Expected one of: {list(STRATEGY_FOLDER_NAMES)}")

    source_dir = (
        instance_data_dir(instance_id)
        / "recovery"
        / disruption_id
        / STRATEGY_FOLDER_NAMES[strategy]
    )
    filenames = STRATEGY_FILE_NAMES[strategy]
    paths = {
        "schedule_csv": source_dir / filenames["schedule"],
        "summary": source_dir / filenames["summary"],
        "schedule_excel": instance_generated_dir(instance_id) / disruption_id / GENERATED_FILE_NAMES[strategy],
    }
    for key in ("recovered_fractions", "unrecovered_fractions", "shifted_course_fractions"):
        if key in filenames:
            paths[key] = source_dir / filenames[key]
    return paths


def ensure_generated_directories(instance_id: str) -> None:
    root = instance_generated_dir(instance_id)
    root.mkdir(parents=True, exist_ok=True)
    for disruption_id in DISRUPTION_DEFINITIONS:
        (root / disruption_id).mkdir(parents=True, exist_ok=True)


def validate_structure(instance_id: str) -> list[tuple[str, Path, bool]]:
    checks: list[tuple[str, Path, bool]] = []
    baseline = baseline_paths(instance_id)
    for label, key in (
        ("Baseline schedule", "schedule_csv"),
        ("Baseline summary", "summary"),
        ("Machine utilization", "machine_utilization"),
        ("Patient summary", "patient_summary"),
        ("RTT utilization", "rtt_utilization"),
    ):
        path = baseline[key]
        checks.append((label, path, path.exists()))

    for disruption_id, disruption in disruptions_for(instance_id).items():
        for label, key in (("affected fractions", "affected"), ("no-recovery summary", "summary")):
            path = Path(disruption[key])
            checks.append((f"{disruption_id} {label}", path, path.exists()))
        if "disrupted_rtt_capacity" in disruption:
            path = Path(disruption["disrupted_rtt_capacity"])
            checks.append((f"{disruption_id} disrupted RTT capacity", path, path.exists()))
        for strategy in STRATEGY_FOLDER_NAMES:
            for key, label in (
                ("schedule_csv", "schedule"), ("summary", "summary"),
                ("recovered_fractions", "recovered fractions"),
                ("unrecovered_fractions", "unrecovered fractions"),
                ("shifted_course_fractions", "shifted-course fractions"),
            ):
                paths = strategy_paths(instance_id, disruption_id, strategy)
                if key in paths:
                    path = paths[key]
                    checks.append((f"{disruption_id} {strategy} {label}", path, path.exists()))
    return checks


if __name__ == "__main__":
    instances = get_available_instances()
    if not instances:
        print(f"No valid instances found under: {DATA_DIR}")
    for instance_id in instances:
        ensure_generated_directories(instance_id)
        print(f"\nInstance: {instance_id}")
        for description, path, exists in validate_structure(instance_id):
            print(f"{'OK' if exists else 'MISSING':8} {description}: {path}")