import re


def combine_diagnoses(diagnose_list: list[str]) -> list[str]:
    combined = []
    current = ""

    diagnosis_start_re = re.compile(r"^Ch(?:ẩ|ấ|ầ)n đoán[:：]", re.IGNORECASE)

    for entry in diagnose_list:
        entry = entry.strip().strip("'").strip()

        if diagnosis_start_re.match(entry):
            if current:
                combined.append(current.strip())
            current = entry
        else:
            current += " " + entry

    if current:
        combined.append(current.strip())

    return combined


def combine_usage_instructions(raw_usage: list[str]) -> list[str]:
    cleaned_usage = []
    i = 0
    while i < len(raw_usage):
        item = raw_usage[i].strip("'").strip()
        if item.startswith("Ghi chú Uống"):
            # Check if next element is an instruction (and not another 'Ghi chú Uống')
            if i + 1 < len(raw_usage) and not raw_usage[i + 1].strip(
                "'"
            ).strip().startswith("Ghi chú"):
                instruction = raw_usage[i + 1].strip("'").strip()
                cleaned_usage.append(f"{item} {instruction}")
                i += 2
            else:
                cleaned_usage.append(f"{item}")
                i += 1
        else:
            i += 1
    return cleaned_usage


def extract_metadata(other_list: list[str]) -> dict:
    metadata = {"Re-examination": [], "Doctor": [], "Notion": []}
    cleaned = [item.strip().strip("'").strip() for item in other_list]

    for entry in cleaned:
        if entry.startswith("Ngày hẹn tái khám"):
            metadata["Re-examination"].append(entry)
        elif entry.startswith("BS."):
            metadata["Doctor"].append(entry)
        elif entry.startswith("Tái khám xin mang theo đơn"):
            metadata["Notion"].append(entry)

    return metadata


def reformat_dict(d: dict) -> dict:
    """
    Reformat the label dictionary to combine diagnoses and usage instructions.
    """
    if "diagnose" in d:
        d["diagnose"] = combine_diagnoses(d["diagnose"])

    if "usage" in d:
        d["usage"] = combine_usage_instructions(d["usage"])
    if "other" in d:
        n_d = extract_metadata(d["other"])
        d = d | n_d

    return d
