import json
import math

def validate_results(file_path="results_log.json"):
    with open(file_path) as f:
        results = json.load(f)

    errors = []
    for idx, entry in enumerate(results, start=1):
        if "box_details" not in entry:  # skip pure rejection cases
            print(f"⚠️  Entry {idx}: Skipped (no box_details, only rejections)")
            continue

        inputs = entry["inputs"]
        box = entry["box_details"]
        insert = entry["insert_details"]

        ok = True
        details = []

        # 1. Max Parts check
        expected_max_parts = insert["units_per_insert"] * box["Layers"]
        if not (box["Max Parts"] == expected_max_parts or box["Max Parts"] <= expected_max_parts):
            ok = False
            details.append(f"Max Parts mismatch: got {box['Max Parts']}, expected {expected_max_parts}")

        # 2. Weight check
        weights = box["Weight Breakdown"]
        total_calc = sum(weights.values())
        if not math.isclose(total_calc, box["Total Weight"], rel_tol=1e-2):
            ok = False
            details.append(f"Weight mismatch: got {box['Total Weight']}, expected {total_calc:.2f}")

        # 3. Cost per part check
        cb = entry["cost_breakdown"]
        calc_cost_per_part = cb["total_cost_per_box"] / box["Max Parts"]
        if not math.isclose(cb["cost_per_part"], calc_cost_per_part, rel_tol=1e-6):
            ok = False
            details.append(f"Cost mismatch: got {cb['cost_per_part']}, expected {calc_cost_per_part:.2f}")

        # 4. Boxes per year check
        expected_boxes = math.ceil(inputs["Annual Parts"] / box["Max Parts"])
        if box["Boxes/Year"] != expected_boxes:
            ok = False
            details.append(f"Boxes/year mismatch: got {box['Boxes/Year']}, expected {expected_boxes}")

        # Report
        if ok:
            print(f"✅ Entry {idx}: OK")
        else:
            print(f"❌ Entry {idx}: Errors found -> {', '.join(details)}")
            errors.append((idx, details))

    print("\n--- Summary ---")
    print(f"Total entries: {len(results)}")
    print(f"Valid entries: {len(results) - len(errors)}")
    print(f"Invalid entries: {len(errors)}")

if __name__ == "__main__":
    validate_results()
