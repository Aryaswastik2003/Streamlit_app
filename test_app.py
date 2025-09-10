import json
import math

def test_results_log_validations():
    with open("results_log.json") as f:
        results = json.load(f)

    for entry in results:
        inputs = entry["inputs"]
        box = entry["box_details"]
        weights = box["Weight Breakdown"]

        # 1. Weight consistency
        total_weight_calc = sum(weights.values())
        assert math.isclose(total_weight_calc, box["Total Weight"], rel_tol=1e-2)

        # 2. Capacity check
        assert box["Total Weight"] <= box["Capacity"]

        # 3. Cost check
        cb = entry["cost_breakdown"]
        assert math.isclose(
            cb["total_cost_per_box"] / box["Max Parts"],
            cb["cost_per_part"],
            rel_tol=1e-6
        )

        # 4. Boxes per year check
        parts = inputs["Annual Parts"]
        assert box["Boxes/Year"] == math.ceil(parts / box["Max Parts"])
