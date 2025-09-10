import json
import math

def test_results_log_validations():
    with open("results_log.json") as f:
        results = json.load(f)

    for entry in results:
        if "box_details" not in entry:
            continue
        box = entry["box_details"]
        weights = box["Weight Breakdown"]

        total_weight_calc = sum(weights.values())
        assert math.isclose(total_weight_calc, box["Total Weight"], rel_tol=1e-2)

def test_maths_validations():
    with open("results_log.json") as f:
        results = json.load(f)

    for entry in results:
        if "box_details" not in entry:  # skip pure rejection cases
            continue

        inputs = entry["inputs"]
        box = entry["box_details"]
        insert = entry["insert_details"]

        # 1. Max parts
        expected_max_parts = insert["units_per_insert"] * box["Layers"]
        assert box["Max Parts"] == expected_max_parts or box["Max Parts"] <= expected_max_parts

        # 2. Weight
        weights = box["Weight Breakdown"]
        total_calc = sum(weights.values())
        assert math.isclose(total_calc, box["Total Weight"], rel_tol=1e-2)

        # 3. Cost
        cb = entry["cost_breakdown"]
        calc_cost_per_part = cb["total_cost_per_box"] / box["Max Parts"]
        assert math.isclose(cb["cost_per_part"], calc_cost_per_part, rel_tol=1e-6)

        # 4. Boxes per year
        expected_boxes = math.ceil(inputs["Annual Parts"] / box["Max Parts"])
        assert box["Boxes/Year"] == expected_boxes
