import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# -----------------------------
# Data Models & Constants (updated with PRD requirements)
# -----------------------------

@dataclass(frozen=True)
class Truck:
    name: str
    dims: Tuple[int, int, int]  # (L, W, H) in mm
    payload_kg: float
    trip_cost: float  # dummy cost per trip

# --- Box dataclass now carries an estimated empty box mass (kg) ---
@dataclass(frozen=True)
class Box:
    box_type: str
    dims: Tuple[int, int, int]  # (L, W, H) in mm
    capacity_kg: float
    empty_weight_kg: float = 0.0   # estimated empty box mass (includes lid where applicable)
    asset_cost: float = 0.0       # PRD requirement: packaging asset cost

# --- BOX_DATABASE updated with PRD-informed complete 6 PP Box sizes ---
BOX_DATABASE: List[Box] = [
    # Complete PP Box database (6 standard sizes as per PRD, capacity 15-16kg)
    Box("PP Box", (300, 200, 150), 15, 0.6, 450),   # Small
    Box("PP Box", (400, 300, 200), 16, 0.8, 580),   # Medium-Small
    Box("PP Box", (400, 300, 235), 16, 0.8, 620),   # Medium
    Box("PP Box", (500, 350, 250), 15, 1.2, 720),   # Medium-Large
    Box("PP Box", (600, 400, 300), 16, 1.6, 850),   # Large
    Box("PP Box", (600, 400, 348), 16, 1.8, 920),   # Extra-Large

    # Foldable Crate (explicit type in DB) - matches PRD small crate sizes
    Box("Foldable Crate", (600, 400, 348), 15, 1.9, 1200),

    # FLCs (PRD: includes lid weight ~5.13 kg; typical total empty mass often in 4-9 kg range)
    Box("FLC", (1200, 1000, 595), 700, 6.8, 2500),    # standard FLC (includes lid mass)
    Box("FLC", (1200, 1000, 1200), 1000, 10.5, 3200), # taller FLC variant

    # PLS (pallet + lid sleeve) — heavier structure
    Box("PLS", (1500, 1200, 1000), 600, 28.0, 4800),
]

TRUCKS: List[Truck] = [
    Truck("9T Truck", (5500, 2200, 2400), 9000, 15000),
    Truck("16T Truck", (7500, 2500, 2600), 16000, 20000),
    Truck("22T Truck", (9500, 2600, 2800), 22000, 28000),
]

CO2_FACTORS = {"Highway": 0.08, "Semi-Urban": 0.12, "Village": 0.15}
ERGONOMIC_LIFT_KG = 25

LOCATIONS = ["Select", "Chennai", "Bangalore", "Delhi", "Pune", "Hyderabad", "Mumbai", "Kolkata"]

# Route distance matrix (in km) - PRD requirement for route-based optimization
ROUTE_DISTANCES = {
    ("Chennai", "Bangalore"): 346,
    ("Chennai", "Delhi"): 2165,
    ("Chennai", "Pune"): 1100,
    ("Chennai", "Hyderabad"): 626,
    ("Chennai", "Mumbai"): 1338,
    ("Chennai", "Kolkata"): 1670,
    ("Bangalore", "Chennai"): 346,
    ("Bangalore", "Delhi"): 2072,
    ("Bangalore", "Pune"): 835,
    ("Bangalore", "Hyderabad"): 569,
    ("Bangalore", "Mumbai"): 982,
    ("Bangalore", "Kolkata"): 1871,
    # Add more routes as needed - this is a sample
}

# Fuel costs per km by truck type and route type
FUEL_COST_PER_KM = {
    "9T Truck": {"Highway": 8.5, "Semi-Urban": 12.0, "Village": 15.5},
    "16T Truck": {"Highway": 12.0, "Semi-Urban": 16.5, "Village": 21.0},
    "22T Truck": {"Highway": 15.5, "Semi-Urban": 21.0, "Village": 27.0},
}

# Box handling costs (Rs per box)
BOX_HANDLING_COST = {
    "PP Box": 25,
    "Foldable Crate": 30,
    "FLC": 50,
    "PLS": 75,
}

# -----------------------------
# Updated Material Selection Constants (PRD requirements)
# -----------------------------
CUSHION_FACTORS = {
    "PP Partition Grid": 0.90,
    "Honeycomb Layer Pad": 0.85,
    "Thermo-vac PP Tray": 0.80,
    "Woven PP Pouch": 0.75
}
PP_DENSITY_G_CM3 = 0.9
PP_DENSITY_G_MM3 = 0.0009

# PRD-specified spacing requirements
SIDE_CLEARANCE = 2.5      # PRD: 2.5mm space on all 4 sides
TOP_CLEARANCE = 5         # PRD: 5mm space on top
BOTTOM_CLEARANCE = 0      # PRD: 0mm space on bottom
PARTITION_THICKNESS = 5   # PRD: Insert thickness 5mm or 6mm (using 5mm)

# -----------------------------
# Helpers (updated with PRD spacing)
# -----------------------------
def get_internal_dims(box: Box) -> Tuple[int, int, int]:
    L, W, H = box.dims
    if box.box_type == "PP Box":
        return (L - 34, W - 34, H - 8)
    elif box.box_type == "PLS":
        return (L - 34, W - 34, H - 210)
    elif box.box_type == "FLC":
        return (L - 30, W - 30, H - 30)
    else:
        return (L, W, H)

def calculate_part_size_factor(part_dim):
    L, W, H = part_dim
    volume_cm3 = (L * W * H) / 1000
    if volume_cm3 <= 50:
        return 0.2
    elif volume_cm3 <= 200:
        return 0.4
    elif volume_cm3 <= 500:
        return 0.6
    elif volume_cm3 <= 1000:
        return 0.8
    else:
        return 1.0

def calculate_load_factor(part_weight, units_per_insert):
    total_weight_per_insert = part_weight * units_per_insert
    if total_weight_per_insert <= 5:
        return 0.3
    elif total_weight_per_insert <= 15:
        return 0.6
    elif total_weight_per_insert <= 30:
        return 0.8
    else:
        return 1.0

def select_material_specs(fragility, part_dim, part_weight, units_per_insert, insert_area_m2):
    size_factor = calculate_part_size_factor(part_dim)
    load_factor = calculate_load_factor(part_weight, units_per_insert)
    
    # PRD requirement: Insert thickness should be ONLY 5mm or 6mm
    # Choose thickness based on load requirements
    total_load = part_weight * units_per_insert
    if total_load > 20 or fragility == "High":
        insert_thickness = 6.0  # Heavier load or high fragility = 6mm
    else:
        insert_thickness = 5.0  # Standard case = 5mm
    
    if fragility == "High":
        if insert_area_m2 >= 0.02 or size_factor >= 0.6:
            volume_mm3 = insert_area_m2 * 1e6 * insert_thickness
            weight_kg = (volume_mm3 * PP_DENSITY_G_MM3) / 1000.0
            return {
                "type": "Thermo-vac PP Tray",
                "gsm_or_thickness": f"{insert_thickness}mm PP sheet",
                "weight_kg": round(weight_kg, 2),
                "note": f"Form-fit tray for fragile parts (thickness: {insert_thickness}mm)"
            }
        else:
            # For pouches, use fixed thickness equivalent weight
            weight_kg = insert_area_m2 * (insert_thickness / 10.0)  # Simplified weight calc
            return {
                "type": "Woven PP Pouch",
                "gsm_or_thickness": f"{insert_thickness}mm equivalent thickness",
                "weight_kg": round(weight_kg, 2),
                "note": f"Soft pouch for small/fragile parts (thickness: {insert_thickness}mm)"
            }
    else:
        # PP Partition Grid - use fixed thickness instead of GSM
        volume_mm3 = insert_area_m2 * 1e6 * insert_thickness
        weight_kg = (volume_mm3 * PP_DENSITY_G_MM3) / 1000.0
        return {
            "type": "PP Partition Grid",
            "gsm_or_thickness": f"{insert_thickness}mm corrugated PP",
            "weight_kg": round(weight_kg, 2),
            "note": f"Grid partition (thickness: {insert_thickness}mm, load factor: {load_factor:.1f})"
        }

# -----------------------------
# Updated Orientation Categorization (PRD Point 7)
# -----------------------------
def categorize_orientations(part_dim):
    """PRD requirement: 3 categorical approaches for part standing positions"""
    L, W, H = part_dim
    
    return {
        "Height Standing": [(L, W, H), (W, L, H)],  # H is vertical
        "Length Standing": [(H, W, L), (W, H, L)],  # L is vertical
        "Width Standing": [(L, H, W), (H, L, W)]    # W is vertical
    }

# -----------------------------
# Truck Optimization Functions (unchanged)
# -----------------------------
def get_route_distance(source, destination):
    """Get distance between two cities"""
    if source == destination:
        return 0
    
    route = (source, destination)
    reverse_route = (destination, source)
    
    if route in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[route]
    elif reverse_route in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[reverse_route]
    else:
        # Default distance if route not found
        return 500

def calculate_boxes_per_truck(box, truck):
    """Calculate how many boxes fit in a truck based on dimensions and weight"""
    box_l, box_w, box_h = box.dims
    truck_l, truck_w, truck_h = truck.dims
    
    # Calculate how many boxes fit by dimensions
    boxes_by_length = truck_l // box_l
    boxes_by_width = truck_w // box_w
    boxes_by_height = truck_h // box_h
    
    # Total boxes that fit geometrically
    boxes_by_dims = boxes_by_length * boxes_by_width * boxes_by_height
    
    return max(0, boxes_by_dims)

def calculate_truck_utilization(box, box_weight, boxes_per_truck, truck):
    """Calculate truck space and weight utilization"""
    box_l, box_w, box_h = box.dims
    truck_l, truck_w, truck_h = truck.dims
    
    # Volume utilization
    box_volume = box_l * box_w * box_h
    truck_volume = truck_l * truck_w * truck_h
    volume_utilization = (boxes_per_truck * box_volume) / truck_volume * 100
    
    # Weight utilization
    total_weight = boxes_per_truck * box_weight
    weight_utilization = (total_weight / truck.payload_kg) * 100
    
    return volume_utilization, weight_utilization

def calculate_trip_cost(truck, distance, route_percentages, boxes_per_truck, box_type, box_asset_cost):
    """Calculate total cost per trip based on PRD requirements (including packaging asset cost)"""
    if distance <= 0:
        return 0, 0, 0, 0, 0
    
    # Fuel cost calculation based on route type distribution
    fuel_cost = 0
    co2_emission = 0
    
    for route_type, percentage in route_percentages.items():
        if percentage > 0:
            route_distance = distance * (percentage / 100)
            fuel_rate = FUEL_COST_PER_KM.get(truck.name, {}).get(route_type, 10)
            co2_rate = CO2_FACTORS.get(route_type, 0.1)
            
            fuel_cost += route_distance * fuel_rate
            co2_emission += route_distance * co2_rate
    
    # Box handling cost
    handling_cost_per_box = BOX_HANDLING_COST.get(box_type, 40)
    total_handling_cost = boxes_per_truck * handling_cost_per_box
    
    # Packaging asset cost (PRD requirement)
    total_asset_cost = boxes_per_truck * box_asset_cost
    
    # Total trip cost (PRD: Trip cost + box handling cost + packaging asset cost)
    total_trip_cost = truck.trip_cost + fuel_cost + total_handling_cost + total_asset_cost
    
    return total_trip_cost, fuel_cost, total_handling_cost, total_asset_cost, co2_emission

def optimize_truck_loading(box_recommendation, source, destination, route_percentages):
    """Main truck optimization function as per PRD requirements"""
    if not box_recommendation or "box_details" not in box_recommendation:
        return []
    
    box_details = box_recommendation["box_details"]
    box_type = box_details["Box Type"]
    box_dims = box_details["Box Dimensions"]
    box_weight = box_details["Total Weight"]
    parts_per_box = box_details["Max Parts"]
    
    # Find the box object for calculations
    selected_box = None
    for box in BOX_DATABASE:
        if box.box_type == box_type and box.dims == box_dims:
            selected_box = box
            break
    
    if not selected_box:
        return []
    
    distance = get_route_distance(source, destination)
    
    truck_recommendations = []
    
    # Analyze all truck types as per PRD requirement
    for truck in TRUCKS:
        # Calculate how many boxes fit in this truck
        boxes_per_truck = calculate_boxes_per_truck(selected_box, truck)
        
        if boxes_per_truck <= 0:
            continue
        
        # Check weight constraints
        total_weight = boxes_per_truck * box_weight
        if total_weight > truck.payload_kg:
            # Reduce boxes to fit weight limit
            boxes_per_truck = int(truck.payload_kg // box_weight)
            total_weight = boxes_per_truck * box_weight
        
        if boxes_per_truck <= 0:
            continue
        
        # Calculate utilization
        volume_util, weight_util = calculate_truck_utilization(
            selected_box, box_weight, boxes_per_truck, truck
        )
        
        # Calculate costs and CO2 (including asset cost)
        trip_cost, fuel_cost, handling_cost, asset_cost, co2_emission = calculate_trip_cost(
            truck, distance, route_percentages, boxes_per_truck, box_type, selected_box.asset_cost
        )
        
        # Calculate parts per truck (PRD requirement)
        total_parts_per_truck = boxes_per_truck * parts_per_box
        
        # Cost per part per trip (PRD key metric)
        cost_per_part = trip_cost / total_parts_per_truck if total_parts_per_truck > 0 else float('inf')
        
        # CO2 per part per trip (PRD key metric)
        co2_per_part = co2_emission / total_parts_per_truck if total_parts_per_truck > 0 else 0
        
        truck_recommendations.append({
            "truck_name": truck.name,
            "truck_dims": truck.dims,
            "payload_capacity": truck.payload_kg,
            "boxes_per_truck": boxes_per_truck,
            "total_parts_per_truck": total_parts_per_truck,
            "volume_utilization": volume_util,
            "weight_utilization": weight_util,
            "total_trip_cost": trip_cost,
            "fuel_cost": fuel_cost,
            "handling_cost": handling_cost,
            "asset_cost": asset_cost,
            "cost_per_part": cost_per_part,
            "co2_emission": co2_emission,
            "co2_per_part": co2_per_part,
            "efficiency_score": (volume_util + weight_util) / 2,  # Combined efficiency
        })
    
    # Sort by cost per part (ascending) - best recommendation first
    truck_recommendations.sort(key=lambda x: x["cost_per_part"])
    
    return truck_recommendations

# -----------------------------
# Updated Recommendation Functions (PRD Points 1, 2, 3, 7)
# -----------------------------
def design_insert_for_box(part_dim, box_internal_dim, fragility, part_weight=1.0, orientation_restriction="None"):
    best_fit = {
        "units_per_insert": 0,
        "matrix": (0, 0),
        "cell_dims": (0, 0, 0),
        "outer_dims": (0, 0, 0),
        "part_orientation": part_dim,
        "volume_efficiency": 0,
        "standing_category": "None"
    }

    L, W, H = part_dim
    box_L, box_W, box_H = box_internal_dim

    # Guard against invalid box dimensions
    if box_L <= 0 or box_W <= 0 or box_H <= 0:
        return None

    box_volume = box_L * box_W * box_H
    if box_volume <= 0:
        return None

    # PRD Point 7: Categorize orientations by standing positions
    categories = categorize_orientations(part_dim)
    
    # Build orientations based on orientation_restriction
    if orientation_restriction in (None, "None"):
        orientations = []
        for category_orientations in categories.values():
            orientations.extend(category_orientations)
    elif orientation_restriction == "Length Standing":
        orientations = categories["Length Standing"]
    elif orientation_restriction == "Width Standing":
        orientations = categories["Width Standing"]
    elif orientation_restriction == "Height Standing":
        orientations = categories["Height Standing"]
    else:
        orientations = []
        for category_orientations in categories.values():
            orientations.extend(category_orientations)

    for pL, pW, pH in orientations:
        # PRD Point 2: Updated spacing requirements
        available_height = box_H - TOP_CLEARANCE - BOTTOM_CLEARANCE
        if pH > available_height:
            continue
        if (pL + PARTITION_THICKNESS) <= 0 or (pW + PARTITION_THICKNESS) <= 0:
            continue

        # PRD Point 2: Use 2.5mm side clearance (convert to int for floor division)
        cols = max(0, int(box_L - (2 * SIDE_CLEARANCE)) // (pL + PARTITION_THICKNESS))
        rows = max(0, int(box_W - (2 * SIDE_CLEARANCE)) // (pW + PARTITION_THICKNESS))
        units_this_orientation = cols * rows
        
        if units_this_orientation > 0:
            insert_L = (cols * pL) + ((cols + 1) * PARTITION_THICKNESS)
            insert_W = (rows * pW) + ((rows + 1) * PARTITION_THICKNESS)
            insert_H = min(box_H, pH + TOP_CLEARANCE)
            
            part_volume = pL * pW * pH
            used_volume_parts = units_this_orientation * part_volume
            volume_efficiency = (used_volume_parts / box_volume) * 100

            # Determine standing category
            standing_category = "None"
            for cat_name, cat_orientations in categories.items():
                if (pL, pW, pH) in cat_orientations:
                    standing_category = cat_name
                    break

            if (volume_efficiency > best_fit["volume_efficiency"] or 
                (volume_efficiency == best_fit["volume_efficiency"] and units_this_orientation > best_fit["units_per_insert"])):
                
                best_fit["units_per_insert"] = units_this_orientation
                best_fit["matrix"] = (cols, rows)
                best_fit["cell_dims"] = (pL, pW, pH)
                best_fit["outer_dims"] = (insert_L, insert_W, insert_H)
                best_fit["part_orientation"] = (pL, pW, pH)
                best_fit["volume_efficiency"] = volume_efficiency
                best_fit["standing_category"] = standing_category

    if best_fit["units_per_insert"] == 0:
        return None

    insert_L, insert_W, insert_H = best_fit["outer_dims"]
    insert_area_m2 = (insert_L / 1000) * (insert_W / 1000)

    material_specs = select_material_specs(
        fragility, part_dim, part_weight, best_fit["units_per_insert"], insert_area_m2
    )
    
    best_fit.update(material_specs)
    return best_fit

def get_separator_details(insert, stacking_allowed):
    if not stacking_allowed or not insert:
        return {"needed": False, "type": "N/A", "weight_kg": 0.0, "note": "Stacking disabled."}
    if insert["type"] in ("PP Partition Grid", "Thermo-vac PP Tray"):
        return {"needed": True, "type": "Honeycomb Layer Pad", "weight_kg": 1.49, "note": "Adds strength between stacked layers."}
    else:
        return {"needed": True, "type": "PP Sheet Separator", "weight_kg": 1.0, "note": "General separator for multiple layers."}

def calculate_cost_per_part(box, insert, separator, layers, fit_count, annual_parts, route_info):
    """Calculate cost per part per trip as per PRD requirement (Point 1)"""
    if not route_info or fit_count <= 0:
        return float('inf'), {}
    
    # Basic cost calculation without truck optimization
    insert_weight_total = insert["weight_kg"] * layers
    separator_weight_total = separator["weight_kg"] * max(0, layers - 1)
    part_total_weight = fit_count * route_info.get("part_weight", 1.0)
    box_empty_mass = getattr(box, "empty_weight_kg", 0.0)
    total_weight = part_total_weight + insert_weight_total + separator_weight_total + box_empty_mass
    
    # Estimated cost per part (simplified for box selection)
    handling_cost = BOX_HANDLING_COST.get(box.box_type, 40)
    asset_cost = getattr(box, "asset_cost", 0.0)
    
    cost_per_box = handling_cost + asset_cost
    cost_per_part = cost_per_box / fit_count if fit_count > 0 else float('inf')
    
    cost_breakdown = {
        "handling_cost_per_box": handling_cost,
        "asset_cost_per_box": asset_cost,
        "total_cost_per_box": cost_per_box,
        "cost_per_part": cost_per_part
    }
    
    return cost_per_part, cost_breakdown

def recommend_boxes(part_dim, part_weight, stacking_allowed, fragility, forklift_available,
                    forklift_capacity, forklift_dim, annual_parts, orientation_restriction):
    """Updated recommendation function implementing PRD Points 1, 3"""
    
    all_viable_options = []
    rejection_log = {}
    
    # Simple route info for cost calculation
    route_info = {"part_weight": part_weight}
    
    for box in BOX_DATABASE:
        log_key = f"{box.box_type} ({box.dims[0]}x{box.dims[1]}x{box.dims[2]})"
        internal_dims = get_internal_dims(box)

        if forklift_available and forklift_dim:
            if not (box.dims[0] <= forklift_dim[0] and box.dims[1] <= forklift_dim[1]):
                rejection_log[log_key] = f"Rejected: Box footprint ({box.dims[0]}x{box.dims[1]}) exceeds forklift dimensions ({forklift_dim[0]}x{forklift_dim[1]})."
                continue

        insert = design_insert_for_box(part_dim, internal_dims, fragility, part_weight, orientation_restriction)
        if not insert or insert["units_per_insert"] == 0:
            rejection_log[log_key] = f"Rejected: Part does not fit in any orientation inside the box's internal dimensions ({internal_dims[0]}x{internal_dims[1]}x{internal_dims[2]})."
            continue

        separator = get_separator_details(insert, stacking_allowed)
        insert_height = insert["outer_dims"][2]
        if insert_height <= 0: continue

        layers = internal_dims[2] // insert_height if stacking_allowed else 1
        if layers < 1: layers = 1
        fit_count = layers * insert["units_per_insert"]
        if fit_count == 0: continue

        # Limit parts by box weight capacity
        if part_weight <= 0:
            rejection_log[log_key] = f"Rejected: Invalid part weight ({part_weight})."
            continue

        max_parts_by_weight = int(box.capacity_kg // part_weight)
        if max_parts_by_weight < 1:
            rejection_log[log_key] = f"Rejected: Single part weight ({part_weight} kg) exceeds box capacity ({box.capacity_kg} kg)."
            continue

        fit_count = min(fit_count, max_parts_by_weight)

        # Weight breakdown
        part_total_weight = fit_count * part_weight
        insert_weight_total = insert["weight_kg"] * layers
        separator_weight_total = separator["weight_kg"] * max(0, layers - 1)
        box_empty_mass = getattr(box, "empty_weight_kg", 0.0)
        total_weight = part_total_weight + insert_weight_total + separator_weight_total + box_empty_mass

        # Adjust fit_count if overweight
        while fit_count > 0 and total_weight > box.capacity_kg:
            fit_count -= 1
            part_total_weight = fit_count * part_weight
            total_weight = part_total_weight + insert_weight_total + separator_weight_total + box_empty_mass

        if fit_count <= 0:
            rejection_log[log_key] = f"Rejected: Even 1 part + packaging exceeds box capacity ({box.capacity_kg} kg)."
            continue

        # Forklift capacity check
        if forklift_available and forklift_capacity and total_weight > forklift_capacity:
            rejection_log[log_key] = f"Rejected: Total weight ({total_weight:.1f} kg) exceeds forklift capacity ({forklift_capacity} kg)."
            continue

        # PRD Point 1: Calculate cost per part per trip
        cost_per_part, cost_breakdown = calculate_cost_per_part(
            box, insert, separator, layers, fit_count, annual_parts, route_info
        )

        # Volume/waste metrics (unchanged)
        part_volume = part_dim[0] * part_dim[1] * part_dim[2]
        box_volume = internal_dims[0] * internal_dims[1] * internal_dims[2]
        used_volume_parts = fit_count * part_volume
        insert_outer_vol = insert["outer_dims"][0] * insert["outer_dims"][1] * insert["outer_dims"][2]
        used_volume_insert = insert_outer_vol * layers

        wasted_pct_parts = 100 * (1 - (used_volume_parts / box_volume)) if box_volume > 0 else 100
        wasted_pct_insert = 100 * ((used_volume_insert - used_volume_parts) / box_volume) if box_volume > 0 and used_volume_insert > used_volume_parts else 0
        volume_efficiency_total = 100 - wasted_pct_parts - wasted_pct_insert

        insert_material_pct = 100 * ((used_volume_insert - used_volume_parts) / used_volume_insert) if used_volume_insert > 0 else 0
        combined_efficiency = volume_efficiency_total + insert_material_pct

        boxes_per_year = -(-annual_parts // fit_count) if fit_count > 0 else 0

        option = {
            "insert_details": insert,
            "separator_details": separator,
            "cost_per_part": cost_per_part,
            "cost_breakdown": cost_breakdown,
            "box_details": {
                "Box Type": box.box_type,
                "Box Dimensions": box.dims,
                "Internal Dimensions": internal_dims,
                "Max Parts": fit_count,
                "Total Weight": total_weight,
                "Weight Breakdown": {
                    "Parts": part_total_weight,
                    "Inserts": insert_weight_total,
                    "Separators": separator_weight_total,
                    "FLC Lid": box_empty_mass
                },
                "Capacity": box.capacity_kg,
                "Asset Cost": getattr(box, "asset_cost", 0.0),
                "Wasted Volume % (parts)": wasted_pct_parts,
                "Wasted Volume % (insert)": wasted_pct_insert,
                "Volume Efficiency %": volume_efficiency_total,
                "Parts Efficiency %": combined_efficiency,
                "Insert Material Value %": insert_material_pct,
                "Insert Outer Volume (mm^3)": insert_outer_vol,
                "Boxes/Year": boxes_per_year,
                "Layers": layers,
                "Orientation Restriction": orientation_restriction,
                "Standing Category": insert.get("standing_category", "None")
            }
        }

        all_viable_options.append(option)

    if not all_viable_options:
        return {"rejection_log": rejection_log}

    # PRD Point 1 & 3: Sort by cost per part (cheapest first) and provide dual recommendations
    all_viable_options.sort(key=lambda x: x["cost_per_part"])

    return {
        "system_recommendation": all_viable_options[0],  # Best choice by cost per part
        "alternative_options": all_viable_options[1:5],  # Up to 4 alternatives for user choice
        "all_options": all_viable_options,
        "total_options": len(all_viable_options),
        "rejection_log": rejection_log
    }

# -----------------------------
# Truck 2D Visualization Function
# -----------------------------
def generate_truck_2d_visualization(truck_rec, box_details):
    """Generate 2D visualization showing boxes arranged in the optimal truck"""
    if not truck_rec or not box_details:
        return ""
    
    truck_dims = truck_rec['truck_dims']
    box_dims = box_details['Box Dimensions']
    boxes_per_truck = truck_rec['boxes_per_truck']
    
    # Calculate box arrangement
    boxes_length = truck_dims[0] // box_dims[0]
    boxes_width = truck_dims[1] // box_dims[1]
    boxes_height = truck_dims[2] // box_dims[2]
    
    # Display settings for visualization
    max_display_length = min(boxes_length, 12)
    max_display_width = min(boxes_width, 8)
    
    # Box styling
    box_style = (
        "display:inline-block;"
        "border:2px solid #2196F3;"
        "border-radius:3px;"
        "width:35px;"
        "height:35px;"
        "margin:2px;"
        "background-color:#E3F2FD;"
        "position:relative;"
        "font-size:8px;"
        "text-align:center;"
        "line-height:31px;"
        "color:#1976D2;"
        "font-weight:bold;"
    )
    
    # Generate visualization HTML
    viz_html = f"""
    <div style="border:2px solid #2196F3; border-radius:8px; padding:15px; background-color:#F8FFFE; margin:10px 0;">
        <h4 style="margin:0 0 10px 0; color:#1976D2;">📦 Truck Loading Pattern - {truck_rec['truck_name']}</h4>
        <div style="font-size:12px; margin-bottom:10px; color:#555;">
            <strong>Truck Dimensions:</strong> {truck_dims[0]} × {truck_dims[1]} × {truck_dims[2]} mm<br>
            <strong>Box Dimensions:</strong> {box_dims[0]} × {box_dims[1]} × {box_dims[2]} mm<br>
            <strong>Total Boxes:</strong> {boxes_per_truck} ({boxes_length}L × {boxes_width}W × {boxes_height}H)
        </div>
        <div style="background-color:white; padding:10px; border-radius:5px; border:1px solid #ddd;">
            <div style="font-size:11px; margin-bottom:8px; color:#666;"><strong>Top View (Length × Width):</strong></div>
    """
    
    # Generate grid
    for w in range(max_display_width):
        viz_html += "<div style='display:flex; justify-content:flex-start;'>"
        for l in range(max_display_length):
            box_number = (w * boxes_length) + l + 1
            if box_number <= boxes_per_truck:
                viz_html += f"<div style='{box_style}' title='Box {box_number}'>{box_number}</div>"
            else:
                # Empty space
                empty_style = box_style.replace("background-color:#E3F2FD;", "background-color:#f5f5f5;").replace("border:2px solid #2196F3;", "border:1px dashed #ccc;").replace("color:#1976D2;", "color:#999;")
                viz_html += f"<div style='{empty_style}'>-</div>"
        viz_html += "</div>"
    
    # Add truncation notice if needed
    if boxes_length > max_display_length or boxes_width > max_display_width:
        viz_html += f"<div style='margin-top:8px; font-size:10px; color:#666; font-style:italic;'>Displaying {max_display_width}×{max_display_length} of {boxes_width}×{boxes_length} total arrangement</div>"
    
    if boxes_height > 1:
        viz_html += f"<div style='margin-top:8px; font-size:11px; color:#1976D2;'><strong>Stacking:</strong> {boxes_height} layers high</div>"
    
    # Add utilization info
    viz_html += f"""
        </div>
        <div style="margin-top:10px; font-size:11px; color:#555;">
            <strong>Utilization:</strong> Volume {truck_rec['volume_utilization']:.1f}% | Weight {truck_rec['weight_utilization']:.1f}%
        </div>
    </div>
    """
    
    return viz_html

# -----------------------------
# Login Page (unchanged)
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Customer Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

# -----------------------------
# Main App (Updated with PRD requirements and truck visualization)
# -----------------------------
def packaging_app():
    st.title("🚚 Auto Parts Packaging Optimization")
    st.caption("🎯 PRD-Compliant: Optimized for minimum cost per part with dual recommendations and truck visualization")

    part_length = st.number_input("Part Length (mm)", min_value=1, value=350, key="part_length")
    part_width = st.number_input("Part Width (mm)", min_value=1, value=250, key="part_width")
    part_height = st.number_input("Part Height (mm)", min_value=1, value=150, key="part_height")
    part_weight = st.number_input("Part Weight (kg)", min_value=0.1, step=0.1, value=2.5, key="part_weight")

    fragility_level = st.selectbox("Fragility Level", ["Low", "Medium", "High"], key="fragility_level")
    stacking_allowed = st.toggle("Stacking Allowed", value=True, key="stacking_allowed")

    forklift_available = st.checkbox("Is forklift available?", key="forklift_available")
    forklift_capacity, forklift_dim = None, None
    if forklift_available:
        forklift_capacity = st.number_input("Forklift Capacity (kg)", min_value=1, value=1000, key="forklift_capacity")
        fl_l = st.number_input("Forklift Max Length (mm)", min_value=1, value=1600, key="forklift_l")
        fl_w = st.number_input("Forklift Max Width (mm)", min_value=1, value=1300, key="forklift_w")
        fl_h = st.number_input("Forklift Max Height (mm)", min_value=1, value=2000, key="forklift_h")
        forklift_dim = (fl_l, fl_w, fl_h)

    orientation_restriction = st.selectbox(
        "Orientation Restriction (PRD: 3 Standing Categories)",
        ["None", "Length Standing", "Width Standing", "Height Standing"],
        key="orientation_restriction"
    )

    annual_parts = st.number_input("Annual Auto Parts Quantity", min_value=1, step=1000, value=50000, key="annual_qty")

    st.subheader("Route Information")
    source = st.selectbox("Route Source", LOCATIONS, key="route_source")
    destination = st.selectbox("Route Destination", LOCATIONS, key="route_destination")

    selected_routes = []
    highway = st.checkbox("Highway", key="route_highway")
    if highway: selected_routes.append("Highway")
    semiurban = st.checkbox("Semi-Urban", key="route_semiurban")
    if semiurban: selected_routes.append("Semi-Urban")
    village = st.checkbox("Village", key="route_village")
    if village: selected_routes.append("Village")

    route_pct = {}
    if selected_routes:
        pct = 100 / len(selected_routes)
        for r in selected_routes:
            route_pct[r] = pct

    if highway:
        st.write(f"➡️ Highway Share: {route_pct.get('Highway', 0):.1f}%")
    if semiurban:
        st.write(f"➡️ Semi-Urban Share: {route_pct.get('Semi-Urban', 0):.1f}%")
    if village:
        st.write(f"➡️ Village Share: {route_pct.get('Village', 0):.1f}%")

    if st.button("Get PRD-Optimized Packaging", key="optimize_button"):
        part_dim = (part_length, part_width, part_height)
        result = recommend_boxes(
            part_dim, part_weight, stacking_allowed, fragility_level,
            forklift_available, forklift_capacity, forklift_dim, annual_parts, orientation_restriction
        )
        
        if "system_recommendation" in result:
            system_rec = result["system_recommendation"]
            alternatives = result.get("alternative_options", [])
            
            # Display system recommendation
            st.divider()
            st.subheader("🥇 System Recommendation (PRD: Cheapest Cost per Part)")
            
            insert = system_rec["insert_details"]
            separator = system_rec["separator_details"]
            best_box = system_rec["box_details"]
            cost_info = system_rec["cost_breakdown"]

            col1, col2 = st.columns([1, 1.3])
            with col1:
                st.markdown("### 🧩 Insert & Separator Design")
                st.markdown(f"""
                    <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                        <b>Insert Details (PRD-Compliant)</b>
                        <ul>
                            <li>Type: {insert['type']}</li>
                            <li>Matrix Pattern: {insert['matrix'][0]} × {insert['matrix'][1]} (cols × rows)</li>
                            <li>Outer Dimensions: {insert['outer_dims'][0]} × {insert['outer_dims'][1]} × {insert['outer_dims'][2]} mm</li>
                            <li>Cell (Part Orientation): {insert['cell_dims'][0]} × {insert['cell_dims'][1]} × {insert['cell_dims'][2]} mm</li>
                            <li>Standing Category: {insert.get('standing_category', 'None')}</li>
                            <li>Orientation Restriction: {best_box['Orientation Restriction']}</li>
                            <li>Auto-parts per insert: {insert['units_per_insert']}</li>
                            <li>Weight per Layer: {insert['weight_kg']} kg</li>
                            <li>Material Specification: {insert.get('gsm_or_thickness','N/A')}</li>
                            <li>Thickness: {insert.get('thickness_mm', 5):.0f}mm (PRD Compliant)</li>
                        </ul>
                        <p style="font-size:12px; color:gray;"><i>{insert.get('note', '')}</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                    <b>Separator Details</b><br>
                    Type: {separator['type'] if separator['needed'] else 'Not Required'}<br>
                    Note: {separator.get('note', 'N/A')}<br>
                    Weight per Unit: {separator.get('weight_kg', 'N/A')} kg
                </div>
                """, unsafe_allow_html=True)

                # PRD Spacing compliance info
                st.markdown(f"""
                <div style="border:2px solid #4CAF50; border-radius:10px; padding:12px; margin-bottom:10px; background-color:#f8fff9;">
                    <b>🎯 PRD Spacing Compliance</b><br>
                    ✅ Side Clearance: {SIDE_CLEARANCE}mm (all 4 sides)<br>
                    ✅ Top Clearance: {TOP_CLEARANCE}mm<br>
                    ✅ Bottom Clearance: {BOTTOM_CLEARANCE}mm<br>
                    ✅ Insert Thickness: {PARTITION_THICKNESS}mm
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### Matrix Pattern Visualization")
                cell_style = "display:inline-block;border:2px solid #b7e4c7;border-radius:4px;width:44px;height:44px;margin:3px;background-color:#f8fff9;"
                rows, cols = insert["matrix"][1], insert["matrix"][0]
                display_rows, display_cols = min(rows, 8), min(cols, 8)
                row_html = "".join([
                    "<div style='display:flex;flex-direction:row;'>" +
                    "".join([f"<div style='{cell_style}'></div>" for _ in range(display_cols)]) +
                    "</div>" for _ in range(display_rows)
                ])
                if best_box["Layers"] > 1:
                    row_html += f"<div style='margin-top:8px;font-size:13px;color:#555;'>Layers stacked: {best_box['Layers']}</div>"

                if rows > display_rows or cols > display_cols:
                    row_html += f"<small><i>Displaying {display_rows}x{display_cols} of {rows}x{cols} total.</i></small>"

                st.markdown(row_html, unsafe_allow_html=True)

            st.divider()
            st.subheader("🏆 Outer Box Recommendation")
            box_dims = best_box["Box Dimensions"]
            internal_dims = best_box["Internal Dimensions"]
            weight_breakdown = best_box["Weight Breakdown"]

            st.markdown(f"""
<div style="border:2px solid #2a9d8f; border-radius:10px; padding:15px;">
    <b>Recommended Type</b>: {best_box['Box Type']} ({box_dims[0]}×{box_dims[1]}×{box_dims[2]} mm)<br><br>
    <b>🎯 PRD Key Metric - Cost per Part:</b> 
    <span style="color:#E91E63; font-weight:bold; font-size:1.2em;">₹{cost_info['cost_per_part']:.2f}</span><br><br>
    <b>Configuration:</b> {best_box['Layers']} layer(s) of {insert['units_per_insert']} parts each.<br>
    <b>Max Parts per Box:</b> <b>{best_box['Max Parts']}</b><br>
    <hr style="border-top: 1px solid #ddd;">
    <b>Cost Breakdown (PRD Compliant):</b><br>
    ➤ Box Handling Cost: ₹{cost_info['handling_cost_per_box']:.2f} per box<br>
    ➤ Packaging Asset Cost: ₹{cost_info['asset_cost_per_box']:.2f} per box<br>
    ➤ Total Cost per Box: ₹{cost_info['total_cost_per_box']:.2f}<br><br>
    <b>Box Capacity:</b> {best_box['Capacity']:.1f} kg<br>
    <b>Total Weight</b> inside the box: {best_box['Total Weight']:.1f} kg<br>
    <small>
    ➤ Parts: {best_box['Max Parts']} × {part_weight:.2f} kg = {weight_breakdown['Parts']:.1f} kg<br>
    ➤ Inserts: {insert['weight_kg']} kg × {best_box['Layers']} = {weight_breakdown['Inserts']:.1f} kg<br>
    ➤ Separators: {separator.get('weight_kg',0)} kg × {max(0,best_box['Layers']-1)} = {weight_breakdown['Separators']:.1f} kg<br>
    {f"➤ FLC Lid: {weight_breakdown['FLC Lid']:.2f} kg<br>" if best_box['Box Type'].lower().startswith("flc") else ""}<br>
    </small><br>
    <b>Boxes Required per Year:</b> {best_box['Boxes/Year']}<br>
</div>
""", unsafe_allow_html=True)

            # PRD Point 3: Display alternative options
            if alternatives:
                st.divider()
                st.subheader("🎛️ Alternative Options (PRD: User Custom Choice)")
                
                alternative_data = []
                for i, alt in enumerate(alternatives[:4], 1):  # Show top 4 alternatives
                    alt_box = alt["box_details"]
                    alt_cost = alt["cost_breakdown"]
                    alternative_data.append({
                        "Option": f"Alt {i}",
                        "Box Type": f"{alt_box['Box Type']} ({alt_box['Box Dimensions'][0]}×{alt_box['Box Dimensions'][1]}×{alt_box['Box Dimensions'][2]})",
                        "Parts/Box": alt_box['Max Parts'],
                        "Cost/Part (₹)": f"{alt_cost['cost_per_part']:.2f}",
                        "Volume Eff %": f"{alt_box['Volume Efficiency %']:.1f}%",
                        "Standing Category": alt["insert_details"].get("standing_category", "None")
                    })
                
                st.table(alternative_data)

            # Truck Optimization Section with 2D Visualization
            if source != "Select" and destination != "Select" and route_pct:
                st.divider()
                st.subheader("🚛 Truck Load Optimization with 2D Visualization")
                
                truck_recommendations = optimize_truck_loading(system_rec, source, destination, route_pct)
                
                if truck_recommendations:
                    # Display route information
                    distance = get_route_distance(source, destination)
                    st.markdown(f"""
                    <div style="border:1px solid #ffc107; border-radius:10px; padding:12px; margin-bottom:15px;">
                        <b>Route Distribution:</b> {', '.join([f'{k}: {v:.0f}%' for k, v in route_pct.items()])}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display best truck with 2D visualization
                    best_truck = truck_recommendations[0]
                    st.markdown("### 🥇 Optimal Truck Configuration")
                    
                    # Display truck info
                    st.markdown(f"""
                        <div style="border:3px solid #28a745; border-radius:10px; padding:15px; margin-bottom:15px; background-color:#f8fff9;">
                            <h4 style="margin:0; color:#28a745;">🏆 Best Choice: {best_truck['truck_name']}</h4>
                            <div style="display:flex; flex-wrap:wrap; gap:20px; margin-top:15px;">
                                <div style="flex:1; min-width:300px;">
                                    <b style="color:#28a745;">📐 Truck Specifications:</b><br>
                                    <span style="color:#222;">• Dimensions: {best_truck['truck_dims'][0]} × {best_truck['truck_dims'][1]} × {best_truck['truck_dims'][2]} mm</span><br>
                                    <span style="color:#222;">• Payload Capacity: {best_truck['payload_capacity']:,} kg</span><br>
                                    <b style="color:#28a745;">📦 Loading Configuration:</b><br>
                                    <span style="color:#222;">• Boxes per Truck: <b>{best_truck['boxes_per_truck']}</b></span><br>
                                    <span style="color:#222;">• Total Parts per Truck: <b>{best_truck['total_parts_per_truck']:,}</b></span><br>
                                    <span style="color:#222;">• Volume Utilization: {best_truck['volume_utilization']:.1f}%</span><br>
                                    <span style="color:#222;">• Weight Utilization: {best_truck['weight_utilization']:.1f}%</span><br>
                                </div>
                                <div style="flex:1; min-width:300px;">
                                    <b style="color:#28a745;">💰 Cost Analysis (PRD Key Metrics):</b><br>
                                    <span style="color:#E91E63; font-weight:bold; font-size:1.1em;">• Cost per Part: ₹{best_truck['cost_per_part']:.2f}</span><br>
                                    <span style="color:#222;">• Total Trip Cost: ₹{best_truck['total_trip_cost']:,.0f}</span><br>
                                    <span style="color:#222;">• Fuel Cost: ₹{best_truck['fuel_cost']:,.0f}</span><br>
                                    <span style="color:#222;">• Handling Cost: ₹{best_truck['handling_cost']:,.0f}</span><br>
                                    <span style="color:#222;">• Asset Cost: ₹{best_truck['asset_cost']:,.0f}</span><br><br>
                                    <b style="color:#28a745;">🌱 Environmental Impact:</b><br>
                                    <span style="color:#4CAF50; font-weight:bold;">• CO₂ per Part: {best_truck['co2_per_part']:.3f} kg</span><br>
                                    <span style="color:#222;">• Total CO₂ per Trip: {best_truck['co2_emission']:.2f} kg</span><br>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 2D Truck Loading Visualization
                    st.markdown("### 📐 2D Truck Loading Visualization")
                    truck_viz_html = generate_truck_2d_visualization(best_truck, best_box)
                    st.markdown(truck_viz_html, unsafe_allow_html=True)
                    
                    # Show comparison with other trucks if available
                    if len(truck_recommendations) > 1:
                        st.markdown("### 📊 Alternative Trucks Comparison")
                        comparison_data = []
                        for truck_rec in truck_recommendations[:3]:
                            comparison_data.append({
                                "Truck": truck_rec['truck_name'],
                                "Parts/Truck": f"{truck_rec['total_parts_per_truck']:,}",
                                "Cost/Part (₹)": f"{truck_rec['cost_per_part']:.2f}",
                                "CO₂/Part (kg)": f"{truck_rec['co2_per_part']:.3f}",
                                "Volume Util (%)": f"{truck_rec['volume_utilization']:.1f}%",
                                "Weight Util (%)": f"{truck_rec['weight_utilization']:.1f}%"
                            })
                        
                        st.table(comparison_data)
                
                else:
                    st.warning("⚠️ No suitable truck configurations found for the selected box and route.")
            else:
                st.info("ℹ️ Select source, destination, and route types to see truck optimization recommendations.")

        else:
            st.error("❌ No suitable box and insert combination found.", icon="🚨")
            st.warning("Here is a diagnostics report showing why each standard box was rejected:", icon="🔬")
            log = result.get("rejection_log", {})
            if not log:
                st.info("No boxes were even attempted. This may indicate a problem with the initial inputs.")
            else:
                for box_name, reason in log.items():
                    st.markdown(f"- **{box_name}**: {reason}")

# -----------------------------
# Controller (unchanged)
# -----------------------------
if not st.session_state.logged_in:
    login()
else:
    packaging_app()