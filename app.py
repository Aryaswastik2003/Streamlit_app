import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# -----------------------------
# Data Models & Constants (updated with truck optimization)
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

# --- BOX_DATABASE updated with PRD-informed approximate empty weights ---
BOX_DATABASE: List[Box] = [
    # Foldable Crates / PP Box (PRD: 6 standard sizes, capacity 15-16kg)
    # Using light-weight PP estimates (examples: 0.6 - 2.2 kg depending on size/thickness)
    Box("PP Box", (400, 300, 235), 16, 0.8),
    Box("PP Box", (600, 400, 348), 20, 1.8),

    # Foldable Crate (explicit type in DB) - matches PRD small crate sizes
    Box("Foldable Crate", (600, 400, 348), 15, 1.9),

    # FLCs (PRD: includes lid weight ~5.13 kg; typical total empty mass often in 4-9 kg range)
    # These values are PRD-informed conservative estimates (replace with measured PRD value if you have it)
    Box("FLC", (1200, 1000, 595), 700, 6.8),    # standard FLC (includes lid mass)
    Box("FLC", (1200, 1000, 1200), 1000, 10.5), # taller FLC variant

    # PLS (pallet + lid sleeve) — heavier structure
    Box("PLS", (1500, 1200, 1000), 600, 28.0),
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
# Insert Material Selection Constants (unchanged)
# -----------------------------
CUSHION_FACTORS = {
    "PP Partition Grid": 0.90,
    "Honeycomb Layer Pad": 0.85,
    "Thermo-vac PP Tray": 0.80,
    "Woven PP Pouch": 0.75
}
PP_DENSITY_G_CM3 = 0.9
PP_DENSITY_G_MM3 = 0.0009

# -----------------------------
# Helpers (unchanged)
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
    insert_L, insert_W = part_dim[0], part_dim[1]
    
    if fragility == "High":
        if insert_area_m2 >= 0.02 or size_factor >= 0.6:
            base_thickness = 1.5
            thickness_mm = base_thickness + (size_factor * 1.0) + (load_factor * 0.5)
            thickness_mm = min(thickness_mm, 3.0)
            thickness_mm = round(thickness_mm, 1)
            volume_mm3 = insert_area_m2 * 1e6 * thickness_mm
            weight_kg = (volume_mm3 * PP_DENSITY_G_MM3) / 1000.0
            return {
                "type": "Thermo-vac PP Tray",
                "gsm_or_thickness": f"{thickness_mm}mm PP sheet",
                "weight_kg": round(weight_kg, 2),
                "note": f"Form-fit tray for fragile parts (size factor: {size_factor:.1f})"
            }
        else:
            base_gsm = 250
            gsm = base_gsm + (size_factor * 50) + (load_factor * 50)
            gsm = min(gsm, 350)
            gsm = round(gsm)
            weight_kg = insert_area_m2 * (gsm / 1000.0)
            return {
                "type": "Woven PP Pouch",
                "gsm_or_thickness": f"{gsm} GSM woven fabric",
                "weight_kg": round(weight_kg, 2),
                "note": f"Soft pouch for small/fragile parts (size factor: {size_factor:.1f})"
            }
    else:
        base_gsm = 650
        if fragility == "Medium":
            gsm_range = [800, 1200]
        else:
            gsm_range = [650, 1000]
        gsm_increment = (gsm_range[1] - gsm_range[0]) * max(size_factor, load_factor)
        gsm = gsm_range[0] + gsm_increment
        gsm = min(gsm, 1600)
        gsm = round(gsm / 50) * 50
        weight_kg = insert_area_m2 * (gsm / 1000.0)
        return {
            "type": "PP Partition Grid",
            "gsm_or_thickness": f"{gsm} GSM corrugated PP",
            "weight_kg": round(weight_kg, 2),
            "note": f"Grid partition (size: {size_factor:.1f}, load: {load_factor:.1f})"
        }

# -----------------------------
# NEW: Truck Optimization Functions (Based on PRD requirements)
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

def calculate_trip_cost(truck, distance, route_percentages, boxes_per_truck, box_type):
    """Calculate total cost per trip based on PRD requirements"""
    if distance <= 0:
        return 0, 0, 0, 0
    
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
    
    # Total trip cost (PRD: Trip cost + box handling cost + packaging asset one time cost)
    total_trip_cost = truck.trip_cost + fuel_cost + total_handling_cost
    
    return total_trip_cost, fuel_cost, total_handling_cost, co2_emission

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
        
        # Calculate costs and CO2
        trip_cost, fuel_cost, handling_cost, co2_emission = calculate_trip_cost(
            truck, distance, route_percentages, boxes_per_truck, box_type
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
            "cost_per_part": cost_per_part,
            "co2_emission": co2_emission,
            "co2_per_part": co2_per_part,
            "efficiency_score": (volume_util + weight_util) / 2,  # Combined efficiency
        })
    
    # Sort by cost per part (ascending) - best recommendation first
    truck_recommendations.sort(key=lambda x: x["cost_per_part"])
    
    return truck_recommendations

# -----------------------------
# Recommendation Functions (Updated Material Logic) - UNCHANGED
# -----------------------------
def design_insert_for_box(part_dim, box_internal_dim, fragility, part_weight=1.0, orientation_restriction="None"):
    best_fit = {
        "units_per_insert": 0,
        "matrix": (0, 0),
        "cell_dims": (0, 0, 0),
        "outer_dims": (0, 0, 0),
        "part_orientation": part_dim,
        "volume_efficiency": 0,
    }

    PARTITION_THICKNESS = 5
    WALL_CLEARANCE = 5
    TOP_CLEARANCE = 5

    L, W, H = part_dim

    # build orientations and filter by orientation_restriction if requested
    all_orients = [
        (L, W, H), (L, H, W), (W, L, H),
        (W, H, L), (H, L, W), (H, W, L)
    ]
    if orientation_restriction in (None, "None"):
        orientations = set(all_orients)
    else:
        if orientation_restriction == "Length Standing":
            orientations = {o for o in all_orients if o[2] == L}
        elif orientation_restriction == "Width Standing":
            orientations = {o for o in all_orients if o[2] == W}
        elif orientation_restriction == "Height Standing":
            orientations = {o for o in all_orients if o[2] == H}
        else:
            orientations = set(all_orients)

    box_L, box_W, box_H = box_internal_dim

    # Guard against invalid box dimensions
    if box_L <= 0 or box_W <= 0 or box_H <= 0:
        return None

    box_volume = box_L * box_W * box_H
    if box_volume <= 0:
        return None

    for pL, pW, pH in orientations:
        if pH > (box_H - TOP_CLEARANCE):
            continue
        if (pL + PARTITION_THICKNESS) <= 0 or (pW + PARTITION_THICKNESS) <= 0:
            continue

        cols = max(0, (box_L - WALL_CLEARANCE) // (pL + PARTITION_THICKNESS))
        rows = max(0, (box_W - WALL_CLEARANCE) // (pW + PARTITION_THICKNESS))
        units_this_orientation = cols * rows
        
        if units_this_orientation > 0:
            insert_L = (cols * pL) + ((cols + 1) * PARTITION_THICKNESS)
            insert_W = (rows * pW) + ((rows + 1) * PARTITION_THICKNESS)
            insert_H = min(box_H, pH + TOP_CLEARANCE)  # clamp to box height
            
            part_volume = pL * pW * pH
            used_volume_parts = units_this_orientation * part_volume
            volume_efficiency = (used_volume_parts / box_volume) * 100

            if (volume_efficiency > best_fit["volume_efficiency"] or 
                (volume_efficiency == best_fit["volume_efficiency"] and units_this_orientation > best_fit["units_per_insert"])):
                
                best_fit["units_per_insert"] = units_this_orientation
                best_fit["matrix"] = (cols, rows)
                best_fit["cell_dims"] = (pL, pW, pH)
                best_fit["outer_dims"] = (insert_L, insert_W, insert_H)
                best_fit["part_orientation"] = (pL, pW, pH)
                best_fit["volume_efficiency"] = volume_efficiency

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


# This is the corrected function with the bug fix
def recommend_boxes(part_dim, part_weight, stacking_allowed, fragility, forklift_available,
                    forklift_capacity, forklift_dim, annual_parts,orientation_restriction
                    ):
    best_option = None
    rejection_log = {}
    best_volume_efficiency_total = 0  # Changed variable name to avoid confusion
    
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

        # 🔧 Limit parts by box weight capacity so large boxes can still be used with fewer parts
        if part_weight <= 0:
            rejection_log[log_key] = f"Rejected: Invalid part weight ({part_weight})."
            continue

        max_parts_by_weight = int(box.capacity_kg // part_weight)
        if max_parts_by_weight < 1:
            rejection_log[log_key] = f"Rejected: Single part weight ({part_weight} kg) exceeds box capacity ({box.capacity_kg} kg)."
            continue

        # Use the smaller of geometric fit and weight-based fit
        fit_count = min(fit_count, max_parts_by_weight)

        # Weight breakdown (recomputed after limiting fit_count)
        # Weight breakdown (recomputed after limiting fit_count)
        part_total_weight = fit_count * part_weight
        insert_weight_total = insert["weight_kg"] * layers
        separator_weight_total = separator["weight_kg"] * max(0, layers - 1)

        # use the box DB empty mass (includes lid if present); default to 0.0 if missing
        box_empty_mass = getattr(box, "empty_weight_kg", 0.0)

        total_weight = part_total_weight + insert_weight_total + separator_weight_total + box_empty_mass

        # if overweight, reduce fit_count until it fits the box capacity
        while fit_count > 0 and total_weight > box.capacity_kg:
            fit_count -= 1
            part_total_weight = fit_count * part_weight
            total_weight = part_total_weight + insert_weight_total + separator_weight_total + box_empty_mass


        # If nothing fits even after reduction → reject
        if fit_count <= 0:
            rejection_log[log_key] = f"Rejected: Even 1 part + packaging exceeds box capacity ({box.capacity_kg} kg)."
            continue
              # skip this box option, exceeds capacity



        # Volume/waste metrics
        part_volume = part_dim[0] * part_dim[1] * part_dim[2]
        box_volume = internal_dims[0] * internal_dims[1] * internal_dims[2]
        used_volume_parts = fit_count * part_volume
        insert_outer_vol = insert["outer_dims"][0] * insert["outer_dims"][1] * insert["outer_dims"][2]
        used_volume_insert = insert_outer_vol * layers
        partition_volume_est = max(insert_outer_vol - (insert["units_per_insert"] * part_volume), 0)

        # Updated: Wasted % is based on box volume, not insert volume
        wasted_pct_parts = 100 * (1 - (used_volume_parts / box_volume)) if box_volume > 0 else 100
        wasted_pct_insert = 100 * ((used_volume_insert - used_volume_parts) / box_volume) if box_volume > 0 and used_volume_insert > used_volume_parts else 0
        
        # New calculation for total volume efficiency
        volume_efficiency_total = 100 - wasted_pct_parts - wasted_pct_insert

        # Calculate the variables needed for the display section
        insert_material_pct = 100 * ((used_volume_insert - used_volume_parts) / used_volume_insert) if used_volume_insert > 0 else 0
        combined_efficiency = volume_efficiency_total + insert_material_pct

        # Reject by capacity / forklift limits
# Reject by capacity / forklift limits (allow partial fills for larger boxes)
        if total_weight > box.capacity_kg:
            # Instead of rejecting outright, calculate how many parts actually fit safely
            max_safe_parts = int(box.capacity_kg // part_weight)
            if max_safe_parts <= 0:
                rejection_log[log_key] = f"Rejected: Box capacity ({box.capacity_kg} kg) is too low for even 1 part."
                continue
            # Adjust fit_count down to safe capacity
            fit_count = min(fit_count, max_safe_parts)
            part_total_weight = fit_count * part_weight
            total_weight = part_total_weight + insert_weight_total + separator_weight_total + box_empty_mass

        if forklift_available and forklift_capacity and total_weight > forklift_capacity:
            rejection_log[log_key] = f"Rejected: Total weight ({total_weight:.1f} kg) exceeds forklift capacity ({forklift_capacity} kg)."
            continue

        # Choose box with HIGHEST total volume efficiency
        if (best_option is None or 
            volume_efficiency_total > best_volume_efficiency_total or
            (volume_efficiency_total == best_volume_efficiency_total and fit_count > best_option["box_details"]["Max Parts"])):
            
            boxes_per_year = -(-annual_parts // fit_count) if fit_count > 0 else 0
            best_volume_efficiency_total = volume_efficiency_total
            
            best_option = {
                "insert_details": insert,
                "separator_details": separator,
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
                    "Capacity": box.capacity_kg,   # 👈 added here

                    "Wasted Volume % (parts)": wasted_pct_parts,
                    "Wasted Volume % (insert)": wasted_pct_insert,
                    "Volume Efficiency %": volume_efficiency_total,
                    "Parts Efficiency %": combined_efficiency, # Renamed metric
                    "Insert Material Value %": insert_material_pct, # Added new metric
                    "Insert Outer Volume (mm^3)": insert_outer_vol,
                    "Partition Volume Estimate (mm^3)": partition_volume_est,
                    "Boxes/Year": boxes_per_year,
                    "Layers": layers,
                    "Orientation Restriction": orientation_restriction
                    },
                "rejection_log": rejection_log
            }

    if best_option: 
        return best_option
    else: 
        return {"rejection_log": rejection_log}
        
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
# Main App (Updated with truck optimization display)
# -----------------------------
def packaging_app():
    st.title("🚚 Auto Parts Packaging Optimization")
    st.caption("🎯 Now optimized for minimum volume wastage with smart material selection and truck optimization")

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

    # 👇 Add this dropdown here
    orientation_restriction = st.selectbox(
        "Orientation Restriction (if any)",
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

    if st.button("Get Optimized Packaging", key="optimize_button"):
        part_dim = (part_length, part_width, part_height)
        result = recommend_boxes(
            part_dim, part_weight, stacking_allowed, fragility_level,
            forklift_available, forklift_capacity, forklift_dim, annual_parts,orientation_restriction
        )
        if "box_details" in result:
            insert = result["insert_details"]
            separator = result["separator_details"]
            best_box = result["box_details"]

            st.divider()
            col1, col2 = st.columns([1, 1.3])
            with col1:
                st.markdown("### 🧩 Insert & Separator Design")
                st.markdown(f"""
                    <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                        <b>Insert Details</b>
                        <ul>
                            <li>Type: {insert['type']}</li>
                            <li>Matrix Pattern: {insert['matrix'][0]} × {insert['matrix'][1]} (cols × rows)</li>
                            <li>Outer Dimensions: {insert['outer_dims'][0]} × {insert['outer_dims'][1]} × {insert['outer_dims'][2]} mm</li>
                            <li>Cell (Part Orientation): {insert['cell_dims'][0]} × {insert['cell_dims'][1]} × {insert['cell_dims'][2]} mm</li>
                            <li>Orientation Restriction: {best_box['Orientation Restriction']}</li>
                            <li>Auto-parts per insert: {insert['units_per_insert']}</li>
                            <li>Weight per Layer: {insert['weight_kg']} kg</li>
                            <li>Material Specification: {insert.get('gsm_or_thickness','N/A')}</li>
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

                insert_outer_vol = insert['outer_dims'][0] * insert['outer_dims'][1] * insert['outer_dims'][2]
                part_vol = part_dim[0] * part_dim[1] * part_dim[2]
                cells = insert['units_per_insert']
                used_by_parts_in_insert = cells * part_vol
                partition_vol = max(insert_outer_vol - used_by_parts_in_insert, 0)
                st.markdown("---")
                st.markdown("**Insert utilization estimates**")
                st.write(f"- Insert outer volume (mm³): {insert_outer_vol:,}")
                st.write(f"- Sum of part volumes inside insert (mm³): {used_by_parts_in_insert:,}")
                st.write(f"- Estimated partition/void volume (mm³): {partition_vol:,}")
                insert_waste_pct = 100 * (1 - (used_by_parts_in_insert / insert_outer_vol)) if insert_outer_vol > 0 else 0
                st.write(f"- Wasted / partition % inside insert: {insert_waste_pct:.1f}%")

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
            volume_efficiency_total = best_box.get("Volume Efficiency %", 0)
            wasted_pct_insert = best_box['Wasted Volume % (insert)']
            
            # Retrieve the newly calculated variables
            parts_efficiency = best_box['Parts Efficiency %']
            insert_material_pct = best_box['Insert Material Value %']
            
            st.markdown(f"""
<div style="border:2px solid #2a9d8f; border-radius:10px; padding:15px;">
    <b>Recommended Type</b>: {best_box['Box Type']} ({box_dims[0]}×{box_dims[1]}×{box_dims[2]} mm)<br><br>
    <b>🎯 Part Efficiency:</b> 
    <span style="color:#00b894; font-weight:bold; font-size:1.1em;">{parts_efficiency:.2f}%</span><br>
    <b>Configuration:</b> {best_box['Layers']} layer(s) of {insert['units_per_insert']} parts each.<br>
    <b>Max Parts per Box:</b> <b>{best_box['Max Parts']}</b><br>
    <hr style="border-top: 1px solid #ddd;">
    <b>Box Capacity:</b> {best_box['Capacity']:.1f} kg<br>
    <b>Total Weight</b> inside the box: {best_box['Total Weight']:.1f} kg<br>
    <small>
        ➤ Parts: {best_box['Max Parts']} × {part_weight:.2f} kg = {weight_breakdown['Parts']:.1f} kg<br>
        ➤ Inserts: {insert['weight_kg']} kg × {best_box['Layers']} = {weight_breakdown['Inserts']:.1f} kg<br>
        ➤ Separators: {separator.get('weight_kg',0)} kg × {max(0,best_box['Layers']-1)} = {weight_breakdown['Separators']:.1f} kg<br>
        ➤ FLC Lid: {weight_breakdown['FLC Lid']:.2f} kg<br>
    </small><br>
    <b>Boxes Required per Year:</b> {best_box['Boxes/Year']}<br>
    <hr style="border-top: 1px solid #ddd;">
    <small><b>Internal Dims:</b> {internal_dims[0]} × {internal_dims[1]} × {internal_dims[2]} mm</small>
</div>
""", unsafe_allow_html=True)

            # NEW: Truck Optimization Section (PRD Requirement)
            if source != "Select" and destination != "Select" and route_pct:
                st.divider()
                st.subheader("🚛 Truck Load Optimization")
                
                truck_recommendations = optimize_truck_loading(result, source, destination, route_pct)
                
                if truck_recommendations:
                    # Display route information
                    distance = get_route_distance(source, destination)
                    st.markdown(f"""
                    <div style="border:1px solid #ffc107; border-radius:10px; padding:12px; margin-bottom:15px; background-color:#1;">
                        <!-- <b>🗺️ Route Details:</b> {source} → {destination} ({distance} km)<br> -->
                        <b>Route Distribution:</b> {', '.join([f'{k}: {v:.0f}%' for k, v in route_pct.items()])}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display top 3 truck recommendations
                    st.markdown("### 🥇 Top Truck Recommendations (Based on Cost per Part)")
                    
                    for i, truck_rec in enumerate(truck_recommendations[:3], 1):
                        # Color coding for ranking
                        if i == 1:
                            border_color = "#28a745"  # Green for best
                            bg_color = "#f8fff9"
                            rank_emoji = "🥇"
                        elif i == 2:
                            border_color = "#ffc107"  # Yellow for second
                            bg_color = "#fffbf0"
                            rank_emoji = "🥈"
                        else:
                            border_color = "#fd7e14"  # Orange for third
                            bg_color = "#fff8f0"
                            rank_emoji = "🥉"
                        
                        st.markdown(f"""
                            <div style="border:3px solid {border_color}; border-radius:10px; padding:15px; margin-bottom:15px; background-color:{bg_color};">
                                <h4 style="margin:0; color:{border_color};">{rank_emoji} Rank {i}: {truck_rec['truck_name']}</h4>
                                <div style="display:flex; flex-wrap:wrap; gap:20px; margin-top:15px;">
                                    <div style="flex:1; min-width:300px;">
                                        <b style="color:{border_color};">📐 Truck Specifications:</b><br>
                                        <span style="color:#222;">• Dimensions: {truck_rec['truck_dims'][0]} × {truck_rec['truck_dims'][1]} × {truck_rec['truck_dims'][2]} mm</span><br>
                                        <span style="color:#222;">• Payload Capacity: {truck_rec['payload_capacity']:,} kg</span><br><br>
                                        
                                        <b style="color:{border_color};">📦 Loading Configuration:</b><br>
                                        <span style="color:#222;">• Boxes per Truck: <b>{truck_rec['boxes_per_truck']}</b></span><br>
                                        <span style="color:#222;">• Total Parts per Truck: <b>{truck_rec['total_parts_per_truck']:,}</b></span><br>
                                        <span style="color:#222;">• Volume Utilization: {truck_rec['volume_utilization']:.1f}%</span><br>
                                        <span style="color:#222;">• Weight Utilization: {truck_rec['weight_utilization']:.1f}%</span><br>
                                    </div>
                                    <div style="flex:1; min-width:300px;">
                                        <b style="color:{border_color};">💰 Cost Analysis (PRD Key Metrics):</b><br>
                                        <span style="color:#E91E63; font-weight:bold; font-size:1.1em;">• Cost per Part: ₹{truck_rec['cost_per_part']:.2f}</span><br>
                                        <span style="color:#222;">• Total Trip Cost: ₹{truck_rec['total_trip_cost']:,.0f}</span><br>
                                        <span style="color:#222;">• Fuel Cost: ₹{truck_rec['fuel_cost']:,.0f}</span><br>
                                        <span style="color:#222;">• Handling Cost: ₹{truck_rec['handling_cost']:,.0f}</span><br><br>

                                        <b style="color:{border_color};">🌱 Environmental Impact:</b><br>
                                        <span style="color:#4CAF50; font-weight:bold;">• CO₂ per Part: {truck_rec['co2_per_part']:.3f} kg</span><br>
                                        <span style="color:#222;">• Total CO₂ per Trip: {truck_rec['co2_emission']:.2f} kg</span><br>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    
                    # Summary comparison table
                    if len(truck_recommendations) > 1:
                        st.markdown("### 📊 Quick Comparison")
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
                        
                        # Create comparison table
                        st.table(comparison_data)
                        
                        # Cost savings analysis
                        best_cost = truck_recommendations[0]['cost_per_part']
                        worst_cost = truck_recommendations[-1]['cost_per_part']
                        if worst_cost > best_cost:
                            savings_pct = ((worst_cost - best_cost) / worst_cost) * 100
                            st.success(f"💡 **Insight**: Using {truck_recommendations[0]['truck_name']} saves {savings_pct:.1f}% cost per part compared to {truck_recommendations[-1]['truck_name']}")
                
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