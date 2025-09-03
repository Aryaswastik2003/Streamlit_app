import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# Data Models & Constants
# -----------------------------
@dataclass(frozen=True)
class Box:
    box_type: str
    dims: Tuple[int, int, int]  # (L, W, H) in mm
    capacity_kg: float


@dataclass(frozen=True)
class Truck:
    name: str
    dims: Tuple[int, int, int]  # (L, W, H) in mm
    payload_kg: float
    trip_cost: float  # dummy cost per trip


BOX_DATABASE: List[Box] = [
    Box("PP Box", (400, 300, 235), 16),
    Box("PP Box", (600, 400, 348), 20),
    Box("Foldable Crate", (600, 400, 348), 15),
    Box("FLC", (1200, 1000, 595), 700),
    Box("FLC", (1200, 1000, 1200), 1000),
    Box("PLS", (1500, 1200, 1000), 600),
]

TRUCKS: List[Truck] = [
    Truck("9T Truck", (5500, 2200, 2400), 9000, 15000),
    Truck("16T Truck", (7500, 2500, 2600), 16000, 20000),
    Truck("22T Truck", (9500, 2600, 2800), 22000, 28000),
]

CO2_FACTORS = {"Highway": 0.08, "Semi-Urban": 0.12, "Village": 0.15}
ERGONOMIC_LIFT_KG = 25

LOCATIONS = ["Select", "Chennai", "Bangalore", "Delhi", "Pune", "Hyderabad", "Mumbai", "Kolkata"]

# -----------------------------
# Helpers
# -----------------------------
def get_internal_dims(box: Box) -> Tuple[int, int, int]:
    """Apply PRD rules for internal dimensions per box type."""
    L, W, H = box.dims
    if box.box_type == "PP Box":
        return (L - 34, W - 34, H - 8)
    elif box.box_type == "PLS":
        return (L - 34, W - 34, H - 210)
    elif box.box_type == "FLC":
        # FLCs often have near-identical internal dims, but let's assume a small wall thickness
        return (L - 30, W - 30, H - 30)
    else:
        return (L, W, H)

# -----------------------------
# Recommendation Functions (New Dynamic Logic)
# -----------------------------
def design_insert_for_box(part_dim, box_internal_dim, fragility):
    """
    Designs the best possible insert matrix for a given part inside a specific box.
    Returns a dictionary with insert details if a fit is found, otherwise None.
    """
    best_fit = {
        "units_per_insert": 0,
        "matrix": (0, 0),
        "cell_dims": (0, 0, 0),
        "outer_dims": (0, 0, 0),
        "part_orientation": part_dim,
    }

    PARTITION_THICKNESS = 5
    WALL_CLEARANCE = 5
    TOP_CLEARANCE = 5

    L, W, H = part_dim
    orientations = set([(L, W, H), (L, H, W), (W, L, H), (W, H, L), (H, L, W), (H, W, L)])
    
    box_L, box_W, box_H = box_internal_dim

    for pL, pW, pH in orientations:
        if pH > (box_H - TOP_CLEARANCE):
            continue

        if (pL + PARTITION_THICKNESS) <= 0 or (pW + PARTITION_THICKNESS) <= 0:
            continue
            
        cols = (box_L - WALL_CLEARANCE) // (pL + PARTITION_THICKNESS)
        rows = (box_W - WALL_CLEARANCE) // (pW + PARTITION_THICKNESS)
        
        units_this_orientation = cols * rows

        if units_this_orientation > best_fit["units_per_insert"]:
            insert_L = (cols * pL) + ((cols + 1) * PARTITION_THICKNESS)
            insert_W = (rows * pW) + ((rows + 1) * PARTITION_THICKNESS)
            insert_H = pH + TOP_CLEARANCE

            best_fit["units_per_insert"] = units_this_orientation
            best_fit["matrix"] = (cols, rows)
            best_fit["cell_dims"] = (pL, pW, pH)
            best_fit["outer_dims"] = (insert_L, insert_W, insert_H)
            best_fit["part_orientation"] = (pL, pW, pH)

    if best_fit["units_per_insert"] == 0:
        return None

    if fragility == "High":
        best_fit["type"] = "Thermo-vac PP Tray"
        best_fit["note"] = "Best for fragile / Class-A parts."
        best_fit["weight_kg"] = 50.0
    elif fragility == "Medium":
        best_fit["type"] = "PP Partition Grid"
        best_fit["note"] = "General purpose for medium parts."
        best_fit["weight_kg"] = 32.78
    else:
        best_fit["type"] = "Woven PP Pouch"
        best_fit["note"] = "For scratch-sensitive or small batches."
        best_fit["weight_kg"] = 80.0
        
    return best_fit


def get_separator_details(insert, stacking_allowed):
    """Determines separator needs based on the designed insert."""
    if not stacking_allowed or not insert:
        return {"needed": False, "type": "N/A", "weight_kg": 0.0, "note": "Stacking disabled."}

    if insert["type"] in ("PP Partition Grid", "Thermo-vac PP Tray"):
        return {"needed": True, "type": "Honeycomb Layer Pad", "weight_kg": 1.49, "note": "Adds strength between stacked layers."}
    else:
        return {"needed": True, "type": "PP Sheet Separator", "weight_kg": 1.0, "note": "General separator for multiple layers."}


def recommend_boxes(part_dim, part_weight, stacking_allowed, fragility, forklift_available,
                    forklift_capacity, forklift_dim, annual_parts):
    
    best_option = None
    rejection_log = {} # diagnostics tool
    
    for box in BOX_DATABASE:
        log_key = f"{box.box_type} ({box.dims[0]}x{box.dims[1]}x{box.dims[2]})"
        internal_dims = get_internal_dims(box)

        if forklift_available and forklift_dim:
            # Forklift check is mainly for footprint (L, W)
            if not (box.dims[0] <= forklift_dim[0] and box.dims[1] <= forklift_dim[1]):
                rejection_log[log_key] = f"Rejected: Box footprint ({box.dims[0]}x{box.dims[1]}) exceeds forklift dimensions ({forklift_dim[0]}x{forklift_dim[1]})."
                continue
        
        insert = design_insert_for_box(part_dim, internal_dims, fragility)
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

        part_total_weight = fit_count * part_weight
        insert_weight_total = insert["weight_kg"] * layers
        separator_weight_total = separator["weight_kg"] * max(0, layers - 1)
        
        total_weight = part_total_weight + insert_weight_total + separator_weight_total
        if box.box_type == "FLC":
            total_weight += 5.13

        if total_weight > box.capacity_kg:
            rejection_log[log_key] = f"Rejected: Total weight ({total_weight:.1f} kg) exceeds box capacity ({box.capacity_kg} kg)."
            continue
        if forklift_available and forklift_capacity and total_weight > forklift_capacity:
            rejection_log[log_key] = f"Rejected: Total weight ({total_weight:.1f} kg) exceeds forklift capacity ({forklift_capacity} kg)."
            continue

        if best_option is None or fit_count > best_option["box_details"]["Max Parts"]:
            boxes_per_year = -(-annual_parts // fit_count) if fit_count > 0 else 0
            
            best_option = {
                "insert_details": insert,
                "separator_details": separator,
                "box_details": {
                    "Box Type": box.box_type,
                    "Box Dimensions": box.dims,
                    "Internal Dimensions": internal_dims,
                    "Max Parts": fit_count,
                    "Total Weight": total_weight,
                    "Boxes/Year": boxes_per_year,
                    "Layers": layers
                },
                "rejection_log": rejection_log
            }
            
    if best_option:
        return best_option
    else:
        # If no solution is found at all, return the log
        return {"rejection_log": rejection_log}

# -----------------------------
# Login Page
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
# Main App
# -----------------------------
def packaging_app():
    st.title("🚚 Auto Parts Packaging Optimization")

    # Part input
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

    annual_parts = st.number_input("Annual Auto Parts Quantity", min_value=1, step=1000, value=50000, key="annual_qty")

    st.subheader("Route Information")
    source = st.selectbox("Route Source", LOCATIONS, key="route_source")
    destination = st.selectbox("Route Destination", LOCATIONS, key="route_destination")
    highway = st.checkbox("Highway", key="route_highway")
    semiurban = st.checkbox("Semi-Urban", key="route_semiurban")
    village = st.checkbox("Village", key="route_village")

    if st.button("Get Optimized Packaging", key="optimize_button"):
        part_dim = (part_length, part_width, part_height)
        
        result = recommend_boxes(
            part_dim, part_weight, stacking_allowed, fragility_level,
            forklift_available, forklift_capacity, forklift_dim, annual_parts
        )
        
        # Check if a valid "box_details" key exists in the result
        if "box_details" in result:
            insert = result["insert_details"]
            separator = result["separator_details"]
            best_box = result["box_details"]

            st.divider()
            
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.markdown("### 🧩 Insert & Separator Design")
                st.markdown(f"""
                <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                    <b>Insert Details</b><br>
                    Type: {insert['type']}<br>
                    Matrix Pattern: {insert['matrix'][0]} × {insert['matrix'][1]}<br>
                    Outer Dimensions: {insert['outer_dims'][0]} × {insert['outer_dims'][1]} × {insert['outer_dims'][2]} mm<br>
                    Cell (Part Orientation): {insert['cell_dims'][0]} × {insert['cell_dims'][1]} × {insert['cell_dims'][2]} mm<br>
                    Weight per Layer: {insert['weight_kg']} kg
                </div>
                <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                    <b>Separator Details</b><br>
                    Type: {separator['type'] if separator['needed'] else 'Not Required'}<br>
                    Note: {separator.get('note', 'N/A')}<br>
                    Weight per Unit: {separator.get('weight_kg', 'N/A')} kg
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### Matrix Pattern Visualization")
                cell_style = "display:inline-block;border:2px solid #b7e4c7;border-radius:4px;width:40px;height:40px;margin:2px;background-color:#f8fff9;"
                rows, cols = insert["matrix"][1], insert["matrix"][0]
                display_rows, display_cols = min(rows, 8), min(cols, 8)
                
                row_html = "".join([
                    "<div style='display:flex;flex-direction:row;'>" +
                    "".join([f"<div style='{cell_style}'></div>" for _ in range(display_cols)]) +
                    "</div>" for _ in range(display_rows)
                ])
                if rows > display_rows or cols > display_cols:
                    row_html += f"<small><i>Displaying {display_rows}x{display_cols} of {rows}x{cols} total.</i></small>"
                st.markdown(row_html, unsafe_allow_html=True)

            st.divider()
            st.subheader("🏆 Outer Box Recommendation")
            box_dims = best_box["Box Dimensions"]
            internal_dims = best_box["Internal Dimensions"]
            st.markdown(f"""
            <div style="border:2px solid #2a9d8f; border-radius:10px; padding:15px; background-color:#f0fff4;">
                <b>Recommended Type</b>: {best_box['Box Type']} ({box_dims[0]}×{box_dims[1]}×{box_dims[2]} mm)<br><br>
                <b>Configuration:</b> {best_box['Layers']} layer(s) of {insert['units_per_insert']} parts each.<br>
                <b>Max Parts per Box:</b> <b>{best_box['Max Parts']}</b><br>
                <b>Total Weight:</b> {best_box['Total Weight']:.1f} kg<br>
                <b>Boxes Required per Year:</b> {best_box['Boxes/Year']}<br>
                <hr style="border-top: 1px solid #ddd;">
                <small><b>Internal Dims:</b> {internal_dims[0]} × {internal_dims[1]} × {internal_dims[2]} mm</small>
            </div>
            """, unsafe_allow_html=True)

        else:
            # --- NEW: Diagnostics Report ---
            st.error("❌ No suitable box and insert combination found.", icon="🚨")
            st.warning("Here is a diagnostics report showing why each standard box was rejected:", icon="🔬")
            
            log = result.get("rejection_log", {})
            if not log:
                st.info("No boxes were even attempted. This may indicate a problem with the initial inputs.")
            else:
                for box_name, reason in log.items():
                    st.markdown(f"- **{box_name}**: {reason}")

# -----------------------------
# Controller
# -----------------------------
if not st.session_state.logged_in:
    login()
else:
    packaging_app()