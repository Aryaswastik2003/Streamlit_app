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
def fits_in_cell(part_dim: Tuple[int, int, int], cell_dim: Tuple[float, float, float]) -> Tuple[bool, Tuple[int,int,int]]:
    """Check if part can fit in a cell in any orientation; returns (fits, chosen_orientation)."""
    L, W, H = part_dim
    cL, cW, cH = cell_dim
    orientations = [(L, W, H), (L, H, W), (W, L, H), (W, H, L), (H, L, W), (H, W, L)]
    for o in orientations:
        if o[0] <= cL and o[1] <= cW and o[2] <= cH:
            return True, o
    return False, part_dim

# -----------------------------
# Recommendation Functions
# -----------------------------
def recommend_insert(part_dim, part_weight, fragility, stacking_allowed):
    # Default recommendation based on fragility (unchanged core logic)
    if fragility == "High":
        insert = {"Type": "Thermo-vac PP Tray", "Note": "Best for fragile / Class-A parts.",
                  "Matrix": (2, 2), "Units per Insert": 4,
                  "Outer Dimensions": (1100, 900, 410), "Weight": 50.0}
    elif fragility == "Medium":
        insert = {"Type": "PP Partition Grid", "Note": "General purpose for medium parts.",
                  "Matrix": (4, 3), "Units per Insert": 12,
                  "Outer Dimensions": (1100, 900, 410), "Weight": 328.78}
    elif fragility == "Low" and part_weight > 20:
        insert = {"Type": "Honeycomb Layer Pad", "Note": "For heavy stacks, adds strength.",
                  "Matrix": (3, 2), "Units per Insert": 6,
                  "Outer Dimensions": (1100, 900, 410), "Weight": 120.0}
    else:
        insert = {"Type": "Woven PP Pouch", "Note": "For scratch-sensitive or small batches.",
                  "Matrix": (5, 2), "Units per Insert": 10,
                  "Outer Dimensions": (1100, 900, 410), "Weight": 80.0}

    # --- Apply PRD tolerance rules (5 mm clearance + 5 mm partition) ---
    outer_L, outer_W, outer_H = insert["Outer Dimensions"]
    cols, rows = insert["Matrix"]
    # (cols+1) ribs in L and (rows+1) ribs in W → 5mm each (partition thickness & wall clearance)
    net_L = outer_L - (cols + 1) * 5
    net_W = outer_W - (rows + 1) * 5
    cell_L = round(net_L / cols, 2)
    cell_W = round(net_W / rows, 2)
    cell_H = round(outer_H - 5, 2)  # top clearance
    insert["Cell Dimensions"] = (cell_L, cell_W, cell_H)

    # Descriptive reasoning (point-wise)
    reasons = []
    reasons.append(f"Fragility level **{fragility}** → selected **{insert['Type']}** ({insert['Note']}).")
    reasons.append(f"Matrix **{cols} × {rows}** → **{insert['Units per Insert']} units/layer**.")
    reasons.append("Applied **5 mm wall & partition clearance** per PRD (manufacturing tolerance).")
    reasons.append(f"Outer size **{outer_L}×{outer_W}×{outer_H} mm** → cell inner **{cell_L}×{cell_W}×{cell_H} mm**.")
    # Fit check (non-blocking, for reasoning only)
    fits, orient = fits_in_cell(part_dim, insert["Cell Dimensions"])
    if fits:
        reasons.append(f"Part fits inside cell (example orientation **{orient} mm**).")
    else:
        reasons.append("⚠️ Part dimensions exceed a single cell in all orientations; consider larger matrix cells or alternate insert.")
    if not stacking_allowed:
        reasons.append("Stacking disabled → insert chosen prioritizes single-layer protection/handling.")
    insert["Reasons"] = reasons
    return insert


def recommend_separator(insert, stacking_allowed):
    if not stacking_allowed:
        return {"Needed": False, "Type": None, "Note": "Stacking disabled.",
                "Dimensions": None, "Weight": 0.0,
                "Reasons": ["Stacking is disabled → no inter-layer separator is required per PRD."]}

    if insert["Type"] in ("PP Partition Grid", "Thermo-vac PP Tray"):
        sep = {"Needed": True, "Type": "Honeycomb Layer Pad",
               "Note": "Adds strength between stacked layers.",
               "Dimensions": (insert["Outer Dimensions"][0], insert["Outer Dimensions"][1]),
               "Weight": 1.49}
        reasons = [
            "Stacking is enabled → separator required between layers (PRD).",
            f"Insert type **{insert['Type']}** benefits from **Honeycomb Layer Pad** to prevent deflection & contact marks.",
            f"Separator spans full footprint **{sep['Dimensions'][0]}×{sep['Dimensions'][1]} mm** for load spread.",
            f"Separator unit weight **{sep['Weight']} kg** (multiplies by layers−1).",
        ]
        sep["Reasons"] = reasons
        return sep

    sep = {"Needed": True, "Type": "PP Sheet Separator",
           "Note": "General separator for multiple layers.",
           "Dimensions": (insert["Outer Dimensions"][0], insert["Outer Dimensions"][1]),
           "Weight": 1.0}
    sep["Reasons"] = [
        "Stacking is enabled → separator required between layers (PRD).",
        "General-purpose **PP Sheet Separator** adequate for the selected insert.",
        f"Separator spans full footprint **{sep['Dimensions'][0]}×{sep['Dimensions'][1]} mm**.",
        f"Lightweight at **{sep['Weight']} kg** each.",
    ]
    return sep


def orientations(part_dim, restriction):
    L, W, H = part_dim
    if restriction == "Length Standing":
        return [(H, W, L)]
    if restriction == "Width Standing":
        return [(L, H, W)]
    if restriction == "Height Standing":
        return [(L, W, H)]
    return [(L, W, H), (L, H, W), (W, L, H), (W, H, L), (H, L, W), (H, W, L)]


def check_fit(part_dim, box_dim, orientation_restriction):
    part_L, part_W, part_H = part_dim
    box_L, box_W, box_H = box_dim
    best_fit, wasted_volume, chosen_orientation = 0, 100.0, part_dim
    for l, w, h in orientations(part_dim, orientation_restriction):
        fit_l = box_L // l
        fit_w = box_W // w
        fit_h = box_H // h
        fit_count = fit_l * fit_w * fit_h
        used_volume = fit_count * (part_L * part_W * part_H)
        total_volume = box_L * box_W * box_H
        waste = ((total_volume - used_volume) / total_volume) * 100.0 if total_volume else 100.0
        if fit_count > best_fit:
            best_fit, wasted_volume, chosen_orientation = fit_count, waste, (l, w, h)
    return best_fit, wasted_volume, chosen_orientation


def get_internal_dims(box: Box) -> Tuple[int, int, int]:
    """Apply PRD rules for internal dimensions per box type."""
    L, W, H = box.dims
    if box.box_type == "PP Box":
        return (L - 34, W - 34, H - 8)
    elif box.box_type == "PLS":
        return (L - 34, W - 34, H - 210)
    elif box.box_type == "FLC":
        return (L, W, H)  # FLC = same as external
    else:
        return (L, W, H)


def recommend_boxes(part_dim, part_weight, orientation_restriction, stacking_allowed,
                    insert, separator, forklift_available, forklift_capacity, forklift_dim, annual_parts):
    best_box = None
    for box in BOX_DATABASE:
        # Forklift dimension check
        if forklift_available and forklift_dim:
            if not (box.dims[0] <= forklift_dim[0] and box.dims[1] <= forklift_dim[1] and box.dims[2] <= forklift_dim[2]):
                continue

        # Layers based on insert height
        insert_height = insert["Outer Dimensions"][2]
        layers = box.dims[2] // insert_height
        if layers < 1:
            continue

        # Parts per box = layers × units per insert
        fit_count = layers * insert["Units per Insert"]

        # Weight calculation
        part_total_weight = fit_count * part_weight
        insert_weight = insert.get("Weight", 0)
        separator_weight = 0
        if separator.get("Needed"):
            separator_weight = separator.get("Weight", 0) * max(0, layers - 1)
        total_weight = part_total_weight + insert_weight + separator_weight

        # Add lid weight if FLC (PRD)
        if box.box_type == "FLC":
            total_weight += 5.13

        # Capacity check
        if total_weight > box.capacity_kg:
            max_parts_by_weight = int((box.capacity_kg - (insert_weight + separator_weight)) // part_weight)
            fit_count = max(0, max_parts_by_weight)
            part_total_weight = fit_count * part_weight
            total_weight = part_total_weight + insert_weight + separator_weight
            if box.box_type == "FLC":
                total_weight += 5.13

        if forklift_available and forklift_capacity and total_weight > forklift_capacity:
            continue

        if fit_count == 0:
            continue

        boxes_per_year = -(-annual_parts // fit_count)  # ceil division
        internal_dims = get_internal_dims(box)

        # Descriptive reasoning (point-wise)
        reasons = []
        reasons.append(f"Insert **{insert['Type']}** provides **{insert['Units per Insert']} units/layer**.")
        reasons.append(f"Box height allows **{layers} layer(s)** → **{fit_count} parts/box**.")
        reasons.append(f"Weight breakdown: parts **{part_total_weight:.1f} kg**, insert **{insert_weight:.1f} kg**, separators **{separator_weight:.1f} kg**.")
        if box.box_type == "FLC":
            reasons.append("Added **FLC lid weight 5.13 kg** per PRD.")
        reasons.append(f"Total load **{total_weight:.1f} kg** ≤ box capacity **{box.capacity_kg} kg**.")
        reasons.append(f"Internal dims (PRD): **{internal_dims[0]}×{internal_dims[1]}×{internal_dims[2]} mm** derived from box type **{box.box_type}**.")
        if forklift_available:
            reasons.append("Forklift constraints validated (dims & capacity).")
        reasons.append(f"Annual demand **{annual_parts}** → **{boxes_per_year} box(es)/year** at {fit_count} parts/box.")

        if (best_box is None) or (fit_count > best_box["Max Parts"]):
            best_box = {
                "Box Type": box.box_type,
                "Box Dimensions": box.dims,
                "Internal Dimensions": internal_dims,
                "Max Parts": fit_count,
                "Total Weight": total_weight,
                "Boxes/Year": boxes_per_year,
                "Reasons": reasons,
            }
    return best_box


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
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid username or password")


# -----------------------------
# Main App
# -----------------------------
def packaging_app():
    st.title("🚚 Auto Parts Packaging Optimization")

    # Part input
    part_length = st.number_input("Part Length (mm)", min_value=1, key="part_length")
    part_width = st.number_input("Part Width (mm)", min_value=1, key="part_width")
    part_height = st.number_input("Part Height (mm)", min_value=1, key="part_height")
    part_weight = st.number_input("Part Weight (kg)", min_value=0.1, step=0.1, key="part_weight")

    fragility_level = st.selectbox("Fragility Level", ["Low", "Medium", "High"], key="fragility_level")
    stacking_allowed = st.toggle("Stacking Allowed", value=True, key="stacking_allowed")
    orientation_restriction = st.selectbox(
        "Orientation Restriction",
        ["Free", "Length Standing", "Width Standing", "Height Standing"],
        key="orientation_restriction"
    )

    forklift_available = st.checkbox("Is forklift available?", key="forklift_available")
    forklift_capacity, forklift_dim = None, None
    if forklift_available:
        forklift_capacity = st.number_input("Forklift Capacity (kg)", min_value=1, key="forklift_capacity")
        fl_l = st.number_input("Forklift Max Length (mm)", min_value=1, key="forklift_l")
        fl_w = st.number_input("Forklift Max Width (mm)", min_value=1, key="forklift_w")
        fl_h = st.number_input("Forklift Max Height (mm)", min_value=1, key="forklift_h")
        forklift_dim = (fl_l, fl_w, fl_h)

    # Annual parts
    annual_parts = st.number_input("Annual Auto Parts Quantity", min_value=1, step=1000, key="annual_qty")

    # Route inputs
    st.subheader("Route Information")
    source = st.selectbox("Route Source", LOCATIONS, key="route_source")
    destination = st.selectbox("Route Destination", LOCATIONS, key="route_destination")
    highway = st.checkbox("Highway", key="route_highway")
    semiurban = st.checkbox("Semi-Urban", key="route_semiurban")
    village = st.checkbox("Village", key="route_village")

    selected_routes = [r for r, flag in [("Highway", highway), ("Semi-Urban", semiurban), ("Village", village)] if flag]
    if selected_routes:
        pct = 100 / len(selected_routes)
        route_distribution = {r: pct for r in selected_routes}
    else:
        route_distribution = {}

    if st.button("Get Optimized Packaging", key="optimize_button"):
        part_dim = (part_length, part_width, part_height)
        insert = recommend_insert(part_dim, part_weight, fragility_level, stacking_allowed)
        separator = recommend_separator(insert, stacking_allowed)
        best_box = recommend_boxes(
            part_dim, part_weight, orientation_restriction, stacking_allowed,
            insert, separator, forklift_available, forklift_capacity, forklift_dim, annual_parts
        )

        # -----------------------------
        # Insert & Separator Cards
        # -----------------------------
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### Design of Inserts (PP material)")
            st.markdown(f"""
            <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                <b>Insert Details</b><br>
                Type: {insert['Type']}<br>
                Matrix Pattern: {insert['Matrix'][0]} × {insert['Matrix'][1]}<br>
                Outer Dimensions: {insert['Outer Dimensions'][0]} × {insert['Outer Dimensions'][1]} × {insert['Outer Dimensions'][2]} mm<br>
                Cell Inner Dimensions: {insert['Cell Dimensions'][0]} × {insert['Cell Dimensions'][1]} × {insert['Cell Dimensions'][2]} mm<br>
                Weight: {insert['Weight']} kg
            </div>
            <div style="border:1px solid #d3d3d3; border-radius:10px; padding:12px; margin-bottom:10px;">
                <b>Separator Details</b><br>
                Type: {separator['Type'] if separator['Needed'] else 'Not Required'}<br>
                Dimensions: {separator.get('Dimensions', 'N/A')}<br>
                Weight: {separator.get('Weight', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Matrix Pattern Visualization")
            cell_style = """
                display:inline-block;border:2px solid #b7e4c7;border-radius:4px;
                width:50px;height:50px;margin:2px;background-color:#f8fff9;
            """
            row_html = ""
            for y in range(insert["Matrix"][1]):
                row_html += "<div style='display:flex;flex-direction:row;'>"
                for x in range(insert["Matrix"][0]):
                    row_html += f"<div style='{cell_style}'></div>"
                row_html += "</div>"
            st.markdown(row_html, unsafe_allow_html=True)

            # Descriptive, point-wise reasoning (right column, under visualization)
            st.markdown("**✅ Insert Recommendation Reasoning**")
            for r in insert["Reasons"]:
                st.markdown(f"- {r}")

            st.markdown("**✅ Separator Recommendation Reasoning**")
            for r in separator["Reasons"]:
                st.markdown(f"- {r}")

        # -----------------------------
        # Box Recommendation Card
        # -----------------------------
        if best_box:
            st.subheader("🏆 Outer Box Recommendation")
            box_dims = best_box["Box Dimensions"]
            internal_dims = best_box["Internal Dimensions"]
            st.markdown(f"""
            <div style="border:2px solid #b7e4c7; border-radius:10px; padding:15px; background-color:#f0fff4;">
                <b>Recommended Type</b>: {best_box['Box Type']} ({box_dims[0]}×{box_dims[1]}×{box_dims[2]})<br><br>
                <b>Internal (L×W×H):</b> {internal_dims[0]} × {internal_dims[1]} × {internal_dims[2]} mm<br>
                <b>External (L×W×H):</b> {box_dims[0]} × {box_dims[1]} × {box_dims[2]} mm<br>
                <b>Max Parts per Box:</b> {best_box['Max Parts']}<br>
                <b>Total Weight (incl. insert + separators):</b> {best_box['Total Weight']:.1f} kg<br>
                <b>Boxes Required per Year:</b> {best_box['Boxes/Year']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**✅ Box Recommendation Reasoning**")
            for r in best_box["Reasons"]:
                st.markdown(f"- {r}")
        else:
            st.error("❌ No suitable box found for the given part dimensions and constraints.")


# -----------------------------
# Controller
# -----------------------------
if not st.session_state.logged_in:
    login()
else:
    packaging_app()
