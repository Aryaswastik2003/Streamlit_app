import streamlit as st

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_boxes(part_dim, part_weight, fragility_level,
                    forklift_available, forklift_capacity, forklift_dim,
                    orientation_restriction, stacking_allowed):

    recommendations = []
    fragility_factor = {"Low": 1.0, "Medium": 0.7, "High": 0.5}[fragility_level]
    part_volume = part_dim[0] * part_dim[1] * part_dim[2]

    # ✅ Orientation sets
    if orientation_restriction == "No Restriction":
        orientations = [
            (part_dim[0], part_dim[1], part_dim[2]),
            (part_dim[0], part_dim[2], part_dim[1]),
            (part_dim[1], part_dim[0], part_dim[2]),
            (part_dim[1], part_dim[2], part_dim[0]),
            (part_dim[2], part_dim[0], part_dim[1]),
            (part_dim[2], part_dim[1], part_dim[0]),
        ]
    elif orientation_restriction == "Length Standing":
        orientations = [(part_dim[1], part_dim[2], part_dim[0]),
                        (part_dim[2], part_dim[1], part_dim[0])]
    elif orientation_restriction == "Width Standing":
        orientations = [(part_dim[0], part_dim[2], part_dim[1]),
                        (part_dim[2], part_dim[0], part_dim[1])]
    elif orientation_restriction == "Height Standing":
        orientations = [(part_dim[0], part_dim[1], part_dim[2]),
                        (part_dim[1], part_dim[0], part_dim[2])]
    else:
        orientations = [(part_dim[0], part_dim[1], part_dim[2])]

    for box_type, box_data in boxes.items():
        for box_dim in box_data["sizes"]:
            box_volume = box_dim[0] * box_dim[1] * box_dim[2]

            best_fit_dim = 0
            best_orientation = None
            best_wasted_percent = 100.0

            for orient in orientations:
                if stacking_allowed:
                    fit = (box_dim[0] // orient[0]) * \
                          (box_dim[1] // orient[1]) * \
                          (box_dim[2] // orient[2])
                else:
                    fit = (box_dim[0] // orient[0]) * \
                          (box_dim[1] // orient[1])   # only base layer

                used_volume = fit * part_volume
                wasted_percent = (box_volume - used_volume) / box_volume * 100 if box_volume > 0 else 0

                if fit > best_fit_dim:
                    best_fit_dim = fit
                    best_orientation = orient
                    best_wasted_percent = wasted_percent

            # ✅ Apply fragility factor
            max_fit = int(best_fit_dim * fragility_factor)

            # ✅ Apply weight capacity
            if max_fit * part_weight > box_data["weight_capacity"]:
                max_fit = box_data["weight_capacity"] // part_weight

            if max_fit < 1:
                continue

            # ✅ Forklift constraint check
            forklift_ok = True
            if forklift_available:
                if (max_fit * part_weight) > forklift_capacity or \
                   any(d > fd for d, fd in zip(box_dim, forklift_dim)):
                    forklift_ok = False

            # ✅ Store recommendation only if forklift allows OR forklift not required
            if not forklift_available or forklift_ok:
                reasons = []
                reasons.append(f" Part Dimensions: {part_dim}")
                reasons.append(f" Box Dimensions: {box_dim}")
                reasons.append(f" Part Weight: {part_weight} kg × {max_fit} = {part_weight * max_fit} kg")
                reasons.append(f" Box Weight Capacity: {box_data['weight_capacity']} kg")
                reasons.append(f" Orientation Restriction: {orientation_restriction}")
                # reasons.append(f" Stacking Allowed: {stacking_allowed}")
                reasons.append(f" Best Orientation Used: {best_orientation}")
                # reasons.append(f" Practical Fit: {best_fit_dim} parts")
                # reasons.append(f" Wasted Volume With Orientation: {best_wasted_percent:.1f}%")


                if fragility_factor < 1.0:
                    reasons.append(f"✅ Adjusted for {fragility_level} fragility")

                if forklift_available:
                    reasons.append("✅ Compatible with forklift capacity & dimensions")

                recommendations.append({
                    "Box Type": box_type,
                    "Box Dimensions": box_dim,
                    "Max Parts": int(max_fit),
                    "Wasted Volume %": best_wasted_percent,
                    "Reasons": reasons
                })

    # ✅ Pick optimized box → minimum wasted volume
    best_box = None
    if recommendations:
        best_box = min(recommendations, key=lambda r: r["Wasted Volume %"])

    return best_box


# -----------------------------
# Streamlit UI
# -----------------------------
st.title(" Packaging Optimization System")

# Part inputs
st.header("Part Details")
part_length = st.number_input("Part Length (mm)", min_value=1, value=198)
part_width = st.number_input("Part Width (mm)", min_value=1, value=230)
part_height = st.number_input("Part Height (mm)", min_value=1, value=200)
part_weight = st.number_input("Part Weight (kg)", min_value=0.1, value=5.0)
fragility_level = st.selectbox("Fragility Level", ["Low", "Medium", "High"])

# Orientation restriction input
st.header("Orientation Restriction")
orientation_restriction = st.selectbox(
    "Select Orientation Restriction",
    ["No Restriction", "Length Standing", "Width Standing", "Height Standing"]
)

# Stacking input
st.header("Stacking Option")
stacking_allowed = st.toggle("Allow Stacking?", value=True)

# Forklift inputs
st.header("Forklift Details")
forklift_available = st.checkbox("Forklift Available?")
forklift_capacity, forklift_dim = None, None
if forklift_available:
    forklift_capacity = st.number_input("Forklift Capacity (kg)", min_value=1, value=1000)
    fl_length = st.number_input("Forklift Length Capacity (mm)", min_value=1, value=1200)
    fl_width = st.number_input("Forklift Width Capacity (mm)", min_value=1, value=1000)
    fl_height = st.number_input("Forklift Height Capacity (mm)", min_value=1, value=1000)
    forklift_dim = (fl_length, fl_width, fl_height)

# Production details
st.header("Production Details")
annual_parts = st.number_input("Enter total number of auto parts per year",
                               min_value=0, step=1000, format="%d")
st.write(f"📦 Total Auto Parts per Year: **{annual_parts}**")

# Distance details
st.header("Distance Details")
cities = ["Pune", "Mumbai", "Chennai", "Delhi", "Bangalore", "Hyderabad"]

source_city = st.selectbox("Select Source City", cities, index=0)
destination_city = st.selectbox("Select Destination City", cities, index=1)

if source_city == destination_city:
    st.warning("⚠️ Source and Destination cannot be the same!")
else:
    st.write(f"🚚 Transport from **{source_city}** to **{destination_city}**")

# Route type distribution
st.header("Route Type Distribution")
st.write("✅ Select applicable route types for transportation planning:")

selected_routes = []
if st.checkbox("🛣️ Highway"):
    selected_routes.append("Highway")
if st.checkbox("🏙️ Semi-Urban"):
    selected_routes.append("Semi-Urban")
if st.checkbox("🏡 Village"):
    selected_routes.append("Village")

if selected_routes:
    percent_each = 100 / len(selected_routes)
    route_distribution = {route: percent_each for route in selected_routes}
    st.success("📊 Route Distribution (auto-calculated):")
    for route, percent in route_distribution.items():
        st.write(f"- {route}: {percent:.1f}%")
else:
    st.warning("⚠️ Please select at least one route type.")

# Box definitions
boxes = {
    "PP Box": {"sizes": [(400, 300, 120), (600, 400, 348)], "weight_capacity": 16},
    "FLC": {"sizes": [(1200, 1000, 975)], "weight_capacity": 500},
    "Crate": {"sizes": [(800, 600, 400)], "weight_capacity": 100},
}

# Run optimization
if st.button("Get Optimized Box"):
    part_dim = (part_length, part_width, part_height)
    best_box = recommend_boxes(part_dim, part_weight, fragility_level,
                               forklift_available, forklift_capacity, forklift_dim,
                               orientation_restriction, stacking_allowed)

    if best_box:
        st.subheader("Best Optimized Box Recommendation")
        st.write(f"**Box Type:** {best_box['Box Type']}")
        st.write(f"**Box Dimensions:** {best_box['Box Dimensions']}")
        st.write(f"**Max Parts in Box:** {best_box['Max Parts']}")
        st.write(f"**Wasted Volume %:** {best_box['Wasted Volume %']:.1f}%")
        st.write("**Reasons:**")
        for reason in best_box["Reasons"]:
            st.write(reason)
    else:
        if forklift_available:
            st.error("❌ No box fits forklift constraints! Try increasing forklift size or reducing weight.")
        else:
            st.error("❌ No suitable box found for the given part dimensions and constraints.")
