import pytest
from hypothesis import given, strategies as st

# --- Strategy to generate random dimensions ---
dims_strategy = st.tuples(
    st.integers(min_value=50, max_value=1000),  # L
    st.integers(min_value=50, max_value=1000),  # W
    st.integers(min_value=50, max_value=1000),  # H
)

@given(
    part_dims=dims_strategy,
    box_dims=dims_strategy,
    part_count=st.integers(min_value=1, max_value=100)
)
def test_box_efficiency_consistency(part_dims, box_dims, part_count):
    """
    Hypothesis property test:
    - Parts Efficiency % + Insert Overhead % = Box Used %
    - None of them should exceed 100%
    """
    Lp, Wp, Hp = part_dims
    Lb, Wb, Hb = box_dims

    part_volume = Lp * Wp * Hp
    box_volume = Lb * Wb * Hb

    # Avoid degenerate cases (zero/negative volume)
    if part_volume <= 0 or box_volume <= 0:
        return

    # Assume inserts overhead = 10% of part volume for testing
    inserts_volume = 0.1 * part_count * part_volume

    parts_volume_total = part_count * part_volume
    parts_eff = (parts_volume_total / box_volume) * 100
    insert_overhead = (inserts_volume / box_volume) * 100
    box_used = parts_eff + insert_overhead

    # --- Assertions ---
    assert parts_eff >= 0
    assert insert_overhead >= 0
    assert box_used >= 0

    # Must not exceed 100% (can't use more than total box volume)
    assert parts_eff <= 100.0
    assert insert_overhead <= 100.0
    assert box_used <= 100.0

    # Consistency check
    assert pytest.approx(box_used, rel=1e-5) == parts_eff + insert_overhead
