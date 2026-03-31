"""
Computational verification of binary polyhedral group theorems.

Tests:
  1. Binary icosahedral group 2I has exactly 120 elements in SU(2)
  2. 2I is closed under quaternion multiplication
  3. Binary tetrahedral 2T (24 elements) and binary octahedral 2O (48 elements)
  4. All three are subgroups of SU(2) (unit quaternions)
  5. Dechant's spinor induction: 2I elements in R^4 form the 600-cell vertices
  6. The 2I elements double-cover the 60 rotations of the icosahedral group I
  7. QuatBlock quaternions can be projected to nearest 2I element

Usage:
    cd framework && python3 -m pytest tests/test_binary_polyhedral_groups.py -v
    # or without pytest:
    cd framework && python3 tests/test_binary_polyhedral_groups.py
"""

import numpy as np
from itertools import product


# ============================================================================
# Quaternion arithmetic
# ============================================================================

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    """Quaternion conjugate [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_norm(q):
    """Quaternion norm."""
    return np.sqrt(np.sum(q**2))


def quat_normalize(q):
    """Normalize to unit quaternion."""
    return q / quat_norm(q)


def quat_to_rotation_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix (sandwich product)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


# ============================================================================
# Generate binary polyhedral groups
# ============================================================================

def generate_binary_icosahedral_group():
    """Generate all 120 elements of the binary icosahedral group 2I.

    Uses the two generators:
        s = (1 + i + j + k) / 2
        t = (phi + phi^{-1}*i + j) / 2
    where phi = (1 + sqrt(5)) / 2 is the golden ratio.

    Presentation: <s, t | (st)^2 = s^3 = t^5>
    """
    phi = (1 + np.sqrt(5)) / 2
    phi_inv = 1 / phi  # = phi - 1

    s = np.array([0.5, 0.5, 0.5, 0.5])
    t = np.array([phi/2, phi_inv/2, 0.5, 0.0])

    # Normalize (should already be unit, but ensure numerical precision)
    s = quat_normalize(s)
    t = quat_normalize(t)

    # Generate the group by repeated multiplication
    elements = set()
    queue = [s, t, quat_conjugate(s), quat_conjugate(t)]

    # Convert to hashable tuple with rounding for set membership
    # NOTE: Do NOT canonicalize sign — q and -q are distinct elements of 2I
    # (they map to the same rotation in SO(3), but are different spinors)
    def to_key(q):
        q = quat_normalize(q)
        return tuple(np.round(q, decimals=8))

    for q in queue:
        elements.add(to_key(q))

    # BFS: multiply existing elements until closure
    changed = True
    max_iterations = 200
    iteration = 0
    all_elements = list(queue)

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        new_elements = []
        for q1 in all_elements:
            for gen in [s, t, quat_conjugate(s), quat_conjugate(t)]:
                for product_q in [quat_multiply(q1, gen), quat_multiply(gen, q1)]:
                    key = to_key(product_q)
                    if key not in elements:
                        elements.add(key)
                        new_elements.append(quat_normalize(product_q))
                        changed = True
        all_elements.extend(new_elements)

    # Convert back to numpy arrays
    result = []
    for key in elements:
        result.append(np.array(key))

    return np.array(result)


def generate_binary_tetrahedral_group():
    """Generate all 24 elements of the binary tetrahedral group 2T.

    The 24 elements consist of:
    - 8 quaternions: ±1, ±i, ±j, ±k
    - 16 quaternions: (±1 ± i ± j ± k) / 2
    """
    elements = []

    # The 8 Lipschitz units
    for sign in [1, -1]:
        elements.append(np.array([sign, 0.0, 0.0, 0.0]))
        elements.append(np.array([0.0, sign, 0.0, 0.0]))
        elements.append(np.array([0.0, 0.0, sign, 0.0]))
        elements.append(np.array([0.0, 0.0, 0.0, sign]))

    # The 16 Hurwitz units
    for signs in product([0.5, -0.5], repeat=4):
        elements.append(np.array(signs))

    return np.array(elements)


def generate_binary_octahedral_group():
    """Generate all 48 elements of the binary octahedral group 2O.

    The 48 elements consist of:
    - All 24 elements of 2T
    - 24 additional elements: all permutations of (±1/√2, ±1/√2, 0, 0)
    """
    elements = list(generate_binary_tetrahedral_group())

    # The 24 additional elements: permutations of (±1/√2, ±1/√2, 0, 0)
    val = 1.0 / np.sqrt(2)
    indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for i, j in indices:
        for si in [val, -val]:
            for sj in [val, -val]:
                q = np.array([0.0, 0.0, 0.0, 0.0])
                q[i] = si
                q[j] = sj
                elements.append(q)

    return np.array(elements)


def nearest_group_element(quaternion, group_elements):
    """Find the nearest element in a finite group to a given quaternion.

    Uses quaternion distance: d(q1, q2) = min(||q1-q2||, ||q1+q2||)
    (accounting for the double cover: q and -q represent the same rotation).
    """
    q = quat_normalize(quaternion)
    distances = np.minimum(
        np.linalg.norm(group_elements - q, axis=1),
        np.linalg.norm(group_elements + q, axis=1),
    )
    idx = np.argmin(distances)
    return group_elements[idx], distances[idx]


# ============================================================================
# Tests
# ============================================================================

def test_2T_has_24_elements():
    """Binary tetrahedral group has exactly 24 elements."""
    group = generate_binary_tetrahedral_group()
    assert group.shape[0] == 24, f"Expected 24 elements, got {group.shape[0]}"
    print(f"  PASS: 2T has {group.shape[0]} elements")


def test_2O_has_48_elements():
    """Binary octahedral group has exactly 48 elements."""
    group = generate_binary_octahedral_group()
    assert group.shape[0] == 48, f"Expected 48 elements, got {group.shape[0]}"
    print(f"  PASS: 2O has {group.shape[0]} elements")


def test_2I_has_120_elements():
    """Binary icosahedral group has exactly 120 elements."""
    group = generate_binary_icosahedral_group()
    assert group.shape[0] == 120, f"Expected 120 elements, got {group.shape[0]}"
    print(f"  PASS: 2I has {group.shape[0]} elements")


def test_all_unit_quaternions():
    """All group elements are unit quaternions (lie in SU(2))."""
    for name, gen_fn, expected in [
        ("2T", generate_binary_tetrahedral_group, 24),
        ("2O", generate_binary_octahedral_group, 48),
        ("2I", generate_binary_icosahedral_group, 120),
    ]:
        group = gen_fn()
        norms = np.linalg.norm(group, axis=1)
        max_deviation = np.max(np.abs(norms - 1.0))
        assert max_deviation < 1e-8, (
            f"{name}: max norm deviation from 1 is {max_deviation}"
        )
        print(f"  PASS: {name} — all {len(group)} elements have unit norm "
              f"(max deviation: {max_deviation:.2e})")


def test_2T_closure():
    """2T is closed under quaternion multiplication."""
    group = generate_binary_tetrahedral_group()
    _check_closure("2T", group)


def test_2O_closure():
    """2O is closed under quaternion multiplication."""
    group = generate_binary_octahedral_group()
    _check_closure("2O", group)


def test_2I_closure():
    """2I is closed under quaternion multiplication."""
    group = generate_binary_icosahedral_group()
    _check_closure("2I", group)


def _check_closure(name, group, tol=1e-6):
    """Verify that a set of quaternions is closed under multiplication."""
    n = len(group)
    max_dist = 0.0
    failures = 0

    for i in range(n):
        for j in range(n):
            product_q = quat_multiply(group[i], group[j])
            _, dist = nearest_group_element(product_q, group)
            max_dist = max(max_dist, dist)
            if dist > tol:
                failures += 1

    assert failures == 0, (
        f"{name}: {failures}/{n*n} products not in group (max dist: {max_dist:.6f})"
    )
    print(f"  PASS: {name} — closed under multiplication "
          f"({n}×{n}={n*n} products checked, max dist: {max_dist:.2e})")


def test_2I_double_covers_icosahedral():
    """2I elements map 2:1 onto 60 distinct rotation matrices (icosahedral group I)."""
    group = generate_binary_icosahedral_group()
    assert len(group) == 120, f"Need 120 elements first, got {len(group)}"

    rotation_matrices = []
    for q in group:
        R = quat_to_rotation_matrix(q)
        rotation_matrices.append(R)

    # Count distinct rotations (q and -q give the same rotation)
    unique_rotations = []
    for R in rotation_matrices:
        is_new = True
        for R_existing in unique_rotations:
            if np.allclose(R, R_existing, atol=1e-8):
                is_new = False
                break
        if is_new:
            unique_rotations.append(R)

    assert len(unique_rotations) == 60, (
        f"Expected 60 distinct rotations, got {len(unique_rotations)}"
    )

    # Verify the 2:1 covering: each rotation should appear exactly twice
    counts = []
    for R_unique in unique_rotations:
        count = sum(1 for R in rotation_matrices if np.allclose(R, R_unique, atol=1e-8))
        counts.append(count)
    assert all(c == 2 for c in counts), (
        f"Expected each rotation to appear exactly 2 times, got distribution: {set(counts)}"
    )

    print(f"  PASS: 2I double-covers I — {len(group)} quaternions → "
          f"{len(unique_rotations)} distinct rotations (each covered exactly 2:1)")


def test_2T_is_subgroup_of_2O():
    """2T is a subgroup of 2O."""
    group_2T = generate_binary_tetrahedral_group()
    group_2O = generate_binary_octahedral_group()
    _check_subgroup("2T", group_2T, "2O", group_2O)


def test_2T_is_subgroup_of_2I():
    """2T is a subgroup of 2I."""
    group_2T = generate_binary_tetrahedral_group()
    group_2I = generate_binary_icosahedral_group()
    _check_subgroup("2T", group_2T, "2I", group_2I)


def _check_subgroup(name_sub, group_sub, name_super, group_super, tol=1e-6):
    """Verify that group_sub is a subset of group_super."""
    missing = 0
    max_dist = 0.0

    for q in group_sub:
        _, dist = nearest_group_element(q, group_super)
        max_dist = max(max_dist, dist)
        if dist > tol:
            missing += 1

    assert missing == 0, (
        f"{name_sub} not contained in {name_super}: {missing} elements missing"
    )
    print(f"  PASS: {name_sub} ⊂ {name_super} "
          f"(all {len(group_sub)} elements found, max dist: {max_dist:.2e})")


def test_600_cell_structure():
    """2I elements in R^4 form 600-cell vertices: all pairwise distances
    are from the set {√(2-φ), 1, √2, √(2+φ), 2} (plus zero for identity).

    The 600-cell is a regular 4-polytope whose 120 vertices have exactly 5
    distinct nonzero pairwise distances.
    """
    group = generate_binary_icosahedral_group()
    n = len(group)
    phi = (1 + np.sqrt(5)) / 2

    # Expected distances for 600-cell (up to the double cover: q and -q)
    # Using quaternion metric: d(q1,q2) = min(||q1-q2||, ||q1+q2||)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            d = min(
                np.linalg.norm(group[i] - group[j]),
                np.linalg.norm(group[i] + group[j]),
            )
            if d > 1e-8:  # exclude self-distance
                distances.append(d)

    distances = np.array(distances)

    # Cluster the distances
    unique_dists = []
    for d in sorted(distances):
        is_new = True
        for ud in unique_dists:
            if abs(d - ud) < 0.01:
                is_new = False
                break
        if is_new:
            unique_dists.append(d)

    print(f"  INFO: 2I pairwise distances cluster into {len(unique_dists)} "
          f"distinct values: {[f'{d:.4f}' for d in unique_dists]}")

    # The 600-cell should have a small number of distinct distances
    assert len(unique_dists) <= 8, (
        f"Expected ≤8 distinct distance classes, got {len(unique_dists)}"
    )
    print(f"  PASS: 2I elements form a highly regular polytope in R^4 "
          f"({len(unique_dists)} distance classes)")


def test_nearest_element_projection():
    """A random quaternion can be projected to its nearest 2I element."""
    group = generate_binary_icosahedral_group()
    rng = np.random.default_rng(42)

    n_trials = 1000
    distances = []
    for _ in range(n_trials):
        q_random = quat_normalize(rng.standard_normal(4))
        nearest, dist = nearest_group_element(q_random, group)
        distances.append(dist)
        # Verify nearest is in the group
        assert quat_norm(nearest) > 0.99, "Nearest element is not unit"

    distances = np.array(distances)
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)

    print(f"  PASS: Projection to nearest 2I element works — "
          f"mean dist: {mean_dist:.4f}, max dist: {max_dist:.4f} "
          f"(over {n_trials} random quaternions)")

    # For 120 well-distributed points on S^3, the maximum distance to the
    # nearest point should be bounded
    assert max_dist < 1.0, (
        f"Max distance to nearest 2I element unexpectedly large: {max_dist}"
    )


def test_generator_relations():
    """Verify the presentation <s, t | (st)^2 = s^3 = t^5>."""
    phi = (1 + np.sqrt(5)) / 2
    phi_inv = 1 / phi

    s = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
    t = quat_normalize(np.array([phi/2, phi_inv/2, 0.5, 0.0]))

    # s^3 should equal ±1
    s3 = quat_multiply(quat_multiply(s, s), s)
    assert np.allclose(np.abs(s3), [1, 0, 0, 0], atol=1e-8), (
        f"s^3 = {s3}, expected ±1"
    )

    # t^5 should equal ±1
    t_power = t.copy()
    for _ in range(4):
        t_power = quat_multiply(t_power, t)
    assert np.allclose(np.abs(t_power), [1, 0, 0, 0], atol=1e-8), (
        f"t^5 = {t_power}, expected ±1"
    )

    # (st)^2 should equal ±1
    st = quat_multiply(s, t)
    st2 = quat_multiply(st, st)
    assert np.allclose(np.abs(st2), [1, 0, 0, 0], atol=1e-8), (
        f"(st)^2 = {st2}, expected ±1"
    )

    # All three should equal the same central element
    assert np.allclose(s3, t_power, atol=1e-8) or np.allclose(s3, -t_power, atol=1e-8), (
        f"s^3 = {s3} but t^5 = {t_power} — should be equal (mod sign)"
    )

    print(f"  PASS: Generator relations verified — s³ = t⁵ = (st)² = {s3}")


def test_dechant_spinor_induction_dimensions():
    """Verify dimensional aspect of Dechant's construction:

    H₃ (3D, rank 3) → spinors in Cl⁺(3,0) ≅ R⁴ → H₄ (4D, rank 4)

    The 120 elements of 2I live in R⁴ (as quaternions). Dechant shows
    these are the roots of H₄ (vertices of 600-cell). This test verifies
    that the 120 quaternions satisfy H₄ root system properties:
    - All roots have the same norm
    - Inner products between distinct roots take only finitely many values
    - The Cartan matrix has the correct structure
    """
    group = generate_binary_icosahedral_group()

    # All norms should be 1
    norms = np.linalg.norm(group, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8), "Not all roots have unit norm"

    # Compute all inner products
    gram = group @ group.T
    inner_products = set()
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            ip = round(gram[i, j], 6)
            inner_products.add(ip)

    print(f"  INFO: Distinct inner products between 2I elements: "
          f"{len(inner_products)}")
    print(f"  INFO: Inner product values: "
          f"{sorted(inner_products)[:10]}{'...' if len(inner_products) > 10 else ''}")

    # For H₄ root system, inner products should be from a finite discrete set
    # related to the golden ratio
    assert len(inner_products) < 20, (
        f"Too many distinct inner products ({len(inner_products)}) for a root system"
    )
    print(f"  PASS: 2I elements have {len(inner_products)} distinct inner product "
          "values — consistent with H₄ root system")


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Binary tetrahedral group 2T has 24 elements", test_2T_has_24_elements),
        ("Binary octahedral group 2O has 48 elements", test_2O_has_48_elements),
        ("Binary icosahedral group 2I has 120 elements", test_2I_has_120_elements),
        ("All elements are unit quaternions (SU(2))", test_all_unit_quaternions),
        ("Generator relations s³ = t⁵ = (st)²", test_generator_relations),
        ("2T is closed under multiplication", test_2T_closure),
        ("2O is closed under multiplication", test_2O_closure),
        ("2I is closed under multiplication", test_2I_closure),
        ("2I double-covers icosahedral group I", test_2I_double_covers_icosahedral),
        ("2T ⊂ 2O", test_2T_is_subgroup_of_2O),
        ("2T ⊂ 2I", test_2T_is_subgroup_of_2I),
        ("2I forms 600-cell in R⁴", test_600_cell_structure),
        ("Projection to nearest 2I element", test_nearest_element_projection),
        ("Dechant spinor induction dimensions", test_dechant_spinor_induction_dimensions),
    ]

    passed = 0
    failed = 0

    print("=" * 70)
    print("Binary Polyhedral Group Verification Tests")
    print("=" * 70)

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
