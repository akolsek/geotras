"""Unit tests for the MCNP -> CAD reverse translation input processing.

Covers the MCNP surface card parsing (Get_primitive_surfaces), TR/TRCL
transformation matrices, the universe level structure and the cell/material
selection logic in geouned.GEOReverse.Modules.MCNPinput.
"""

import math

import pytest

import geouned  # noqa: F401  sets up the FreeCAD import path
import FreeCAD

from geouned.GEOReverse.Modules.MCNPinput import (
    Get_primitive_surfaces,
    McnpInput,
    getTransMatrix,
    selectCells,
)
import geouned.GEOReverse.Modules.MCNPinput as mcnp_input_module
from geouned.GEOReverse.Modules.Utils.boundBox import BoxSettings


def parallel(v, w, tol=1e-9):
    v = FreeCAD.Vector(v)
    w = FreeCAD.Vector(w)
    v.normalize()
    w.normalize()
    return abs(abs(v.dot(w)) - 1.0) < tol


def build_surface(stype, params, tr=None):
    """Parse a single MCNP surface card definition (cm units, scaled to mm)."""
    surfaces = Get_primitive_surfaces({1: (stype, params, tr, 1)}, scale=10.0)
    return surfaces.get(1)


# ---------------------------------------------------------------------------
# planes and spheres
# ---------------------------------------------------------------------------


def test_axis_planes():
    for stype, axis in (("PX", (1, 0, 0)), ("PY", (0, 1, 0)), ("PZ", (0, 0, 1))):
        s = build_surface(stype, [5.0])
        assert s.type == "plane"
        normal, d = s.params
        assert parallel(normal, axis)
        assert d == pytest.approx(50.0)  # 5 cm -> 50 mm


def test_general_plane_4_coefficients():
    s = build_surface("P", [0.0, 0.0, 1.0, 5.0])
    assert s.type == "plane"
    normal, d = s.params
    assert parallel(normal, (0, 0, 1))
    assert d == pytest.approx(50.0)


def test_general_plane_3_points():
    # plane z=1 cm through three points
    s = build_surface("P", [0, 0, 1, 1, 0, 1, 0, 1, 1])
    assert s.type == "plane"
    normal, d = s.params
    assert parallel(normal, (0, 0, 1))
    assert abs(d) == pytest.approx(10.0)


def test_spheres():
    s = build_surface("S", [1.0, 2.0, 3.0, 4.0])
    assert s.type == "sphere"
    center, radius = s.params
    assert center.isEqual(FreeCAD.Vector(10, 20, 30), 1e-9)
    assert radius == pytest.approx(40.0)

    s = build_surface("SO", [4.0])
    center, radius = s.params
    assert center.isEqual(FreeCAD.Vector(0, 0, 0), 1e-9)
    assert radius == pytest.approx(40.0)

    for stype, axis in (("SX", (1, 0, 0)), ("SY", (0, 1, 0)), ("SZ", (0, 0, 1))):
        s = build_surface(stype, [2.0, 4.0])
        center, radius = s.params
        assert center.isEqual(FreeCAD.Vector(axis) * 20.0, 1e-9)
        assert radius == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# cylinders and cones
# ---------------------------------------------------------------------------


def test_axis_cylinders():
    for stype, axis in (("CX", (1, 0, 0)), ("CY", (0, 1, 0)), ("CZ", (0, 0, 1))):
        s = build_surface(stype, [2.0])
        assert s.type == "cylinder"
        p, v, radius = s.params
        assert parallel(v, axis)
        assert radius == pytest.approx(20.0)


def test_offset_cylinders():
    # C/Z y x R -> cylinder parallel to z through (x, y)
    s = build_surface("C/Z", [1.0, 2.0, 3.0])
    p, v, radius = s.params
    assert parallel(v, (0, 0, 1))
    assert p.isEqual(FreeCAD.Vector(10, 20, 0), 1e-9)
    assert radius == pytest.approx(30.0)

    s = build_surface("C/X", [1.0, 2.0, 3.0])
    p, v, radius = s.params
    assert parallel(v, (1, 0, 0))
    assert p.isEqual(FreeCAD.Vector(0, 10, 20), 1e-9)


def test_axis_cones():
    # single sheet cone along +x
    s = build_surface("KX", [5.0, 0.25, 1.0])
    assert s.type == "cone"
    p, v, tan, dblsht = s.params
    assert p.isEqual(FreeCAD.Vector(50, 0, 0), 1e-9)
    assert v.isEqual(FreeCAD.Vector(1, 0, 0), 1e-9)
    assert tan == pytest.approx(0.5)
    assert dblsht is False

    # negative sheet flips the axis
    s = build_surface("KX", [5.0, 0.25, -1.0])
    p, v, tan, dblsht = s.params
    assert v.isEqual(FreeCAD.Vector(-1, 0, 0), 1e-9)

    # no sheet value -> double sheet
    s = build_surface("KZ", [5.0, 0.25])
    p, v, tan, dblsht = s.params
    assert dblsht is True


def test_offset_cone():
    s = build_surface("K/Z", [1.0, 2.0, 3.0, 1.0, 1.0])
    assert s.type == "cone"
    p, v, tan, dblsht = s.params
    assert p.isEqual(FreeCAD.Vector(10, 20, 30), 1e-9)
    assert v.isEqual(FreeCAD.Vector(0, 0, 1), 1e-9)
    assert tan == pytest.approx(1.0)
    assert dblsht is False


# ---------------------------------------------------------------------------
# tori
# ---------------------------------------------------------------------------


def test_torus():
    s = build_surface("TZ", [1.0, 2.0, 3.0, 10.0, 2.0, 2.0])
    assert s.type == "torus"
    p, v, Ra, Rb, Rc = s.params
    assert p.isEqual(FreeCAD.Vector(10, 20, 30), 1e-9)
    assert parallel(v, (0, 0, 1))
    assert Ra == pytest.approx(100.0)
    assert Rb == pytest.approx(20.0)
    assert Rc == pytest.approx(20.0)


def test_elliptical_torus_keeps_both_minor_radii():
    s = build_surface("TX", [0.0, 0.0, 0.0, 10.0, 2.0, 3.0])
    p, v, Ra, Rb, Rc = s.params
    assert Rb == pytest.approx(20.0)
    assert Rc == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# point-defined X/Y/Z surfaces (regression tests, were crashing before)
# ---------------------------------------------------------------------------


def test_point_defined_single_pair_is_plane():
    s = build_surface("X", [7.0, 1.0])
    assert s.type == "plane"
    normal, d = s.params
    assert parallel(normal, (1, 0, 0))
    assert d == pytest.approx(70.0)


def test_point_defined_equal_radii_is_cylinder():
    s = build_surface("X", [5.0, 2.0, 8.0, 2.0])
    assert s.type == "cylinder"
    p, v, radius = s.params
    assert parallel(v, (1, 0, 0))
    assert radius == pytest.approx(20.0)  # was unscaled before the fix

    s = build_surface("Z", [5.0, 1.5, 8.0, 1.5])
    p, v, radius = s.params
    assert parallel(v, (0, 0, 1))
    assert radius == pytest.approx(15.0)


def test_point_defined_varying_radii_is_cone():
    # r = 1 at y = 3, r = 2 at y = 5 -> apex at y = 1, tan = 0.5
    s = build_surface("Y", [3.0, 1.0, 5.0, 2.0])
    assert s.type == "cone"
    p, v, tan, dblsht = s.params
    assert p.isEqual(FreeCAD.Vector(0, 10, 0), 1e-6)
    assert v.isEqual(FreeCAD.Vector(0, 1, 0), 1e-9)
    assert tan == pytest.approx(0.5)
    assert dblsht is False


def test_point_defined_zero_radii_is_plane():
    s = build_surface("Z", [4.0, 0.0, 4.0, 0.0])
    assert s.type == "plane"


def test_point_defined_degenerate_card_is_skipped():
    # r1 = r2 = 0 with z1 != z2 defines no surface: must not crash nor
    # produce a surface object with empty parameters
    surfaces = Get_primitive_surfaces({1: ("Z", [1.0, 0.0, 4.0, 0.0], None, 1)}, scale=10.0)
    assert 1 not in surfaces


# ---------------------------------------------------------------------------
# macrobodies
# ---------------------------------------------------------------------------


def test_rpp_box():
    s = build_surface("RPP", [0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    assert s.type == "box"
    p, v1, v2, v3 = s.params
    assert p.isEqual(FreeCAD.Vector(0, 0, 0), 1e-9)
    assert v1.isEqual(FreeCAD.Vector(10, 0, 0), 1e-9)
    assert v2.isEqual(FreeCAD.Vector(0, 20, 0), 1e-9)
    assert v3.isEqual(FreeCAD.Vector(0, 0, 30), 1e-9)


def test_box():
    s = build_surface("BOX", [1, 1, 1, 2, 0, 0, 0, 3, 0, 0, 0, 4])
    p, v1, v2, v3 = s.params
    assert p.isEqual(FreeCAD.Vector(10, 10, 10), 1e-9)
    assert v1.isEqual(FreeCAD.Vector(20, 0, 0), 1e-9)


def test_rcc_can():
    s = build_surface("RCC", [0, 0, 0, 0, 0, 5, 2])
    assert s.type == "cylinder"
    p, v, radius = s.params
    assert v.isEqual(FreeCAD.Vector(0, 0, 50), 1e-9)
    assert radius == pytest.approx(20.0)


def test_rec_elliptic_can():
    # 10-parameter form: minor axis from cross product, minor radius scalar
    s = build_surface("REC", [0, 0, 0, 0, 0, 5, 2, 0, 0, 1])
    assert s.type == "cylinder_elliptic"
    p, v, radii, raxes = s.params
    assert sorted(radii) == pytest.approx([10.0, 20.0])

    # 12-parameter form: both axes as vectors
    s = build_surface("REC", [0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 1, 0])
    p, v, radii, raxes = s.params
    assert sorted(radii) == pytest.approx([10.0, 20.0])


def test_trc_truncated_cone():
    s = build_surface("TRC", [0, 0, 0, 0, 0, 4, 3, 1])
    assert s.type == "cone"
    p, v, r1, r2 = s.params
    assert v.isEqual(FreeCAD.Vector(0, 0, 40), 1e-9)
    assert r1 == pytest.approx(30.0)
    assert r2 == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# GQ / SQ quadric decomposition
# ---------------------------------------------------------------------------


def test_gq_cylinder():
    # x^2 + y^2 - 1 = 0 : cylinder along z, radius 1 cm
    s = build_surface("GQ", [1, 1, 0, 0, 0, 0, 0, 0, 0, -1])
    assert s.type == "cylinder"
    p, v, radius = s.params
    assert parallel(v, (0, 0, 1))
    assert radius == pytest.approx(10.0)


def test_gq_translated_cylinder():
    # (x-1)^2 + (y-2)^2 - 4 = 0
    s = build_surface("GQ", [1, 1, 0, 0, 0, 0, -2, -4, 0, 1])
    assert s.type == "cylinder"
    p, v, radius = s.params
    assert parallel(v, (0, 0, 1))
    assert radius == pytest.approx(20.0)
    assert p.x == pytest.approx(10.0)
    assert p.y == pytest.approx(20.0)


def test_gq_cone():
    # x^2 + y^2 - z^2 = 0 : double cone along z with tan = 1
    s = build_surface("GQ", [1, 1, -1, 0, 0, 0, 0, 0, 0, 0])
    assert s.type == "cone"
    p, v, tan, dblsht = s.params
    assert parallel(v, (0, 0, 1))
    assert tan == pytest.approx(1.0)
    assert dblsht is True


def test_gq_ellipsoid():
    # x^2 + y^2 + 4 z^2 - 4 = 0 : revolution ellipsoid, semi-axes (2, 2, 1) cm
    s = build_surface("GQ", [1, 1, 4, 0, 0, 0, 0, 0, 0, -4])
    assert s.type == "ellipsoid"
    radii = s.params[2]
    assert sorted(radii) == pytest.approx([10.0, 20.0])


def test_gq_elliptic_cylinder():
    # x^2 + 4 y^2 - 4 = 0 : elliptic cylinder along z, semi-axes (2, 1) cm
    s = build_surface("GQ", [1, 4, 0, 0, 0, 0, 0, 0, 0, -4])
    assert s.type == "cylinder_elliptic"
    p, v, radii, raxes = s.params
    assert parallel(v, (0, 0, 1))
    assert sorted(radii) == pytest.approx([10.0, 20.0])


def test_gq_hyperbolic_cylinder():
    # x^2 - y^2 - 1 = 0
    s = build_surface("GQ", [1, -1, 0, 0, 0, 0, 0, 0, 0, -1])
    assert s.type == "cylinder_hyperbolic"
    p, v, radii, raxes = s.params
    assert parallel(v, (0, 0, 1))
    assert sorted(radii) == pytest.approx([10.0, 10.0])


def test_gq_hyperboloid():
    # x^2 + y^2 - z^2 - 1 = 0 : one-sheet hyperboloid along z
    s = build_surface("GQ", [1, 1, -1, 0, 0, 0, 0, 0, 0, -1])
    assert s.type == "hyperboloid"
    p, v, radii, raxes, one_sheet = s.params
    assert parallel(v, (0, 0, 1))
    assert one_sheet
    assert sorted(radii) == pytest.approx([10.0, 10.0])


def test_gq_paraboloid():
    # z = x^2 + y^2 : paraboloid along z, focal distance 1/4 cm
    s = build_surface("GQ", [1, 1, 0, 0, 0, 0, 0, 0, -1, 0])
    assert s.type == "paraboloid"
    p, v, focal = s.params
    assert parallel(v, (0, 0, 1))
    assert focal == pytest.approx(2.5)


def test_sq_sphere():
    # SQ form of unit sphere centered at (1, 2, 3) cm
    s = build_surface("SQ", [1, 1, 1, 0, 0, 0, -1, 1, 2, 3])
    assert s.type == "ellipsoid"
    pos = s.params[0]
    radii = s.params[2]
    assert pos.isEqual(FreeCAD.Vector(10, 20, 30), 1e-6)
    assert sorted(radii) == pytest.approx([10.0, 10.0])


# ---------------------------------------------------------------------------
# TR transformation matrices
# ---------------------------------------------------------------------------


def test_tr_translation_only():
    m = getTransMatrix([1.0, 2.0, 3.0])
    pt = m.multVec(FreeCAD.Vector(0, 0, 0))
    assert pt.isEqual(FreeCAD.Vector(10, 20, 30), 1e-9)


def test_tr_full_rotation():
    # 90 degree rotation about z: x' = y, y' = -x  (rows are the primed axes).
    # The cm -> mm scale applies to the translation part only.
    m = getTransMatrix([0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1])
    pt = m.multVec(FreeCAD.Vector(1, 0, 0))
    assert pt.isEqual(FreeCAD.Vector(0, 1, 0), 1e-9)
    pt = m.multVec(FreeCAD.Vector(0, 1, 0))
    assert pt.isEqual(FreeCAD.Vector(-1, 0, 0), 1e-9)


def test_tr_rotation_with_translation():
    m = getTransMatrix([5, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1])
    pt = m.multVec(FreeCAD.Vector(1, 0, 0))
    assert pt.isEqual(FreeCAD.Vector(50, 1, 0), 1e-9)


def test_tr_two_rows_third_from_cross_product():
    # 6 rotation entries: x' and y' rows, z' completed by cross product
    m = getTransMatrix([0, 0, 0, 0, 1, 0, -1, 0, 0])
    pt = m.multVec(FreeCAD.Vector(0, 0, 1))
    assert pt.isEqual(FreeCAD.Vector(0, 0, 1), 1e-9)
    pt = m.multVec(FreeCAD.Vector(1, 0, 0))
    assert pt.isEqual(FreeCAD.Vector(0, 1, 0), 1e-9)


def test_tr_degrees_form():
    # same 90 degree rotation about z given as angles (in degrees)
    m = getTransMatrix([0, 0, 0, 90, 0, 90, 180, 90, 90, 90, 90, 0], unit="*")
    pt = m.multVec(FreeCAD.Vector(1, 0, 0))
    assert pt.isEqual(FreeCAD.Vector(0, 1, 0), 1e-6)


def test_surface_card_with_tr(tmp_path):
    inp = tmp_path / "tr.mcnp"
    inp.write_text(
        """surface TR test
1 0 -1 imp:n=1
2 0 1 imp:n=0

1 1 so 2

tr1 5 0 0
mode n
"""
    )
    geo = McnpInput(str(inp))
    geo.GetSurfaces()
    sphere = geo.surfaces[1]
    assert sphere.type == "sphere"
    center, radius = sphere.params
    assert center.isEqual(FreeCAD.Vector(50, 0, 0), 1e-6)
    assert radius == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# universe structure, LIKE-BUT cells and error paths
# ---------------------------------------------------------------------------


def make_input(tmp_path, text, name="model.mcnp"):
    f = tmp_path / name
    f.write_text(text)
    return McnpInput(str(f))


def test_level_structure_with_shared_fill(tmp_path):
    # two containers at the same level filling the same universe
    geo = make_input(
        tmp_path,
        """nested universes
1 0 -1 fill=1 imp:n=1
2 0 1 -2 fill=1 imp:n=1
3 0 -4 u=1 fill=2 imp:n=1
4 0 4 -5 u=1 imp:n=1
5 0 -6 u=2 imp:n=1
6 0 6 u=2 imp:n=1
7 0 2 imp:n=0

1 so 10
2 so 20
4 so 3
5 so 5
6 so 1

mode n
""",
    )
    geo.GetSurfaces()
    geo.GetLevelStructure()
    assert geo.levels == {0: (0,), 1: (1,), 2: (2,)}
    assert sorted(geo.Universes[0]) == [1, 2, 7]
    assert sorted(geo.Universes[1]) == [3, 4]
    assert sorted(geo.Universes[2]) == [5, 6]


def test_like_but_trcl_creates_transformed_surfaces(tmp_path):
    geo = make_input(
        tmp_path,
        """like-but cell
1 0 -1 u=1 imp:n=1
2 like 1 but trcl=(5 0 0)
3 0 -10 fill=1 imp:n=1
4 0 10 imp:n=0

1 so 2
10 so 50

mode n
""",
    )
    geo.GetSurfaces()
    geo.GetLevelStructure()
    filtered, surfaces = geo.GetFilteredCells(
        0, -1, {"mat": ("all", None), "cell": ("all", None)}, BoxSettings()
    )
    assert 2 in filtered[1]
    # the like cell must reference a transformed copy of sphere 1
    spheres = [s for s in surfaces.values() if s.type == "sphere"]
    centers = sorted(round(s.params[0].x, 6) for s in spheres)
    assert 50.0 in centers, f"no sphere translated to x=50 mm found: {centers}"


@pytest.mark.xfail(reason="GetCell on a LIKE-BUT cell cannot resolve the referenced cell", strict=False)
def test_get_single_like_but_cell(tmp_path):
    geo = make_input(
        tmp_path,
        """like-but single cell
1 0 -1 imp:n=1
2 like 1 but trcl=(5 0 0)
3 0 1 imp:n=0

1 so 2

mode n
""",
    )
    geo.GetSurfaces()
    cell = geo.GetCell(2, BoxSettings())
    assert cell is not None


def test_unsupported_surface_reference_raises(tmp_path):
    geo = make_input(
        tmp_path,
        """unsupported surface
1 0 -1 -2 imp:n=1
2 0 1 imp:n=0

1 so 10
2 ell 0 0 -2 0 0 2 6

mode n
""",
    )
    geo.GetSurfaces()
    geo.GetLevelStructure()
    with pytest.raises(ValueError, match="references surface 2"):
        geo.GetFilteredCells(0, 0, {"mat": ("all", None), "cell": ("all", None)}, BoxSettings())


def test_select_cells_returns_copy(monkeypatch):
    monkeypatch.setattr(mcnp_input_module, "remove_hash", lambda cells, name: f"geom-{name}")

    class FakeCell:
        def __init__(self):
            self.FILL = None
            self.MAT = 0
            self.geom = None

    cells = {1: FakeCell(), 2: FakeCell()}
    selected = selectCells(cells, {"mat": ("all", None), "cell": ("all", None)})
    assert selected is not cells
    assert selected.keys() == cells.keys()


def test_select_cells_include_exclude(monkeypatch):
    monkeypatch.setattr(mcnp_input_module, "remove_hash", lambda cells, name: f"geom-{name}")

    class FakeCell:
        def __init__(self, mat):
            self.FILL = None
            self.MAT = mat
            self.geom = None

    cells = {1: FakeCell(1), 2: FakeCell(2), 3: FakeCell(3)}

    selected = selectCells(cells, {"mat": ("all", None), "cell": ("include", [1, 3])})
    assert sorted(selected) == [1, 3]

    selected = selectCells(cells, {"mat": ("all", None), "cell": ("exclude", [2])})
    assert sorted(selected) == [1, 3]

    selected = selectCells(cells, {"mat": ("include", [2]), "cell": ("all", None)})
    assert sorted(selected) == [2]

    selected = selectCells(cells, {"mat": ("exclude", [2]), "cell": ("all", None)})
    assert sorted(selected) == [1, 3]
