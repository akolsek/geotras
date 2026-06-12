"""Unit tests for the CSG -> MCNP/OpenMC/Serpent surface writers.

Exercises the surface card generation in geouned.GEOUNED.write.functions for
each surface type, including the quadric (GQ) fallbacks for non axis-aligned
surfaces, the 80-column line wrapping, and the hard errors raised when a used
surface cannot be expressed in the output format.
"""

import io
import math
from types import SimpleNamespace

import pytest

import geouned
import FreeCAD

from geouned.GEOUNED.write.functions import (
    cut_line,
    mcnp_surface,
    open_mc_surface,
    serpent_surface,
    write_sequence_mcnp,
    write_sequence_serpent,
)
from geouned.GEOUNED.write.mcnp_format import McnpInput as McnpWriter
from geouned.GEOUNED.utils.boolean_function import BoolSequence

OPT = geouned.Options()
TOL = geouned.Tolerances()
NF = geouned.NumericFormat()


def plane(axis, position):
    return SimpleNamespace(Axis=FreeCAD.Vector(axis), Position=FreeCAD.Vector(position), pointDef=False)


def cylinder(axis, center, radius):
    return SimpleNamespace(Axis=FreeCAD.Vector(axis), Center=FreeCAD.Vector(center), Radius=radius)


def cone(axis, apex, semi_angle):
    return SimpleNamespace(Axis=FreeCAD.Vector(axis), Apex=FreeCAD.Vector(apex), SemiAngle=semi_angle)


def sphere(center, radius):
    return SimpleNamespace(Center=FreeCAD.Vector(center), Radius=radius)


def torus(axis, center, major, minor):
    return SimpleNamespace(
        Axis=FreeCAD.Vector(axis),
        Center=FreeCAD.Vector(center),
        MajorRadius=major,
        MinorRadius=minor,
        Degenerated=False,
    )


def card_values(card, skip=2):
    """Numeric values of a (possibly multi-line) surface card."""
    return [float(v) for v in card.split()[skip:]]


def gq_value(coeffs, x, y, z):
    a, b, c, d, e, f, g, h, j, k = coeffs
    return a * x * x + b * y * y + c * z * z + d * x * y + e * y * z + f * z * x + g * x + h * y + j * z + k


# ---------------------------------------------------------------------------
# MCNP surface cards
# ---------------------------------------------------------------------------


def test_mcnp_axis_planes():
    out = mcnp_surface(1, "Plane", plane((0, 0, 1), (0, 0, 30)), OPT, TOL, NF)
    assert out.split()[1] == "PZ"
    assert card_values(out) == pytest.approx([3.0])  # 30 mm -> 3 cm

    out = mcnp_surface(1, "Plane", plane((1, 0, 0), (70, 0, 0)), OPT, TOL, NF)
    assert out.split()[1] == "PX"
    assert card_values(out) == pytest.approx([7.0])


def test_mcnp_general_plane():
    n = FreeCAD.Vector(1, 1, 0)
    n.normalize()
    out = mcnp_surface(1, "Plane", plane(n, (10, 0, 0)), OPT, TOL, NF)
    assert out.split()[1] == "P"
    a, b, c, d = card_values(out)
    # point (1,0,0) cm must satisfy a x + b y + c z = d
    assert a * 1.0 + b * 0.0 + c * 0.0 == pytest.approx(d, abs=1e-9)


def test_mcnp_cylinders():
    out = mcnp_surface(2, "Cylinder", cylinder((0, 0, 1), (0, 0, 0), 20), OPT, TOL, NF)
    assert out.split()[1] == "CZ"
    assert card_values(out) == pytest.approx([2.0])

    out = mcnp_surface(2, "Cylinder", cylinder((0, 0, 1), (10, 20, 0), 20), OPT, TOL, NF)
    assert out.split()[1] == "C/Z"
    assert card_values(out) == pytest.approx([1.0, 2.0, 2.0])


def test_mcnp_skew_cylinder_gq_quadric():
    # cylinder along (1,1,0)/sqrt(2) through origin, radius 1 cm
    out = mcnp_surface(3, "Cylinder", cylinder((1, 1, 0), (0, 0, 0), 10), OPT, TOL, NF)
    assert out.split()[1] == "GQ"
    coeffs = card_values(out)
    assert len(coeffs) == 10
    u = 1 / math.sqrt(2)
    # points on the cylinder surface (cm units) must satisfy the quadric
    on_surface = [(0, 0, 1), (0, 0, -1), (u, u, 1), (u, -u, 0)]
    for pt in on_surface:
        assert gq_value(coeffs, *pt) == pytest.approx(0.0, abs=1e-9), pt
    # the axis (inside) must be negative as for all MCNP cylinders
    assert gq_value(coeffs, 0, 0, 0) < 0
    assert gq_value(coeffs, u, u, 0) < 0


def test_mcnp_cones():
    out = mcnp_surface(4, "Cone", cone((1, 0, 0), (10, 0, 0), math.atan(1.0)), OPT, TOL, NF)
    assert out.split()[1] == "KX"
    x, t2, sheet = card_values(out)
    assert x == pytest.approx(1.0)
    assert t2 == pytest.approx(1.0)
    assert sheet == 1

    out = mcnp_surface(4, "Cone", cone((0, 0, -1), (0, 0, 50), math.atan(0.5)), OPT, TOL, NF)
    assert out.split()[1] == "KZ"
    z, t2, sheet = card_values(out)
    assert z == pytest.approx(5.0)
    assert t2 == pytest.approx(0.25)
    assert sheet == -1

    out = mcnp_surface(4, "Cone", cone((0, 0, 1), (10, 20, 30), math.atan(1.0)), OPT, TOL, NF)
    assert out.split()[1] == "K/Z"


def test_mcnp_skew_cone_gq_quadric():
    # cone apex at origin, axis (1,1,0)/sqrt(2), tan = 0.5
    out = mcnp_surface(5, "Cone", cone((1, 1, 0), (0, 0, 0), math.atan(0.5)), OPT, TOL, NF)
    assert out.split()[1] == "GQ"
    coeffs = card_values(out)
    s2 = math.sqrt(2)
    # |P|^2 - (1 + t^2) (P.u)^2 = 0 on the cone (cm units)
    on_surface = [(s2, s2, 1), (s2, s2, -1), (-s2, -s2, 1)]
    for pt in on_surface:
        assert gq_value(coeffs, *pt) == pytest.approx(0.0, abs=1e-6), pt


def test_mcnp_spheres():
    out = mcnp_surface(6, "Sphere", sphere((0, 0, 0), 5), OPT, TOL, NF)
    assert out.split()[1] == "SO"
    assert card_values(out) == pytest.approx([0.5])

    out = mcnp_surface(6, "Sphere", sphere((10, 20, 30), 5), OPT, TOL, NF)
    assert out.split()[1] == "S"
    assert card_values(out) == pytest.approx([1.0, 2.0, 3.0, 0.5])


def test_mcnp_torus():
    out = mcnp_surface(7, "Torus", torus((0, 0, 1), (0, 0, 50), 100, 20), OPT, TOL, NF)
    assert out.split()[1] == "TZ"
    assert card_values(out) == pytest.approx([0.0, 0.0, 5.0, 10.0, 2.0, 2.0])


def test_mcnp_skew_torus_has_no_representation():
    out = mcnp_surface(7, "Torus", torus((1, 1, 0), (0, 0, 0), 100, 20), OPT, TOL, NF)
    assert out == ""


def test_mcnp_surface_lines_fit_80_columns():
    surfs = [
        ("Cylinder", cylinder((1, 2, 3), (123.456, -78.9, 4567.8), 123.456)),
        ("Cone", cone((3, -2, 1), (-987.6, 543.2, -1.234), math.atan(0.789))),
    ]
    for stype, surf in surfs:
        out = mcnp_surface(999999, stype, surf, OPT, TOL, NF)
        for line in out.split("\n"):
            assert len(line) <= 80


def test_cut_line_terminates_on_long_tail():
    # regression: used to loop forever when the last space was past the limit
    line = "1      GQ  " + "1.2345678E+00 " * 7 + "9.9999999E+99"
    out = cut_line(line, 80)
    assert all(len(l) <= 80 for l in out.split("\n"))
    assert out.replace("\n", " ").split() == line.split()


# ---------------------------------------------------------------------------
# OpenMC surfaces
# ---------------------------------------------------------------------------


def test_openmc_xml_surfaces():
    name, coeffs = open_mc_surface("Plane", plane((0, 0, 1), (0, 0, 30)), TOL, NF)
    assert name == "z-plane"
    assert [float(v) for v in coeffs.split()] == pytest.approx([3.0])

    name, coeffs = open_mc_surface("Cylinder", cylinder((0, 0, 1), (10, 20, 0), 20), TOL, NF)
    assert name == "z-cylinder"
    assert [float(v) for v in coeffs.split()] == pytest.approx([1.0, 2.0, 2.0])

    name, coeffs = open_mc_surface("Sphere", sphere((10, 20, 30), 5), TOL, NF)
    assert name == "sphere"

    name, coeffs = open_mc_surface("Cone", cone((0, 1, 0), (0, 10, 0), math.atan(0.5)), TOL, NF)
    assert name == "y-cone"

    name, coeffs = open_mc_surface("Torus", torus((1, 0, 0), (0, 0, 0), 100, 20), TOL, NF)
    assert name == "x-torus"


def test_openmc_skew_torus_returns_none():
    name, coeffs = open_mc_surface("Torus", torus((1, 1, 0), (0, 0, 0), 100, 20), TOL, NF)
    assert name is None


def test_openmc_py_surfaces():
    name, coeffs = open_mc_surface("Sphere", sphere((10, 20, 30), 5), TOL, NF, out_xml=False)
    assert name == "Sphere"
    assert "r=0.5" in coeffs


# ---------------------------------------------------------------------------
# Serpent surfaces (regression tests for dropped/malformed cards)
# ---------------------------------------------------------------------------


def test_serpent_cylz_keeps_position():
    out = serpent_surface(3, "Cylinder", cylinder((0, 0, 1), (10, 20, 0), 5), OPT, TOL, NF)
    fields = out.split()
    assert fields[:3] == ["surf", "3", "cylz"]
    assert [float(v) for v in fields[3:]] == pytest.approx([1.0, 2.0, 0.5])


def test_serpent_cone_card_id_position_and_sheet():
    out = serpent_surface(4, "Cone", cone((0, 0, -1), (0, 0, 50), math.atan(0.5)), OPT, TOL, NF)
    fields = out.split()
    assert fields[:3] == ["surf", "4", "ckz"]
    values = [float(v) for v in fields[3:]]
    assert values == pytest.approx([0.0, 0.0, 5.0, 0.25, -1.0])


def test_serpent_skew_cone_emits_quadratic():
    out = serpent_surface(5, "Cone", cone((1, 1, 0), (0, 0, 0), math.atan(0.5)), OPT, TOL, NF)
    fields = out.split()
    assert fields[:3] == ["surf", "5", "quadratic"]
    assert len(fields[3:]) == 10


def test_serpent_skew_cylinder_emits_quadratic_with_id():
    out = serpent_surface(6, "Cylinder", cylinder((1, 1, 0), (0, 0, 0), 10), OPT, TOL, NF)
    fields = out.split()
    assert fields[:3] == ["surf", "6", "quadratic"]


def test_serpent_writer_does_not_mutate_surface_axis():
    surf = cylinder((0, 0, 2), (0, 0, 0), 5)
    serpent_surface(3, "Cylinder", surf, OPT, TOL, NF)
    assert surf.Axis.isEqual(FreeCAD.Vector(0, 0, 2), 1e-12)


# ---------------------------------------------------------------------------
# boolean sequence writers
# ---------------------------------------------------------------------------


def test_write_sequences_nested():
    seq = BoolSequence("1 (2:-3) 4")
    mcnp = write_sequence_mcnp(seq)
    serpent = write_sequence_serpent(seq)
    assert "2:-3" in mcnp
    assert "2:-3" in serpent


# ---------------------------------------------------------------------------
# unwritable used surfaces are now hard errors
# ---------------------------------------------------------------------------


def make_mcnp_writer(surface_table):
    writer = object.__new__(McnpWriter)
    writer.options = OPT
    writer.tolerances = TOL
    writer.numeric_format = NF
    writer.surfaceTable = surface_table
    writer.inpfile = io.StringIO()
    return writer


def test_used_unwritable_surface_raises():
    writer = make_mcnp_writer({7: {0}})
    surface = SimpleNamespace(Index=7, Type="Torus", Surf=torus((1, 1, 0), (0, 0, 0), 100, 20))
    with pytest.raises(RuntimeError, match="Surface 7"):
        writer.write_surfaces(surface)


def test_unused_unwritable_surface_is_skipped():
    writer = make_mcnp_writer({})
    surface = SimpleNamespace(Index=7, Type="Torus", Surf=torus((1, 1, 0), (0, 0, 0), 100, 20))
    writer.write_surfaces(surface)  # no raise
    assert writer.inpfile.getvalue() == ""


def test_writable_surface_is_written():
    writer = make_mcnp_writer({7: {0}})
    surface = SimpleNamespace(Index=7, Type="Sphere", Surf=sphere((0, 0, 0), 5))
    writer.write_surfaces(surface)
    assert "SO" in writer.inpfile.getvalue()
