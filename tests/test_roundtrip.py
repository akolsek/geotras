"""Round-trip validation: CAD (STEP) -> CSG -> CAD.

Converts a STEP model to MCNP and OpenMC XML with GEOUNED, rebuilds the CAD
geometry from the generated CSG input with GEOReverse, and compares the solid
volumes of the reconstruction against the original model. This validates the
geometric consistency of the two translation directions together.
"""

import re
from pathlib import Path

import pytest

import geouned
import Part

INPUT_STEP = Path("testing/inputSTEP/cylBox.stp")
OUTPUT_DIR = Path("tests_outputs/roundtrip")


@pytest.fixture(scope="module")
def forward_conversion():
    """Convert the STEP file to MCNP and OpenMC XML once for all tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / INPUT_STEP.stem

    geo = geouned.CadToCsg()
    geo.load_step_file(filename=str(INPUT_STEP.resolve()), skip_solids=[])
    geo.start()
    geo.export_csg(
        title="round trip test",
        geometryName=str(stem.resolve()),
        outFormat=("mcnp", "openmc_xml"),
        cellCommentFile=False,
        cellSummaryFile=False,
    )

    mcnp_file = stem.with_suffix(".mcnp")
    header = mcnp_file.read_text()
    n_solid_cells = int(re.search(r"Solid Cells\s*:\s*(\d+)", header).group(1))

    original = Part.Shape()
    original.read(str(INPUT_STEP))
    original_volumes = sorted(s.Volume for s in original.Solids)

    return stem, n_solid_cells, original_volumes


@pytest.mark.parametrize("csg_format", ["mcnp", "openmc_xml"])
def test_roundtrip_volumes(forward_conversion, csg_format):
    stem, n_solid_cells, original_volumes = forward_conversion
    suffix = ".mcnp" if csg_format == "mcnp" else ".xml"

    reverse = geouned.CsgToCad()
    reverse.read_csg_file(input_filename=str(stem.with_suffix(suffix)), csg_format=csg_format)
    # solid cells are numbered first, starting at 1
    reverse.cell_filter("include", list(range(1, n_solid_cells + 1)))
    reverse.build_universe()
    out_stem = OUTPUT_DIR / f"{INPUT_STEP.stem}_rebuilt_{csg_format}"
    reverse.export_cad(output_filename=str(out_stem))

    rebuilt = Part.Shape()
    rebuilt.read(str(out_stem.with_suffix(".stp")))
    rebuilt_volumes = sorted(s.Volume for s in rebuilt.Solids)

    assert len(rebuilt_volumes) == n_solid_cells == len(original_volumes)
    for vol_orig, vol_rebuilt in zip(original_volumes, rebuilt_volumes):
        assert vol_rebuilt == pytest.approx(vol_orig, rel=1e-3), (
            f"solid volume changed in {csg_format} round trip: {vol_orig} -> {vol_rebuilt}"
        )
