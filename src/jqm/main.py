from .water_geo import load_water_geometry
from .qm_models.tight_binding import TightBinding

def test_sim():
    coords, charges = load_water_geometry()

    model = TightBinding(coords, charges)
    print("TB energy:", model.energy())
    print("TB forces:", model.forces())

    # model = HartreeFock(coords, charges)
    # print("HF energy:", model.energy())
