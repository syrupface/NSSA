""" Starter workflows: a few ready-made sequences of steps for common jobs, for users new to AFM image processing.
A workflow applies ordinary steps to the shown channel (they appear in the History and can be undone, removed or
saved as a macro), then optionally opens the result: a tab, the grain analysis or a tool. No GUI code. """

import dataclasses

WORKFLOWS = []


@dataclasses.dataclass
class Workflow:
    key: str
    title: str
    description: str
    steps: list                     # (operation, parameters) applied to the shown channel, in order
    finish: str = ""                # what is shown afterwards: 'statistics', 'grains', 'lattice', 'combine', 'psd'
    channel_hint: str = ""          # a channel name pattern this workflow is meant for (regex, case-insensitive)


def _add(*args, **kwargs):
    WORKFLOWS.append(Workflow(*args, **kwargs))


_add("topography", "Topography: flatten and clean",
     "For a height image: aligns the scan lines and removes the tilt, repairs bad scan lines and scars, and sets the "
     "lowest point to zero. A good start for most topography images.",
     [("level.flatten", {}),
      ("glitch.auto_lines", {}),
      ("glitch.scars", {}),
      ("level.shift", {"mode": "min"})])

_add("roughness", "Roughness (ISO 25178)",
     "Removes the form (a plane after aligning the scan lines) and shows the areal roughness parameters (Sa, Sq, Sz, "
     "Ssk, Sku, Sal, Str, ...) in the Statistics tab. No filter is applied, so the values describe the whole image at "
     "its pixel size: compare images only at equal scan size and pixels.",
     [("level.flatten", {})],
     finish="statistics")

_add("grains", "Particles and grains",
     "For particles on a flat substrate: flattens, masks the particles with an automatic (Otsu) threshold, flattens "
     "again on the substrate only (particles excluded), puts the substrate at zero, and runs the grain analysis "
     "(count, sizes, heights). Check the mask; adjust it in the Mask tab and analyze again if needed.",
     [("level.flatten", {}),
      ("mask.otsu", {"source": "self", "side": "above", "combine": "new"}),
      ("level.flatten", {"fit_on": "exclude"}),
      ("level.shift", {"mode": "median", "fit_on": "exclude"}),
      ("mask.remove_small", {"min_pixels": 5})],
     finish="grains")

_add("psd", "Surface spectrum (PSD)",
     "Flattens the image and shows the power spectral density with a fit of the ABC model and the noise floor: the "
     "correlation length, the slope and the effective lateral resolution.",
     [("level.flatten", {})],
     finish="psd")

_add("lateral", "Lateral force (friction)",
     "For Lateral Force mode: aligns the scan lines of the forward and backward friction images, then opens the channel "
     "combination with (forward - backward) / 2, the friction signal; press 'Find shift' there to align the two scan "
     "directions.",
     [("level.align_rows", {"method": "median"})],
     finish="combine", channel_hint=r"friction|lateral|torsion")

_add("lattice", "Periodic structure (lattice)",
     "For atomic lattices, moiré patterns or 2D gratings: flattens and opens the lattice measurement, which finds the "
     "two lattice vectors, their lengths and angle from the autocorrelation.",
     [("level.flatten", {})],
     finish="lattice")


def get(key: str) -> Workflow:
    for workflow in WORKFLOWS:
        if workflow.key == key:
            return workflow
    raise KeyError(key)
