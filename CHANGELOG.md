# Change log

## 1.1.0 (not yet released)

Needs NSFopen 2.5.0.

- **Tools - Curvature:** principal radii of curvature and their directions, the center and the bow of a surface from a
  quadratic fit (on all pixels, the masked ones or the rest), with the profiles along both directions; *Subtract the fit*.
  The center can be set by hand (click, drag or type it): the fit is then made with its vertex there.
- **Tools - Terrace analysis:** the terraces of a stepped surface, their heights and the common step height, with the
  background (tilt, bow) fitted together with the terraces. Steps found from the slopes or drawn into the mask; a known
  step height can be given. *Level the image* adds the operation *Terrace level* (undo, macros, batch); the table can be
  copied or saved as CSV.
- **Spectroscopy - Maugis-Dugdale** contact model (sphere, retract curve): the modulus, the work of adhesion and the
  Maugis parameter λ (with the Tabor parameter and the adhesive stress), between DMT and JKR.
- **Spectroscopy - thin samples (bottom effect):** Hertz, Sneddon and DMT can correct for the hard substrate under a
  thin sample: one thickness for all curves, or (maps) the height of every curve above the substrate, taken from the
  masked curves. Bonded or not bonded; sphere, cone, pyramid and cylinder punch.
- **Tools - Curvature:** *Highest point* / *Lowest point* buttons for the center.
- **Zoom with a box** in every graph (profile, histogram, PSD, force and I-V curves, statistics): drag with the left
  button; a flat box zooms along x only, a narrow one along y only. The right or middle button pans; images and maps
  still pan with the left button.
- **Spectroscopy - Basic / Advanced:** Basic shows only File, Curve, Calibration, Contact model (I-V: Analysis) and
  Analyze. The switch is shared with the Image page.
- **Spectroscopy:** *All curves* is split into the groups *Analyze* and *Export*.
- **Spectroscopy:** *Fit regions* renamed *Model regions* (files `_model_regions.json`, format `nssa-model-regions`;
  older region files still open).

## 1.0.0 (November 4, 2026)

First public version.
