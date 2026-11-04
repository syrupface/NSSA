# NSF Surface and Spectroscopy Analyzer (NSSA) - User Manual

NSSA, the NSF Surface and Spectroscopy Analyzer (formerly Nanosurf Surface Analyzer and Nanosurf Img), processes and analyzes the **images** and the
**spectroscopy** (force curves, force maps, I-V curves) of Nanosurf instruments (`.nid` from Mobile S / Easyscan, `.nhf` from Nanosurf Studio), and images
saved by Gwyddion (`.gwy`). It is an independent project, not a product of Nanosurf AG; it reads Nanosurf's file formats with NSFopen. Every step is
a recorded operation: it can be undone, saved as a *macro* and replayed on other files, also automatically.

**Contents**: 1 Quick start (1a Basic and Advanced, live preview, starter workflows) - 2 Window - 3 Conventions - 4 Level - 5 Mask - 6 Filter - 7 Glitch correction -
8 Rescale (8a Open-loop XY correction from the position sensors, 8b Resample pixels) - 9 Statistics - 10 Line cut -
11 Spectra - 12 Grains (tool) - 13 Stitching - 13a Combine channels (lateral force) - 13b 3D view - 13c Lattice - 14 Macros and batch - 15 Export (15a Report) -
16 Files and limits - 17 Spectroscopy (force curves, force maps, I-V curves)

---

## 1. Quick start

1. Start the program: the stand-alone program from the Start menu (*NSF Surface and Spectroscopy Analyzer*) or
   `NSSA.exe [file]`, or from the sources `python app_NSSA/main.py` (or `main.py <file>`). To try the program without your own data, use *File - Open example*: NSFopen's example files (sapphire with atomic terraces for roughness, a 10 x 10 force map for the Spectroscopy page, a lateral force image of SBS-PMMA, MoS2, a calibration grid and test patterns for plane removal); they come with NSFopen and with the stand-alone program, and the tutorials use them. Click **Open**, choose a `.nid`, `.nhf` or `.gwy` file, or simply
   **drop the file on the window**; *File - Open recent* lists the last 12 files. The arrow buttons step through the image
   files of the folder. A macro file (`.json`) dropped on the window is applied to the open image.
2. Pick a channel in the drop-down above the image. Closed-loop height (`Z-Axis Sensor`, `Topography`) is shown first.
3. New to AFM processing? Choose a **Starter workflow** under *Quick start* (left) and press **Run**, e.g.
   *Topography: flatten and clean*. The steps appear in *History*, where they can be undone or removed one by one.
   Or process with the tabs on the left (usually: *Level* -> *Glitch* -> *Filter*). Every action appears in *History*.
4. Read the numbers in *Statistics*, draw a *Line cut*, then **Export**.
5. To process more files the same way: **Macro - Save steps as macro**, then apply it to other files or switch on
   *Apply automatically to opened files* (section 14).

### 1a. Basic and Advanced, live preview, starter workflows

- **Basic / Advanced** (buttons above *Quick start*, also *Settings - Advanced mode*): **Basic** shows the common steps with
  good defaults, e.g. on the Level tab *Flatten*, *Level* (plane), *Align rows* and *Zero level*; the Mask tab without the
  outlier and source options; the Filter tab with the smoothing and morphology filters only; and hides the 2D Spectrum and
  Functions tabs and the open-loop sensor correction. **Advanced** shows every method and setting. Both modes apply the
  same operations; hidden settings keep their values. The choice is remembered.
- **Flatten** (Level tab, recommended first step): aligns the scan lines (median of the differences between neighboring
  lines) and subtracts a plane. One step in the History.
- **Live preview** (check box next to the mode buttons, also in the Settings menu): changing a setting in a panel shows the
  result of that group's step on the image, marked *Preview* in its upper left corner, **without applying it**. The button
  of the group applies it; **Esc**, another tab or another channel returns to the processed image. Steps that need clicks
  on the image (3-point, shapes) have no preview. Large images with slow steps (rolling ball, large median) may take a
  moment; the status bar says so, and the preview can be switched off.
- **Starter workflows** (*Quick start* box, also *Macro - Starter workflows*): ready-made sequences of ordinary steps on
  the shown channel; the text under the list says what each does.

| Workflow | Steps | Then |
|---|---|---|
| Topography: flatten and clean | Flatten, repair bad lines, remove scars, minimum to zero | |
| Roughness (ISO 25178) | Flatten (form removal; no filter) | the Statistics tab (Sa, Sq, ..., Sal, Str) |
| Particles and grains | Flatten, Otsu mask, flatten again without the particles, substrate median to zero, remove masked areas < 5 px | grain analysis |
| Surface spectrum (PSD) | Flatten | the PSD tab with the ABC + noise floor fit |
| Lateral force (friction) | Align rows of the forward and backward friction channels | the channel combination ((F - B) / 2) |
| Periodic structure (lattice) | Flatten | the lattice measurement |

## 2. Window

| Area | Content |
|---|---|
| Left tabs | **Level**, **Mask**, **Filter**, **Glitch**, **Rescale**, **History**: parameters and *Apply* buttons |
| Image | The channel with color bar and mask overlay (red). Above it: the channel and the **tools** (below); the *Min* / *Max* fields appear there when the color range is manual. The axes count from **0 at the lower left** |
| Theme | *Settings - Theme*: **System** (the window colors follow Windows' light or dark mode; the graphs are dark), **Dark** (the whole program dark, whatever Windows uses) or **Light** (light windows and light graphs). Applied at once and remembered. In every theme selections and checked buttons use the **Nanosurf orange** (#ea390d, darker #9c390d under the mouse), following the Nanosurf corporate design guidelines 2024. Exports and reports are always black on white |
| Settings menu | How the image is shown: **Color map** (*Sky* is the default of height images, *viridis* of the others; one alphabetical list of matplotlib's maps (Afmhot, Jet, Viridis, ...) and the Nanosurf maps (Caribbean, Cold, Digitalis, Flare, Gold, Maple, Neon, Plum, Sky, Spectrum, Strata, Warm, BlueBlackRed: the same as NSFopen's), each with a small preview, and a wide preview when the mouse rests on an entry; the submenu holding the chosen map shows its name; also in the 3D view. *New or edit color map...* opens the color map editor (section 2a). Your own maps: gradient files (Gwyddion's format, see the README in the folder) in the folder `app_NSSA/colormaps`, in the stand-alone program in the folder `colormaps` next to `NSSA.exe`; they appear under *Your color maps*. If Gwyddion is installed (Windows, macOS or Linux), its gradients appear under *Gwyddion (installed)*, a name that is taken with ' (Gwyddion)'; nothing of Gwyddion comes with the program. A Gwyddion in an unusual place is found with the environment variable GWYDDION_DATADIR (the folder holding `gwyddion/gradients`). Names of earlier versions (Pm3d, Blend2, Code-V) open as Flare, Strata and Spectrum), **Color range** (1-99 %, 0.1-99.9 %, full range, ± 3 σ, or manual: starts from the range shown, then type Min / Max above the image), **Show mask**, **Scale bar** (a scale bar instead of the axis numbers, also in the exported PNG), **Scale bar color** (white or black), **Scale bar background** (*Outline*, default: a thin edge in the opposite color around bar and text, readable on any image; *Box*: a half transparent box in the opposite color behind them; *Plain*) and **Axis labels...** (names of the x and y axis, the color bar and the title of the image). Mask and the scale bar settings are remembered for the next session |
| Renaming labels | **Double-click** an axis label, a title or a color bar label in **any graph** (image, profile, histogram, PSD, ...) to rename it. The unit is added automatically and follows the data ("Distance" becomes "Distance [nm]"); type a unit in brackets to fix it yourself; an empty name restores the default. In the image the x and y names hold for all channels, the color bar name and title for the channel they were given to. The exported PNG uses the names |
| Several channels | **Layout** button above the image (also *Settings - Image layout*): *One image*, *Two side by side* or *Four (2 x 2)*. The main image keeps the tools; each extra image has its own channel list (other channels of the same scan direction are chosen first) and the color map of its kind of data. Images of the same size share **zoom and pan**, and the mouse over one shows a **cross** at the same place on the others and the **values of all shown channels** in the status bar. The layout is remembered |
| Channel thumbnails | A strip of small pictures of all channels above the image (they follow the processing); **click one** to show it. Keys: **Page Down / Page Up** next / previous channel, **F / B** the forward / backward scan of the same channel. Turn the strip off in the *Layout* menu |
| Line profile (top right) | The line cut of the Line cut tool, with width averaging |
| Histogram (bottom left) | All pixels, and the masked pixels in red; fits of Gaussians and other peaks (section 9a). *Bin rule*: *Manual* (type the number of *Bins*) or a rule that picks it from the data: *Auto* (the larger of Sturges and Freedman-Diaconis), *Freedman-Diaconis* (bin width 2 IQR / n^(1/3), robust against outliers), *Scott* (3.5 σ / n^(1/3)), *Doane* (Sturges corrected for skewed data), *Sturges* (log2 n + 1, for small images), *Rice* (2 n^(1/3)), *Square root* (√n), *Stone* (minimizes an estimate of the error). *Image colors*: bars colored with the color map and color range of the image |
| Bottom right tabs | **Statistics**, **2D Spectrum**, **PSD**, **Bearing**, **Functions**, **File Info** |
| Edit menu | **Undo** (Ctrl+Z), **Redo** (Ctrl+Y or Ctrl+Shift+Z: applies again what Undo took back, until a new step is made), **Show original** (Ctrl+B, also the *Original* button above the image: the image before any processing, to compare; again to return), Reset channel, Reset image zoom, Clear line cut (Ctrl+L), **Remove mask** (Ctrl+Shift+M: a *Clear mask* step, so it can be undone), **Duplicate channel** (a copy of the shown channel with its processing so far and its mask, named e.g. *Z-Axis Sensor copy (Forward)*; it is processed on its own from then on, e.g. to compare two leveling methods side by side; Undo removes it, it is part of macros, and the original cannot be removed while the copy exists; also on the thumbnail right-click menu), **Remove channels...** (check the channels you do not need: they and their steps leave the file in this program, so exports, the 3D view, macros and the panes show only the others; not the last one, and not one that another step or a combined channel uses), **Restore removed channels** (the channels of the file come back as recorded, with crop / flip / resample that were applied to all channels, but without their own steps). Also from a right-click on the channel thumbnails |
| Tools menu | **Batch processing**, **Stitch images**, **Combine channels (lateral force)**, **Grain analysis** (Ctrl+G, section 12), **Measure lattice** (section 13c) |
| Pages | **Image** (this manual, sections 1-16), **Spectroscopy** (force curves, force maps and I-V curves, section 17), **3D View**, **Help** |
| Help menu | **User manual** (F1, the *Help* page) and **About** (version of the program, NSFopen, Qt and the other libraries; please give it with any error report) |
| Mouse in the graphs | In **every graph** (image, profile, histogram, PSD, ...): the **mouse wheel zooms** at the cursor (also on logarithmic axes; never further out than the whole image or graph, and panning stops at its edges; the scale bar follows the zoom with a suitable length and unit), **dragging pans** (left button where it has no other use: the *Navigate* tool on the image, empty areas of the other graphs; right or middle button always), a **double-click** on an empty spot (or Home) shows the whole graph again, a **right-click** opens a menu: *Reset view*, *Save image...* (PNG, SVG, PDF), *Copy image*, and on the image *Clear line cut*, *Clear mask* (an undoable step), *Show mask*, and *Scale bar*; on the profile *Clear line cut* |
| Pages | **Image**, **3D View** (section 13b), **Help** (this text) |

**Tools** above the image, in groups separated by lines: *Navigate* | *Measure* (Line cut) | *Level* (3-point) |
*Mask* (Rect, Ellipse, Polygon, Freehand, Fill: icons only, hold the mouse over one for its name) | *Geometry* (Crop) |
*Repair* (Row). One is active; the status bar says what it expects. *Navigate*: zoom with the wheel, drag to pan:

| Tool | Use |
|---|---|
| Line cut | Drag a line; the profile is shown. The **arrow** next to the button chooses the kind of line: *Free line*, *Horizontal line*, *Vertical line* or *Horizontal and vertical* (keys **H**, **V**, **X** while the tool is active; the same key again goes back to a free line). Horizontal and vertical lines run over the whole image and are placed by a click or by dragging (section 10). To refine it, drag one of its ends or the line itself (also in Navigate mode), or type the coordinates under the profile. Used for the lateral calibration and the critical dimension, too |
| Rect, Ellipse, Freehand | Drag to edit the mask (Rect and Ellipse: hold **Ctrl** for a square or circle); the shape is combined with the mask as chosen in the Mask tab (add, remove, replace, intersect, xor) |
| Polygon | Click corners; double-click, right-click or Enter closes it; Esc cancels |
| Fill | Click a pixel: masks the connected region whose height differs from the clicked pixel by less than the tolerance of the Mask tab |
| Crop | Drag the region to keep; hold **Ctrl** to keep it **square** (equal width and height in µm). While dragging, the size is shown next to the rectangle (in µm or nm and in pixels; also for Rect and Ellipse). **Crop, Flip, Rotate and lateral Rescale are applied to all channels of the file** so that they stay the same size |
| Row | Click to add the row (or column) to the list of the Glitch tab |
| 3-point | Click three points on the substrate: the plane through them is subtracted and they become zero. Each point is averaged over a square of *3-point: point size* pixels (Level tab, default 3; 1 = the clicked pixel; larger squares average out noise), drawn around the clicked points. Another tool cancels unfinished points |

### 2a. Color map editor (*Settings - Color map - New or edit color map...*, *Tools - Color map editor...*)

Makes a new color map or changes one of yours, starting from any map (*Start from*). The bar shows the gradient with
its **anchors** (triangles) below and the **histogram of the current image** above, over the color range of the image,
so you see which colors fall on which heights. **Drag an anchor** to move its color, e.g. pull the anchors together
over the heights of a step or a particle to spread more colors there; the arrow keys move the selected anchor in small
steps (Shift: larger). **Click on the bar** to add an anchor (with the color found there), **double-click** an anchor
(or *Color...*) to choose its color, **right-click** or *Delete* to remove it; the two ends stay at 0 and 100 %. *Anchor
at* shows the position in % and the height it stands for. *Even spacing* spreads the anchors evenly, *Spread over the
image* puts them at equal shares of the pixels (histogram equalization: every color covers the same area, small height
differences show up), *Reverse* turns the map around, *Reset* starts again. The preview on the right shows the image
with the map as it is edited.

**Save and use** writes the map as a gradient file into the folder `colormaps` (Gwyddion's format), applies it to the
image and lists it under *Your color maps* (also in the 3D view). The built-in maps are not changed: an edited built-in
map is saved under a new name (default *Sky custom*). One of your maps can be saved again under its name, or deleted
with *Delete this map*. matplotlib's maps (Viridis, ...) are tables of 256 colors; the editor starts from 9 anchors
sampled from them.

## 3. Conventions

- Data are in SI units (m, V, degree); the display picks a readable unit (nm, mV, ...).
- Row 0 is the **top** of the picture. Scans with direction *Up* store the first (bottom) line first, so the reader flips
  them; forward and backward images have the same orientation.
- The **axes** of the image count from 0 at the **lower left**: x to the right, y upwards (the centre of the lower left
  pixel is (0, 0)). Lattice directions and angles use the same axes. The cursor read-out under the image gives x and y
  this way and, in brackets, the pixel (column, row) with row 0 at the top.
- Pixel coordinates are (x = column, y = row); pixel centers are integers.
- NaN pixels (invalid samples) are ignored by all operations and statistics.
- **Masks.** *Fit* operations (levels, shift) can use *All pixels*, *Exclude masked pixels* (typical: mask particles, then
  level the substrate) or *Only masked pixels*. *Filter* and *Glitch* operations can replace the *whole image*, only the pixels
  *inside* or only those *outside* the mask. Statistics can be taken over all, inside or outside the mask.
- Operations that use **another channel** (mask from a threshold of the phase, rescale to a reference) use the *current
  processed* state of that channel. Process the reference as you process the channel, or you compare different things.

## 4. Level tab

| Operation | What it does |
|---|---|
| Polynomial / plane | Least-squares 2D polynomial of total degree n (1 = plane) in normalized coordinates, subtracted. Turn off *Same order in x and y* (Advanced) to give each direction its own order: *Order in x* and *Order in y* (for example 1 in x and 2 in y removes a bow along the slow scan axis and only a tilt along x; the terms are x^i y^j with i up to the x order, j up to the y order and i + j up to the larger of the two). *Outlier rejection* (sigma) repeats the fit up to 5 times without the pixels further than that many standard deviations from it, so particles do not tilt the plane (alternative to a mask) |
| Line by line | Polynomial of order n (or the median) per scan line, rows or columns, subtracted. Same fit selection and outlier rejection |
| Align rows | Removes offsets between lines only. *Median of differences*: the offset of a line is the median of its difference to the previous line (accumulated); robust against features. *Median* / *Mean*: every line gets the image median/mean |
| Shift | Minimum, mean or median of the selected pixels becomes zero |
| Facet level | Subtracts the plane that makes the **most common local slope** horizontal (mean shift on the local gradients, three rounds). For terraces, steps and large particles, where a plane fit tilts the flat areas. Not for random rough surfaces. Respects the fit selection |
| 3-point level | The *3-point* tool above the image (section 2) |
| Median background | Subtracts a median filter with a large window (pixels): features smaller than the window stay. Windows above 15 pixels are computed on a reduced image and interpolated back. Biased near the image border on curved backgrounds |
| Rolling ball | Rolls a ball (or an *arc*: a cylinder along the rows, the columns or both) of the given **radius in m** under the surface and subtracts its envelope. *From the top* rolls it over the surface (keeps holes and pits). Height channels only |

**Choosing the rolling-ball radius.** AFM heights are tiny compared with lateral sizes, so the radius is large: a bump of
width w and height h has a curvature radius of about w² / (8 h) (100 nm wide and 5 nm high: 0.25 µm), a bow of height H over
a length L about L² / (8 H) (10 µm, 50 nm: 250 µm). Choose a radius between the two: the ball then follows the background but
does not enter the features. Too small removes features, too large leaves background. The ball is computed as the paraboloid
with the same curvature at its bottom (exact for AFM data, where heights are far below the radius), at full resolution; the
image is continued past its border by an odd reflection. A 1024 x 1024 image takes under a second, even for radii of 100 µm.

## 5. Mask tab

Besides the drawing tools: **threshold** masks pixels *above*, *below*, *between* or *outside* limits of this channel, of
**another channel** (e.g. phase) or of the **slope** (degrees, from central differences). Limits are percentiles, multiples of
sigma around the mean, or absolute values (SI prefixes such as `5n` work). The result can replace or be combined with the
mask. **Grow/Shrink** use the Euclidean distance (the image border counts as unmasked when shrinking); **Remove small areas**
drops connected areas below a pixel count; **Fill holes**, **Invert**, **Clear**.

**Automatic**: *Otsu mask* finds the threshold that splits the values of the *Source* (this channel, slope, or another
channel) into two classes with the smallest spread inside each (Otsu's method, histogram of the 0.05 to 99.95 percentile
range) and masks the pixels above or below it: particles on a flat substrate in one click (level first). *Mask outliers*
masks pixels further than n robust standard deviations from the median of the image (*Compared with: the whole image*) or
from the median of their n x n neighborhood (local outliers: dust, single spikes, holes). Both use the combination setting
of the threshold group.

**Watershed (touching grains)**: every hill of the image becomes a grain of its own, separated from its neighbors by a
1-pixel unmasked line, so the grain analysis counts touching grains separately (polycrystalline films, close-packed
particles, domains), where a threshold or Otsu joins them into one area. The hills grow downhill from their tops until
they meet in the valleys (watershed), and a hill must rise at least *Min. prominence* (% of the height range, 1 to 99 %)
above the pass to its neighbor, otherwise it is merged with it: raise it if noise splits grains, lower it if touching
grains are merged (default 2 %). Advanced: *Smoothing σ* (only for finding the tops), *Min. distance of maxima*,
*Keep the upper part* (0 %: the whole basin of every grain, i.e. the whole image is divided; 50 %: only the upper half of
every grain, e.g. the tops of particles without the substrate between them), *Pits instead of hills* (holes, pores).
Level the image first. Test: 8 x 8 touching Gaussian bumps: watershed 64 grains, Otsu far fewer.

## 6. Filter tab

| Filter | Notes |
|---|---|
| **Smoothing and morphology** | Choose the filter in the drop-down list, its *Size* (diameter of the footprint in pixels) and *Shape* (disc or square), then *Apply filter*. The line under the list says what the chosen filter does. NaN pixels stay NaN |
| Mean | Average over the footprint (NaN pixels ignored) |
| Median | Median over the footprint: removes spikes and noise, keeps edges better than the mean |
| Gaussian | Sigma in pixels; NaN pixels are ignored (normalized convolution) |
| Conservative denoise | A pixel higher than all its neighbours (or lower) is set to the highest (lowest) neighbour: removes single-pixel spikes, changes nothing else |
| Kuwahara | Each pixel gets the mean of the quadrant of its window with the smallest variance: smooths flat areas, keeps edges sharp |
| Minimum / Maximum | Grey erosion / dilation: the lowest / highest value in the footprint; features shrink / grow by the footprint radius |
| Opening | Erosion then dilation: removes peaks, particles and spikes narrower than the footprint, keeps larger shapes |
| Closing | Dilation then erosion: fills pits, holes and gaps narrower than the footprint |
| Sharpen | Unsharp mask: adds *Amount* x (image - Gaussian(sigma)); enhances detail smaller than sigma, and noise |
| FFT filter | Butterworth (order 4) transfer function of the spatial frequency, cutoffs given as **wavelengths**. Low-pass keeps features longer than the cutoff, high-pass shorter ones, band-pass/-stop use both cutoffs (first = short, second = long). The image is mirror-padded by a quarter of its size to reduce edge ringing; the mean is kept |
| FFT notch | Select spots in the **2D Spectrum** graph (click), set the radius, *Remove spots*. Each spot and its mirror image are attenuated with a Gaussian. Typical for periodic noise, e.g. a vertical streak (kx = 0) from scan-line noise |

**Apply to** (*Whole image*, *Inside the mask only*, *Outside the mask only*) decides which pixels receive the filtered
values; the others keep their values. The filter itself is still computed from all pixels, so masked features (particles,
scars) still influence the unmasked pixels within the filter window near the mask edge. To keep a defect out completely, fill
it first (*Glitch - Fill masked pixels*), then filter.

## 7. Glitch tab

- **Spikes**: a pixel is replaced by the local median (window n) if it deviates from it by more than *threshold* robust standard
  deviations (1.4826 x MAD of the deviations). Passes repeat it.
- **Scan lines**: *Detect* builds the offset profile of the lines (median differences of neighboring lines, accumulated)
  and lists the lines further than *threshold* robust standard deviations from its running median over 2n + 3 lines, so
  clusters of up to n (*Adjacent bad lines up to*) neighboring bad lines are found, and their good neighbors are not.
  *Replace listed lines* interpolates them linearly between the nearest intact lines (the list accepts `12, 40-42`).
  *Repair automatically* does the detection and the replacement in one step. A real feature that runs along the lines and is
  at most n lines wide looks the same and is also listed: check the list before replacing.
- **Scars**: short pieces of scan lines (at most *Max. width* lines thick, at least *Min. length* pixels long) that are
  higher (*Up*) or lower (*Down*) than both the line above and the line below by more than *threshold* times the robust
  standard deviation of the differences between neighboring lines. *Mask scars* adds them to the mask (to check the
  settings), *Remove scars* fills them from their surroundings (Laplace interpolation) without touching the mask.
- **Steps inside lines**: a jump in the middle of a line, e.g. a tip change. Every line is compared with the mean of the
  line above and below; where the mean of that difference over *Min. segment* / 2 pixels after a position differs from the
  mean before it by more than *threshold* robust standard deviations, the line is cut, and every part is shifted to the level
  of the longest one. Run *Align rows* first; uses the direction of the scan lines group.
- **Masked defects**: mask a scar or particle and *Fill masked pixels*: the masked pixels become the solution of Laplace's
  equation with the surrounding pixels as boundary (reproduces planes exactly).

## 8. Rescale tab

- **Rescale to a reference channel**: maps this channel onto another channel of the file by `reference ~ gain * channel +
  offset` (order 1) or a polynomial (orders 2 to 5). The intended use is the open-loop `Z-Axis` against the closed-loop
  `Z-Axis Sensor`. *Show fit* gives the gain (least squares and standard-deviation ratio) and the correlation r **before** you
  apply. Least squares underestimates the gain and the ratio of standard deviations overestimates it when the channels are noisy;
  they agree only for r close to 1. Level both channels the same way first. **If r is low, do not trust the gain**: the open-loop
  and the sensor channel are then not related by a simple factor (tilt, creep, hysteresis and noise dominate).
- **Scale values**: `value * factor + offset`; *Invert sign* is factor -1.
- **Lateral scale**: factors for the image size (X, Y), a manual step of the history (for the scale measured by the position
  sensors see 8a). *Calibrate from line cut*: draw a line over a feature of known size
  (grating pitch, a distance between terraces), enter the true length, choose X, Y or both: the factor is true length / measured
  length (along x, along y, or the line length). Applies to all channels.
- **Geometry** (all channels): Flip vertically or horizontally; **Rotate 90° left** (counterclockwise), **Rotate 90°
  right** (clockwise) or **Rotate 180°**, e.g. to turn the tall mosaic of a vertical stitch into a wide image (width and
  height swap, also for non-square pixels; the line cut and the zoom are reset); crop with the Crop tool.
- **Resample pixels**: see 8b.

### 8a. Open-loop XY correction from the position sensors (bottom of the *Rescale* tab)

Normally scans are taken in **closed loop**: the position sensors control the scanner, and the scale stored in the file is
right. This section is only for the occasional **open-loop** scan, where the scanner moves by the requested range only
approximately (gain error, non-linearity, creep, hysteresis). If the file also holds the **measured** X and Y position per pixel (position sensor channels, in m), the image can be shown on the
scale the scanner really reached. The switch **XY scale of the image** (section *Open-loop XY correction* at the bottom of the Rescale tab) acts on the **whole file** (all channels, both scan
directions), is not a step of the history, and can be toggled at will:

| Setting | Meaning |
|---|---|
| As recorded (closed loop, normal) | The scale stored in the file: the pixels lie on a regular grid of the scan size. The default, right for closed-loop scans |
| Open-loop scan: corrected from the X / Y sensors | Uses the position channels. The forward images use the forward position channels, the backward images the backward ones |
| X / Y position channel | Found by name (`Position X`, `X-Axis Sensor`, `X Sensor`, ...) and shown in the list; if the names are different choose them by hand. They must be in m. Channels of the list that are not positions are rejected with an error |
| Scale only | The image size follows the measured travel of the scanner between the first and the last pixel (median over the lines), one factor per axis. The pixels are not moved or interpolated. Corrects a gain error |
| Resample | The data are interpolated onto a **regular grid of the measured positions** (linear, two passes: along the lines, then along the columns), which also corrects non-linearity, creep and hysteresis. The image keeps its number of pixels; its size becomes the region covered by all lines. The sensor positions are smoothed first with a local cubic fit (*Sensor smoothing*, window about 4 x the value in pixels) because sensor noise of a pixel or more would make them non-monotonic |

Details: the sign of each axis is detected from the data (the reader flips the lines of upward scans, so the Y sensor
decreases along the rows); positions must increase along the scan after that. The position channels themselves keep their
raw data in *Resample* mode so that the measured positions and their non-linearity can still be inspected.
After a switch every channel's operations are **replayed** on the new image, so masks, levels and filters follow (pixel
positions of masks drawn before a *Resample* refer to the pixels, not to the sample positions). An operation that reads another
channel (threshold of the phase, rescale to a reference) reads the **final** state of that channel when it is replayed. If a
replay fails the switch is refused and nothing changes. The chosen mode stays active when you open the next file (arrow
buttons) if that file has position channels; otherwise the file opens as recorded and the status bar says why. The title bar shows
`[XY: sensors, ...]` while the sensor scale is active, and statistics exports and batch results have a `lateral` column.

Checks: with synthetic images sampled at known distorted positions the resampled image reproduces the true surface to 0.1 nm
on a 40 nm relief (the open-loop image is off by 4 nm), including all four sign combinations of the axes, noisy and drifting
positions. The example files available here contain no X/Y position channels, so the recognition of the channel names
(`Position X`, ...) is untested on instrument files: tell the program the names if they differ. The *File Info* tab shows
*XY linearization* for `.nhf` files (`disabled` = the scan was not corrected by the position sensors).

**Square pixels** (*Rescale* tab): the line shows the pixel size in x and y. When they differ (e.g. a 512 x 256 scan of a
square area), *Make pixels square* resamples the axis with the coarser pixels to the pixel size of the finer one, for all
channels, keeping the image size. In a macro the step adapts to each file (nothing happens if its pixels are square).

### 8b. Resample pixels (*Rescale* tab)

Changes the number of pixels of **all channels**, while the image keeps its physical size, so the pixel size changes. Choose
*Number of pixels* (X and Y) or *Pixel size* (the numbers follow from the image size); the line under the settings shows the
pixels before and after. *Interpolation*: *Linear* (default), *Cubic* (smoother, can overshoot at steps) or *Nearest pixel*
(no new values, blocky). NaN pixels stay NaN; the mask is resampled with the nearest pixel. It is a history step like Crop
(undo removes it from all channels).

Use it to compare images at the same pixel size (roughness and PSD depend on it), to reduce large images before slow
operations, or before stitching images of different resolution (the stitching does that by itself).

## 9. Statistics

Computed for the selected region (all / inside / outside the mask) with the mean as reference. **Click a parameter** (or
hold the mouse over it) and the line under the table explains it in plain words. Values are shown with a readable unit
and 4 significant digits; a value that is zero within the numerical noise (e.g. the mean of a leveled image,
1e-17 m) is shown as 0. *Copy table* copies what is shown.

| Parameter | Definition |
|---|---|
| Mean, Median, Min, Max | of the pixel values |
| Sa | mean absolute deviation from the mean |
| Sq | root mean square deviation (ISO 25178 Sq) |
| Ssk, Sku | skewness and kurtosis of the heights (Sku = 3 for a Gaussian distribution) |
| Sp, Sv, Sz | highest peak above the mean, deepest valley below it, Sz = Sp + Sv (= max - min) |
| Sdq | root mean square slope, from central differences (heights in m only) |
| Sdr | developed area ratio, mean of sqrt(1 + slope^2) minus 1, in % (heights in m only) |
| Pixels, Area | number of pixels and their area |
| Step height (mask - rest) | with a mask, for *All pixels*: the masked area minus the rest, from one plane for the whole image plus an offset for the masked pixels (pixels within 3 pixels of the mask edge left out), with its standard error. For a pad, a line or a step standard: mask the upper level (threshold or Otsu), read the height. Tilt does not enter; curvature does, so level a bow first |
| Sal | autocorrelation length: the shortest distance in which the 2D ACF falls to 0.2 (ISO 25178) |
| Str | texture aspect ratio: shortest / longest such distance; near 1 isotropic, near 0 strongly directional (grooves). n/a if the ACF does not fall to 0.2 in some direction within half the image |
| Std | texture direction: the lay in degrees, 0 to 180 counterclockwise from +x (y up), from the angular power spectrum (frequencies at least 10 steps from the center). Meaningful only when Str is clearly below 1 |
| Correlation length | along x (the fast scan axis): where the line-averaged ACF falls to 1/e |

Use **Copy table** to paste into a spreadsheet. The **Bearing** tab shows the Abbott-Firestone curve of the same region.
Roughness depends on the scan size and the pixel size: compare images only at equal size, or use the PSD.
The texture parameters take about 0.2 s for a 512 x 512 image; they use the selected region (the ACF of a masked region
counts only the pixel pairs inside it). Strong scan-line offsets make every surface look directional along x: align the rows
first.

### 9a. Histogram fits

Under the histogram: choose the model (*Fit*), the pixels (*Of*: all, inside or outside the mask) and the range (*From*,
*To*, in the unit of the axis; *Full range* resets it; a typed range is kept for the next images, else the range follows
the data), then **Fit**.

| Model | Parameters |
|---|---|
| Gaussian | center, σ, FWHM = 2.355 σ |
| 2 / 3 Gaussians | per peak: center, σ, FWHM and its share of the pixels; the separations of neighboring centers |
| Lorentzian | center, HWHM γ, FWHM = 2 γ |
| Pseudo-Voigt | center, FWHM, Lorentzian share η (0 Gaussian, 1 Lorentzian) |

The fit is a least squares fit of the histogram counts with Poisson weights; the ± values are the standard errors of the
parameters. Neighboring pixels of an image are correlated, so the real uncertainty is larger, and pixels on sloped walls
between two levels add a background that the model does not include (a large reduced χ²). The **separation of two
Gaussians** is a robust **step height**: level the image first (*Facet level* for terraces), fit *2 Gaussians* to a range
around both levels. The fit (and the dashed components) is drawn on the histogram; *Copy* copies the text. When the image
changes the fit is cleared.

## 10. Line cut

The profile is sampled along the line at the pixel pitch (bilinear interpolation) and averaged over *Width* pixels measured
perpendicular to the line; the band is the standard deviation across the width, min and max are drawn thin. Ra, Rq and Rz of
the mean profile (linear trend removed), the length and the direction (degrees, counterclockwise from +x) are shown in the
title. Export with *File - Export line profile*.

**Horizontal and vertical lines.** Choose the kind of line with the arrow next to *Line cut* (or H, V, X). A
*horizontal* line covers a whole pixel row, a *vertical* line a whole column (the profile runs like the image axes:
left to right, bottom to top); click to place it, or drag to move it while watching the image. *Horizontal and
vertical* places both through the clicked point (a cross): the profile shows the horizontal cut in the usual color
and the vertical one in blue, with Ra and Rq of each, and the legend names their positions (y and x as on the image
axes). Typed coordinates under the profile make it a free line again. Both lines are also shown on the side-by-side
images and drawn in the report.

**Roughness and waviness (ISO 21920).** Choose *Measure: Roughness and waviness (ISO 21920)* under the profile. The
profile is split as in ISO 21920 (successor of ISO 4287 / 4288) with the Gaussian filter of ISO 16610-21:

- *Form removed*: a line (tilt, default), a 2nd order polynomial (tilt and bow) or nothing. What remains is the primary
  profile **P** (optionally smoothed by the short cutoff **λs** = λc / 300; it is skipped when shorter than two point
  spacings, which is usual for AFM sampling).
- **Waviness W** = P smoothed by the Gaussian filter with the cutoff **λc** (wavelengths longer than λc; a sine of
  wavelength λc keeps half its amplitude). **Roughness R** = P - W.
- *λc cutoff*: 0 chooses it automatically: the largest of ... 0.25, 0.8, 2.5, 8, 25, 80 µm ... (the ISO series, extended
  to AFM lengths) for which the line holds six cutoffs. The ISO 4288 table (0.08 to 8 mm) assumes stylus profiles of
  millimeters, longer than most AFM lines; for comparisons with a stylus instrument type its cutoff.
- The parameters are evaluated without **λc / 2 at each end** (shaded in the lower graph), where the filter sees only
  half its window; the evaluation length is the line length minus λc. Rz is the mean of the peak-to-valley heights of
  the sections of one cutoff each (five by default). Lines shorter than three cutoffs are evaluated over their whole
  length (with a note).

The upper graph shows the profile with the waviness (orange-yellow), the lower one the roughness. The table lists
**R**: Ra, Rq, Rp, Rv, Rz, Rt, Rsk, Rku, Rdq (RMS slope, also in degrees), RSm (mean width of the profile elements, height
discrimination 10 % of Rz, width discrimination 1 % of λc), Rmr (material ratio half the total height below the top),
Rdc (height difference between 20 % and 80 % material ratio); **W**: Wa, Wq, Wp, Wv, Wt, WSm; **P**: Pa, Pq, Pt, Psk,
Pku. Hold the mouse over a row for its meaning. *Copy*, *File - Export line cut measurement* (CSV with λc, λs and the
evaluation length) and the report include them. Checked with sine profiles of known Ra, Rq, Rz, RSm, Rku and Rdq.

**Through all shown channels.** With two or four images (section 2), the line is drawn on all of them and the profile
shows each channel of the same size in its own color with **its own y axis** (on the right), so topography and phase or
KPFM can be compared along the same line. Turn it off with *Layout - Line cut through all shown channels*; the roughness
values in the title are of the main channel.

**Refining the line.** Drag an end of the line (the dots) to move only that end, or drag the line between the ends to move
it as a whole (it stays inside the image); this works with the *Line cut* tool and in *Navigate* mode. With the *Line cut*
tool, dragging away from the line draws a new one. Under the profile the ends are shown as *Start x/y* and *End x/y*
(physical positions from the top left corner of the image); type a value (SI prefixes such as `1.25u` or `800n` work) to
place an end exactly. To remove the line: *Clear line* under the profile, *Edit - Clear line cut* (Ctrl+L), or Esc on the
image.

**Cursors.** Under the profile, *Cursors* shows two **horizontal** lines (*Horizontal (height)*), two **vertical** lines
(*Vertical (distance)*) or both. Drag a line with the mouse (dragging elsewhere pans the graph) or
type its position in the fields next to the selector. The line below shows **Δz** (height 2 minus height 1), **Δx**
(distance 2 minus distance 1) and, with both pairs, the slope between the two points. The cursors keep their positions
when the line cut is moved or redrawn; ones that fall outside the new profile are placed again at a quarter and three
quarters.

**Critical dimension (CD).** Choose *Measure: Critical dimension* under the profile, then *Lines* (raised features) or
*Trenches*, and the *Levels* in % of the height measured from the bottom (default `10, 50, 90`: bottom, middle and top CD).
Draw the line across the features, perpendicular to them, with a *Width* of several pixels to average noise.

| Result | Definition |
|---|---|
| Bottom, top, height | the profile is split into a lower and an upper class (Otsu); their medians are the floor and the top (dashed lines), the height (depth for trenches) is the difference |
| CD at a level | for every feature that lies completely on the line: the distance between the outer crossings of its left and right flank with that level (linear interpolation); drawn as bars on the profile |
| Left / right angle | sidewall angle from the horizontal (90° = vertical), from a line fit of each flank between 20 and 80 % of the height; for height data only |
| Feature height | median of the feature top minus the median of the floor on both sides |
| Pitch | mean distance of the feature centers (middle of the 50 % crossings) ± their standard deviation |

Features cut by the ends of the line are not measured (the summary says how many). The measured CD contains the tip:
lines appear wider and trenches narrower by about the tip width at that height, and the sidewall angle cannot be steeper
than the tip's flank. *Copy* copies the summary and the table, *File - Export line cut measurement* writes them in SI
units.

**Step height.** Choose *Measure: Step height* and draw the line across the step(s), perpendicular to the edges, with a
*Width* of 10 to 30 pixels to average roughness. Every step on the line is found and measured (a staircase of terraces
gives one row per step).

| Setting / result | Meaning |
|---|---|
| Min. plateau | the shortest flat part between two steps, in profile points; the step detector compares the mean height over half of it before and after every point |
| Sensitivity | a step must exceed this many times the noise of the detector, which includes the roughness of the line at that scale (5 to 8 is usual; lower finds smaller steps and more false ones) |
| Min. step | steps smaller than this are ignored (0: only the sensitivity counts) |
| Edge exclusion | the part of every plateau left out at each end (edge rounding by the tip, overshoot); 25 % keeps the middle half, ISO 5436-1 uses a third |
| Height | all plateaus are fitted together with **one common slope** and one level each (least squares, the used parts are drawn as thick lines); a step height is the difference of the levels left and right of it, positive for a step up from left to right, so a tilt of the line does not enter it |
| ± | standard error of that difference from the fit (statistical only; the scanner's Z calibration is not included) |
| Tilt | the common slope of the plateaus, in degrees |

On the 0.5 µm sapphire example (aligned rows, facet level, line along x, width 30, sensitivity 5) it finds three terrace
steps of about 0.18 nm.

## 11. Spectra

- **2D Spectrum**: log amplitude of the 2D FFT (Hann window, mean removed), axes in 1/um. Streaks along kx = 0 are
  scan-line noise; spots are periodic structures; a spot at k belongs to a period 1/k.
- **PSD**: radially averaged power spectral density in nm^2 um^2 (Hann window, normalized so that its integral over the
  whole spectrum equals the image variance). *Keep curve* leaves the curve in the graph so that images of different scan size
  can be compared on one axis.

  **PSD fit** (below the graph): choose a model, the range *From* / *To* (1/µm; *Full range* resets it) and **Fit**.
  The fitted curve is drawn dashed, the fitted range shaded, and the results are listed (and can be copied). Until
  *Clear fit*, every new image or processing step is fitted again with the same settings.

  | Model | Fits | Reports |
  |---|---|---|
  | ABC + noise floor (default) | PSD = A / (1 + (B f)²)^((C+1)/2) + N: a plateau at long wavelengths turning into a power law, plus a flat (white) noise floor | **corner** f_c = 1/B (the transition from plateau to power law) and the correlation length ξ = B/2π; the power law slope -(C+1) and, for 2D surfaces, the Hurst exponent H = (C-1)/2; the rms roughness of the model surface; the **noise floor** as nm rms; and the **effective resolution**: the period 1/f_n where the power law falls to the noise floor. Finer detail is below the noise; features down to about half that period are resolved |
  | ABC (k-correlation) | the same without noise floor | corner, slope, H, rms |
  | Two power laws | two straight lines in the log-log graph meeting at f_t | the **transition** f_t and both slopes; when the slope above f_t is flatter (the noise), the period 1/f_t is reported as the effective resolution |

  If the signal never reaches the noise floor, the resolution is set by the pixels (period = 2 pixels). When the corner
  lies outside the fitted range, the program says so instead of giving a correlation length (no plateau: scan a larger
  area). A warning appears when the model fits poorly, typically for periodic samples like gratings, whose spectrum has
  peaks: narrow the range or use *Two power laws*. The lowest one or two points are affected by the window and the
  leveling; leave them out with *From* if they bend the fit. Example (sapphire, 2 µm, Z-Axis after line leveling):
  corner 1.25 1/µm, slope -2.3, noise floor 0.07 nm rms, effective resolution 19 nm period.

- **Functions** (tab): statistical functions of the selected region.

| Function | Definition |
|---|---|
| ACF along a line | autocorrelation along x (rows) or y, every line with its mean removed, averaged over the lines, normalized to 1; the dashed lines mark 1/e (correlation length) and 0.2 |
| HHCF | height-height correlation <(z(x+τ) - z(x))²>, log-log; it rises with slope 2H (H: Hurst exponent) and saturates at 2 Sq² beyond the correlation length |
| 2D autocorrelation | central half of the 2D ACF with its 0.2 contour, and Sal, Str, Std in the title |
| Slope distribution | histogram of the local inclination (degrees, central differences) |
| Slope direction | polar plot of the direction in which the surface rises, weighted by the slope |

*Export curve as CSV* writes the shown curve (SI units); the 2D ACF is saved as a picture with a right-click
(*Save image...*).

## 12. Grains (*Tools - Grain analysis*, Ctrl+G)

**Columns...** chooses the properties shown in the table (hold the mouse over a header for its meaning); the CSV export
always has all of them. For **nanoparticles** the height above base is the reliable size: lateral sizes (diameters,
Feret, areas) are widened by the tip shape.

**Histogram** (under the table, the divider can be dragged): the distribution of one property of the grains. Choose
it in *Histogram of* or **click the header of a column**; mean (line), median (dashed), the number of grains and the
standard deviation are given, and the grain selected in the table is marked in red. *Logarithmic axis* suits sizes that
spread over decades. The graph zooms with the wheel; right-click saves or copies it.

*Grains on the image*: **Each grain in its own color** (default; rainbow colors, consecutive numbers clearly different,
the number cell of the table shows the same color, the selected grain stronger with a white edge), *Outlines* (one
color), or *Not shown*. The grain analysis opens in its own window beside the image (it can stay open while you edit the mask); the Mask tab
has a *Grain analysis...* button too, and the workflow *Particles and grains* opens it with a result.

The grains are the **areas of the mask**. Make the mask first (Mask tab), typically a *threshold* above the background
(percentile or sigma of a leveled image), possibly cleaned with *Remove small areas*, *Grow/Shrink* or *Fill holes*; the
drawing tools add or remove grains by hand. Then **Analyze grains**.

| Setting | Meaning |
|---|---|
| Connectivity | 8: masked pixels that touch at a corner belong to one grain; 4: only edges connect |
| Smallest grain | Grains with fewer pixels are ignored (noise) |
| Split touching grains | *No*: one grain per connected mask area. *By height maxima*: a watershed from every height maximum (after a Gaussian *smoothing*), for touching particles. *By shape*: a watershed of the distance to the mask border, for touching round grains on a flat top |
| Min. distance of maxima | Maxima closer than this (pixels) count as one; raise it if grains are split too often |
| Ignore grains touching the border | They are cut by the image edge, so their size is wrong |

The summary gives the number of grains, their density (per area), the covered area (%) and mean ± standard deviation of
area, equivalent diameter, height above base, maximum height and volume. The table has one row per grain:

| Column | Definition |
|---|---|
| Area, Equivalent diameter | pixel count x pixel area; diameter of the circle with that area |
| Perimeter | length of the pixel outline x pi/4 (unbiased for smooth outlines) |
| Circularity | 4 pi area / perimeter^2 (1 for a circle) |
| Major / Minor axis, Orientation | of the ellipse with the same second moments; orientation of the major axis counterclockwise from +x, 0 to 180° |
| Center x / y | centroid in m (from the top left corner) |
| Max Feret, Min Feret (+ directions) | the caliper sizes (ISO 13322): the largest distance across the grain (its length; the widest gap it could fill) and the narrowest width (the narrowest gap it could pass through), from the convex hull of the pixel corners; directions counterclockwise from +x, 0 to 180° |
| Feret ratio | Max Feret / Min Feret: 1 for round or square grains, large for rods and fibers (aspect ratio) |
| Convex hull area, Solidity | area of the convex hull (a rubber band around the grain); solidity = area / hull area: near 1 for compact grains, lower for agglomerates, branched or concave grains (an L shape: about 0.5) |
| Surface area | true 3D area of the grain surface, from the local slopes (height channels only) |
| Max / Mean / Median / Min height, RMS height | of the grain pixels; RMS = their standard deviation, the roughness on top of the grain |
| Height above base | max height minus the base: the median of the background pixels around the grain (the grain minimum if it is surrounded by other grains) |
| Volume above zero / above base | sum of (height) or of (height minus base, only positive) x pixel area |
| Volume above background | volume above a background interpolated under the grain from the surrounding pixels (Laplace, as *Fill masked pixels*): correct on tilted or curved substrates, where the base level overestimates (flat grains on a tilt get a volume above base, none above background) |
| Touches edge | the grain reaches the image border |

The grains are outlined (and numbered, up to 150) on the image; clicking a row of the table highlights that grain.
*Show grains on the image* switches the outlines off. When the image, the mask or the channel changes the result is cleared:
press *Analyze grains* again. **Clear** removes the analysis (table, histogram and the grains on the image); the mask
stays. *Copy table* (for a spreadsheet) and *Export CSV* (SI units, also in the File menu). With
*Split* on a large mask (hundreds of thousands of pixels) the analysis takes a few seconds.

In **batch** processing the macro must create the mask; a macro saved after a grain analysis contains it and runs it.
Otherwise the option *Grain analysis of the mask* (dialog) or `--grains`
(command line) adds the grain summary to `results.csv` and writes one grain table per image and channel.

## 13. Stitching (*Tools - Stitch images*)

Joins overlapping images into one mosaic. Add the images (the one open in the window is already in the list); the order
of the list is the stitching order: the first image stays fixed and every next one is registered against the mosaic of the
images before it, so put a neighbor of the images above it next in the list.

**Positions** are the scan centers in µm with y pointing up, as the instrument stores them (`X-Pos`/`Y-Pos` in `.nid`,
the scanner offset in `.nhf`). *Positions from the files* fills them in; you can type them into the table; *Positions as a
grid* lays the images out row by row from the top left with the given number of columns and overlap (optionally
serpentine), for files that all have the same position.

| Setting | Meaning |
|---|---|
| Before stitching | Every image (height channels) is processed first: *Plane level* (default), *Align rows + plane*, *None* or a macro of one channel |
| Channel used to register | The channel whose features are matched (default: the closed-loop height). All channels the images have in common are stitched with the same placement unless *Stitch all channels* is off |
| Refine the positions | Moves every image by up to *Largest correction* (% of the image size) to the position of the best normalized cross-correlation with the mosaic, computed only on the overlapping pixels. Images whose overlap is too small (< 3 % of the image) or whose best correlation is below 0.3 stay at their position; a message lists them |
| Remove height offsets | Shifts every image by the median height difference to the mosaic in the overlap |
| Overlaps | *Feather*: weighted average with weights that fall to zero at the image borders (no visible seams); *Average*; *Earlier* or *Later image on top* |

Images with different pixel sizes are resampled to the finest pixel size first. The registration is to the nearest pixel.
The mosaic opens in the window like a measured image (title `<first file>_stitched`; it is **not saved**: use *File -
Export all channels as Gwyddion file*); *File Info* lists how every image was placed. Areas not covered by any image are
NaN (no data). Rotation between the images is not corrected; the mosaic is limited to 8192 pixels per side and about 1 GB.

### 13a. Combine channels (lateral force) (*Tools - Combine channels*)

Makes a **new channel** from two channels of the file; the two stay as they are. Typical use: the **lateral force**
image from the forward and backward scans in Lateral Force mode (channel *Friction force* in .nid files). Friction
twists the cantilever one way on the forward scan and the other way on the backward scan, so its contrast is inverted
between the two, while the topographic crosstalk (slopes, edges) and the offset of the signal look the same in both
directions. The half difference (F - B) / 2 is therefore the lateral force, and the half sum (F + B) / 2 the crosstalk and
offset, a useful check that the lateral force image is free of topography. The dialog opens on a channel whose name
contains *Friction*, *Lateral* or *Torsion*, with the result (A - B) / 2 and the shift to be found from *Z-Axis Sensor*
(or *Z-Axis*, *Topography*). Any other pair of channels can be combined too.

Example (*Lateral Force mode Kit - SBS-PMMA sample*, 25 µm): *Find shift* on the Z-Axis Sensor gives 4.4 px (430 nm)
between forward and backward; (F - B) / 2 shows the PMMA domains clearly, and (F + B) / 2 keeps only faint edges.

| Setting | Meaning |
|---|---|
| Channel A, Channel B | the two channels; A is the shown channel (or the first lateral force channel when the shown one is not), B is preset to the other scan direction of A |
| Result | A - B, A + B, (A - B) / 2, (A + B) / 2, or A or the shifted B alone (to look at the alignment). A and B must have the same unit for sums and differences |
| Shift of B in x / y [px] | B is moved by this many pixels (fractions allowed, linear interpolation; positive x: to the right, positive y: down) before it is combined: the displacement of the tip between the forward and the backward scan (lag, hysteresis, creep). The line under the fields shows it in nm. Pixels moved in from outside the image are NaN (no data) |
| Find the shift from | a forward / backward pair with clear common features, usually the topography of the main scan; *Find shift* cross-correlates the two (lines leveled, sub-pixel peak) and sets the shift. It is refused when the best match does not stand at least 6 σ above chance (featureless or noisy channels): then type the shift |
| Name | of the new channel; preset as e.g. `Friction force ((F - B) / 2)` |

The preview shows the result with its rms. *Create channel* adds it to the channel list and shows it; it is then an
ordinary channel: it can be processed, measured, exported (also in the Gwyddion file) and used as a mask source. It is a
snapshot of A and B at that moment: process A and B first (e.g. *Align rows*), then combine. The creation is a step of
the history: *Undo* on the new channel (after its own steps) removes it, as does *Edit - Remove channels...*; a
channel that another step or channel uses cannot be removed. Crop, flip and resample made after the creation apply to
it too; one made before it cannot be undone while the channel exists. **Macros** record the creation with its settings
(the shift is stored, not searched again), so a saved macro builds the lateral force channel on every file that has the two
channels (others skip it, the status bar says so). A switch of the XY scale (8a) rebuilds it from the new A and B.

### 13b. 3D view (page *3D View*)

The shown channel as a 3D surface, drawn by the graphics card (OpenGL 3.3; nothing is drawn by the CPU plotting library).
It follows the processing: every step on the *Image* page is shown when the 3D page is opened. Modelled on Gwyddion's 3D
window.

**Mouse.** The buttons above the view choose what dragging with the left button does, as in Gwyddion (also the keys,
with the view focused): **Rotate (R)** (horizontal drag turns around the vertical axis, vertical drag tilts), **Scale (S)**
(drag up to enlarge), **Value scale (V)** (drag up to exaggerate the heights) and **Light (L)** (moves the light in the
lighting modes; a dashed wheel and a line show where it comes from). The wheel zooms, Shift + wheel changes the value
scale, the middle button pans, the **right button** opens a quick menu of color gradients and materials. *Top*, *Front*
and *Reset view* set the view; *Controls* hides or shows the settings on the right.

| Tab | Settings |
|---|---|
| Basic | rotation φ and tilt θ, scale, **value scale** (the height of the full data range relative to the longer image side; AFM data is very flat in true proportions, so the default makes the range 15 % of the side), *True proportions* (heights and lengths in the same scale), perspective or parallel projection, axes, axis labels, mask (drawn in red), axis line width, resolution (points per side; larger images are averaged in blocks; *Full* draws every pixel), background, autocrop of saved images |
| Light & material | the **mode**: *Gradient* (colors from the heights), *Overlay, no light* (heights from this channel, colors from another one, chosen under *Colors from*, e.g. the phase), *Lighting* (a material lit by the light), *Overlay with lighting*; the color gradient; the material (presets after the classic OpenGL table: gray, matte, brass, bronze, chrome, copper, gold, silver, pewter, pearl, jade, ruby, turquoise, obsidian), *Edit material* (ambient, diffuse and specular color and shininess, as Gwyddion's material editor); the light direction φ (around the vertical) and θ (above the plane) |
| Labels | font size of the axis labels |
| Color bar | show or hide it; color range full or 1 to 99 % (outliers do not stretch it) |

The axes run along the two edges of the image base that meet at the corner nearest to you, the height axis stands at
the corner further left; ticks are in the units of the data. **Save image** (and *File - Export 3D view*) saves the view
as shown, with axes, labels and color bar, as PNG, JPEG or TIFF by the file extension; *Autocrop* removes the empty
border; *Saved image size* (Basic tab) renders the saved image at 2, 3 or 4 times the size of the view on the graphics
card, with lines and text scaled to match, for print. If the 3D view stays blank or reports that OpenGL 3.3 is missing
(old graphics drivers, remote desktop), start the program with the environment variable `QT_OPENGL=software`. A 1024 x 1024 image at full resolution has two million triangles: any graphics card from the last ten years
turns it in real time.

If the computer has no OpenGL 3.3 (some virtual machines and remote desktop sessions), the page says so and the rest of
the program works as before.

### 13c. Lattice (*Tools - Measure lattice*)

Measures the **lattice vectors** of a periodic image (atomic lattice, moiré pattern, 2D grating, colloidal crystal) from its
autocorrelation (ACF). The dialog shows the ACF of the shown channel with the vectors **a** and **b** and the ACF maxima
used (circles).

1. *Find automatically*: the two shortest independent vectors between strong ACF maxima, reduced to the shortest basis (angle
   between 60° and 90°).
2. *Refine with all maxima* (on by default): every ACF maximum that lies on the lattice is fitted by least squares
   (hundreds of maxima on a good image), which is much more precise than the nearest maxima alone and averages over the
   whole image.
3. To choose other vectors, click near an ACF maximum: it sets **a** or **b** (choose with the buttons; the next click sets
   the other one). The maximum is located to sub-pixel precision.

Results: |a|, |b|, the angle between them (marked *hexagonal*, *square* or *rectangular* when it fits within 3 %/3°), the
direction of a (counterclockwise from the x axis, y up), the vector components, the unit cell area and the number of cells
per µm². *Region* restricts the ACF to the pixels inside or outside the mask. Level the image first (line leveling for
scan-line offsets). Drift and creep distort the lattice: a hexagonal moiré may come out with 63° instead of 60°; compare the
forward and backward images, or use the sensor correction (8a). A line grating (periodic in one direction only) or an
image without order gives a message; measure a grating pitch with a line cut. Example: *Moire_TRM_1.nhf*, Phase 2 after line
leveling: |a| = 13.34 nm, |b| = 13.28 nm, 63.1°, 440 maxima; the Amplitude channel of the same scan gives 13.32 / 13.26 nm.

## 14. Macros and batch

A **macro** is the list of steps applied to a file, saved so that other files can be processed the same way.

1. Process one image as you want it, on as many channels as needed (e.g. level the height and the phase, mask the height
   from a threshold of the phase, crop). Optionally run the grain analysis.
2. **Macro - Save steps as macro** (Ctrl+S, or *Save macro* in the History tab). The macro holds **every step of every
   channel in the order you applied them**, which matters when a step reads another channel, and the grain analysis with
   its settings if its result is shown when you save. *Macro - Show the steps of this file* lists them first.
3. Open another file and **Macro - Apply macro** (Ctrl+M, or *Apply macro* in the History tab), or pick it under
   *Apply recent macro*. All steps run in order and the grain analysis follows. If one step fails nothing is changed and
   the message names the step.
4. **Macro - Apply automatically to opened files**: choose a macro once; every file you open afterwards (also with the
   arrow buttons that step through the folder) is processed by it. The title bar shows `[macro: name]`; select the menu
   entry again to switch it off.

How the steps find their channels: a step belongs to a channel by its full name (`Z-Axis Sensor (Forward)`); crop, flip,
resample and lateral rescale belong to all channels. Steps of a channel the new file does not have are skipped (the
status bar says which). A macro whose steps all belong to **one** channel is applied to the **shown** channel when that
channel is missing, so a macro made on `Z-Axis Sensor` also works on `Topography` of a Studio file. Recipes of older
versions (one list of operations) are macros for the shown channel. The applied steps are ordinary history steps: undo,
reset and further steps work as usual.

The **History** tab lists the operations of the current channel. *Apply to other channels* repeats them on chosen
channels of the same file (operations that act on all channels are skipped there, they are already applied). **Undo**
removes the last operation of the channel; for crop/flip/resample/lateral scale it removes it from all channels, which
needs that no channel has later operations.

**Remove selected step** (or the Delete key in the list) takes out **any** step, also one in the middle: select it in the
list first. The original data is then processed again with all the other steps of the file, in the order they were made,
so the later steps of the channel, steps of other channels that read it (e.g. a mask taken from it) and channels made
from it with *Combine channels* all follow. A crop, flip, resample or lateral scale is removed from every channel (you are
asked first). If a later step no longer works without the removed one (e.g. a three-point level whose points lie outside
the image once a resample is removed), nothing changes and a message names that step.

**Tools - Batch processing** applies a macro to all `.nid`/`.nhf`/`.gwy` files of a folder (not to `_processed.gwy` results of an earlier batch) (the last used macro is preselected)
and writes `results.csv` (one row per file and channel: statistics of the whole image, and inside and outside the mask if
the macro creates one, and the grain summary if it contains a grain analysis), optionally the
processed images as `.gwy` and `.png`. Channels: `height` (all channels in m), a channel name (`Z-Axis Sensor`, both
directions) or a full name. A file or channel that fails gets a row with the error. A macro with steps for several
channels is applied once per file (the first selected channel counts as the shown one); a macro of one channel is applied
to every selected channel.

Without window:

```
python app_NSSA/main.py --batch macro.json folder_or_files... --out results --channels "Z-Axis Sensor" --gwy --png
NSSA-batch.exe macro.json folder_or_files... --out results ...      (the stand-alone program: same options)
```

Use `none` instead of a macro file to tabulate the raw statistics. The exit code is 1 if any row has an error. Add
`--lateral sensor` (and `--lateral-correction scale` or `resample`) to correct open-loop scans with the X/Y position sensors (section 8a; the default `nominal` keeps the recorded
scale, right for closed-loop scans); a file
without position channels then gives an error row. The batch dialog has the same choice (*XY scale*). `--grains` adds the
grain analysis of section 12 (`--grain-split height|distance`, `--grain-min-pixels n`, `--grain-exclude-edge`).

## 15. Export (File menu)

PNG (black on white with color bar and the color map and range of the view; with a scale bar when *Settings - Scale bar* is on
(in the scale bar color of the Settings menu), otherwise with numbered axes from 0 at the lower left; the mask as
shown; non-square pixels are drawn with their true aspect), image data as CSV (SI unit, with header),
all channels of the file as one Gwyddion `.gwy` (processed data and masks), the line profile, the statistics of all channels
(CSV), the grain table (CSV).

### 15a. Report (*File - Report (PDF or PowerPoint)...*, Ctrl+P)

A PDF or a PowerPoint file of the opened file for a customer, a colleague, a presentation or the lab book. The
dialog asks for:

- **Title** (the file name at first), **Sample**, **Operator**, **Comments** (shown under the title), an optional
  **Logo** (PNG or JPG, top right), the **Format** (PDF or PowerPoint) and the **Page size** of a PDF (Letter or A4).
- **Channels**: one section each; the shown channels are checked at first (*Shown*, *All*, *None*).
- **Contents**: image with color bar; the channels shown side by side (small, under the main image); statistics table;
  line cut; histogram; PSD; bearing curve; grain analysis; processing steps; file info.

The report follows what is on the screen: each image with its color map, color range, mask, scale bar or axes and
renamed labels (the other channels with the map of their side-by-side image, or the default of their kind of data);
the grains in their colors when *Grains on the image* is not *off*; the statistics of the region chosen under
*Statistics of*; the line cut of the Line cut tool on every channel of that size (width and roughness in the title; both lines of a horizontal-and-vertical cut); the **critical dimension** or **step height** chosen under the profile, for the shown channel, with its marks in the plot (levels and widths, or the fitted plateaus and numbered steps), the summary line and the full table (long tables continue on the next page); the
histogram with the bins of the Histogram panel and the masked pixels outlined in red. The grain section (summary and
size distribution) belongs to the channel that was analyzed. *Processing steps* lists the steps of each channel in plain
words (and the XY scale correction, if one is used); *File info* has the instrument and scan parameters and the
warnings of the file. Blocks that do not fit go to the next page; every page has the file name, the date and the page
number. Text and graphs are vectors, the images keep their pixels. The choices are remembered; *Open the file after
saving* opens it in the default program.

**PowerPoint** (16:9 slides, needs `pip install python-pptx`): a title slide (title, sample, operator, date, file,
channels, comments, logo), then for each channel: the image with its statistics as a table and the main numbers
(Sq, Sa, Sz, area) under the title; the channels side by side; the line cut with the critical dimension or step
height table (continued on more slides if long); histogram, PSD and bearing curve on one slide; the grains (summary
table and size distribution). Then the processing steps of all channels and the file info. Tables and text are
PowerPoint objects that can be edited and restyled; images and graphs are pictures (200 dpi). The processing steps of
each channel are also in the notes of its image slide.

## 16. Files and limits

- **.nid**: read by the program itself (text header, then the data blocks; the same values as NSFopen and Nanosurf's
  reader, and two channels with the same name are both kept, the second as e.g. *Amplitude 2*). Channel names, units,
  sizes and the scan direction come from the header. Spectroscopy files: the curves go to the Spectroscopy page (section
  17), their images (if any) to the Image page.
- **.nhf**: with **Nanosurf's own reader** (`nanosurf.utils.io.nhf_reader`, package `nanosurf` version 2 or newer:
  `pip install -U nanosurf`) when it is installed: all calibration types and file versions it knows; *File Info - File -
  Read with* says which reader was used. Without it (or if it fails on a file), the built-in reader: read directly with h5py. Image groups with `rect_axis_range`; each channel is calibrated linearly from the stored
  min/max of its signal (`signal_calibration_min/max` over `signal_minmax`). Channels without these attributes are skipped with
  a warning. Validated against the Studio example file only.
- **.gwy** (Gwyddion 2): every image channel (`/N/data`) with its title, size and unit; a mask saved with it becomes the
  starting mask; metadata goes to *File Info*. The scan direction is taken from the title ("Height (Forward)", "Phase
  backward", "fwd" / "bwd", "trace" / "retrace"); equal titles get a number. Graphs, spectra, volume and XYZ data are not
  read; channels whose lateral unit is not m are skipped with a warning. Old Gwyddion 1 files (GWYO) must be saved again
  in Gwyddion 2.
- The `.gwy` writer follows the published Gwyddion format; it is covered by a read-back test in this program but has not been
  opened in Gwyddion itself.
- Rotation of the scan frame is not applied. Time-resolved or multi-image Studio files show every image group's channels, prefixed
  by the group name.
- Not included: drift or distortion correction (except the position sensors, 8a), tip deconvolution, rotation by
  angles other than 90°, sub-pixel registration of stitched images, terrace analysis, curvature fitting.
- **Stand-alone program** (`NSSA.exe`, nothing else to install): the first version leaves out the
  optional parts: no PowerPoint report (PDF works) and the built-in `.nhf` reader instead of Nanosurf's (own color maps:
  gradient files in `colormaps` next to the .exe). An unexpected error is shown in a
  message box and written to `%LOCALAPPDATA%\NSSA\error.log`. Settings are kept in the same
  place as from the sources, so both share recent files and choices; those of the earlier names are taken over once.


## 17. Spectroscopy (page *Spectroscopy*)

Force-distance curves, force maps and I-V curves of Nanosurf `.nid` (Mobile S, Easyscan, FlexAFM, DriveAFM) and `.nhf`
(Nanosurf Studio) files, read with NSFopen. **Open** them like images (Open, drop on the window, Open recent) or with
*Open spectroscopy file* on the page: a spectroscopy file shows its curves here, and the images it holds (a force map
records the topography too) in the *Image* page.

**Layout** (like the Image page): the settings on the left; for a **force map** the **maps** in the center; the
current **curve** with its fit top right; below it the tabs **This curve** (all values), **Statistics**, **Histogram**
and **File Info**. A file that is not a map (single curves, I-V curves) has no maps: the curve gets the whole width.
Left panel: the **curve** (number, ◀ ▶; the position on the map), **Data**: *Analysis* (force curves or I-V curves,
guessed from the channels) and the **X** and **Y channel**. The **bar above the curve**: *X axis*, *Approach*, *Retract*,
*Zoom to contact* and *Other curves* (up to 40 in gray) for force curves, the *Plot* for I-V curves. All settings are
remembered. **Positions** (the curve, the map axes) count from **(0, 0) at the lower left corner** of the map like the
axes of images; *Origin (0, 0) at the lower left* off shows the stage coordinates of the file (Studio .nhf files record
absolute positions).

**Maps** (force maps, after *Analyze all curves*): **1, 2, 3 or 4 maps** side by side (*Maps*), each of its own value
(*Map 1* ... *Map 4*), one pixel per curve, gray where the fit failed; the current curve is framed on all of them.
Tools: **Pick curve** (click a pixel to show its curve), **Mask: add** and **Mask: erase** (drag a rectangle or click
pixels; on any map, the mask is shared). From a value: *Mask where* a value lies between *from* and *to* (**Mask
range**), or above its automatic **Otsu** threshold (e.g. the high parts of *height (contact)*, or the stiff parts of
*E*); **Invert**, **Clear mask**. The masked curves are red on the maps.

**Statistics**: N, mean, standard deviation, median, minimum and maximum of every value over **all curves**, the curves
**inside** or **outside the mask** (*Curves*). **Histogram** of any value (*Histogram of*, or click a row of the
statistics) with everything of the image histogram (section 9a): bin rules, the masked curves in red, and **fits** of
1 to 3 Gaussians, a Lorentzian or a pseudo-Voigt over a range, of all curves or inside / outside the mask (*Of*).

### 17a. Force curves

**Calibration**: spring constant (from the file) and deflection sensitivity (only used if the deflection is stored in V).
**Poisson ratio** of the sample (hold the mouse over it for typical values): 0.5 rubber, gels, cells and tissue; 0.45-0.5
soft polymers; 0.33-0.4 glassy polymers and most metals; 0.2-0.3 ceramics, glass and silicon; about 0.1-0.2 very stiff
materials. Unknown: 0.3 for hard, 0.5 for soft samples (E changes by at most about 30 %).

**Processing**, applied to every curve in this order:
1. *Filter*: moving average, median or Savitzky-Golay over a window of points (none by default).
2. *Baseline*, fitted to the part of the curve far from the sample (*From* / *To*, in % of the Z range from the far
   end; default 0-40 %) and subtracted from the whole curve: **offset**, **offset and tilt**, or **offset, tilt and
   interference**: a sine in Z on top of the line, from the laser light reflected by the sample (period about half the
   laser wavelength in Z, found automatically; the retract curve uses the period of the approach). Use it when the
   baseline is wavy; the shaded band in the graph is the region.
3. *Contact point*: where the approach force last crosses zero before the maximum.
4. Force F = deflection (N; m × k; V × sensitivity × k) and the tip position δ = Z − F/k (piezo Z minus the bending of
   the cantilever). **X axis** of the graph: piezo Z, tip-sample separation, or indentation (both relative to the
   contact point). *Zoom to the contact* shows the contact region; off, the whole curve.

**Values of every curve (no model)**: max force, contact point (piezo Z) and *height (contact)* (its negative: the
topography of a force map), max indentation, snap-in force (attraction on the approach), slope of the top half of the
approach (N/m: the stiffness of cantilever and sample in series), adhesion force (minimum of the retract), pull-off
distance (piezo retraction from the contact point to the pull-off), adhesion energy (area below zero of the retract),
dissipated energy (area between approach and retract).

**Contact model** (Young's modulus E of the sample for a rigid tip; E* = E / (1 − ν²) is the reduced modulus):

| Model | Shapes | Fitted on | Use for |
|---|---|---|---|
| **Hertz** | sphere (paraboloid formula, indentation << R) | approach, up to *Fit up to % of max force* | elastic samples without adhesion |
| **Sneddon** | sphere (exact), cone, pyramid, cylinder punch, hyperboloid | approach | elastic samples, any tip shape |
| **DMT** | all five | retract, from the maximum to the pull-off | stiff samples with small adhesion: F = E* g(h) − F_ad with the measured pull-off force; work of adhesion W = F_ad / 2πR for a sphere |
| **JKR** | sphere | retract, through the pull-off | soft samples with strong adhesion: E* and the work of adhesion W |
| **Oliver-Pharr** | all five | retract (unloading), between *Unloading from* and *to* % of the max force | elastic-plastic samples: power law F = α (h − h_f)^m, stiffness S at the maximum, contact depth h_c = h_max − ε F_max / S, area A(h_c), E_r = √π S / (2√A) |

Shapes: **sphere** (radius R), **cone** (half opening angle α), **pyramid** with 4 faces (α between the axis and a
face), **cylinder punch** (flat end of radius R), **hyperboloid** (apex radius R, asymptotic half angle α). For all
models but JKR and Oliver-Pharr only the contact point is searched; the modulus follows from it by least squares.
*Fit on*: the model's default segment, or approach / retract. The values of the fit are added to *This curve*: E, E*,
contact point, max indentation, the adhesion and work of adhesion (DMT, JKR), S, h_c, A, h_f, m (Oliver-Pharr), the
residual rms and R².

Notes: the tip radius and angle are not stored in the files: set them (the modulus of Hertz scales with 1/√R). Stiff
samples with a soft cantilever indent only a few nm: the modulus is then uncertain (look at R² and the spread in
*Statistics*). Oliver-Pharr assumes no adhesion (an adhesion offset makes the modulus too low) and needs a reliable
contact point, since the contact area depends on the depth from it.

**Fit regions: other models for some curves.** Different parts of a surface can need different models (a soft,
sticky polymer domain: JKR or Hertz; a stiff substrate: DMT; plastic deformation: Oliver-Pharr), and different tips or
Poisson ratios. Workflow: analyze all curves; mask the curves of a part (Map: *Mask: add*, *Otsu* or *Mask range*);
set the model, tip and Poisson ratio in the panel; **Fit the masked curves with these settings**. Those curves are
fitted again and keep the other curves' results; repeat for more parts. The list *Fit regions* shows *All curves* (their
model settings) and every region (later regions win where they overlap). Select an entry for: **Settings to the panel**
(to look at them), **Use the panel settings** (that entry gets the model settings of the panel; everything is fitted
again), **Curves to the mask**, **Remove**, **Save...** / **Load...** (the regions with their curves and settings, to
apply them again). The processing (filter, baseline, spring constant) is the same for all regions; **Analyze**
applies everything again: the settings of all curves, then every region. The results get a **model** text and a
**fit region** value (0 = all curves): show *fit region* as a map next to *E* to see which model made which pixel, and
choose *Curves: Fit region k* (or *All curves outside the fit regions*) for the statistics and the histogram. The CSV has
both columns. *Model of its region* (bar above the curve) fits the curve with the settings of its region as in the
results; off, with the settings of the panel (to try other settings on that curve).
Caution: the moduli of different models are not on the same footing (DMT and JKR treat the adhesion differently,
Oliver-Pharr uses the unloading stiffness), so part of the contrast between regions can come from the models. Compare
both models on a few curves at the border before trusting a contrast.

### 17b. I-V curves

Current against bias (channels with *current* / A and *voltage* / *bias*; the ramp of the file is used if the voltage is
not stored). **Sweeps**: forward, backward or the mean of both; *Subtract the current at 0 V* removes an amplifier
offset; filter as above. **Plot**: I(V), **dI/dV** (Savitzky-Golay derivative over *dI/dV window* points), the
normalized conductance (dI/dV)/(I/V) (broadened near 0 V), log |I|, and the Fowler-Nordheim plot ln(I/V²) against
1/V. **Values**: resistance of a linear fit, zero-bias conductance and resistance (fit within ± *Zero-bias range*), the
current at 0 V, the **gap** around 0 V where |I| stays below *Gap: \|I\| below* (width, from, to; marked in red), the
rectification ratio |I(+V)/I(−V)| at the largest common |V|, and the max |current|.

### 17c. All curves

**Partial maps**: *Every n-th line and column* analyzes a coarse map (n = 4: one curve in 16, 16 times faster) to look
at the data first; *Curves: The region of the mask* analyzes the rectangle of lines and columns around the masked
curves (at any step), e.g. a region of the coarse map at full resolution. The maps then show the analyzed part; the
label below says how many curves will be fitted. **Analyze** fits the chosen curves with the current settings (a progress bar for more than 50 curves; changing a
setting afterwards marks the results as old) and fills *Statistics* and *Map*. **Export results (CSV)**: one row per
curve with its position and every value (SI units in the column names) and the error of curves that could not be
fitted. **Export this curve (CSV)**: the processed curve (piezo Z, force, tip position, force before the baseline; for
I-V: bias, current, dI/dV); with a mask the results have a *masked* column (1 / 0). **Result maps to the Image tab**
(force maps): every value as a channel of a new image (one pixel per curve, with the mask), to level, measure, export
or put in a report like any image. **3D: value on the topography** (force maps): the 3D View of the topography of the
file (the images recorded with the map; leveled with *Flatten* unless it was processed in the Image page already)
colored by the value of *Map 1*, e.g. the modulus on the topography. The result maps are put onto the pixels of the
image by position, so every curve colors the area around its point; the topography and the maps are then a document
of the Image page, where the channel of the colors can be changed in the 3D View (*Colors from*). A map without images
uses the height of the contact points as topography.

**Quality filter**: *Leave out fits with R² below* (tab *Statistics*) leaves the fits that do not describe their curve
out of the statistics, the histogram, the maps (empty cells) and the 3D view; the label below says how many curves are
left out. The CSV keeps every curve and marks those with *left out (R² < x)*. 0 (the default) keeps all fits.

**Curves on the topography**: a map cell can show an image recorded with the map (e.g. the topography, leveled)
instead of a result. The analyzed curve points are drawn on it; with the *Pick* tool a click opens the curve closest
to the click, and the mask tools mark the curves inside the rectangle. This way curves can be chosen by what the
surface looks like, e.g. only the curves on a particle.

### 17d. Report, settings and batch

**Report (PDF or PowerPoint)...** writes a report of the page: analysis settings, the curve shown with its fit, the
maps as they are shown (with the mask), the statistics of the curves chosen under *Curves*, the histogram with its fit,
the fit regions and the file info. Pick the sections in the dialog; the choice is remembered. PowerPoint needs the
optional python-pptx package.

**Save settings... / Load settings...** (group *Batch*) store the whole analysis in a JSON file: analysis type,
processing, model, the step of a partial map, the quality filter and the fit regions. Load it to repeat the same
analysis on another file.

**Analyze a folder...** runs these settings on every .nid and .nhf in a folder (not its subfolders) and writes one CSV per
file (`<file>_spectroscopy.csv`, like *Export results*) and `results.csv` with one row per file: number of curves,
analyzed, failed, left out, then mean, standard deviation, median and N of every value, and an error column for files
that could not be read. By default each file's own spring constant and sensitivity are used; untick the box to use
those of the page for all files. Fit regions only fit maps of the same size as the one they were drawn on.

The same from the command line (stand-alone: `NSSA-batch.exe --spectroscopy ...`):

```
python main.py --spectroscopy settings.json D:\data\maps --out D:\data\maps\results
python main.py --spectroscopy settings.json a.nid b.nhf --out results --step 4 --page-spring-constant
```

`--step n` overrides the saved step (every n-th line and column), `--page-spring-constant` uses the spring constant and
sensitivity saved with the settings. The exit code is 1 if a file failed.

**Scripts**: the calculations (contact models, force curve processing, results, I-V) are NSFopen's
(`NSFopen.process.contact`, `spectroscopy`, `iv`, NSFopen 2.4 or newer), so a script and the app give the same
numbers. Example (`from NSFopen.process import spectroscopy`): `spec = spectroscopy.SpecFile("map.nid")`,
`results = spectroscopy.ForceAnalysis(spec, spectroscopy.ForceSettings(model="hertz", radius=10e-9)).run()`,
`results.statistics()`, `results.write_csv("out.csv")`.
