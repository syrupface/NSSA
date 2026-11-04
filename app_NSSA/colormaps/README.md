# Your own color maps

Put gradient files here to add color maps to the program (menu *Color map - Your color maps*). The format is
Gwyddion's gradient resource; the file name (no extension, or .txt) is the name of the color map:

    Gwyddion resource GwyGradient
    0 0 0 0 1
    0.5 0.8 0.3 0.1 1
    1 1 1 1 1

Each line: position, red, green, blue (and alpha), all from 0 to 1, positions rising from 0 to 1. Gwyddion's own
files (share/gwyddion/gradients of a Gwyddion installation) can be copied here as they are; they are not distributed
with the program.
