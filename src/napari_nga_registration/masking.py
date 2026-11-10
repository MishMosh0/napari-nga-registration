"""Manual polygon masking of tiles, one tile at a time.

Each "Save Mask" press records the current tile's Shapes layer and advances
to the next tile; state lives on the ``rough_register`` sequence object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari
from napari.utils.notifications import show_info
from natsort import natsorted

if TYPE_CHECKING:
    from .rough_register import rough_register


def begin_masking(sequence: rough_register) -> None:
    """Hide all tiles, then present the first tile for masking."""
    viewer = napari.current_viewer()
    sequence.mask_order = natsorted(
        name for name in sequence.tiles if name in viewer.layers
    )
    sequence.mask_index = 0
    sequence.masks = {}

    if not sequence.mask_order:
        show_info("No tiles to mask")
        return

    for tile_name in sequence.mask_order:
        viewer.layers[tile_name].visible = False

    sequence.show_widget(sequence.save_mask)
    show_mask_target(sequence)


def show_mask_target(sequence: rough_register) -> None:
    """Show the current tile and add a fresh shapes layer ready for polygon drawing."""
    viewer = napari.current_viewer()
    tile_name = sequence.mask_order[sequence.mask_index]
    viewer.layers[tile_name].visible = True
    mask_layer = viewer.add_shapes(name=f"{tile_name}_mask")
    mask_layer.mode = "add_polygon"
    viewer.layers.selection.active = mask_layer
    show_info(f"Mask tile {sequence.mask_index + 1}/{len(sequence.mask_order)}: {tile_name}")


def advance_mask(sequence: rough_register) -> None:
    """Record the current mask, hide the tile, and move to the next one."""
    viewer = napari.current_viewer()
    if sequence.mask_index >= len(sequence.mask_order):
        return

    tile_name = sequence.mask_order[sequence.mask_index]
    mask_layer = viewer.layers[f"{tile_name}_mask"]
    sequence.masks[tile_name] = mask_layer
    mask_layer.visible = False
    viewer.layers[tile_name].visible = False

    sequence.mask_index += 1
    if sequence.mask_index < len(sequence.mask_order):
        show_mask_target(sequence)
        return

    # all tiles masked: show everything again and hand off to stitching
    for name in sequence.mask_order:
        viewer.layers[name].visible = True
    sequence.show_widget(sequence.rough_stitch)
    show_info("All masks saved")
