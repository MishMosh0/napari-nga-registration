import napari
from napari import viewer
from napari.utils.notifications import show_info
from .mosiac_viewer import mosiac_viewer
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Shapes
from natsort import natsorted
import distinctipy
from .rough_mosaic_core import rough_mosaic, cumulative_offsets
from .masking import begin_masking, advance_mask


class rough_register:
    def __init__(self):
        self.tiles: dict[str, str] = {}
        self.reference: Image | None = None
        self.rel_reg: np.ndarray | None = None
        self.offsets: np.ndarray | None = None
        self.masks: dict[str, Shapes] = {}
        # masking state: ordered tile names and the index of the tile being masked
        self.mask_order: list[str] = []
        self.mask_index: int = 0
        # the dock widget currently shown for this step, so it can be swapped out
        self._dock = None

    def show_widget(self, widget) -> None:
        """Replace the current step's dock widget with the next one."""
        window = napari.current_viewer().window
        if self._dock is not None:
            window.remove_dock_widget(self._dock)
        self._dock = window.add_dock_widget(widget)
        
    @magicgui(call_button="Save Tiles")
    def save_tiles(self):
        """Save all layers as tiles."""
        show_info("TILES SAVED!!")
        self.tiles = mosiac_viewer.get_layer_filepaths(napari.current_viewer())
        self.show_widget(self.mask_tiles)
        color_tiles(self)
        return self.tiles
    
    @magicgui(call_button="Mask Tiles")
    def mask_tiles(self):
        """Start masking: show the first tile with an empty shapes layer."""
        begin_masking(self)

    @magicgui(call_button="Save Mask")
    def save_mask(self):
        """Save the current tile's mask and move on to the next tile."""
        advance_mask(self)
    
    @magicgui(call_button="Rough Stitch")
    def rough_stitch(self):
        """Perform rough stitching of the saved tiles."""
        if self._dock is not None:
            napari.current_viewer().window.remove_dock_widget(self._dock)
            self._dock = None
        stitch_tiles(self)
    
    

def rough_register_begin() -> None:
    """begin the sequence with saving tiles"""
    sequence = rough_register()
    widget = napari.current_viewer().window.dock_widgets["Start Registration (napari-nga-registration)"]
    napari.current_viewer().window.remove_dock_widget(widget.native)
    sequence.show_widget(sequence.save_tiles)


def color_tiles(sequence: rough_register) -> None:
    """roughly register all saved tiles in the current viewer"""
    
    # clear all layers in the current viewer before reopening
    napari.current_viewer().layers.clear()
    
    #reopen all files in the viewer with additive blending and normalized positions
    for file_path in sequence.tiles.values():
        napari.current_viewer().open(file_path, blending="additive", opacity=1.0)
    tile_names = natsorted([layer.name for layer in napari.current_viewer().layers if file_path in layer.source.path])
    
    #color each tile with a distinct color
    colors = distinctipy.get_colors(len(tile_names))
    for i in range (0, len(colors), 1):
        tile_name = tile_names[i]
        color = colors[i]
        layer = napari.current_viewer().layers[i]
        layer.colormap = color
        layer.contrast_limits = (0, napari.current_viewer().layers[0].data.max()/2)

def ordered_tile_layers() -> list[Image]:
    """Image layers in the current viewer, in natural order of their names."""
    layers = [layer for layer in napari.current_viewer().layers if isinstance(layer, Image)]
    return natsorted(layers, key=lambda layer: layer.name)


def layer_to_strip(layer: Image) -> np.ndarray:
    """Reduce a layer's data to a 2D float array for template matching."""
    data = np.squeeze(np.asarray(layer.data))
    if data.ndim > 2:
        # collapse leading (e.g. spectral / channel) axes onto the spatial plane
        data = data.reshape(-1, *data.shape[-2:]).mean(axis=0)
    return data.astype(np.float64)


def stitch_tiles(sequence: rough_register, ov: int = 32, numbest: int = 16) -> None:
    """Perform rough stitching of the saved tiles in the current viewer"""
    layers = ordered_tile_layers()
    if len(layers) < 2:
        show_info("Need at least two tiles to stitch")
        return

    strips = [layer_to_strip(layer) for layer in layers]
    sequence.rel_reg = rough_mosaic(strips, ov=ov, numbest=numbest)
    sequence.offsets = cumulative_offsets(sequence.rel_reg)

    for layer, (row, col) in zip(layers, sequence.offsets):
        # translate is given in data-axis order, so pad leading (non-spatial) axes with 0
        layer.translate = (0,) * (layer.ndim - 2) + (int(row), int(col))

    napari.current_viewer().reset_view()
    show_info(f"Rough stitched {len(layers)} tiles")
    
