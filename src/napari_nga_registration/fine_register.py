import napari
from napari.layers import Image, Shapes 
from . import rough_register
from . import register_core
from napari.utils import progress

class fine_register:
    def __init__(self, sequence: rough_register):
        self.tiles: dict[str, str] = sequence.tiles
        self.reference = sequence.reference
        self.rel_reg = sequence.rel_reg
        self.offsets = sequence.offsets
        self.masks: dict[str, object] = sequence.masks
    
    def fine_register(self):
        """Perform fine registration of the saved tiles."""
        if self._dock is not None:
            napari.current_viewer().window.remove_dock_widget(self._dock)
            self._dock = None
        sequence = register_core.Registration(self.tiles, self.reference, self.rel_reg, self.offsets, self.masks)
        pbar = progress(total=2+len(self.tiles), desc="Registering")

        pbar.update(1, desc="Preparing")
        sequence.prepare()
        pbar.update(2, desc="Finding anchor")
        sequence.find_anchor()
        pbar.update(3, desc="Finalizing")
        sequence.step_sequential_anchor_registration()
        
        
        
