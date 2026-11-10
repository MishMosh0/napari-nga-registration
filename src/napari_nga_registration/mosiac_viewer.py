import napari
from napari.utils.notifications import *
from magicgui import magicgui


    

class mosiac_viewer:
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer

    @staticmethod
    def get_layer_filepaths(viewer: napari.Viewer) -> dict[str, str]:
        """Get file paths of all layers in the viewer."""
        filepaths = {}
        for layer in viewer.layers:
            filepaths[layer.name] = layer.source.path if hasattr(layer.source, 'path') else None
        return filepaths
        

def print_files() -> None:
    """print all files in the current viewer"""
    for name, path in mosiac_viewer.get_layer_filepaths(napari.current_viewer()).items():
        print(f'Layer {name} file path: {path}')
    
