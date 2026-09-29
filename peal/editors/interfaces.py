"""Base config shared by all PEAL editors.

An editor is an inversion recipe (e.g. edit-friendly DDPM inversion) that
turns an image into latents a sampler can re-run after an edit.
``EditorConfig`` is the pydantic base every concrete editor config extends;
``editor_type`` names the concrete class and ``category`` tells the yaml
loader which config family the file belongs to.
"""

from pydantic import BaseModel


class EditorConfig(BaseModel):
    """Pydantic base config of an editor.

    Parameters
    ----------
    editor_type : str
        Name of the concrete editor class (e.g. ``"DDPMInversion"``); the yaml
        loader uses it to pick the concrete config subclass.
    category : str
        Config family marker, always ``"editor"`` for editors.
    """

    editor_type: str
    category: str = "editor"
