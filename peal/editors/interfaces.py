from pydantic import BaseModel


class EditorConfig(BaseModel):
    """
    This class defines the config of a EditorConfig.
    """
    editor_type: str
    category: str = 'editor'