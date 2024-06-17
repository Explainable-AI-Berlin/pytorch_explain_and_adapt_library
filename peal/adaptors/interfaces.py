
class Adaptor:
    def run(self):
        """
        Run the adaptor.
        """
        raise NotImplementedError

class AdaptorConfig:
    """
    The config template for an adaptor.
    """
    adaptor_type: str
    """
    The category of the config.
    """
    category: str = 'adaptor'