class ExplainerInterface:
    def explain_batch(self, batch, **args):
        raise NotImplementedError
