from peal.dependencies.rrclarc.model_training.correction_methods.base_correction_method import Vanilla
from peal.dependencies.rrclarc.model_training.correction_methods.clarc import AClarc, PClarc
from peal.dependencies.rrclarc.model_training.correction_methods.rrc import RRClarc
from peal.dependencies.rrclarc.model_training.correction_methods.rrr import RRR_ExpLogSum


def get_correction_method(method_name):
    CORRECTION_METHODS = {
        'Vanilla': Vanilla,
        'AClarc': AClarc,
        'PClarc': PClarc,
        'RRClarc': RRClarc,
        'RRR_ExpLogSum': RRR_ExpLogSum,
    }

    assert method_name in CORRECTION_METHODS.keys(), f"Correction method '{method_name}' unknown," \
                                                     f" choose one of {list(CORRECTION_METHODS.keys())}"
    return CORRECTION_METHODS[method_name]
