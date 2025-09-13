"""
Adapted from the code for "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization
for Worst-Case Generalization"

doi: https://doi.org/10.48550/arXiv.1911.08731
Repository: https://github.com/kohpangwei/group_DRO/tree/master
"""

model_attributes = {
    #    'bert': {
    #        'feature_type': 'text'
    #    },
    #    'inception_v3': {
    #        'feature_type': 'image',
    #        'target_resolution': (299, 299),
    #        'flatten': False
    #    },
    "wideresnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet34": {"feature_type": "image", "target_resolution": None, "flatten": False},
    #    'raw_logistic_regression': {
    #        'feature_type': 'image',
    #        'target_resolution': None,
    #        'flatten': True,
    #    }
}
