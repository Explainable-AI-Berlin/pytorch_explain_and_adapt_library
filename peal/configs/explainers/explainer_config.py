from pydantic import BaseModel


class ExplainerConfig(BaseModel):
    """
    This class defines the config of a ExplainerConfig.
    """
    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explainer_type: str
    """
    The category of the config.
    """
    category: str = 'explainer'



'''class DiffeoCFConfig(ExplainerConfig):
    """
    This class defines the config of a DiffeoCF.
    """

    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explanation_type: str = "DiffeoCFConfig"
    """
    The maximum number of gradients step done for explaining the network
    """
    gradient_steps: PositiveInt
    """
    The optimizer used for searching the counterfactual
    """
    optimizer: str = "Adam"
    """
    The learning rate used for finding the counterfactual
    """
    learning_rate: float = None
    """
    The desired target confidence.
    Consider the tradeoff between minimality and clarity of counterfactual
    """
    y_target_goal_confidence: float = 0.65
    """
    Whether samples in the current search batch are masked after reaching y_target_goal_confidence
    or whether they are continued to be updated until the last surpasses the threshhold
    """
    use_masking: bool = True
    """
    How much noise to inject into the image while passing through it in the forward pass.
    Helps avoiding adversarial attacks in the case of a weak generator
    """
    img_noise_injection: float = 0.01
    """
    Regularizing factor of the L1 distance in latent space between the latent code of the
    original image and the counterfactual
    """
    l1_regularization: float = 1.0
    """
    Keeps the counterfactual in the high density area of the generative model
    """
    log_prob_regularization: float = 0.0
    """
    Regularization between counterfactual and original in image space to keep similariy high.
    """
    img_regularization: float = 0.0
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.ExplainerConfig"'''
