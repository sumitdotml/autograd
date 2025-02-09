class Context:
    """
    A context object to store information needed for the backward pass
    """

    def __init__(self):
        self.saved_tensors = ()
        self.input_shape = None
        self.a_shape = None
        self.b_shape = None

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
