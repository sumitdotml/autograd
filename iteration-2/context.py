class Context:
    def __init__(self):
        self._saved_tensors = []

    def save_for_backward(self, *tensors):
        self._saved_tensors.extend(tensors)

    @property
    def saved_tensors(self):
        return self._saved_tensors
