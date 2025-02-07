from tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = []
        
    def parameters(self):
        """Returns all parameters of this module"""
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")