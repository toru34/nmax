from typing import Any, Union, Literal

import jax


Parameter = Union[jax.Array, jax.interpreters.partial_eval.JaxprTracer]
Constant = Any
RandomState = Union[jax.Array, jax.interpreters.partial_eval.JaxprTracer, None]


class MetaModule(type):
    def __call__(cls, *args, **kwargs):
        """
        """
        instance = super(MetaModule, cls).__call__(*args, **kwargs)

        instance._register_fields()

        assert "forward" in dir(instance), f"forward function must be implemented in {instance.__class__}"

        return instance


@jax.tree_util.register_pytree_node_class
class Module(metaclass=MetaModule):
    _key: RandomState = None
    _mode: Literal["train", "eval"] = "train"
    _modules: tuple[str, ...]
    _parameters: tuple[str, ...]
    _constants: tuple[str, ...]

    def __init_subclass__(cls):
        """
        Register Module class as pytree node, enabling a recursive traverse.
        This is needed to use jax.grad.

        For jax.tree_util.register_pytree_node_class, see https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.register_pytree_node_class.html.
        For pytree in general, see https://jax.readthedocs.io/en/latest/pytrees.html#pytrees.
        """
        jax.tree_util.register_pytree_node_class(cls)
    
    def __call__(self, *args, **kwargs):
        """
        Perform forward propagation.
        """
        return self.forward(*args, **kwargs)

    def _register_fields(self):
        """
        Register each field (member) to appropriate list? based on their type annotations.

        If no type annotation provided, it's considered to constant value and added to _constants.

        _modules: TODO: add doc
        _parameters: TODO: add doc
        _constants: TODO: add doc
        """
        _modules, _parameters, _constants = list(), list(), list()

        for _name, _var in vars(self).items():
            if _name in ("_key", "_mode", "_modules", "_parameters", "_constants"):
                continue
            if _name in self.__annotations__:
                _type = self.__annotations__[_name]
            else:
                _type = Constant # All member variable without type annotation are considered constant.
            
            if _type == Module:
                if _name not in _modules:
                    _modules.append(_name)
            elif _type == Parameter:
                if _name not in _parameters:
                    _parameters.append(_name)
            elif _type == Constant:
                if _name not in _constants:
                    _constants.append(_name)
            else:
                raise NotImplementedError(f"Type annotation for module member must be Module, Parameter, or Constant, but {_type} provided.")

        self._modules = tuple(_modules)
        self._parameters = tuple(_parameters)
        self._constants = tuple(_constants)

    def add_module(self, name, module):
        """
        TODO: add doc
        """
        setattr(self, name, module)
        self.__annotations__[name] = Module
    
    def add_parameter(self, name, parameter):
        """
        TODO: add doc
        """
        setattr(self, name, parameter)
        self.__annotations__[name] = Parameter
    
    def add_constant(self, name, constant):
        """
        TODO: add doc
        """
        setattr(self, name, constant)
        self.__annotations__[name] = Constant
    
    def eval(self):
        """
        Switch from training mode to evaluation mode.
        This often disables the randomness of forward pass, such as masking in dropout layer.
        """
        self._mode = "eval"
        for name in self._modules:
            getattr(self, name).eval()
    
    def train(self):
        """
        Switch from evaluation mode to training mode.
        """
        self._mode = "train"
        for name in self._modules:
            getattr(self, name).train()

    def get_random_state(self) -> RandomState:
        """
        TODO: add doc
        """
        assert isinstance(
            self._key, RandomState
        ), "Random state has not been initialised."
        return self._key

    def initialise_random_state(self, key: RandomState):
        """
        Initialise every module's random state recursively and return a newly initialised one.
        """
        leaves, aux = self.tree_flatten()

        return self.tree_unflatten(aux, leaves, key)

    def tree_flatten(self):
        """
        TODO: add doc
        """
        leaves = []
        aux = {
            "_mode": self._mode,
            "_modules": list(),
            "_parameters": list(),
            "_constants": list()
        }

        for name in self._modules:
            child_leaves, child_aux = getattr(self, name).tree_flatten()
            leaves += child_leaves

            aux["_modules"].append({
                "class": getattr(self, name).__class__,
                "name": name,
                "n_vars": len(child_leaves),
                "aux": child_aux,
            })
        
        for name in self._parameters:
            leaves.append(getattr(self, name))

            aux["_parameters"].append({
                "name": name,
            })
        
        for name in self._constants:
            aux["_constants"].append({
                "name": name,
                "value": getattr(self, name),
            })

        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves, key=None):
        """
        TODO: add doc
        """
        module = cls.__new__(cls)

        pointer = 0
        module._key = key

        for module_info in aux["_modules"]:
            child_leaves = leaves[pointer : pointer + module_info["n_vars"]]
            child_aux = module_info["aux"]

            if key is not None:
                key = jax.random.split(key, num=1)[0]

            module.add_module(
                name=module_info["name"],
                module=module_info["class"].__new__(module_info["class"]).tree_unflatten(child_aux, child_leaves, key)
            )

            pointer += module_info["n_vars"]
        
        for parameter_info in aux["_parameters"]:
            module.add_parameter(
                name=parameter_info["name"],
                parameter=leaves[pointer],
            )

            pointer += 1
        
        for constant_info in aux["_constants"]:
            module.add_constant(
                name=constant_info["name"],
                constant=constant_info["value"],
            )
        
        module._register_fields()

        return module


class ModuleTuple(Module):
    def __init__(self, module_tuple):
        """
        TODO: add doc
        """
        for i, module in enumerate(module_tuple):
            self.add_module(f"module{i}", module)
    
    def forward(self, x):
        """
        TODO: add doc
        """
        for _module in self._modules:
            x = getattr(self, _module)(x)
        return x