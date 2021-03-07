from typing import Any, Union, Literal

import jax
import jax.numpy as jnp


Parameter = Union[jax.interpreters.xla.DeviceArray, jax.interpreters.partial_eval.JaxprTracer]
Constant = Union[Any]


class MetaModule(type):
    def __call__(cls, *args, **kwargs):
        instance = super(MetaModule, cls).__call__(*args, **kwargs)

        instance._register_fields()

        assert 'forward' in dir(instance), f'forward function must be implemented in {instance.__class__}'

        return instance

@jax.tree_util.register_pytree_node_class
class Module(metaclass=MetaModule):

    _mode: Literal['train', 'eval'] = 'train'
    _modules: tuple[str, ...]
    _parameters: tuple[str, ...]
    _constants: tuple[str, ...]

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _register_fields(self):
        _modules = list() if '_modules' not in dir(self) else self._modules
        _parameters = list() if '_parameters' not in dir(self) else self._parameters
        _constants = list() if '_constants' not in dir(self) else self._constants

        for _name, _var in vars(self).items():
            if _name in ('_modules', '_parameters', '_constants'):
                continue
    
            if _name in self.__annotations__:
                _type = self.__annotations__[_name]
            else:
                _type = Constant
            
            if _type == Module:
                _modules.append(_name)
            elif _type == Parameter:
                _parameters.append(_name)
            elif _type == Constant:
                _constants.append(_name)
            else:
                raise NotImplementedError

        self._modules = tuple(_modules)
        self._parameters = tuple(_parameters)
        self._constants = tuple(_constants)
    
    def add_module(self, name, module):
        setattr(self, name, module)

        self._modules = (name,) if '_modules' not in dir(self) else tuple(list(self._modules) + [name])
    
    def add_parameter(self, name, parameter):
        setattr(self, name, parameter)

        self._parameters = (name,) if '_parameters' not in dir(self) else tuple(list(self._parameters) + [name])
    
    def add_constant(self, name, constant):
        setattr(self, name, constant)

        self._constants = (name,) if '_constants' not in dir(self) else tuple(list(self._constants) + [name])
    
    def eval(self):
        self._mode = 'eval'
        for name in self._modules:
            getattr(self, name).eval()
    
    def train(self):
        self._mode = 'train'
        for name in self._modules:
            getattr(self, name).train()

    def tree_flatten(self):
        leaves = []
        aux = {
            '_mode': self._mode,
            '_modules': list(),
            '_parameters': list(),
            '_constants': list()
        }

        for name in self._modules:
            child_leaves, child_aux = getattr(self, name).tree_flatten()
            leaves += child_leaves

            aux['_modules'].append({
                'class': getattr(self, name).__class__,
                'name': name,
                'n_vars': len(child_leaves),
                'aux': child_aux,
            })
        
        for name in self._parameters:
            leaves.append(getattr(self, name))

            aux['_parameters'].append({
                'name': name,
            })
        
        for name in self._constants:
            aux['_constants'].append({
                'name': name,
                'value': getattr(self, name),
            })

        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        module = cls.__new__(cls)

        pointer = 0

        for module_info in aux['_modules']:
            child_leaves = leaves[pointer:pointer+module_info['n_vars']]
            child_aux = module_info['aux']

            setattr(
                module,
                module_info['name'],
                module_info['class'].__new__(module_info['class']).tree_unflatten(child_aux, child_leaves)
            )

            pointer += module_info['n_vars']
        
        for parameter_info in aux['_parameters']:
            setattr(
                module,
                parameter_info['name'],
                leaves[pointer]
            )

            pointer += 1
        
        for constant_info in aux['_constants']:
            setattr(
                module,
                constant_info['name'],
                constant_info['value']
            )
        
        # The following two lines must be in this order (so as not to register '_mode' as constant).
        module._register_fields()
        module._mode = aux['_mode']
        
        return module