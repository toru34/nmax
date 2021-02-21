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

    mode: Literal['train', 'eval'] = 'train'
    modules: tuple[str, ...]
    parameters: tuple[str, ...]
    constants: tuple[str, ...]

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _register_fields(self):
        modules, parameters, constants = list(), list(), list()
        for _name, _var in vars(self).items():
            if _name in self.__annotations__:
                _type = self.__annotations__[_name]
            else:
                _type = Constant
            
            if _type == Module:
                modules.append(_name)
            elif _type == Parameter:
                parameters.append(_name)
            elif _type == Constant:
                constants.append(_name)
            else:
                raise NotImplementedError

        self.modules = tuple(modules)
        self.parameters = tuple(parameters)
        self.constants = tuple(constants)
    
    def eval(self):
        self.mode = 'eval'
        for name in self.modules:
            getattr(self, name).eval()
    
    def train(self):
        self.mode = 'train'
        for name in self.modules:
            getattr(self, name).train()

    def tree_flatten(self):
        leaves = []
        aux = {
            'mode': self.mode,
            'modules': list(),
            'parameters': list(),
            'constants': list()
        }

        for name in self.modules:
            child_leaves, child_aux = getattr(self, name).tree_flatten()
            leaves += child_leaves

            aux['modules'].append({
                'class': getattr(self, name).__class__,
                'name': name,
                'n_vars': len(child_leaves),
                'aux': child_aux,
            })
        
        for name in self.parameters:
            leaves.append(getattr(self, name))

            aux['parameters'].append({
                'name': name,
            })
        
        for name in self.constants:
            aux['constants'].append({
                'name': name,
                'value': getattr(self, name),
            })

        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        module = cls.__new__(cls)

        pointer = 0

        for module_info in aux['modules']:
            child_leaves = leaves[pointer:pointer+module_info['n_vars']]
            child_aux = module_info['aux']

            setattr(
                module,
                module_info['name'],
                module_info['class'].__new__(module_info['class']).tree_unflatten(child_aux, child_leaves)
            )

            pointer += module_info['n_vars']
        
        for parameter_info in aux['parameters']:
            setattr(
                module,
                parameter_info['name'],
                leaves[pointer]
            )

            pointer += 1
        
        for constant_info in aux['constants']:
            setattr(
                module,
                constant_info['name'],
                constant_info['value']
            )
        
        # The following two lines must be in this order (so as not to register 'mode' as constant).
        module._register_fields()
        module.mode = aux['mode']
        
        return module