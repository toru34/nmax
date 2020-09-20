import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class BaseModule:
    # TODO: make member variables immutable (e.g. dataclass)
    def __init__(self):
        self._eval = False
        # TODO: add option to make some variables unlearnable
    
    def __repr__(self, depth=0):
        depth += 1
        rep = f"{'  ' * depth}{self.__class__.__name__}\n"
        for name, var in vars(self).items():
            if isinstance(var, BaseModule):
                rep += var.__repr__(depth)
            elif isinstance(var, jax.interpreters.xla.DeviceArray):
                # TODO
                pass
            else:
                # TODO
                pass
        return rep
    
    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)
    
    def __call__(self, x):
        return self.forward(x)
    
    def _count_parameters(self):
        n_parameters = 0
        for name, var in vars(self).items():
            if isinstance(var, BaseModule):
                n_parameters += var._count_parameters()
            elif isinstance(var, jax.interpreters.xla.DeviceArray):
                n_parameters += var.size
            else:
                pass
        return n_parameters
    
    def forward(self, x):
        raise NotImplementedError
    
    def eval(self):
        self._eval = True
        for name, var in vars(self).items():
            if isinstance(var, BaseModule):
                get(self, name).eval()
    
    def train(self):
        self._eval = False
        for name, var in vars(self).items():
            if isinstance(var, BaseModule):
                get(self, name).train()
    
    def tree_flatten(self):
        leaves = []
        aux = []

        for name, var in vars(self).items():
            if isinstance(var, BaseModule):
                child_leaves, child_aux = var.tree_flatten()
                leaves += child_leaves

                var_info = {
                    'type': 'module',
                    'class': var.__class__,
                    'name': name,
                    'n_vars': len(child_leaves),
                    'aux': child_aux
                }
            elif isinstance(var, jax.interpreters.xla.DeviceArray) or isinstance(var, jax.interpreters.partial_eval.JaxprTracer):
                leaves.append(var)

                var_info = {
                    'type': 'jaxarray',
                    'name': name
                }
            else:
                var_info = {
                    'type': 'others',
                    'name': name,
                    'var': var
                }
            
            aux.append(var_info)
        return leaves, aux
    
    @classmethod
    def tree_unflatten(cls, aux, leaves):
        module = cls.__new__(cls)
        pointer = 0

        for var_info in aux:
            if var_info['type'] == 'module':
                var_list = leaves[pointer:pointer+var_info['n_vars']]
                pointer += var_info['n_vars']

                var = var_info['class'].__new__(var_info['class']).tree_unflatten(var_info['aux'], var_list)

                setattr(module, var_info['name'], var)
            elif var_info['type'] == 'jaxarray':
                var = leaves[pointer]
                pointer += 1

                setattr(module, var_info['name'], var)
            else:
                setattr(module, var_info['name'], var_info['var'])
        return module