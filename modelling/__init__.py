import dataclasses
import json
import typing as t

import numpy as np

lognormal_around = lambda p10, p90, **kwargs: np.random.lognormal((np.log(p10)+np.log(p90))/2 , (np.log(p90)-np.log(p10))/(2*np.sqrt(2)*0.906194), **kwargs)
# xs = lognormal_around(28, 39, size=100000); xs.sort(); assert abs(28-xs[len(xs)//10]) < 0.5; assert abs(39-xs[len(xs)*9//10]) < 0.5

Func = t.Callable[..., t.Any]
_F = t.TypeVar('_F', bound=Func)

@dataclasses.dataclass(frozen=True)
class NodeInfo:
  display_name: str
  parents: t.Sequence[Func]

class Graph:
  def __init__(self):
    self.node_infos: t.MutableMapping[Func, NodeInfo] = {}
    self.nsims = 0

  def def_node(self, display_name: t.Optional[str] = None, parents: t.Sequence[Func] = ()) -> t.Callable[[_F], _F]:
    def decorate(f: _F) -> _F:
      self.node_infos[f] = NodeInfo(
        display_name=display_name if (display_name is not None) else f.__name__,
        parents=tuple(parents),
      )
      return f
    return decorate

  def graphviz(self, simulation: t.Optional[t.Mapping[Func, t.Any]] = None) -> str:
    node_names = {
      node: info.display_name if simulation is None
        else f'{info.display_name}\n(mean = {simulation[node].mean():g})' if np.isreal(simulation[node]).all()
        else info.display_name
      for node, info in self.node_infos.items()
    }
    return '\n'.join([
      'digraph G {',
      *[f'  {json.dumps(node_names[parent])} -> {json.dumps(node_names[child])};' for child, info in self.node_infos.items() for parent in info.parents],
      '}',
    ])

  def simulate(self, nsims: int) -> t.Mapping[Func, t.Any]:
    self.nsims = nsims
    world: t.MutableMapping[Func, t.Any] = {}
    for func, node_info in self.node_infos.items():
      vals = func(*[world[parent] for parent in node_info.parents])
      if not isinstance(vals, np.ndarray):
        raise TypeError(f'{func} returned {type(vals)}; expected Numpy array')
      if vals.shape[0] != nsims:
        raise ValueError(f'{func} returned an array of shape {vals.shape}; expected first dimension to be nsims (i.e. {nsims})')
      world[func] = vals
    self.nsims = 0
    return world
