import numpy as np
import matplotlib.pyplot as plt
import json
from numpy.core.numeric import isscalar

from numpy.lib.type_check import isreal

lognormal_around = lambda p10, p90, **kwargs: np.random.lognormal((np.log(p10)+np.log(p90))/2 , (np.log(p90)-np.log(p10))/(2*np.sqrt(2)*0.906194), **kwargs)
# xs = lognormal_around(28, 39, size=100000); xs.sort(); assert abs(28-xs[len(xs)//10]) < 0.5; assert abs(39-xs[len(xs)*9//10]) < 0.5

import dataclasses
import typing as t

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
        else f'{info.display_name}\n(value = {simulation[node]:g})' if np.isscalar(simulation[node]) and np.isreal(simulation[node])
        else f'{info.display_name}\n(value = {simulation[node]:r})' if np.isscalar(simulation[node])
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
      world[func] = func(*[world[parent] for parent in node_info.parents])
    self.nsims = 0
    return world

graph = Graph()

@graph.def_node(display_name='uCov/wk')
def microcovids_per_week():
  return lognormal_around(100, 500, size=graph.nsims)

@graph.def_node(display_name='uCov/umort')
def microcovids_per_micromort():
  return lognormal_around(100, 1000, size=graph.nsims)

@graph.def_node(parents=[microcovids_per_week, microcovids_per_micromort])
def micromorts_per_week(mcpw, mcpmm):
  return mcpw / mcpmm

@graph.def_node(display_name='hr/Talk')
def hour_cost_of_budget_discussion():
  return 2

@graph.def_node(display_name='hr/umort')
def hour_cost_of_micromort():
  return 0.5

@graph.def_node(display_name='uCov/wk budget')
def microcovid_per_week_budget():
  return 500

@graph.def_node(display_name='Over budget?', parents=[microcovids_per_week, microcovid_per_week_budget])
def is_week_over_budget(mcpw, budget):
  return mcpw > budget

@graph.def_node(display_name='lost hr/wk to discussion', parents=[is_week_over_budget, hour_cost_of_budget_discussion])
def weekly_hour_cost_to_disc(over_budget, cpbd):
  return over_budget*cpbd

@graph.def_node(display_name='lost hr/wk to disease', parents=[micromorts_per_week, hour_cost_of_micromort])
def weekly_hour_cost_to_disease(mmpw, cpmm):
  return mmpw*cpmm

@graph.def_node(display_name='lost hr/wk', parents=[weekly_hour_cost_to_disc, weekly_hour_cost_to_disease])
def weekly_hour_cost(disc, disease):
  return disc + disease

sim = graph.simulate(nsims=100000)
print(graph.graphviz(sim))
print({f.__name__: v for f, v in sim.items()})
