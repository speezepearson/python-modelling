from .. import Graph, lognormal_around
import numpy as np

graph = Graph()

@graph.def_node(display_name='uCov/wk')
def microcovids_per_week():
  return lognormal_around(100, 500, size=graph.nsims)

@graph.def_node(display_name='uCov/umort')
def microcovids_per_micromort():
  return lognormal_around(100, 1000, size=graph.nsims)

@graph.def_node(parents={'mcpw': microcovids_per_week, 'mcpmm': microcovids_per_micromort})
def micromorts_per_week(mcpw, mcpmm):
  return mcpw / mcpmm

@graph.def_node(display_name='hr/Talk')
def hour_cost_of_budget_discussion():
  return 2 * np.ones(graph.nsims)

@graph.def_node(display_name='hr/umort')
def hour_cost_of_micromort():
  return 0.5 * np.ones(graph.nsims)

@graph.def_node(display_name='uCov/wk budget')
def microcovid_per_week_budget():
  return 500 * np.ones(graph.nsims)

@graph.def_node(display_name='Over budget?', parents={'mcpw':microcovids_per_week, 'budget':microcovid_per_week_budget})
def is_week_over_budget(mcpw, budget):
  return mcpw > budget

@graph.def_node(display_name='lost hr/wk to discussion', parents={'over_budget':is_week_over_budget, 'cpbd':hour_cost_of_budget_discussion})
def weekly_hour_cost_to_disc(over_budget, cpbd):
  return over_budget*cpbd

@graph.def_node(display_name='lost hr/wk to disease', parents={'mmpw':micromorts_per_week, 'cpmm':hour_cost_of_micromort})
def weekly_hour_cost_to_disease(mmpw, cpmm):
  return mmpw*cpmm

@graph.def_node(display_name='lost hr/wk', parents={'disc':weekly_hour_cost_to_disc, 'disease':weekly_hour_cost_to_disease})
def weekly_hour_cost(disc, disease):
  return disc + disease

sim = graph.simulate(nsims=10000)
print(graph.graphviz(sim))
# print({f.__name__: v for f, v in sim.items()})
