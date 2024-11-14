from GridWorld import GridWorld
from GridWorldConstants import *
from actor_critic import ActorCritic, convertStateToArray, evaluate, softmaxPolicy, render_env, runACExperiments, RbfFeaturizer
import numpy as np

env = GridWorld(render_mode=None, num_cops=2, vision=3)

featurizer = RbfFeaturizer(env, 100)

Theta, w, eval_returns = ActorCritic(env, featurizer, evaluate)

env.render_mode = "human"
render_env(env, featurizer, Theta, softmaxPolicy)

env.render_mode = None
#results = runACExperiments(featurizer, evaluate)  # this would produce a figure
