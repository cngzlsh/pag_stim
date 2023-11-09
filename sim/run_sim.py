from fcutils.path import from_yaml
from tpd import recorder
import matplotlib.pyplot as plt
import numpy as np

from experiments import SimulationExperiment
from models import FeedForwardModel

EXP_PARAMS_FILES = './pag_stim_exp.yaml'
MODEL_PARAMS_FILES = './feedforward.yaml'

EXPERIMENTS = dict(pag_stim=SimulationExperiment)

experiment = EXPERIMENTS[from_yaml(EXP_PARAMS_FILES)["experiment_name"]]

exp = experiment(FeedForwardModel, MODEL_PARAMS_FILES, EXP_PARAMS_FILES) 

exp.simulate()
exp.make_plots()

# save data
recorder.add_figures(svg=False)
recorder.describe()
exp.save(recorder)


    