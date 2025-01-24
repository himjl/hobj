# %% Instantiate benchmarks
from hobj.benchmarks import MutatorHighVarBenchmark

experiment1_benchmark = MutatorHighVarBenchmark()

# %% Instantiate learner
from hobj.learning_models import DummyBinaryLearner
learner = DummyBinaryLearner()

# %% Score
experiment1_result = experiment1_benchmark(learner=learner)
print(experiment1_result)

# %% Visualize results
import matplotlib.pyplot as plt

