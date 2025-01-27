# %% Instantiate benchmarks
from hobj.benchmarks import MutatorHighVarBenchmark, MutatorOneshotBenchmark

experiment1_benchmark = MutatorHighVarBenchmark()


# %% Instantiate learner
from hobj.learning_models import RandomGuesser
learner = RandomGuesser()

# %% Score
#experiment1_result = experiment1_benchmark(learner=learner)
#print(experiment1_result)

# %% Visualize results
import matplotlib.pyplot as plt

experiment2_benchmark = MutatorOneshotBenchmark()
experiment2_result = experiment2_benchmark(learner=learner)