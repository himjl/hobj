import hobj.benchmarks.mut_highvar_benchmark as highvar_benchmark
import matplotlib.pyplot as plt
import numpy as np

# %% View data for Experiment 1
experiment1_benchmark = highvar_benchmark.MutatorHighVarBenchmark()
data = experiment1_benchmark.target_statistics

# %%
plt.figure()
plt.imshow(data.phat, aspect='auto')
plt.xlabel('trial')
plt.ylabel('session')
plt.title("Performance data")
plt.show()

# %%
plt.figure()
plt.errorbar(x=np.arange(100) + 1, y=data.phat.values[0], yerr=np.sqrt(data.varhat_phat.values[0]))
plt.show()

# %%
yerr = data.boot_phat.mean('subtask').std('boot_iter')
glc = data.phat.mean('subtask')
plt.errorbar(x=np.arange(100) + 1, y=glc, yerr=yerr)
plt.show()
