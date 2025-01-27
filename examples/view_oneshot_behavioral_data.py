
from hobj.benchmarks import MutatorOneshotBenchmark
import matplotlib.pyplot as plt
import numpy as np

# %% View processed data for Experiment 2
experiment2_benchmark = MutatorOneshotBenchmark()
target_statistics = experiment2_benchmark.target_statistics

# %%

plt.figure(figsize=(12, 4))
xx = np.arange(len(target_statistics.transformation.values))
xlabels = target_statistics.transformation.values
plt.errorbar(x=xx, y=target_statistics.phat, yerr=np.sqrt(target_statistics.varhat_phat))
plt.xlabel('trial')
plt.ylabel('generalization accuracy')
plt.xticks(xx, xlabels, rotation=45, ha='right')
plt.ylim([0.3, 1.05])
plt.tight_layout()
plt.show()

# %%
plt.plot(target_statistics.varhat_phat, target_statistics.boot_phat.var('boot_iter'), '.')
plt.axis([0, 0.002, 0, 0.002])
plt.show()