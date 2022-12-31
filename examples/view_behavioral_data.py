import hobj.benchmarks.leedicarlo_benchmarks.MutatorHighVar.mut_highvar_benchmark as highvar_benchmark
import hobj.benchmarks.leedicarlo_benchmarks.MutatorOneshot.mut_oneshot_benchmark as oneshot_benchmark
import matplotlib.pyplot as plt
import numpy as np

# %% View processed data for Experiment 1
experiment1_benchmark = highvar_benchmark.MutatorHighVarBenchmark()
ds_experiment1 = experiment1_benchmark.ds_behavioral_statistics

plt.figure()
plt.errorbar(x=np.arange(100) + 1, y=ds_experiment1.glc, yerr=ds_experiment1.glc_std)
plt.xlabel('trial')
plt.ylabel('subtask-and-subject averaged accuracy')
plt.show()

# %% View processed data for Experiment 2
experiment2_benchmark = oneshot_benchmark.MutatorOneshotBenchmark()
ds_experiment2 = experiment2_benchmark.ds_behavioral_statistics

plt.figure(figsize = (12, 4))
xx = np.arange(len(ds_experiment2.transformation_id))
xlabels = ds_experiment2.transformation_id.values
plt.errorbar(x=xx, y=ds_experiment2.perf, yerr=np.sqrt(ds_experiment2.hat_var_perf))
plt.xlabel('trial')
plt.ylabel('generalization accuracy')
plt.xticks(xx, xlabels, rotation = 45, ha = 'right')
plt.tight_layout()
plt.show()
