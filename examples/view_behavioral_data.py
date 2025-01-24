import hobj.benchmarks.mut_highvar_benchmark as highvar_benchmark
#import hobj.benchmarks.todo.mut_oneshot_benchmark as oneshot_benchmark
import matplotlib.pyplot as plt
import numpy as np

# %% View data for Experiment 1
experiment1_benchmark = highvar_benchmark.MutatorHighVarBenchmark()
data = experiment1_benchmark.data

stacked_data = []
for subtask in data:
    subtask_data = np.array([
        data[subtask][worker_id] for worker_id in data[subtask]
    ])
    stacked_data.append(subtask_data)
stacked_data = np.concat(stacked_data, axis=0)

plt.figure()
plt.imshow(stacked_data,  aspect = 'auto')
plt.xlabel('trial')
plt.ylabel('session')
plt.title("Performance data")
plt.show()


# %% View processed data for Experiment 2
raise NotImplementedError
experiment2_benchmark = oneshot_benchmark.MutatorOneshotBenchmark()
ds_experiment2 = experiment2_benchmark.ds_behavioral_statistics

plt.figure(figsize=(12, 4))
xx = np.arange(len(ds_experiment2.transformation_id))
xlabels = ds_experiment2.transformation_id.values
plt.errorbar(x=xx, y=ds_experiment2.perf, yerr=np.sqrt(ds_experiment2.hat_var_perf))
plt.xlabel('trial')
plt.ylabel('generalization accuracy')
plt.xticks(xx, xlabels, rotation=45, ha='right')
plt.tight_layout()
plt.show()