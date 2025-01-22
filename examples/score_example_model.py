import hobj.learning_models.learning_model as lm
import hobj.benchmarks.leedicarlo_benchmarks.MutatorHighVar.mut_highvar_benchmark as highvar_benchmark
import hobj.benchmarks.leedicarlo_benchmarks.MutatorOneshot.mut_oneshot_benchmark as oneshot_benchmark

dummy_model = lm.BinaryLearningModel(learner_id='dummylearner')

hv = highvar_benchmark.MutatorHighVarBenchmark()
raise
os = oneshot_benchmark.MutatorOneshotBenchmark()

result = hv.evaluate_model(
    learner=dummy_model,
)
print(result)

result2 = os.evaluate_model(
    learner=dummy_model,
)