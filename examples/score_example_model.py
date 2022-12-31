import hobj.learning_models.learning_model as lm
import hobj.benchmarks.leedicarlo_benchmarks.MutatorHighVar.mut_highvar_benchmark as highvar_benchmark
import hobj.benchmarks.leedicarlo_benchmarks.MutatorOneshot.mut_oneshot_benchmark as oneshot_benchmark

dummy_model = lm.LearningModel(learner_id='dummylearner')
benchmarks = [
    highvar_benchmark.MutatorHighVarBenchmark(),
    oneshot_benchmark.MutatorOneshotBenchmark()
]

force_recompute = False
for benchmark in benchmarks:
    result = benchmark.evaluate_model(
        learner=dummy_model,
        force_recompute=force_recompute
    )
    print(result)
