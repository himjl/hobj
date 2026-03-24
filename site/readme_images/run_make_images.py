if __name__ == '__main__':

    # %%
    import matplotlib.pyplot as plt
    from hobj.images import MutatorHighVarImageset
    from hobj.benchmarks import MutatorHighVarBenchmark
    import numpy as np

    imageset = MutatorHighVarImageset()
    benchmark = MutatorHighVarBenchmark()

    target_stats = benchmark.target_statistics
    x = np.arange(100) + 1

    # %%
    plt.figure(figsize=(6, 4))

    for subtask in target_stats.subtask.values:
        y = target_stats.phat.sel(subtask=subtask)
        yerr = np.sqrt(target_stats.varhat_phat.sel(subtask=subtask))
        plt.plot(
            x, y,
            alpha=0.05,
            color='gray',
            lw=1,
        )

    glc = target_stats.phat.mean(dim='subtask')
    glc_se = target_stats.boot_phat.mean('subtask').std('boot_iter', ddof=1)

    plt.errorbar(
        x,
        glc,
        yerr=glc_se,
        color='blue',
        label='human average'
    )
    plt.axhline(0.5, ls=':', color='black', label='chance', zorder=0)
    plt.ylabel('test accuracy')
    plt.xscale('log')
    plt.xlabel('# examples')
    plt.title('Human learning curves')
    plt.xlim([None, 100])
    plt.legend(loc=(1.05, 0.5))

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('./human_learning_curves.svg', transparent=True)
    plt.show()
