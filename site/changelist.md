
We note the following changes since this codebase was released alongisde the paper:


## 1/27/2025: 
* Benchmarks now report MSEn scores which subtract the sampling variance of the human target, meaning that a "correct model" will have an expected score of 0. Before, these benchmarks did not perform this subtraction, meaning a "correct model" would have an expected score equal to the human sampling variance.
* Statistical analyses other than the calculation of MSEn scores, their 95% confidence intervals, and generation of bootstrap resamples have been removed from this codebase.  
* Experiment 2: The lapse-rate corrected performance is now clipped between [0, 1].
* Experiment 2: The estimation of the variance of a lapse-rate corrected performance estimate from using bootstrapping to using the formula $\frac{1}{1-\hat p_{l}} \hat \sigma^2$, which treats the estimated lapse rate as a constant. Because a high number (~1000s) of Binomial observations are used to estimate $\hat p_{l}$, we consider this a decent approximation that is less computationally expensive.
