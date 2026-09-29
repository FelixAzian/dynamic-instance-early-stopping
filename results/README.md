# Reported results

`benchmark_results.csv` is a machine-readable transcription of Tables 5–8 and 17–20 in the non-confidential internship report. Values are reported as mean and 95% confidence interval over five runs.

The `saved_sample_ratio` is the share of sample/epoch pairs excluded from backpropagation, not a direct FLOP measurement. `time_mean_s` is measured wall-clock time and therefore depends on the original hardware and software environment.

## Known report inconsistency

Table 19 reports the CIFAR-10 linear/square-root result with derivative switching as `94.1 ± 0.1%` accuracy, `66.5 ± 4.4%` saved samples, and `1994.1 ± 126.4 s`. Those values match the `0.05` threshold row in Table 13, although the preceding text says the final benchmark uses a first-order threshold of `0.001`. The CSV preserves the final-benchmark value from Table 19. This discrepancy should be resolved from the original run logs before using that row in a publication.
