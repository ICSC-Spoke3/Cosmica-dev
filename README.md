# COSMICA Develpment folder

Contains:

- the folders of code versions with following optimizations
- The performance plot with a test sample of ions and 5 even distributed input energies

(The test are run on a single NVIDIA A30 board for benchmark consistency)

## Version history

- V1 Milestone 7 version of the code (with usage of shared memory)
- V2 Improving internal structure using customized compilation flags
- V3 Optimization of the partial computation of stochastic differential equations coefficients
- V5 (under development) Use of the rigidity as main variable instead of kinetic energy

## Performance

All performance indicators are evalueted in 'SimTimePlot_speedup.ipynb'

Performance benchmark on single A30 GPU ![plot1](test_plots/SimExeTimes_compare_codes.jpg)
