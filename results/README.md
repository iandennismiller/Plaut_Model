# Results

### Notes
> * For simulations run after Feb 20, 2020, a simulation specific "label" will be used to distinguish unique simulations.
> * As a result, simulation folders will be named with the following format: `<label>-SXDXOX-<date>` where `X` is a placeholder for random seed, dilation, and anchor order.
> * `.gz` files will be named with the following format:
`warping-dilation-<label>-S<seed>D<dilation>O<order>-<date>.csv.gz`
> * For sets of simulations, the simulations will be placed in a directory with format:
`/results/<label>-<date>`

### Feb 20, 2020
>#### Base Simulation
> * **Description**: "Base" simulation for comparison purposes - essentially identical to the ones in the Feb 2020 paper, with the exception of resetting Adam after adding anchors
> * **Simulation Folders**: `BASE-SXDXOX-feb20` inside the `BASE-feb20` directory
> * **`.gz` files**: inside `BASE-feb20` directory, and also inside simulation folders, with format
`warping-dilation-BASE-SXDXOX-feb20.csv.gz`

>#### Fixed Plaut Frequency
> * **Description**: Frequency for Plaut dataset is kept unchanged (instead of replacing with `ln(2)`) after the anchors are added
> * **Simulation Folders**: `FPF-SXDXOX-feb20` inside the `FPF-feb20` directory
> * **`.gz` files**: inside `FPF-feb20` directory, and also inside simulation folders, with format
`warping-dilation-FPF-SXDXOX-feb20.csv.gz`

### Feb 19, 2020
>#### Frequency Test A
> * **Description**: Simulations with frequency calculated as `ln(10/N+2)`
> * **Simulation Folders**: /freq_tests/feb19_test04 to /freq_tests/feb19_test06
> * **`.gz` files**: inside simulation folders, with format `warping-dilation-FREQTESTA-S<seed>D<dilation>O<order>FEB19.csv.gz`

>#### Frequency Test B
> * **Description**: Simulations with frequency calculated as `ln(10+2)/N`
> * **Simulation Folders**: /freq_tests/feb19_test01 to /freq_tests/feb19_test03
> * **`.gz` files**: inside simulation folders, with format `warping-dilation-FREQTESTB-S<seed>D<dilation>O<order>FEB19.csv.gz`

>#### Frequency Test C
> * **Description**: Simulations with only anchor 1, and frequencies `ln(10+2)`, `ln(5+2)`, `ln(3.33+2)`
> * **Simulation Folders**: /freq_tests/feb19_test07 to /freq_tests/feb19_test09
> * .gz files: inside simulation folders, with format `warping-dilation-FREQTESTC-S<seed>D<dilation>O<order>FEB19.csv.gz`

>#### Notes
>    * `<seed> = {1}` indicates random number generator seed
>    * `<dilation> = {1, 2, 3}` indicates N, N/2, N/3 dilation for Frequency Tests A and B
>    * `<dilation> = {0, 5, 3}` indicates `ln(10+2)`, `ln(5+2)`, `ln(3.33+2)` respectively for Frequency Test C
>    * `<order> = {1}` indicates starting anchor set
>    * **Typo**: For Frequency Test C, order is shown as 3 in the title and inside the csv file. This is a typo; order should be 1.


### Notes
> * For simulations run after Feb 05, 2020, the Adam optimizer is reset after the anchors are added.

### Feb 05, 2020
> * **Description**: Individual anchor simulations from anchors_new1.csv
> * **Simulation Folders**: `feb05_test01` to `feb05_test27` inside `/results/single_anchor`
> * **`.gz` files**: inside `results/single_anchor`, named: `warping-dilation-<anchor>.csv.gz`
>    * `<anchor>` is the orthography of the anchor

### Jan 21, 2020
 > * **Description**: Simulation results used in Feb 2020 paper
 > * **Simulation Folders**: `jan21_test01` to `jan21_test10`, inside `/results/PAPER-jan21`
 > * **`.gz` files**: `warping-dilation-seed-<seed>-dilation-<dilation>-order-<order>-date-jan21.csv.gz`
 >    * `<seed> = {1, 2}` indicates random number generator seed
 >    * `<dilation> = {1, 2, 3}` indicates N, N/2, N/3 dilation
 >    * `<order> = {1, 3}` indicates starting anchor set
     

