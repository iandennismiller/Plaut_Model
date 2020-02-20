# Results

### Feb 19, 2020
>#### Frequency Test A
> * Description: Simulations with frequency calculated as `ln(10/N+2)`
> * Simulation Folders: /freq_tests/feb19_test04 to /freq_tests/feb19_test06
> * .gz files: inside simulation folders, with format `warping-dilation-FREQTESTA-S<seed>D<dilation>O<order>FEB19.csv.gz`

>#### Frequency Test B
> * Description: Simulations with frequency calculated as `ln(10+2)/N`
> * Simulation Folders: /freq_tests/feb19_test01 to /freq_tests/feb19_test03
> * .gz files: inside simulation folders, with format `warping-dilation-FREQTESTB-S<seed>D<dilation>O<order>FEB19.csv.gz`

>#### Frequency Test C
> * Description: Simulations with only anchor 1, and frequencies `ln(10+2)`, `ln(5+2)`, `ln(3.33+2)`
> * Simulation Folders: /freq_tests/feb19_test07 to /freq_tests/feb19_test09
> * .gz files: inside simulation folders, with format `warping-dilation-FREQTESTC-S<seed>D<dilation>O<order>FEB19.csv.gz`



Note: Simulations run after Feb 05, 2020 have the Adam optimizer reset after anchors are added.

### Feb 05, 2020
> * Description: Individual anchor simulations from anchors_new1.csv
> * Simulation Folders: feb05_test01 to feb05_test27
> * .gz files: inside `results/single_anchor`, named: `warping-dilation-<anchor>.csv.gz`
>    * `<anchor>` is the orthography of the anchor

### Jan 21, 2020
 > * Description: Simulation results used in Feb 2020 paper
 > * Simulation Folders: jan21_test01 to jan21_test10
 > * .gz files: `warping-dilation-seed-<seed>-dilation-<dilation>-order-<order>-date-jan21.csv.gz`
 >    * `<seed> = {1, 2}` indicates random number generator seed
 >    * `<dilation> = {1, 2, 3}` indicates N, N/2, N/3 dilation
 >    * `<order> = {1, 3}` indicates starting anchor set
     

