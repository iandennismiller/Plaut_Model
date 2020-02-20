NOTES REGARDING DATASET

PLAUT DATASET
1. plaut_dataset.csv is the set of words from PMSPdata.txt, with KFFRQ frequecies, and log(f+2) log frequencies.

2. plaut_dataset_collapsed.csv is same as 3. but with collapsed categories:
   > HFEEXPT, HFE are combined as HFE
   > EXC, EXPT, LFE, LFEEXPT are combined as LFE

NEW ANCHORS AND PROBES (based on Warping_dilutionMP.xlsx)
frequency of the files below is given by ln(10/N + 2)
1. anchors_new1.csv is Anchor1 only
2. anchors_new2.csv is Anchor1 and Anchor2
3. anchors_new3.csv is Anchor1, Anchor2, and Anchor3
4. anchors_swap1.csv is Anchor3 only
5. anchors_swap2.csv is Anchor3 and Anchor2
6. anchors_swap3.csv is Anchor3, Anchor2, and Anchor1 (i.e. identical to anchors_new3.csv


FREQUENCY TEST
frequency of the files below is given by ln(10+2)/N
1. anchors_new1.csv is Anchor1 only
2. anchors_new2.csv is Anchor1 and Anchor2
3. anchors_new3.csv is Anchor1, Anchor2, and Anchor3


OLD ANCHORS AND PROBES
1. anchors.csv is the old set of anchors
2. probes.csv is the old set of probes

OTHER
1. PMSPdata.txt is the original Plaut dataset
2. word_freq.csv is the KFFRQ frequencies.