# noise_vs_heterog
These are the codes for the paper "Coding in homogeneous and heterogeneous neural population" by Manuel Beiran, Alexandra Kruscha, Jan Benda and Benjamin Lindner (submitted, 2017).

The file "f_network.py" is the library that includes all the functions I used in this project, and needs to be imported to run all other programs.

Figs. 1 and 2 are directly obtained by running the corresponding programs.

The theoretical curves in Fig. 6 can also be obtained by running the fig6 related files. The theoretical calculations are slow (in total about an hour).

The files "sim_two_hom.py" and "sim_two_het.py" are the codes we used (changing the parameters) to obtain the data plotted in Figs. 3, 4 and 5, for the homogeneous and heterogeneous network respectively. Running the code once for a fixed parameter takes ~1 hour. To avoid calculating the equivalent heterogeneous distribution of inputs P(mu) to a homogeneous network with given noise intensity D, I calculated this distributions once for the parameters mu=1.05 and mu=1.3 (as shown in the paper). These data are included in the folders "muA105" and "muA" respectively, and the file "sim_two_het.py" loads it automatically.

