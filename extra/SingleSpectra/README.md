# caso singolo

in questo caso si cerca di simulare i flusso di protoni visti da AMS-02
i dati sono nel file
`SingleSpectra/Dati/Rigidity_AMS-02_PhysRep2021_Proton.dat`

il LIS è un file in formato fits
`SingleSpectra/LIS/all_nuclei_sets_nuclei_57_All_2023_final`

un esempio di output del codice montecarlo è 
- `SingleSpectra/ContiFiniti/Deuteron_20110509_20180516_r00100_lat00000_matrix_26399.dat`
- `SingleSpectra/ContiFiniti/Proton_20110509_20180516_r00100_lat00000_matrix_146895.dat`
che poi abbiamo unito in unico file
`SingleSpectra/ContiFiniti/Proton_RawMatrixFile.pkl`
composto da una serie di dizionari


un esempio di output finale è
`SingleSpectra/Figure_Proton_AMS-02_PhysRep2021.png`
`SingleSpectra/Proton_AMS-02_PhysRep2021.txt`

per trasformare il file pkl in flusso modulato ho usato lo script 
`SingleSpectra/PlotFigures_AMS_singlePlot.py`
(i path sono tutti sballati perchè qui ho fatto una selezione)

