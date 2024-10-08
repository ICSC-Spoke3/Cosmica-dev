set -e
HERE=$PWD
#for dire in  AMS-02 ;
#do
#        cd $dire ;
#        echo $dire
#               python3 $HERE/Script/EvaluateSimulationResult.py
#                python3 $HERE/Script/EvaluateFlux.py      
#        cd .. ; 
#done
#exit
cd D001_00AUfast_v02;
for dire in HelMod-4_ods_Z*;
do
       cd $dire ;
       echo $dire
              python3 $HERE/Script/EvaluateSimulationResult.py
              python3 $HERE/Script/EvaluateFlux.py      
       cd .. ; 
done
cd ..
