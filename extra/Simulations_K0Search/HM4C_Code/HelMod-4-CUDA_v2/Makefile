COMPILER = nvcc 
OUTPUT = HelMod-4-CUDA
SOURCE = HelMod-4.cu
SOURCE += sources/*.cu

# compiling options
ptxas = --ptxas-options="-v"  # mostra dettagli registri
el    = -rdc=true             # Enable or disable the generation of relocatable device code. If disabled, executable device code is generated. Relocatable device code must be linked before it can be executed.
                              # --> risolve il l'errore ptxas fatal   : Unresolved extern function
openmp = -Xcompiler -fopenmp  # per usare OpenMP per il multithread
opt   = --use_fast_math

#Optimization variables
VAR = -DSetWarpPerBlock=10 
# da rivedere per ottimizzazione tempo
VAR += -DMaxValueTimeStep=50
VAR += -DMinValueTimeStep=0.01

#old - TBD
#CXXFLAG  = -g -w -I  $(INC) 

#profiling
#VAR += -Xcompiler "-g -pg"

all:  
	@echo "Compiling and Linking the Generic code ..."
	$(COMPILER) $(ptxas) $(el) $(VAR) $(openmp) $(opt)  -I headers/ -o $(OUTPUT) $(SOURCE) 
	@echo "done"	
	

	
.PHONY : clean

clean:  
	rm -f $(OUTPUT)
	
	

	


