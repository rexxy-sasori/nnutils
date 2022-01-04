A_dim=$1
B_dim=$2
C_dim=$3
mode=$4
# BLOCK=$5
# default value in python scipt
C_blocks=1
Gy=1
infile=matrix_transposed.npy

AT=6
B_blocks=1
CT=2
# input: infile(sparse matrix transpose) infile_bias():bias=np.random.normal(size=(M))
python code_gen_cpu.py --A_dim $A_dim --B_dim $B_dim --C_dim $C_dim --AT $AT --CT $CT --B_blocks $B_blocks --C_blocks $C_blocks --Gy $Gy --infile $infile --outfile testing.cpp --outfile_asm test1.s --x86 --no_relu --infile_bias bias.npy --fuse
gcc -shared -g test1.s -o test.so
icc -fopenmp -I . -qmkl -O3 -march=native -D AT=$AT -D CT=$1 -D C_Blocks=$C_blocks -DA_dim=$A_dim -DINFILE=$infile -D B_dim=$B_dim -D C_dim=$C_dim -D C_blocks=$C_blocks -D X86=1 -D MULTI=0 driver_cpu.cpp -lcnpy -o test -std=c++17
./test