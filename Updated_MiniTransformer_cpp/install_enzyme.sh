
# Dependencies: all of llvm, cmake, ninja, libxml2

cd Enzyme/enzyme
mkdir build & cd build
export CC=clang
export CXX=clang++
cmake -G Ninja .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_DIR=`llvm-config --cmakedir`
ninja

