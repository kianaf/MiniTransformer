all: main

input.ll: main.cpp
	clang++ main.cpp -std=c++23 -fmodules -fbuiltin-module-map -S -emit-llvm -o input.ll -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

output.ll: input.ll
	opt input.ll -load-pass-plugin=Enzyme/enzyme/build/Enzyme/LLVMEnzyme-17.so -passes=enzyme -o output.ll -S

output_opt.ll: output.ll
	opt output.ll -O2 -o output_opt.ll -S

main: output_opt.ll
	clang++ output_opt.ll -o main

clean:
	rm -f main input.ll output.ll output_opt.ll
	