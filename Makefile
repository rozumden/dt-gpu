all:
	g++ -c -I. main.cpp -o main.cpp.o
	nvcc -arch sm_20 -c -I. hello.cu -o hello.cu.o
	nvcc -arch sm_20 -c -I. local-maxima.cu -o local-maxima.cu.o
	nvcc -arch sm_20 -c -I. distance-transform.cu -o distance-transform.cu.o
	make link
link:
	g++ -o exec hello.cu.o local-maxima.cu.o distance-transform.cu.o init.h main.cpp.o -lcudart
