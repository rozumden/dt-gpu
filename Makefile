all:
	g++ -std=c++11 -c -I. main.cpp -o main.cpp.o
	g++ -std=c++11 -c -I. processing-gpu.cpp -o processing-gpu.cpp.o
	/usr/local/cuda/bin/nvcc -arch sm_21 -c -I. local-maxima.cu -o local-maxima.cu.o
	/usr/local/cuda/bin/nvcc -arch sm_21 -c -I. distance-transform.cu -o distance-transform.cu.o
	/usr/local/cuda/bin/nvcc -arch sm_21 -c -I. distance-transform-5x5.cu -o distance-transform-5x5.cu.o
	/usr/local/cuda/bin/nvcc -arch sm_21 -c -I. distance-transform-fast.cu -o distance-transform-fast.cu.o
	g++ -std=c++11 -o dt local-maxima.cu.o distance-transform.cu.o \
		 distance-transform-5x5.cu.o distance-transform-fast.cu.o \
		 init.h processing-gpu.cpp.o main.cpp.o \
		-L/usr/local/cuda/lib64 \
		-I/usr/local/cuda/include -lcudart -I/usr/local/include  -L/usr/local/lib \
		`pkg-config --libs opencv` \
		`pkg-config --libs --cflags opencv`

test:
	g++ -c -I. test.cpp -o test.cpp.o
	/usr/local/cuda/bin/nvcc -arch sm_20 -c -I. hello.cu -o hello.cu.o
	g++ -o testgpu hello.cu.o init.h test.cpp.o -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcudart -I/usr/local/include  -L/usr/local/lib
