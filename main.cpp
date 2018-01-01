#include <add-gpu.h>
#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char const *argv[])
{
	int dev = 0;
	if (argc == 2) dev = atoi(argv[1]);
	hello(dev);
	return 0;
}