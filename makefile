all: hello

hello: hello.o
	g++ -o hello hello.o -Wall -Iinclude -lGL -lGLU -lX11

hello.o: hello.cpp
	g++ -c hello.cpp

clean:
	rm hello.o hello
