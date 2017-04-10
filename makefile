all: hello

hello: hello.o
	g++ -o hello hello.o -lX11 -lGL


hello.o: hello.cpp
	g++ -c hello.cpp

clean:
	rm hello.o hello
