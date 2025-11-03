# ********     Author: Lijun Chang    ******
# ******** Email: ljchang@outlook.com ******
#
CC=g++
CFLAGS=-c -I. -std=c++11 -g

all: ged

ged: .obj/main.o .obj/Application.o
	${CC} .obj/main.o .obj/Application.o -o ged
	rm -rf *o .obj/
	mkdir .obj

.obj/main.o: main.cpp
	${CC} ${CFLAGS} -o .obj/main.o main.cpp

.obj/Application.o: Application.cpp
	${CC} ${CFLAGS} -o .obj/Application.o Application.cpp

clean:
	rm -rf *o .obj/
	mkdir .obj
