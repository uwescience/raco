CPPC = g++
CC = gcc

CFLAGS = -ggdb #-O3 #-m64 -Wno-deprecated -fPIC
LIBS = 
INCL = 
COMP = $(CPPC) $(CFLAGS) -c $< $(INCL) 
LINK = $(CPPC) $(CFLAGS) -o $@ $^ $(LIBS)
COMPLINK = $(CPPC) $(CFLAGS) -o $@ $^ $(LIBS) $@.cc


# for 'triangle'
CSWITCHES = -O -DLINUX -I/usr/X11R6/include -L/usr/X11R6/lib
TRILIBDEFS = -DTRILIBRARY

all: query

query: query.o io_util.o counters_util.o
	$(LINK) 

clean:
	rm *.o
	rm query

query.o : query.cc io_util.cc io_util.h counters_util.cc counters_util.h
	$(COMP) 

io_util.o : io_util.cc io_util.h
	$(COMP) 

counters_util.o : counters_util.cc counters_util.h
	$(COMP) 
