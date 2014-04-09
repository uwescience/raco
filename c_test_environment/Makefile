# WARNING: name environment source files .h or .cc, but not .cpp, which is reserved (.gitignore laziness)

CXX = clang++

CFLAGS = -ggdb -std=c++11 #-O3 #-m64 -Wno-deprecated -fPIC
LIBS = 
INCL = 
COMP = $(CXX) $(CFLAGS) -c $< $(INCL) 
LINK = $(CXX) $(CFLAGS) -o $@ $^ $(LIBS)
COMPLINK = $(CXX) $(CFLAGS) -o $@ $^ $(LIBS) $@.cc


# for 'triangle'
CSWITCHES = -O -DLINUX -I/usr/X11R6/include -L/usr/X11R6/lib
TRILIBDEFS = -DTRILIBRARY

#all: 

%.exe: %.o io_util.o counters_util.o strings.o
	$(LINK) 

clean:
	rm *.o *.exe

io_util.o : io_util.cc io_util.h
	$(COMP) 

counters_util.o : counters_util.cc counters_util.h
	$(COMP) 

strings.o : strings.cc strings.h
	$(COMP) 

%.o : %.cpp 
	$(COMP) 

grappa-install: strings.cc strings.h utils.h
	ln -sf `pwd`/strings.cc $(GRAPPA_HOME)/applications/join
	ln -sf `pwd`/strings.h $(GRAPPA_HOME)/applications/join
	ln -sf `pwd`/utils.h $(GRAPPA_HOME)/applications/join

