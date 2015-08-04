# WARNING: name environment source files .h or .cc, but not .cpp, which is reserved (.gitignore laziness)

#CXX = clang++
CXX=g++

CFLAGS =-ggdb -std=c++11 -O3
#-O3 #-m64 -Wno-deprecated -fPIC

ifneq ($(shell uname), Darwin)
LIBS = -lrt
endif

INCL = 
COMP = $(CXX) $(CFLAGS) -c $< $(INCL) 
LINK = $(CXX) $(CFLAGS) -o $@ $^ $(LIBS)
COMPLINK = $(CXX) $(CFLAGS) -o $@ $^ $(LIBS) $@.cc


# for 'triangle'
CSWITCHES = -O -DLINUX -I/usr/X11R6/include -L/usr/X11R6/lib
TRILIBDEFS = -DTRILIBRARY

#all: 

%.convert: %.convert.cpp convert2bin.h
	g++ -std=c++11 -o $@ $< 

%.exe: %.o io_util.o counters_util.o strings.o
	$(LINK) 

clean:
	rm -f *.o *.exe

io_util.o : io_util.cc io_util.h
	$(COMP) 

counters_util.o : counters_util.cc counters_util.h
	$(COMP) 

strings.o : strings.cc strings.h
	$(COMP) 

dates.o : dates.cc dates.h
	$(COMP)

radish_utils.o : radish_utils.cc radish_utils.h
	$(COMP) 

%.o : %.cpp 
	$(COMP) 

dates_test : dates.o dates_test.cc
	$(LINK)

run_dates_test : dates_test
	./dates_test

libracoc.a: strings.o radish_utils.o dates.o
	ar rcs $@ $^
