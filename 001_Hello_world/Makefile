
target := square
OBJ = square.o 

CC:=nvcc

LDIR =../lib


CFLAGS=--device-debug --debug -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -ccbin g++

LFLAGS=--cudart=static -ccbin g++ -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75

all: $(target)

deps := $(patshust %.o,%.d,$(OBJ))
-include $(deps)
DEPFLAGS = -MMD -MF $(@:.o=.d)

$(target): $(OBJ)
	$(CC) $(LFLAGS) -o $@ $^ 

%.o: %.cu 
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $< 

.PHONY: clean

clean:
	rm -f *.o *~ core *.d $(target)

run:
	./$(target)

