CC = mpicc
CFLAGS = -O3 -Wall -fopenmp
LDFLAGS = -lm -fopenmp
TARGET = hybrid_spmv

all: $(TARGET)

$(TARGET): hybrid_spmv.c
	$(CC) $(CFLAGS) -o $(TARGET) hybrid_spmv.c $(LDFLAGS)

clean:
	rm -f $(TARGET)