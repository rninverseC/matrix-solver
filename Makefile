CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude
SRC := $(wildcard src/*.cpp)
BIN ?= matrix_calculator

.PHONY: all run clean

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)
