CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra
INCLUDES := -Iinclude
SRCDIR := src
TESTDIR := tests

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
MAIN_SRC := main.cpp
TEST_SRC := $(TESTDIR)/test_backprop.cpp

MAIN_TARGET := main.exe
TEST_TARGET := $(TESTDIR)/test_backprop.exe

.PHONY: all compile test run clean

all: compile test run

compile: $(MAIN_TARGET)

$(MAIN_TARGET): $(MAIN_SRC) $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(TEST_TARGET): $(TEST_SRC) $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

run: $(MAIN_TARGET)
	./$(MAIN_TARGET)

clean:
	rm -f $(MAIN_TARGET) $(TEST_TARGET)