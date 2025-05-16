# NVCC_FLAGS=--std=c++20 -Icccl/libcudacxx/include -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_100,code=compute_100 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120,code=compute_120 
NVCC_FLAGS=--std=c++20 -Icccl/libcudacxx/include -arch=native
TEST_C_FLAGS=/Od -fsanitize=address
TEST_LIBS=-I$(CURDIR)/lib/libheatmap -I$(CURDIR)/lib/lodepng -I$(CURDIR)/lib/gif-h
TEST_FILES=./lib/lodepng/lodepng.cpp ./lib/libheatmap/heatmap.c ./lib/libheatmap/colorschemes/Spectral.c ./lib/libheatmap/colorschemes/gray.c

libheatmap:
	-git clone https://github.com/lucasb-eyer/libheatmap.git $(CURDIR)/lib/libheatmap

lodepng:
	-git clone https://github.com/lvandeve/lodepng.git $(CURDIR)/lib/lodepng

gif-h:
	-git clone https://github.com/charlietangora/gif-h $(CURDIR)/lib/gif-h

lib: libheatmap lodepng gif-h

make_folders:
	-makedir "./lib"
	-makedir "./build"
	-makedir "./build/obj"
	-makedir "./build/exe"
	-makedir "./out"

init: make_folders lib

clear:
	rd /s /q "./lib/" "./build/" "./out/"

clear_linux:
	rm -r "./lib/" "./build/" "./out/"

./build/obj/character.obj: character.cu character.h
	nvcc -dc $(NVCC_FLAGS) character.cu -o ./build/obj/character.obj

./build/obj/character_debug.obj: character.cu character.h
	nvcc -dc $(NVCC_FLAGS) character.cu -g -Xcompiler "$(TEST_C_FLAGS)" -o ./build/obj/character_debug.obj

./build/obj/simulator.obj: simulator.cu simulator_data.h simulator_methods.h
	nvcc -dc $(NVCC_FLAGS) simulator.cu -o ./build/obj/simulator.obj

./build/obj/simulator_debug.obj: simulator.cu simulator_data.h simulator_methods.h
	nvcc -dc $(NVCC_FLAGS) simulator.cu -g -Xcompiler "$(TEST_C_FLAGS)" -o ./build/obj/simulator_debug.obj

./build/obj/ai.obj: ai.cu ai_data.h ai_methods.h
	nvcc -dc $(NVCC_FLAGS) ai.cu -o ./build/obj/ai.obj

./build/obj/ai_debug.obj: ai.cu ai_data.h ai_methods.h
	nvcc -dc $(NVCC_FLAGS) ai.cu -g -Xcompiler "$(TEST_C_FLAGS)" -o ./build/obj/ai_debug.obj

./build/obj/test.obj: test.cu
	nvcc -dc $(NVCC_FLAGS) $(TEST_LIBS) test.cu -o ./build/obj/test.obj

./build/obj/test_debug.obj: test.cu
	nvcc -dc $(NVCC_FLAGS) $(TEST_LIBS) test.cu -g -Xcompiler "$(TEST_C_FLAGS)" -o ./build/obj/test_debug.obj

./build/exe/test.exe: ./build/obj/character.obj ./build/obj/simulator.obj ./build/obj/ai.obj ./build/obj/test.obj
	nvcc $(NVCC_FLAGS) $(TEST_LIBS) $(TEST_FILES) ./build/obj/character.obj ./build/obj/simulator.obj ./build/obj/ai.obj ./build/obj/test.obj -o ./build/exe/test.exe

./build/exe/test_debug.exe: ./build/obj/character_debug.obj ./build/obj/simulator_debug.obj ./build/obj/ai_debug.obj ./build/obj/test_debug.obj
	nvcc $(NVCC_FLAGS) $(TEST_LIBS) $(TEST_FILES) $(CURDIR)/build/obj/character_debug.obj ./build/obj/simulator_debug.obj ./build/obj/ai_debug.obj ./build/obj/test_debug.obj -g -Xcompiler "$(TEST_C_FLAGS)" -o ./build/exe/test_debug.exe

test_mem: ./build/exe/test_debug.exe
	compute-sanitizer --tool memcheck --leak-check=full ./build/exe/test_debug.exe

test_race: ./build/exe/test_debug.exe
	compute-sanitizer --tool racecheck ./build/exe/test_debug.exe

test_init: ./build/exe/test_debug.exe
	compute-sanitizer --tool initcheck ./build/exe/test_debug.exe

test_sync: ./build/exe/test_debug.exe
	compute-sanitizer --tool synccheck ./build/exe/test_debug.exe

all: ./build/exe/test.exe ./build/exe/test_debug.exe

test: ./build/exe/test.exe
	./build/exe/test.exe

test_debug: ./build/exe/test_debug.exe
	./build/exe/test_debug.exe

./build/obj/character.o: character.cu character.h
	nvcc -dc $(NVCC_FLAGS) character.cu -o ./build/obj/character.o

./build/obj/simulator.o: simulator.cu simulator_data.h simulator_methods.h
	nvcc -dc $(NVCC_FLAGS) simulator.cu -o ./build/obj/simulator.o

./build/obj/ai.o: ai.cu ai_data.h ai_methods.h
	nvcc -dc $(NVCC_FLAGS) ai.cu -o ./build/obj/ai.o

./build/obj/test.o: test.cu
	nvcc -dc $(NVCC_FLAGS) $(TEST_LIBS) test.cu -o ./build/obj/test.o

./build/exe/test.out: ./build/obj/character.o ./build/obj/simulator.o ./build/obj/ai.obj ./build/obj/test.o
	nvcc $(NVCC_FLAGS) $(TEST_LIBS) $(TEST_FILES) ./build/obj/character.o ./build/obj/simulator.o ./build/obj/ai.o ./build/obj/test.o -o ./build/exe/test.out

test_linux: ./build/exe/test.out