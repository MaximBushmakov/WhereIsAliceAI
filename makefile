NVCC_FLAGS=--std=c++20 -lcudnn -Icccl/libcudacxx/include -arch=native
TEST_C_FLAGS=/Od -fsanitize=address
TEST_LIBS=-I$(CURDIR)/lib/libheatmap -I$(CURDIR)/lib/lodepng
TEST_FILES=./lib/lodepng/lodepng.cpp ./lib/libheatmap/heatmap.c ./lib/libheatmap/colorschemes/Spectral.c

libheatmap:
	-git clone https://github.com/lucasb-eyer/libheatmap.git $(CURDIR)/lib/libheatmap

lodepng:
	-git clone https://github.com/lvandeve/lodepng.git $(CURDIR)/lib/lodepng

lib: libheatmap lodepng

make_folders:
	md "./lib"
	md "./build"
	md "./build/obj"
	md "./build/exe"
	md "./out"

init: make_folders lib

clear:
	rd /s /q "./lib/" "./build/" "./out/"

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