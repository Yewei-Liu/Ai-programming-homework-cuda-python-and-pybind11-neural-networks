# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/liuyewei/ai programming homework/final/task3"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/liuyewei/ai programming homework/final/task3/build"

# Include any dependencies generated for this target.
include CMakeFiles/mytensor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mytensor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mytensor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mytensor.dir/flags.make

CMakeFiles/mytensor.dir/src/tmp.cu.o: CMakeFiles/mytensor.dir/flags.make
CMakeFiles/mytensor.dir/src/tmp.cu.o: ../src/tmp.cu
CMakeFiles/mytensor.dir/src/tmp.cu.o: CMakeFiles/mytensor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/liuyewei/ai programming homework/final/task3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/mytensor.dir/src/tmp.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/mytensor.dir/src/tmp.cu.o -MF CMakeFiles/mytensor.dir/src/tmp.cu.o.d -x cu -c "/home/liuyewei/ai programming homework/final/task3/src/tmp.cu" -o CMakeFiles/mytensor.dir/src/tmp.cu.o

CMakeFiles/mytensor.dir/src/tmp.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/mytensor.dir/src/tmp.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/mytensor.dir/src/tmp.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/mytensor.dir/src/tmp.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target mytensor
mytensor_OBJECTS = \
"CMakeFiles/mytensor.dir/src/tmp.cu.o"

# External object files for target mytensor
mytensor_EXTERNAL_OBJECTS =

mytensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/mytensor.dir/src/tmp.cu.o
mytensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/mytensor.dir/build.make
mytensor.cpython-310-x86_64-linux-gnu.so: /home/liuyewei/miniconda3/envs/backup/lib/libcudart_static.a
mytensor.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/librt.a
mytensor.cpython-310-x86_64-linux-gnu.so: /home/liuyewei/miniconda3/envs/backup/lib/libpython3.10.so
mytensor.cpython-310-x86_64-linux-gnu.so: CMakeFiles/mytensor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/liuyewei/ai programming homework/final/task3/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA shared module mytensor.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mytensor.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/liuyewei/ai\ programming\ homework/final/task3/build/mytensor.cpython-310-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/mytensor.dir/build: mytensor.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/mytensor.dir/build

CMakeFiles/mytensor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mytensor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mytensor.dir/clean

CMakeFiles/mytensor.dir/depend:
	cd "/home/liuyewei/ai programming homework/final/task3/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/liuyewei/ai programming homework/final/task3" "/home/liuyewei/ai programming homework/final/task3" "/home/liuyewei/ai programming homework/final/task3/build" "/home/liuyewei/ai programming homework/final/task3/build" "/home/liuyewei/ai programming homework/final/task3/build/CMakeFiles/mytensor.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/mytensor.dir/depend

