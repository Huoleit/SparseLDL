cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)
project(SparseLDL VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ##############################################################################
# Code generation with python
# ##############################################################################
set(NUM_STAGES 50)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(CodeGen
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/CodeGenerator/CodeGenerator.py ${NUM_STAGES}
  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/include/SparseLDL/CodeGen/SparseLDLGenerated.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CodeGenerator/templates/Macros.jinja ${CMAKE_CURRENT_SOURCE_DIR}/CodeGenerator/templates/SparseLDL.cpp.jinja
)

# ##############################################################################
# ## Build ##
# ##############################################################################
include_directories(include)
add_library(${PROJECT_NAME}
  src/Types.cpp
)
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
)
target_compile_options(${PROJECT_NAME} PUBLIC -Wfatal-errors -msse3 -msse4.1 -msse4.2)
add_dependencies(${PROJECT_NAME}
  CodeGen
)

# ##############################################################################
# ## Benchmark ##
# ##############################################################################
add_subdirectory(benchmark)

# ##############################################################################
# ## Test ##
# ##############################################################################
enable_testing()
add_subdirectory(test)
