FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ##########
# # Target #
# ##########
add_executable(${PROJECT_NAME}-test
  TestLDL.cpp
)
target_link_libraries(${PROJECT_NAME}-test
  ${PROJECT_NAME}
  GTest::Main
)
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}-test)