include(GoogleTest)
find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME}-test
  TestLDL.cpp
)
target_link_libraries(${PROJECT_NAME}-test
  SparseLDL
  Eigen3::Eigen
  GTest::Main
)
gtest_discover_tests(${PROJECT_NAME}-test)