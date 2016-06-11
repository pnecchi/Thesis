add_executable(first main_first.cpp)
target_link_libraries(first thesis)

add_executable(main main.cpp)
target_link_libraries(main thesis)
