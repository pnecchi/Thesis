add_executable(main main.cpp)
target_link_libraries(main thesis)

add_executable(main_thesis main_thesis.cpp)
target_link_libraries(main_thesis thesis)

add_executable(main_multiple main_multiple.cpp)
target_link_libraries(main_multiple thesis)
