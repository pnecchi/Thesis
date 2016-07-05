add_executable(main main.cpp)
target_link_libraries(main thesis)

add_executable(MainArac MainArac.cpp)
target_link_libraries(MainArac thesis)

add_executable(MainArrsac MainArrsac.cpp)
target_link_libraries(MainArrsac thesis)

add_executable(MainPgpeArac MainPgpeArac.cpp)
target_link_libraries(MainPgpeArac thesis)

add_executable(MainPgpeArrsac MainPgpeArrsac.cpp)
target_link_libraries(MainPgpeArrsac thesis)

add_executable(MainNpgpe MainNpgpe.cpp)
target_link_libraries(MainNpgpe thesis)
