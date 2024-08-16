cmake_minimum_required(VERSION 3.1)
project(lcm_position)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

set(ARX_cpp_source
    arx/src/utility.cpp
    arx/src/Hardware/motor.cpp
    arx/src/Hardware/math_ops.cpp
    arx/src/App/arm_control.cpp
)

set(ARX_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/arx)
add_definitions(-DARX_DIRECTORY="${ARX_DIRECTORY}")


include_directories(
    include
    ../lcm_types
    /usr/include/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/arx/include
    ${CMAKE_CURRENT_SOURCE_DIR}/arx/src
)

message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
  set(ARM_ARCH x86_64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(ARCH arm64)
  set(ARM_ARCH arm64)
endif()

link_directories(
  lib  
  lib/unitree/cpp/${ARCH}
#   arx/libcan/${ARCH}
)
  

add_compile_options(-std=c++14)

set(EXTRA_LIBS -pthread -lboost_system -lboost_thread libunitree_legged_sdk.so lcm)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "-O3 -fPIC")

add_executable(lcm_position lcm_position.cpp ${ARX_cpp_source})
add_executable(lcm_position_arm lcm_position_arm.cpp ${ARX_cpp_source})
# add_executable(lcm_position lcm_position.cpp)
# add_executable(lcm_position_arm lcm_position_arm.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    target_link_libraries(lcm_position
        ${EXTRA_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmcan.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmscan.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmarx_s.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmkey.so

    )
    target_link_libraries(lcm_position_arm
        ${EXTRA_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmcan.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmscan.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmarx_s.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarmkey.so
    )

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    target_link_libraries(lcm_position
        ${EXTRA_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libcana.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libcan.a
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarx_s.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libkey.so
    
    )

    target_link_libraries(lcm_position_arm
        ${EXTRA_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libcana.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libcan.a
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libarx_s.so
        ${CMAKE_CURRENT_SOURCE_DIR}/arx/libcan/${ARCH}/libkey.so
    )
endif()



set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME "ARX_CPP")
set(CPACK_PACKAGE_VERSION "0.0.1")
set(CPACK_PACKAGE_CONTACT "js011107@163.com")
set(CMAKE_BUILD_TYPE Debug)
include(CPack)