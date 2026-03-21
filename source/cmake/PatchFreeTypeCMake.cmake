if(NOT DEFINED FT_SOURCE_DIR)
  message(FATAL_ERROR "FT_SOURCE_DIR is required")
endif()

set(_ft_cmake "${FT_SOURCE_DIR}/CMakeLists.txt")
if(NOT EXISTS "${_ft_cmake}")
  message(FATAL_ERROR "FreeType CMakeLists.txt not found at ${_ft_cmake}")
endif()

file(READ "${_ft_cmake}" _ft_contents)
string(REPLACE "cmake_minimum_required(VERSION 3.0...3.5)"
               "cmake_minimum_required(VERSION 3.0...3.10)"
               _ft_contents
               "${_ft_contents}")
file(WRITE "${_ft_cmake}" "${_ft_contents}")
