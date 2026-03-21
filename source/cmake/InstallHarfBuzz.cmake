if(NOT DEFINED CHROMASPACE_HARFBUZZ_PYTHON OR CHROMASPACE_HARFBUZZ_PYTHON STREQUAL "")
  message(FATAL_ERROR "CHROMASPACE_HARFBUZZ_PYTHON is required")
endif()

if(NOT DEFINED CHROMASPACE_HARFBUZZ_BUILD_DIR OR CHROMASPACE_HARFBUZZ_BUILD_DIR STREQUAL "")
  message(FATAL_ERROR "CHROMASPACE_HARFBUZZ_BUILD_DIR is required")
endif()

if(NOT DEFINED CHROMASPACE_HARFBUZZ_INSTALL_DIR OR CHROMASPACE_HARFBUZZ_INSTALL_DIR STREQUAL "")
  message(FATAL_ERROR "CHROMASPACE_HARFBUZZ_INSTALL_DIR is required")
endif()

if(NOT DEFINED CHROMASPACE_HARFBUZZ_REPACK_APPLE)
  set(CHROMASPACE_HARFBUZZ_REPACK_APPLE OFF)
endif()

function(_chromaspace_repack_apple_archive object_glob output_archive)
  file(GLOB _objects LIST_DIRECTORIES FALSE ${object_glob})
  if(_objects STREQUAL "")
    message(FATAL_ERROR "No object files matched for ${output_archive}: ${object_glob}")
  endif()

  if(EXISTS "${output_archive}")
    file(REMOVE "${output_archive}")
  endif()

  execute_process(
    COMMAND /usr/bin/libtool -static -o "${output_archive}" ${_objects}
    RESULT_VARIABLE _libtool_status)
  if(NOT _libtool_status EQUAL 0)
    message(FATAL_ERROR "Failed to repack ${output_archive} with Apple libtool")
  endif()

  execute_process(
    COMMAND /usr/bin/ranlib "${output_archive}"
    RESULT_VARIABLE _ranlib_status)
  if(NOT _ranlib_status EQUAL 0)
    message(FATAL_ERROR "Failed to run ranlib for ${output_archive}")
  endif()
endfunction()

execute_process(
  COMMAND "${CHROMASPACE_HARFBUZZ_PYTHON}" -m mesonbuild.mesonmain install -C "${CHROMASPACE_HARFBUZZ_BUILD_DIR}" --no-rebuild
  RESULT_VARIABLE _install_status)
if(NOT _install_status EQUAL 0)
  message(FATAL_ERROR "Meson install failed for bundled HarfBuzz")
endif()

if(CHROMASPACE_HARFBUZZ_REPACK_APPLE)
  _chromaspace_repack_apple_archive(
    "${CHROMASPACE_HARFBUZZ_BUILD_DIR}/src/libharfbuzz.a.p/*.o"
    "${CHROMASPACE_HARFBUZZ_INSTALL_DIR}/lib/libharfbuzz.a")

  if(EXISTS "${CHROMASPACE_HARFBUZZ_BUILD_DIR}/src/libharfbuzz-subset.a.p")
    _chromaspace_repack_apple_archive(
      "${CHROMASPACE_HARFBUZZ_BUILD_DIR}/src/libharfbuzz-subset.a.p/*.o"
      "${CHROMASPACE_HARFBUZZ_INSTALL_DIR}/lib/libharfbuzz-subset.a")
  endif()
endif()
