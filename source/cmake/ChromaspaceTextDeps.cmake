include_guard(GLOBAL)

include(CMakeParseArguments)
include(ExternalProject)

set(CHROMASPACE_TEXT_DEPS_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(CHROMASPACE_FREETYPE_URL
  "https://github.com/freetype/freetype/archive/refs/tags/VER-2-13-3.tar.gz")
set(CHROMASPACE_FREETYPE_URL_HASH
  "SHA256=bc5c898e4756d373e0d991bab053036c5eb2aa7c0d5c67e8662ddc6da40c4103")
set(CHROMASPACE_HARFBUZZ_URL
  "https://github.com/harfbuzz/harfbuzz/archive/refs/tags/10.2.0.tar.gz")
set(CHROMASPACE_HARFBUZZ_URL_HASH
  "SHA256=11749926914fd488e08e744538f19329332487a6243eec39ef3c63efa154a578")

function(_chromaspace_append_cmake_cache_arg out_var cache_type variable_name)
  set(_args "${${out_var}}")
  if(DEFINED ${variable_name} AND NOT "${${variable_name}}" STREQUAL "")
    set(_value "${${variable_name}}")
    string(REPLACE ";" "|" _value "${_value}")
    list(APPEND _args "-D${variable_name}:${cache_type}=${_value}")
  endif()
  set(${out_var} "${_args}" PARENT_SCOPE)
endfunction()

function(_chromaspace_get_apple_flag_string out_var)
  set(_apple_flags)
  if(APPLE)
    if(DEFINED CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
      foreach(_arch IN LISTS CMAKE_OSX_ARCHITECTURES)
        list(APPEND _apple_flags "-arch" "${_arch}")
      endforeach()
    endif()
    if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
      list(APPEND _apple_flags "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
  endif()

  if(_apple_flags)
    string(JOIN " " _apple_flag_string ${_apple_flags})
  else()
    set(_apple_flag_string "")
  endif()
  set(${out_var} "${_apple_flag_string}" PARENT_SCOPE)
endfunction()

function(_chromaspace_text_lib_filename out_var base_name)
  if(WIN32)
    set(${out_var} "${base_name}.lib" PARENT_SCOPE)
  else()
    set(${out_var} "lib${base_name}.a" PARENT_SCOPE)
  endif()
endfunction()

function(_chromaspace_require_meson_module)
  set(Python3_FIND_VIRTUALENV FIRST)
  set(Python3_FIND_STRATEGY LOCATION)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import mesonbuild"
    RESULT_VARIABLE _meson_status
    OUTPUT_QUIET
    ERROR_QUIET)
  if(NOT _meson_status EQUAL 0)
    message(FATAL_ERROR
      "Chromaspace bundled text dependencies require Meson's Python module.\n"
      "Install it in a virtual environment, for example:\n"
      "  ${Python3_EXECUTABLE} -m venv .meson-venv\n"
      "  .meson-venv/bin/python -m pip install meson\n"
      "Or opt into system text dependencies with:\n"
      "  -DCHROMASPACE_USE_BUNDLED_TEXT_DEPS=OFF -DCHROMASPACE_TEXT_DEPS_ALLOW_SYSTEM=ON")
  endif()
endfunction()

function(_chromaspace_build_meson_env out_var install_prefix)
  set(_env)
  if(APPLE)
    _chromaspace_get_apple_flag_string(_apple_flag_string)
    if(NOT _apple_flag_string STREQUAL "")
      list(APPEND _env
        "CFLAGS=${_apple_flag_string}"
        "CXXFLAGS=${_apple_flag_string}"
        "LDFLAGS=${_apple_flag_string}")
    endif()
  endif()

  if(UNIX AND NOT WIN32)
    list(APPEND _env "PKG_CONFIG_PATH=${install_prefix}/lib/pkgconfig")
  endif()

  set(${out_var} "${_env}" PARENT_SCOPE)
endfunction()

function(chromaspace_configure_text_deps)
  set(options)
  set(one_value_args OUT_TARGET OUT_EXTERNAL_TARGET)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "" ${ARGN})

  if(NOT ARG_OUT_TARGET)
    message(FATAL_ERROR "chromaspace_configure_text_deps requires OUT_TARGET.")
  endif()

  if(CHROMASPACE_USE_BUNDLED_TEXT_DEPS)
    _chromaspace_require_meson_module()

    if(CMAKE_MAKE_PROGRAM AND EXISTS "${CMAKE_MAKE_PROGRAM}")
      set(_chromaspace_ninja "${CMAKE_MAKE_PROGRAM}")
    else()
      find_program(_chromaspace_ninja NAMES ninja ninja-build REQUIRED)
    endif()

    set(_text_root "${CMAKE_BINARY_DIR}/text-deps")
    set(_text_install_prefix "${_text_root}/install")
    file(MAKE_DIRECTORY "${_text_install_prefix}")

    set(_freetype_cmake_args
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_INSTALL_LIBDIR:STRING=lib
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DBUILD_SHARED_LIBS:BOOL=OFF
      -DFT_DISABLE_ZLIB:BOOL=ON
      -DFT_DISABLE_BZIP2:BOOL=ON
      -DFT_DISABLE_PNG:BOOL=ON
      -DFT_DISABLE_HARFBUZZ:BOOL=ON
      -DFT_DISABLE_BROTLI:BOOL=ON)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args FILEPATH CMAKE_C_COMPILER)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args FILEPATH CMAKE_CXX_COMPILER)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args FILEPATH CMAKE_MAKE_PROGRAM)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args FILEPATH CMAKE_TOOLCHAIN_FILE)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args STRING CMAKE_OSX_ARCHITECTURES)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args STRING CMAKE_OSX_DEPLOYMENT_TARGET)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args PATH CMAKE_OSX_SYSROOT)
    _chromaspace_append_cmake_cache_arg(_freetype_cmake_args STRING CMAKE_MSVC_RUNTIME_LIBRARY)
    if(APPLE)
      _chromaspace_get_apple_flag_string(_freetype_apple_flag_string)
      if(NOT _freetype_apple_flag_string STREQUAL "")
        list(APPEND _freetype_cmake_args
          "-DCMAKE_C_FLAGS:STRING=${_freetype_apple_flag_string}"
          "-DCMAKE_CXX_FLAGS:STRING=${_freetype_apple_flag_string}"
          "-DCMAKE_ASM_FLAGS:STRING=${_freetype_apple_flag_string}")
      endif()
    endif()

    set(_freetype_generator_args CMAKE_GENERATOR "${CMAKE_GENERATOR}")
    if(DEFINED CMAKE_GENERATOR_PLATFORM AND NOT CMAKE_GENERATOR_PLATFORM STREQUAL "")
      list(APPEND _freetype_generator_args CMAKE_GENERATOR_PLATFORM "${CMAKE_GENERATOR_PLATFORM}")
    endif()
    if(DEFINED CMAKE_GENERATOR_TOOLSET AND NOT CMAKE_GENERATOR_TOOLSET STREQUAL "")
      list(APPEND _freetype_generator_args CMAKE_GENERATOR_TOOLSET "${CMAKE_GENERATOR_TOOLSET}")
    endif()
    _chromaspace_text_lib_filename(_freetype_library_name "freetype")
    set(_freetype_library "${_text_install_prefix}/lib/${_freetype_library_name}")

    ExternalProject_Add(chromaspace_freetype_ep
      LIST_SEPARATOR |
      URL "${CHROMASPACE_FREETYPE_URL}"
      URL_HASH "${CHROMASPACE_FREETYPE_URL_HASH}"
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      PREFIX "${_text_root}/freetype"
      INSTALL_DIR "${_text_install_prefix}"
      BUILD_BYPRODUCTS "${_freetype_library}"
      PATCH_COMMAND ${CMAKE_COMMAND} -DFT_SOURCE_DIR=<SOURCE_DIR> -P "${CHROMASPACE_TEXT_DEPS_CMAKE_DIR}/PatchFreeTypeCMake.cmake"
      ${_freetype_generator_args}
      CMAKE_ARGS ${_freetype_cmake_args}
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
      INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target install
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE)

    add_library(chromaspace_freetype STATIC IMPORTED GLOBAL)
    set_target_properties(chromaspace_freetype PROPERTIES
      IMPORTED_LOCATION "${_freetype_library}"
      INTERFACE_INCLUDE_DIRECTORIES "${_text_install_prefix}/include/freetype2")

    _chromaspace_build_meson_env(_meson_env "${_text_install_prefix}")
    set(_meson_command "${Python3_EXECUTABLE}" -m mesonbuild.mesonmain)
    set(_harfbuzz_library "${_text_install_prefix}/lib/libharfbuzz.a")
    if(APPLE)
      set(_harfbuzz_repack_apple ON)
    else()
      set(_harfbuzz_repack_apple OFF)
    endif()
    set(_harfbuzz_install_command
      ${CMAKE_COMMAND}
      -DCHROMASPACE_HARFBUZZ_PYTHON=${Python3_EXECUTABLE}
      -DCHROMASPACE_HARFBUZZ_BUILD_DIR=<BINARY_DIR>
      -DCHROMASPACE_HARFBUZZ_INSTALL_DIR=<INSTALL_DIR>
      -DCHROMASPACE_HARFBUZZ_REPACK_APPLE=${_harfbuzz_repack_apple}
      -P "${CHROMASPACE_TEXT_DEPS_CMAKE_DIR}/InstallHarfBuzz.cmake")

    ExternalProject_Add(chromaspace_harfbuzz_ep
      URL "${CHROMASPACE_HARFBUZZ_URL}"
      URL_HASH "${CHROMASPACE_HARFBUZZ_URL_HASH}"
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      PREFIX "${_text_root}/harfbuzz"
      INSTALL_DIR "${_text_install_prefix}"
      DEPENDS chromaspace_freetype_ep
      BUILD_BYPRODUCTS "${_harfbuzz_library}"
      PATCH_COMMAND ${CMAKE_COMMAND} -DHB_SOURCE_DIR=<SOURCE_DIR> -P "${CHROMASPACE_TEXT_DEPS_CMAKE_DIR}/PatchHarfBuzzMeson.cmake"
      CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env ${_meson_env}
        ${_meson_command}
        setup <BINARY_DIR> <SOURCE_DIR>
        --backend=ninja
        --buildtype=release
        --default-library=static
        --prefix=<INSTALL_DIR>
        --libdir=lib
        --wrap-mode=nodownload
        -Dcmake_prefix_path=${_text_install_prefix}
        -Dfreetype=enabled
        -Dglib=disabled
        -Dgobject=disabled
        -Dcairo=disabled
        -Dchafa=disabled
        -Dicu=disabled
        -Dgraphite2=disabled
        -Dgdi=disabled
        -Ddirectwrite=disabled
        -Dcoretext=disabled
        -Dtests=disabled
        -Dintrospection=disabled
        -Ddocs=disabled
        -Ddoc_tests=false
        -Dutilities=disabled
        -Dbenchmark=disabled
        -Dexperimental_api=false
        -Dwith_libstdcxx=false
      BUILD_COMMAND
        ${CMAKE_COMMAND} -E env ${_meson_env}
        ${_meson_command} compile -C <BINARY_DIR>
      INSTALL_COMMAND ${_harfbuzz_install_command}
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE)

    add_library(chromaspace_harfbuzz STATIC IMPORTED GLOBAL)
    set_target_properties(chromaspace_harfbuzz PROPERTIES
      IMPORTED_LOCATION "${_harfbuzz_library}"
      INTERFACE_INCLUDE_DIRECTORIES "${_text_install_prefix}/include/harfbuzz")

    add_custom_target(ChromaspaceTextDepsExternal)
    add_dependencies(ChromaspaceTextDepsExternal chromaspace_harfbuzz_ep)

    add_library(ChromaspaceTextDeps INTERFACE)
    target_link_libraries(ChromaspaceTextDeps INTERFACE chromaspace_harfbuzz chromaspace_freetype)
    target_include_directories(ChromaspaceTextDeps INTERFACE
      "${_text_install_prefix}/include/freetype2"
      "${_text_install_prefix}/include/harfbuzz")
    if(UNIX AND NOT APPLE)
      target_link_libraries(ChromaspaceTextDeps INTERFACE Threads::Threads m)
    endif()

    set(CHROMASPACE_TEXT_DEPS_PROVIDER "bundled" CACHE INTERNAL "Chromaspace text dependency provider" FORCE)
    set(${ARG_OUT_TARGET} ChromaspaceTextDeps PARENT_SCOPE)
    if(ARG_OUT_EXTERNAL_TARGET)
      set(${ARG_OUT_EXTERNAL_TARGET} ChromaspaceTextDepsExternal PARENT_SCOPE)
    endif()
    return()
  endif()

  if(NOT CHROMASPACE_TEXT_DEPS_ALLOW_SYSTEM)
    message(FATAL_ERROR
      "Chromaspace text dependencies are configured to avoid system packages.\n"
      "Use the default bundled path, or explicitly enable system packages with:\n"
      "  -DCHROMASPACE_TEXT_DEPS_ALLOW_SYSTEM=ON")
  endif()

  find_package(Freetype REQUIRED)
  find_package(harfbuzz CONFIG QUIET)
  if(TARGET harfbuzz::harfbuzz)
    set(_chromaspace_harfbuzz_target harfbuzz::harfbuzz)
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(CHROMASPACE_HARFBUZZ QUIET IMPORTED_TARGET harfbuzz)
    endif()
    if(TARGET PkgConfig::CHROMASPACE_HARFBUZZ)
      set(_chromaspace_harfbuzz_target PkgConfig::CHROMASPACE_HARFBUZZ)
    else()
      message(FATAL_ERROR
        "System HarfBuzz was requested but no usable harfbuzz package was found.\n"
        "Either install HarfBuzz development files for your platform, or use the bundled default.")
    endif()
  endif()

  add_library(ChromaspaceTextDeps INTERFACE)
  target_link_libraries(ChromaspaceTextDeps INTERFACE Freetype::Freetype ${_chromaspace_harfbuzz_target})
  set(CHROMASPACE_TEXT_DEPS_PROVIDER "system" CACHE INTERNAL "Chromaspace text dependency provider" FORCE)
  set(${ARG_OUT_TARGET} ChromaspaceTextDeps PARENT_SCOPE)
  if(ARG_OUT_EXTERNAL_TARGET)
    set(${ARG_OUT_EXTERNAL_TARGET} "" PARENT_SCOPE)
  endif()
endfunction()

function(chromaspace_attach_text_assets target_name)
  set(options)
  set(one_value_args)
  set(multi_value_args ASSETS DESTINATIONS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  foreach(_font_asset IN LISTS ARG_ASSETS)
    if(EXISTS "${_font_asset}")
      foreach(_destination IN LISTS ARG_DESTINATIONS)
        add_custom_command(TARGET ${target_name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory "${_destination}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_font_asset}" "${_destination}"
          VERBATIM)
      endforeach()
    endif()
  endforeach()
endfunction()
