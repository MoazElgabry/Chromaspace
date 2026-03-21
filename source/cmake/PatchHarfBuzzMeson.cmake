if(NOT DEFINED HB_SOURCE_DIR OR HB_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "HB_SOURCE_DIR is required")
endif()

set(_files
  "${HB_SOURCE_DIR}/meson.build"
  "${HB_SOURCE_DIR}/src/meson.build"
  "${HB_SOURCE_DIR}/src/hb-blob.cc")

foreach(_file IN LISTS _files)
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "Expected HarfBuzz Meson file not found: ${_file}")
  endif()
endforeach()

file(READ "${HB_SOURCE_DIR}/meson.build" _root_meson)
string(REPLACE
  "cpp = meson.get_compiler('cpp')\nnull_dep = dependency('', required: false)"
  "cpp = meson.get_compiler('cpp')\ncc = meson.get_compiler('c')\nnull_dep = dependency('', required: false)"
  _root_meson
  "${_root_meson}")
string(REPLACE
  "if cpp.get_id() == 'clang' and cpp.get_define('_MSC_FULL_VER') != ''"
  "if cpp.get_id() == 'clang' and cpp.get_argument_syntax() == 'msvc'"
  _root_meson
  "${_root_meson}")
string(REPLACE
  "if cpp.has_header(name)"
  "if cpp.has_header(name) or cc.has_header(name)"
  _root_meson
  "${_root_meson}")
file(WRITE "${HB_SOURCE_DIR}/meson.build" "${_root_meson}")

file(READ "${HB_SOURCE_DIR}/src/meson.build" _src_meson)
string(REPLACE
  "if cpp.get_define('_MSC_FULL_VER') != ''"
  "if cpp.get_argument_syntax() == 'msvc'"
  _src_meson
  "${_src_meson}")
string(REPLACE
  "if cpp.get_define('_MSC_FULL_VER') == ''"
  "if cpp.get_argument_syntax() != 'msvc'"
  _src_meson
  "${_src_meson}")
file(WRITE "${HB_SOURCE_DIR}/src/meson.build" "${_src_meson}")

file(READ "${HB_SOURCE_DIR}/src/hb-blob.cc" _hb_blob)
if(NOT _hb_blob MATCHES "#if !defined\\(_WIN32\\) && \\(defined\\(__APPLE__\\) \\|\\| defined\\(__unix__\\) \\|\\| defined\\(__unix\\)\\)")
  string(REPLACE
    "#endif /* HAVE_SYS_MMAN_H */"
    "#endif /* HAVE_SYS_MMAN_H */\n\n#if !defined(_WIN32) && (defined(__APPLE__) || defined(__unix__) || defined(__unix))\n#include <unistd.h>\n#include <sys/mman.h>\n#endif"
    _hb_blob
    "${_hb_blob}")
endif()
file(WRITE "${HB_SOURCE_DIR}/src/hb-blob.cc" "${_hb_blob}")
