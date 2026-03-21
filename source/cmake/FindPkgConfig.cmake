set(PkgConfig_FOUND FALSE)
set(PKG_CONFIG_FOUND FALSE)
set(PKG_CONFIG_EXECUTABLE "" CACHE FILEPATH "pkg-config executable" FORCE)

macro(pkg_check_modules _prefix)
  set(${_prefix}_FOUND FALSE)
endmacro()

macro(pkg_search_module _prefix)
  set(${_prefix}_FOUND FALSE)
endmacro()

function(pkg_get_variable _outvar)
  set(${_outvar} "" PARENT_SCOPE)
endfunction()
