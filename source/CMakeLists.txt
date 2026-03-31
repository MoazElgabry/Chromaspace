cmake_minimum_required(VERSION 3.20)

project(Chromaspace VERSION 1.0.7 LANGUAGES CXX)
set(CUBE_DISPLAY_VERSION "1.0.7Beta")
set(PLUGIN_BASENAME "Chromaspace")
set(PLUGIN_IDENTIFIER "com.moazelgabry.chromaspace")
set(LOCAL_OFX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ofx-sdk")

set(_CHROMASPACE_COMMON_CANDIDATES
  "${CMAKE_CURRENT_SOURCE_DIR}/../common"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../common"
)
set(WORKSHOP_COMMON_DIR "")
foreach(_candidate IN LISTS _CHROMASPACE_COMMON_CANDIDATES)
  if(EXISTS "${_candidate}/color/ColorManagement.cpp")
    set(WORKSHOP_COMMON_DIR "${_candidate}")
    break()
  endif()
endforeach()
if(WORKSHOP_COMMON_DIR STREQUAL "")
  message(FATAL_ERROR "Could not locate shared common sources beside Chromaspace/source.")
endif()

set(_CHROMASPACE_ICON_CANDIDATES
  "${CMAKE_CURRENT_SOURCE_DIR}/../icons"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../icons"
)
set(SHARED_ICON_DIR "")
foreach(_candidate IN LISTS _CHROMASPACE_ICON_CANDIDATES)
  if(EXISTS "${_candidate}/Icon_Candidate.png")
    set(SHARED_ICON_DIR "${_candidate}")
    break()
  endif()
endforeach()
if(SHARED_ICON_DIR STREQUAL "")
  message(FATAL_ERROR "Could not locate shared icon assets beside Chromaspace/source.")
endif()

set(WORKSHOP_COLOR_SOURCES
  "${WORKSHOP_COMMON_DIR}/color/ColorManagement.cpp"
)
set(WORKSHOP_TEXT_SOURCES
  "${WORKSHOP_COMMON_DIR}/text/FontRenderer.cpp"
)
set(WORKSHOP_TEXT_ASSETS
  "${WORKSHOP_COMMON_DIR}/text/assets/OpenSans-Regular.ttf"
)
set(PLUGIN_ICON_PNG "${SHARED_ICON_DIR}/Icon_Candidate.png")
set(VIEWER_ICON_PNG "${SHARED_ICON_DIR}/Icon_Candidate.png")
set(GENERATED_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

if(NOT EXISTS "${LOCAL_OFX_SDK_DIR}/include/ofxImageEffect.h")
  message(FATAL_ERROR "Missing local OFX SDK at ${LOCAL_OFX_SDK_DIR}.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
  set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig ON CACHE BOOL
      "pkg-config is not required for Chromaspace Windows builds" FORCE)
endif()
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Threads REQUIRED)
find_package(OpenCL QUIET)
if(WIN32 AND NOT DEFINED CUDAToolkit_ROOT)
  set(_CHROMASPACE_WINDOWS_CUDA_CANDIDATES
    "$ENV{CUDA_PATH_V12_9}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
    "$ENV{CUDA_PATH_V12_8}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
    "$ENV{CUDA_PATH_V11_8}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
    "$ENV{CUDA_PATH}"
  )
  foreach(_cuda_candidate IN LISTS _CHROMASPACE_WINDOWS_CUDA_CANDIDATES)
    if(_cuda_candidate AND EXISTS "${_cuda_candidate}/bin/nvcc.exe")
      set(CUDAToolkit_ROOT "${_cuda_candidate}" CACHE PATH "Preferred CUDA toolkit root" FORCE)
      set(CMAKE_CUDA_COMPILER "${_cuda_candidate}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
      break()
    endif()
  endforeach()
endif()
find_package(CUDAToolkit QUIET)
include(CheckLanguage)

set(CHROMASPACE_CUDA_RUNTIME_TARGET CUDA::cudart)
if(UNIX AND NOT APPLE AND CUDAToolkit_FOUND)
  set(CHROMASPACE_CUDA_RUNTIME_TARGET CUDA::cudart_static)
endif()

option(CHROMASPACE_USE_BUNDLED_TEXT_DEPS "Use pinned bundled FreeType/HarfBuzz for viewer text rendering" ON)
option(CHROMASPACE_TEXT_DEPS_ALLOW_SYSTEM "Allow resolving system FreeType/HarfBuzz when bundled text deps are disabled" OFF)
option(CHROMASPACE_FORCE_BUNDLED_FREETYPE "Deprecated compatibility alias for bundled text deps" OFF)
option(CHROMASPACE_FORCE_BUNDLED_HARFBUZZ "Deprecated compatibility alias for bundled text deps" OFF)
mark_as_advanced(CHROMASPACE_FORCE_BUNDLED_FREETYPE CHROMASPACE_FORCE_BUNDLED_HARFBUZZ)
if(CHROMASPACE_FORCE_BUNDLED_FREETYPE OR CHROMASPACE_FORCE_BUNDLED_HARFBUZZ)
  set(CHROMASPACE_USE_BUNDLED_TEXT_DEPS ON CACHE BOOL
      "Use pinned bundled FreeType/HarfBuzz for viewer text rendering" FORCE)
endif()

if(APPLE)
  enable_language(OBJCXX)
endif()
set(CHROMASPACE_VIEWER_HAS_CUDA OFF)
set(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS OFF)
if(CUDAToolkit_FOUND AND NOT APPLE)
  if((NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "-NOTFOUND$")
     AND DEFINED CUDAToolkit_NVCC_EXECUTABLE
     AND EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    # Some cached Visual Studio configure trees keep CMAKE_CUDA_COMPILER at NOTFOUND
    # even though CUDAToolkit already found nvcc successfully.
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
  check_language(CUDA)
endif()
if(CUDAToolkit_FOUND AND NOT APPLE AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
  endif()
  if(WIN32 AND CMAKE_CUDA_FLAGS MATCHES "(^| )--allow-unsupported-compiler( |$)")
  elseif(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
  endif()
  set(CHROMASPACE_VIEWER_HAS_CUDA ON)
  set(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS ON)
endif()

set(OFX_SUPPORT_SOURCES
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsCore.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsImageEffect.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsInteract.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsLog.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsMultiThread.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsParams.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsProperty.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsPropertyValidation.cpp
)

set(CHROMASPACE_PLUGIN_SOURCES
  src/Chromaspace.cpp
  ${WORKSHOP_COLOR_SOURCES}
  $<$<PLATFORM_ID:Darwin>:src/metal/ChromaspaceMetal.mm>
  ${OFX_SUPPORT_SOURCES}
)
if(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS)
  list(APPEND CHROMASPACE_PLUGIN_SOURCES
    src/cuda/ChromaspaceCloudCuda.cu
  )
endif()

add_library(${PLUGIN_BASENAME} SHARED
  ${CHROMASPACE_PLUGIN_SOURCES}
)

target_include_directories(${PLUGIN_BASENAME} PRIVATE
  ${LOCAL_OFX_SDK_DIR}/include
  ${LOCAL_OFX_SDK_DIR}/Support/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${WORKSHOP_COMMON_DIR}
)

if(CUDAToolkit_FOUND)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE CHROMASPACE_HAS_CUDA)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE ${CHROMASPACE_CUDA_RUNTIME_TARGET})
endif()
if(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS)
  set_target_properties(${PLUGIN_BASENAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(OpenCL_FOUND)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE CHROMASPACE_HAS_OPENCL)
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE OpenCL::OpenCL)
endif()

if(MSVC)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${PLUGIN_BASENAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /bigobj /MP /EHsc>)
endif()

if(APPLE)
  target_compile_features(${PLUGIN_BASENAME} PRIVATE cxx_std_17)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE "-framework Foundation" "-framework Metal" "-framework OpenGL")
endif()

if(WIN32)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE opengl32)
endif()

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  OUTPUT_NAME "${PLUGIN_BASENAME}"
)

if(WIN32)
  set(OFX_STAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle")
  set(OFX_RESOURCES_DIR "${OFX_STAGE_ROOT}/Contents/Resources")
  set(OFX_OUTPUT_DIR "${OFX_STAGE_ROOT}/Contents/Win64")
elseif(APPLE)
  set(OFX_STAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle")
  set(OFX_BUNDLE_CONTENTS_DIR "${OFX_STAGE_ROOT}/Contents")
  set(OFX_OUTPUT_DIR "${OFX_BUNDLE_CONTENTS_DIR}/MacOS")
  set(OFX_RESOURCES_DIR "${OFX_BUNDLE_CONTENTS_DIR}/Resources")
elseif(UNIX)
  set(OFX_STAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle")
  set(OFX_RESOURCES_DIR "${OFX_STAGE_ROOT}/Contents/Resources")
  set(OFX_OUTPUT_DIR "${OFX_STAGE_ROOT}/Contents/Linux-x86-64")
endif()
file(MAKE_DIRECTORY "${OFX_OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OFX_RESOURCES_DIR}")
file(MAKE_DIRECTORY "${GENERATED_ASSET_DIR}")

if(APPLE)
  file(MAKE_DIRECTORY "${OFX_BUNDLE_CONTENTS_DIR}")
  file(MAKE_DIRECTORY "${OFX_RESOURCES_DIR}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist.in" "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" "${OFX_BUNDLE_CONTENTS_DIR}/Info.plist")
endif()

set(BUILD_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
set(BUILD_PLUGIN_DIR "${BUILD_ARTIFACT_DIR}/plugin")
set(BUILD_VIEWER_DIR "${BUILD_ARTIFACT_DIR}/viewer")
file(MAKE_DIRECTORY "${BUILD_PLUGIN_DIR}")
file(MAKE_DIRECTORY "${BUILD_VIEWER_DIR}")

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
  PDB_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  PDB_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  PDB_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY "${BUILD_PLUGIN_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG "${BUILD_PLUGIN_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY_RELEASE "${BUILD_PLUGIN_DIR}"
)

find_package(OpenGL REQUIRED)
include(ChromaspaceTextDeps)
chromaspace_configure_text_deps(
  OUT_TARGET CHROMASPACE_TEXT_DEPS_TARGET
  OUT_EXTERNAL_TARGET CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET)
option(CHROMASPACE_FORCE_BUNDLED_GLFW "Use bundled GLFW for Cube Viewer builds" ${APPLE})
if(NOT CHROMASPACE_FORCE_BUNDLED_GLFW)
  find_package(glfw3 QUIET)
  if(NOT glfw3_FOUND AND NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
    endif()
  endif()
endif()
if(CHROMASPACE_FORCE_BUNDLED_GLFW OR (NOT glfw3_FOUND AND NOT TARGET PkgConfig::GLFW3))
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  if(APPLE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  endif()
  FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
    URL_HASH SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  )
  FetchContent_MakeAvailable(glfw)
endif()

set(CHROMASPACE_VIEWER_SOURCES
  tools/cube_viewer_stub/main.cpp
  ${WORKSHOP_COLOR_SOURCES}
  ${WORKSHOP_TEXT_SOURCES}
)
if(APPLE)
  list(APPEND CHROMASPACE_VIEWER_SOURCES
    tools/cube_viewer_stub/metal/ChromaspaceMetal.mm
  )
elseif(CHROMASPACE_VIEWER_HAS_CUDA)
  list(APPEND CHROMASPACE_VIEWER_SOURCES
    tools/cube_viewer_stub/cuda/ChromaspaceCuda.cu
  )
endif()

add_executable(Chromaspace_CubeViewer WIN32 ${CHROMASPACE_VIEWER_SOURCES})
target_compile_features(Chromaspace_CubeViewer PRIVATE cxx_std_17)
target_include_directories(Chromaspace_CubeViewer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${WORKSHOP_COMMON_DIR}
)
if(MSVC)
  target_compile_definitions(Chromaspace_CubeViewer PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(CHROMASPACE_VIEWER_HAS_CUDA)
  target_compile_definitions(Chromaspace_CubeViewer PRIVATE CHROMASPACE_VIEWER_HAS_CUDA)
  target_include_directories(Chromaspace_CubeViewer PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(Chromaspace_CubeViewer PRIVATE ${CHROMASPACE_CUDA_RUNTIME_TARGET})
  set(_CHROMASPACE_VIEWER_CUDA_RUNTIME Shared)
  if(UNIX AND NOT APPLE)
    set(_CHROMASPACE_VIEWER_CUDA_RUNTIME Static)
  endif()
  set_target_properties(Chromaspace_CubeViewer PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY ${_CHROMASPACE_VIEWER_CUDA_RUNTIME}
    CUDA_SEPARABLE_COMPILATION OFF)
endif()
target_link_libraries(Chromaspace_CubeViewer PRIVATE ${CHROMASPACE_TEXT_DEPS_TARGET})
if(CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET)
  add_dependencies(Chromaspace_CubeViewer ${CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET})
endif()
chromaspace_attach_text_assets(Chromaspace_CubeViewer
  ASSETS "${WORKSHOP_TEXT_ASSETS}"
  DESTINATIONS "${BUILD_VIEWER_DIR}" "${OFX_OUTPUT_DIR}")

if(WIN32)
  if(NOT EXISTS "${PLUGIN_ICON_PNG}")
    message(FATAL_ERROR "Missing plugin icon PNG at ${PLUGIN_ICON_PNG}")
  endif()
  if(NOT EXISTS "${VIEWER_ICON_PNG}")
    message(FATAL_ERROR "Missing viewer icon PNG at ${VIEWER_ICON_PNG}")
  endif()

  set(PLUGIN_ICON_ICO "${GENERATED_ASSET_DIR}/Chromaspace.ico")
  set(PLUGIN_ICON_RC "${GENERATED_ASSET_DIR}/Chromaspace.rc")
  set(VIEWER_ICON_ICO "${GENERATED_ASSET_DIR}/Chromaspace_CubeViewer.ico")
  set(VIEWER_ICON_RC "${GENERATED_ASSET_DIR}/Chromaspace_CubeViewer.rc")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/plugin_icon.rc.in" "${PLUGIN_ICON_RC}" @ONLY)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/viewer_icon.rc.in" "${VIEWER_ICON_RC}" @ONLY)

  add_custom_command(
    OUTPUT "${PLUGIN_ICON_ICO}"
    COMMAND powershell -ExecutionPolicy Bypass -File "${CMAKE_CURRENT_SOURCE_DIR}/cmake/png_to_ico.ps1" -InputPng "${PLUGIN_ICON_PNG}" -OutputIco "${PLUGIN_ICON_ICO}"
    DEPENDS "${PLUGIN_ICON_PNG}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/png_to_ico.ps1"
    COMMENT "Generating Windows ICO from ${PLUGIN_ICON_PNG}"
    VERBATIM)

  add_custom_command(
    OUTPUT "${VIEWER_ICON_ICO}"
    COMMAND powershell -ExecutionPolicy Bypass -File "${CMAKE_CURRENT_SOURCE_DIR}/cmake/png_to_ico.ps1" -InputPng "${VIEWER_ICON_PNG}" -OutputIco "${VIEWER_ICON_ICO}"
    DEPENDS "${VIEWER_ICON_PNG}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/png_to_ico.ps1"
    COMMENT "Generating Windows ICO from ${VIEWER_ICON_PNG}"
    VERBATIM)

  target_sources(${PLUGIN_BASENAME} PRIVATE "${PLUGIN_ICON_RC}" "${PLUGIN_ICON_ICO}")
  target_sources(Chromaspace_CubeViewer PRIVATE "${VIEWER_ICON_RC}" "${VIEWER_ICON_ICO}")
endif()

if(TARGET glfw)
  target_link_libraries(Chromaspace_CubeViewer PRIVATE glfw OpenGL::GL)
elseif(TARGET PkgConfig::GLFW3)
  target_link_libraries(Chromaspace_CubeViewer PRIVATE PkgConfig::GLFW3 OpenGL::GL)
else()
  target_link_libraries(Chromaspace_CubeViewer PRIVATE glfw OpenGL::GL)
endif()
if(APPLE)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE GL_SILENCE_DEPRECATION)
  target_compile_definitions(Chromaspace_CubeViewer PRIVATE GL_SILENCE_DEPRECATION)
  target_link_libraries(Chromaspace_CubeViewer PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreVideo" "-framework Metal" "-framework Foundation")
  set(VIEWER_OUTPUT_DIR "${BUILD_VIEWER_DIR}")
elseif(WIN32)
  target_link_libraries(Chromaspace_CubeViewer PRIVATE winmm)
  if(MSVC)
    target_link_options(Chromaspace_CubeViewer PRIVATE $<HOST_LINK:/ENTRY:mainCRTStartup>)
  endif()
  set(VIEWER_OUTPUT_DIR "${BUILD_VIEWER_DIR}")
else()
  target_link_libraries(Chromaspace_CubeViewer PRIVATE Threads::Threads)
  set(VIEWER_OUTPUT_DIR "${BUILD_VIEWER_DIR}")
endif()
set_target_properties(Chromaspace_CubeViewer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${VIEWER_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${VIEWER_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${VIEWER_OUTPUT_DIR}"
  PDB_OUTPUT_DIRECTORY "${VIEWER_OUTPUT_DIR}"
  PDB_OUTPUT_DIRECTORY_DEBUG "${VIEWER_OUTPUT_DIR}"
  PDB_OUTPUT_DIRECTORY_RELEASE "${VIEWER_OUTPUT_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY "${VIEWER_OUTPUT_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG "${VIEWER_OUTPUT_DIR}"
  COMPILE_PDB_OUTPUT_DIRECTORY_RELEASE "${VIEWER_OUTPUT_DIR}")

add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_OUTPUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${PLUGIN_BASENAME}>" "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:${PLUGIN_BASENAME}>"
  COMMAND ${CMAKE_COMMAND} -E rm -f "${OFX_RESOURCES_DIR}/Cube.png"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PLUGIN_ICON_PNG}" "${OFX_RESOURCES_DIR}/${PLUGIN_IDENTIFIER}.png"
  COMMAND ${CMAKE_COMMAND} -E rm -f
    "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.ilk"
    "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.pdb"
    "${OFX_OUTPUT_DIR}/Chromaspace_CubeViewer.ilk"
    "${OFX_OUTPUT_DIR}/Chromaspace_CubeViewer.pdb")

add_custom_command(TARGET Chromaspace_CubeViewer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_OUTPUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:Chromaspace_CubeViewer>" "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_CubeViewer>"
  COMMAND ${CMAKE_COMMAND} -E rm -f
    "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.ilk"
    "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.pdb"
    "${OFX_OUTPUT_DIR}/Chromaspace_CubeViewer.ilk"
    "${OFX_OUTPUT_DIR}/Chromaspace_CubeViewer.pdb")

if(UNIX AND NOT APPLE)
  # Linux hosts launch the bundled viewer directly, so keep the execute bit on both
  # the built helper and the copy staged into the OFX bundle.
  add_custom_command(TARGET Chromaspace_CubeViewer POST_BUILD
    COMMAND /bin/chmod +x "$<TARGET_FILE:Chromaspace_CubeViewer>"
    COMMAND /bin/chmod +x "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_CubeViewer>"
    VERBATIM)
endif()
