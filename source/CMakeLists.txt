cmake_minimum_required(VERSION 3.20)

project(Chromaspace VERSION 2.0.0 LANGUAGES CXX)
set(CUBE_DISPLAY_VERSION "2.0 beta")
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
set(PLUGIN_ICON_PNG "${SHARED_ICON_DIR}/Chromaspace 2 to 1.png")
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
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT APPLE)
  find_package(OpenCL QUIET)
endif()
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
if((WIN32 OR (UNIX AND NOT APPLE)) AND CUDAToolkit_FOUND AND TARGET CUDA::cudart_static)
  set(CHROMASPACE_CUDA_RUNTIME_TARGET CUDA::cudart_static)
endif()

option(CHROMASPACE_USE_BUNDLED_TEXT_DEPS "Use pinned bundled FreeType/HarfBuzz for viewer text rendering" ON)
option(CHROMASPACE_TEXT_DEPS_ALLOW_SYSTEM "Allow resolving system FreeType/HarfBuzz when bundled text deps are disabled" OFF)
option(CHROMASPACE_FORCE_BUNDLED_FREETYPE "Deprecated compatibility alias for bundled text deps" OFF)
option(CHROMASPACE_FORCE_BUNDLED_HARFBUZZ "Deprecated compatibility alias for bundled text deps" OFF)
mark_as_advanced(CHROMASPACE_FORCE_BUNDLED_FREETYPE CHROMASPACE_FORCE_BUNDLED_HARFBUZZ)

# The globally discoverable IOSurface source export is migration-only code.
# Keep that lane out of the normal Runtime V2 OFX target; enable it explicitly
# for a diagnostic compatibility build.  The viewer's OpenGL compositor still
# has a separate extraction gate (see the source-boundary note in the runtime
# document) and is not controlled by this option.
option(CHROMASPACE_MACOS_LEGACY_IOSURFACE_COMPAT
  "Enable the explicitly named macOS global-IOSurface compatibility lane"
  OFF)
if(CHROMASPACE_MACOS_LEGACY_IOSURFACE_COMPAT AND NOT APPLE)
  message(FATAL_ERROR
    "CHROMASPACE_MACOS_LEGACY_IOSURFACE_COMPAT is only supported on Apple builds.")
endif()

option(CHROMASPACE_MACOS_OPENGL_VIEWER_COMPAT
  "Enable the legacy OpenGL migration fallback in the monolithic macOS viewer"
  OFF)
if(CHROMASPACE_MACOS_OPENGL_VIEWER_COMPAT AND NOT APPLE)
  message(FATAL_ERROR
    "CHROMASPACE_MACOS_OPENGL_VIEWER_COMPAT is only supported on Apple builds.")
endif()

option(CHROMASPACE_PACKAGE_MACOS_EXCHANGE_HELPERS
  "Package the legacy broker/template exchange helpers inside the OFX bundle"
  OFF)
if(CHROMASPACE_PACKAGE_MACOS_EXCHANGE_HELPERS AND NOT APPLE)
  message(FATAL_ERROR
    "CHROMASPACE_PACKAGE_MACOS_EXCHANGE_HELPERS is only supported on Apple builds.")
endif()
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
  src/ChromaspaceViewerIpcEndpoint.cpp
  ${WORKSHOP_COLOR_SOURCES}
  $<$<PLATFORM_ID:Darwin>:src/metal/ChromaspaceMetal.mm>
  $<$<PLATFORM_ID:Darwin>:src/metal/ChromaspaceSourceExchangeV2.mm>
  $<$<PLATFORM_ID:Darwin>:src/metal/ChromaspaceSourceExchangeState.cpp>
  $<$<PLATFORM_ID:Darwin>:src/metal/ChromaspaceSourceProducerState.cpp>
  $<$<PLATFORM_ID:Darwin>:src/metal/ChromaspaceSourceProducerClientState.cpp>
  $<$<PLATFORM_ID:Darwin>:src/metal/ChromaspaceSourceProducerClient.mm>
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
  target_compile_options(${PLUGIN_BASENAME} PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE
    "-framework Foundation"
    "-framework Metal")
  if(CHROMASPACE_MACOS_LEGACY_IOSURFACE_COMPAT)
    target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
      CHROMASPACE_MACOS_LEGACY_IOSURFACE_COMPAT)
    target_link_libraries(${PLUGIN_BASENAME} PRIVATE
      "-framework IOSurface"
      "-framework OpenGL")
  endif()
  set_target_properties(${PLUGIN_BASENAME} PROPERTIES
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${PLUGIN_IDENTIFIER}"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
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
endif()

set(BUILD_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
set(BUILD_PLUGIN_DIR "${BUILD_ARTIFACT_DIR}/plugin")
set(BUILD_VIEWER_DIR "${BUILD_ARTIFACT_DIR}/viewer")
file(MAKE_DIRECTORY "${BUILD_PLUGIN_DIR}")
file(MAKE_DIRECTORY "${BUILD_VIEWER_DIR}")

if(APPLE)
  if(NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR
      "Python 3 is required for macOS Runtime V2 qualification.")
  endif()
  find_program(CHROMASPACE_XCRUN_EXECUTABLE xcrun REQUIRED)
  set(CHROMASPACE_VIEWER_METAL_SOURCE
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal/ChromaspaceViewer.metal")
  set(CHROMASPACE_VIEWER_METAL_AIR
      "${GENERATED_ASSET_DIR}/metal/ChromaspaceViewer.air")
  set(CHROMASPACE_VIEWER_METALLIB_NAME "ChromaspaceViewer.metallib")
  set(CHROMASPACE_VIEWER_METALLIB
      "${BUILD_VIEWER_DIR}/${CHROMASPACE_VIEWER_METALLIB_NAME}")
  set(_CHROMASPACE_METAL_DEPLOYMENT_ARGS)
  if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    list(APPEND _CHROMASPACE_METAL_DEPLOYMENT_ARGS
         "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()

  add_custom_command(
    OUTPUT "${CHROMASPACE_VIEWER_METALLIB}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_ASSET_DIR}/metal"
    COMMAND "${CHROMASPACE_XCRUN_EXECUTABLE}" -sdk macosx metal
            ${_CHROMASPACE_METAL_DEPLOYMENT_ARGS}
            -c "${CHROMASPACE_VIEWER_METAL_SOURCE}"
            -o "${CHROMASPACE_VIEWER_METAL_AIR}"
    COMMAND "${CHROMASPACE_XCRUN_EXECUTABLE}" -sdk macosx metallib
            "${CHROMASPACE_VIEWER_METAL_AIR}"
            -o "${CHROMASPACE_VIEWER_METALLIB}"
    DEPENDS "${CHROMASPACE_VIEWER_METAL_SOURCE}"
    BYPRODUCTS "${CHROMASPACE_VIEWER_METAL_AIR}"
    COMMENT "Compiling Chromaspace viewer Metal library"
    VERBATIM)
  add_custom_target(Chromaspace_ViewerMetallib
    DEPENDS "${CHROMASPACE_VIEWER_METALLIB}")
endif()

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

include(ChromaspaceTextDeps)
chromaspace_configure_text_deps(
  OUT_TARGET CHROMASPACE_TEXT_DEPS_TARGET
  OUT_EXTERNAL_TARGET CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET)
if(NOT APPLE)
  find_package(OpenGL REQUIRED)
  option(CHROMASPACE_FORCE_BUNDLED_GLFW
    "Use bundled GLFW for Cube Viewer builds" OFF)
  if(NOT CHROMASPACE_FORCE_BUNDLED_GLFW)
    find_package(glfw3 QUIET)
    if(NOT glfw3_FOUND AND NOT WIN32)
      find_package(PkgConfig QUIET)
      if(PkgConfig_FOUND)
        pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
      endif()
    endif()
  endif()
  if(CHROMASPACE_FORCE_BUNDLED_GLFW OR
     (NOT glfw3_FOUND AND NOT TARGET PkgConfig::GLFW3))
    include(FetchContent)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      glfw
      URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
      URL_HASH SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    FetchContent_MakeAvailable(glfw)
  endif()
endif()

set(CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES
  tools/cube_viewer_metal_native/main.mm
  src/ChromaspaceViewerFrameCoordinator.cpp
  src/ChromaspaceViewerDomain.cpp
  src/ChromaspaceViewerCamera.cpp
  src/ChromaspaceViewerCameraInteraction.cpp
  src/ChromaspaceViewerLassoInteraction.cpp
  src/ChromaspaceViewerUiProjection.cpp
  src/ChromaspaceViewerTextLayout.cpp
  src/ChromaspaceViewerUiScene.cpp
  src/ChromaspaceViewerFramePlan.cpp
  src/ChromaspaceViewerLayout.cpp
  src/ChromaspaceViewerSession.cpp
  src/ChromaspaceViewerController.cpp
  src/ChromaspaceViewerWorkspace.cpp
  src/ChromaspaceViewerWorkspacePersistence.cpp
  src/ChromaspaceViewerWorkspaceStore.cpp
  src/ChromaspaceViewerLiveCommand.cpp
  src/ChromaspaceViewerIpcEndpoint.cpp
  src/ChromaspaceViewerCommandServer.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalWorkspaceFrame.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.mm
  tools/cube_viewer_stub/metal/ChromaspaceMetalViewerRuntime.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalViewerRuntime.mm
  tools/cube_viewer_stub/metal/ChromaspaceMetalMemoryPressureMailbox.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalFrameFailure.cpp
  tools/cube_viewer_stub/metal/ChromaspaceFrameRecoveryPolicy.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalTransientArena.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalSubmissionRetention.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.mm
  tools/cube_viewer_stub/metal/ChromaspaceMetalPlotCompiler.cpp
  tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSession.cpp
  tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSession.mm
  tools/cube_viewer_stub/metal/ChromaspaceMetal.mm
  tools/cube_viewer_stub/metal/ChromaspaceSourceViewerClient.mm
  src/metal/ChromaspaceSourceExchangeV2.mm
  src/metal/ChromaspaceSourceViewerClientState.cpp
  src/metal/ChromaspaceSourceExchangeState.cpp
  ${WORKSHOP_COLOR_SOURCES}
  ${WORKSHOP_TEXT_SOURCES}
)

# These sources exist only in the separately named native qualification
# executable.  Keeping them out of the product list makes it impossible for
# fault injection or synthetic-source machinery to ship by source-list drift.
set(CHROMASPACE_METAL_QUALIFICATION_SOURCES
  tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationCampaign.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationFaultBackend.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationSourceAdapter.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationWorkspace.cpp
  tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationNativeSourceBackend.mm
)

if(APPLE)
  set(CHROMASPACE_VIEWER_SOURCES
    ${CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES})
else()
  set(CHROMASPACE_VIEWER_SOURCES
    tools/cube_viewer_stub/main.cpp
    src/ChromaspaceViewerLayout.cpp
    src/ChromaspaceViewerUiScene.cpp
    src/ChromaspaceViewerFramePlan.cpp
    src/ChromaspaceViewerSession.cpp
    src/ChromaspaceViewerController.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerWorkspacePersistence.cpp
    src/ChromaspaceViewerWorkspaceStore.cpp
    src/ChromaspaceViewerLiveCommand.cpp
    src/ChromaspaceViewerIpcEndpoint.cpp
    src/ChromaspaceViewerCommandServer.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameFailure.cpp
    tools/cube_viewer_stub/metal/ChromaspaceFrameRecoveryPolicy.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalViewerRuntime.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotCompiler.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalWorkspaceFrame.cpp
    tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSession.cpp
    src/ChromaspaceViewerDomain.cpp
    src/ChromaspaceViewerCamera.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp
    ${WORKSHOP_COLOR_SOURCES}
    ${WORKSHOP_TEXT_SOURCES}
  )
  if(CHROMASPACE_VIEWER_HAS_CUDA)
    list(APPEND CHROMASPACE_VIEWER_SOURCES
      tools/cube_viewer_stub/cuda/ChromaspaceCuda.cu)
  endif()
endif()

add_executable(Chromaspace_CubeViewer WIN32 ${CHROMASPACE_VIEWER_SOURCES})
if(APPLE)
  add_dependencies(Chromaspace_CubeViewer Chromaspace_ViewerMetallib)
endif()
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
  if((WIN32 OR (UNIX AND NOT APPLE)) AND CHROMASPACE_CUDA_RUNTIME_TARGET STREQUAL "CUDA::cudart_static")
    set(_CHROMASPACE_VIEWER_CUDA_RUNTIME Static)
  endif()
  set_target_properties(Chromaspace_CubeViewer PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY ${_CHROMASPACE_VIEWER_CUDA_RUNTIME}
    CUDA_SEPARABLE_COMPILATION OFF)
endif()
target_link_libraries(Chromaspace_CubeViewer PRIVATE ${CHROMASPACE_TEXT_DEPS_TARGET})
if(APPLE)
  target_include_directories(Chromaspace_CubeViewer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub")
  target_compile_definitions(Chromaspace_CubeViewer PRIVATE
    CHROMASPACE_METAL_NATIVE_ONLY)
  target_compile_options(Chromaspace_CubeViewer PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
endif()
if(CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET)
  add_dependencies(Chromaspace_CubeViewer ${CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET})
endif()
chromaspace_attach_text_assets(Chromaspace_CubeViewer
  ASSETS "${WORKSHOP_TEXT_ASSETS}"
  DESTINATIONS "${BUILD_VIEWER_DIR}")

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

if(APPLE)
  find_library(CHROMASPACE_SECURITY_FRAMEWORK Security REQUIRED)
  target_link_libraries(Chromaspace_CubeViewer PRIVATE
    "-framework Cocoa"
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
    ${CHROMASPACE_SECURITY_FRAMEWORK}
    Threads::Threads)
  set(VIEWER_OUTPUT_DIR "${BUILD_VIEWER_DIR}")
else()
  if(TARGET glfw)
    target_link_libraries(Chromaspace_CubeViewer PRIVATE glfw OpenGL::GL)
  elseif(TARGET PkgConfig::GLFW3)
    target_link_libraries(Chromaspace_CubeViewer PRIVATE
      PkgConfig::GLFW3 OpenGL::GL)
  else()
    target_link_libraries(Chromaspace_CubeViewer PRIVATE glfw OpenGL::GL)
  endif()
endif()
if(WIN32)
  target_link_libraries(Chromaspace_CubeViewer PRIVATE winmm)
  if(MSVC)
    target_link_options(Chromaspace_CubeViewer PRIVATE $<HOST_LINK:/ENTRY:mainCRTStartup>)
  endif()
  set(VIEWER_OUTPUT_DIR "${BUILD_VIEWER_DIR}")
elseif(NOT APPLE)
  target_link_libraries(Chromaspace_CubeViewer PRIVATE Threads::Threads)
  set(VIEWER_OUTPUT_DIR "${BUILD_VIEWER_DIR}")
endif()

if(APPLE)
  # Keep a separately named qualification executable. It shares the complete
  # production closure and adds only the explicitly allowlisted harness seam.
  set(CHROMASPACE_METAL_NATIVE_CUTOVER_DIR
      "${BUILD_ARTIFACT_DIR}/metal-native-cutover")
  add_executable(Chromaspace_MetalViewerCutover
    ${CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES}
    ${CHROMASPACE_METAL_QUALIFICATION_SOURCES})
  add_dependencies(Chromaspace_MetalViewerCutover Chromaspace_ViewerMetallib)
  target_compile_features(Chromaspace_MetalViewerCutover PRIVATE cxx_std_17)
  target_compile_definitions(Chromaspace_MetalViewerCutover PRIVATE
    CHROMASPACE_METAL_NATIVE_ONLY
    CHROMASPACE_METAL_QUALIFICATION)
  target_compile_options(Chromaspace_MetalViewerCutover PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_include_directories(Chromaspace_MetalViewerCutover PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${WORKSHOP_COMMON_DIR}")
  target_link_libraries(Chromaspace_MetalViewerCutover PRIVATE
    ${CHROMASPACE_TEXT_DEPS_TARGET}
    "-framework Cocoa"
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
    ${CHROMASPACE_SECURITY_FRAMEWORK}
    Threads::Threads)
  if(CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET)
    add_dependencies(Chromaspace_MetalViewerCutover
      ${CHROMASPACE_TEXT_DEPS_EXTERNAL_TARGET})
  endif()
  set_target_properties(Chromaspace_MetalViewerCutover PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CHROMASPACE_METAL_NATIVE_CUTOVER_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CHROMASPACE_METAL_NATIVE_CUTOVER_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CHROMASPACE_METAL_NATIVE_CUTOVER_DIR}"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
  add_custom_command(TARGET Chromaspace_MetalViewerCutover POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CHROMASPACE_VIEWER_METALLIB}"
            "$<TARGET_FILE_DIR:Chromaspace_MetalViewerCutover>/${CHROMASPACE_VIEWER_METALLIB_NAME}"
    VERBATIM)
endif()

include(CTest)
if(BUILD_TESTING)
  function(chromaspace_enable_test_assertions target_name)
    if(MSVC)
      target_compile_options(${target_name} PRIVATE /UNDEBUG)
    else()
      target_compile_options(${target_name} PRIVATE -UNDEBUG)
    endif()
  endfunction()
  add_executable(Chromaspace_ViewerLayoutTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerLayoutTests.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerLayoutTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerLayoutTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerLayoutTests)
  add_test(NAME Chromaspace.ViewerLayout
           COMMAND Chromaspace_ViewerLayoutTests)
  add_executable(Chromaspace_ViewerWorkspaceTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerWorkspaceTests.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerWorkspaceTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerWorkspaceTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerWorkspaceTests)
  add_test(NAME Chromaspace.ViewerWorkspace
           COMMAND Chromaspace_ViewerWorkspaceTests)
  add_executable(Chromaspace_ViewerWorkspacePersistenceTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerWorkspacePersistenceTests.cpp
    src/ChromaspaceViewerWorkspacePersistence.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerWorkspacePersistenceTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerWorkspacePersistenceTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerWorkspacePersistenceTests)
  add_test(NAME Chromaspace.ViewerWorkspacePersistence
           COMMAND Chromaspace_ViewerWorkspacePersistenceTests)
  add_executable(Chromaspace_ViewerWorkspaceStoreTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerWorkspaceStoreTests.cpp
    src/ChromaspaceViewerWorkspaceStore.cpp
    src/ChromaspaceViewerWorkspacePersistence.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerWorkspaceStoreTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerWorkspaceStoreTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerWorkspaceStoreTests)
  add_test(NAME Chromaspace.ViewerWorkspaceStore
           COMMAND Chromaspace_ViewerWorkspaceStoreTests)
  add_executable(Chromaspace_ViewerLiveCommandTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerLiveCommandTests.cpp
    src/ChromaspaceViewerLiveCommand.cpp)
  target_compile_features(Chromaspace_ViewerLiveCommandTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerLiveCommandTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(Chromaspace_ViewerLiveCommandTests PRIVATE Threads::Threads)
  chromaspace_enable_test_assertions(Chromaspace_ViewerLiveCommandTests)
  add_test(NAME Chromaspace.ViewerLiveCommand
           COMMAND Chromaspace_ViewerLiveCommandTests)
  add_executable(Chromaspace_ViewerIpcEndpointTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerIpcEndpointTests.cpp
    src/ChromaspaceViewerIpcEndpoint.cpp)
  target_compile_features(Chromaspace_ViewerIpcEndpointTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerIpcEndpointTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerIpcEndpointTests)
  add_test(NAME Chromaspace.ViewerIpcEndpoint
           COMMAND Chromaspace_ViewerIpcEndpointTests)
  add_executable(Chromaspace_ViewerCommandServerTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerCommandServerTests.cpp
    src/ChromaspaceViewerCommandServer.cpp
    src/ChromaspaceViewerLiveCommand.cpp
    src/ChromaspaceViewerIpcEndpoint.cpp)
  target_compile_features(Chromaspace_ViewerCommandServerTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerCommandServerTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(Chromaspace_ViewerCommandServerTests PRIVATE Threads::Threads)
  chromaspace_enable_test_assertions(Chromaspace_ViewerCommandServerTests)
  add_test(NAME Chromaspace.ViewerCommandServer
           COMMAND Chromaspace_ViewerCommandServerTests)
  add_executable(Chromaspace_ViewerUiSceneTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerUiSceneTests.cpp
    src/ChromaspaceViewerUiScene.cpp
    src/ChromaspaceViewerFramePlan.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerUiSceneTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerUiSceneTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerUiSceneTests)
  add_test(NAME Chromaspace.ViewerUiScene
           COMMAND Chromaspace_ViewerUiSceneTests)
  add_executable(Chromaspace_ViewerUiProjectionTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerUiProjectionTests.cpp
    src/ChromaspaceViewerUiProjection.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerDomain.cpp
    src/ChromaspaceViewerFramePlan.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerUiProjectionTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerUiProjectionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerUiProjectionTests)
  add_test(NAME Chromaspace.ViewerUiProjection
           COMMAND Chromaspace_ViewerUiProjectionTests)
  add_executable(Chromaspace_ViewerTextLayoutTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerTextLayoutTests.cpp
    src/ChromaspaceViewerTextLayout.cpp
    ${WORKSHOP_TEXT_SOURCES})
  target_compile_features(Chromaspace_ViewerTextLayoutTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerTextLayoutTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${WORKSHOP_COMMON_DIR})
  target_link_libraries(Chromaspace_ViewerTextLayoutTests PRIVATE
    ${CHROMASPACE_TEXT_DEPS_TARGET})
  chromaspace_enable_test_assertions(Chromaspace_ViewerTextLayoutTests)
  add_test(NAME Chromaspace.ViewerTextLayout
           COMMAND Chromaspace_ViewerTextLayoutTests)
  add_executable(Chromaspace_ViewerSessionTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerSessionTests.cpp
    src/ChromaspaceViewerSession.cpp)
  target_compile_features(Chromaspace_ViewerSessionTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerSessionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerSessionTests)
  add_test(NAME Chromaspace.ViewerSession
           COMMAND Chromaspace_ViewerSessionTests)
  add_executable(Chromaspace_ViewerControllerTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerControllerTests.cpp
    src/ChromaspaceViewerController.cpp
    src/ChromaspaceViewerUiScene.cpp
    src/ChromaspaceViewerFramePlan.cpp
    src/ChromaspaceViewerLayout.cpp
    src/ChromaspaceViewerSession.cpp)
  target_compile_features(Chromaspace_ViewerControllerTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerControllerTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerControllerTests)
  add_test(NAME Chromaspace.ViewerController
           COMMAND Chromaspace_ViewerControllerTests)
  add_executable(Chromaspace_MetalFrameExecutorTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutorTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameFailure.cpp)
  target_compile_features(Chromaspace_MetalFrameExecutorTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_MetalFrameExecutorTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(Chromaspace_MetalFrameExecutorTests)
  add_test(NAME Chromaspace.MetalFrameExecutor
           COMMAND Chromaspace_MetalFrameExecutorTests)
  add_executable(Chromaspace_MetalViewerRuntimeTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalViewerRuntimeTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalViewerRuntime.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp
    tools/cube_viewer_stub/metal/ChromaspaceFrameRecoveryPolicy.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameFailure.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_MetalViewerRuntimeTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_MetalViewerRuntimeTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(Chromaspace_MetalViewerRuntimeTests)
  add_test(NAME Chromaspace.MetalViewerRuntime
           COMMAND Chromaspace_MetalViewerRuntimeTests)
  add_executable(Chromaspace_MetalMemoryPressureMailboxTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalMemoryPressureMailboxTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalMemoryPressureMailbox.cpp)
  target_compile_features(Chromaspace_MetalMemoryPressureMailboxTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_MetalMemoryPressureMailboxTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  target_link_libraries(Chromaspace_MetalMemoryPressureMailboxTests PRIVATE
    Threads::Threads)
  chromaspace_enable_test_assertions(
    Chromaspace_MetalMemoryPressureMailboxTests)
  add_test(NAME Chromaspace.MetalMemoryPressureMailbox
           COMMAND Chromaspace_MetalMemoryPressureMailboxTests)
  add_executable(Chromaspace_FrameRecoveryPolicyTests
    tools/cube_viewer_stub/metal/ChromaspaceFrameRecoveryPolicyTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceFrameRecoveryPolicy.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameFailure.cpp)
  target_compile_features(Chromaspace_FrameRecoveryPolicyTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_FrameRecoveryPolicyTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(Chromaspace_FrameRecoveryPolicyTests)
  add_test(NAME Chromaspace.FrameRecoveryPolicy
           COMMAND Chromaspace_FrameRecoveryPolicyTests)
  add_executable(Chromaspace_MetalQualificationCampaignTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationCampaignTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationCampaign.cpp)
  target_compile_features(Chromaspace_MetalQualificationCampaignTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_MetalQualificationCampaignTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(
    Chromaspace_MetalQualificationCampaignTests)
  add_test(NAME Chromaspace.MetalQualificationCampaign
           COMMAND Chromaspace_MetalQualificationCampaignTests)
  add_executable(Chromaspace_MetalQualificationWorkspaceTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationWorkspaceTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationWorkspace.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerLayout.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_MetalQualificationWorkspaceTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_MetalQualificationWorkspaceTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(
    Chromaspace_MetalQualificationWorkspaceTests)
  add_test(NAME Chromaspace.MetalQualificationWorkspace
           COMMAND Chromaspace_MetalQualificationWorkspaceTests)
  add_executable(Chromaspace_MetalQualificationFaultBackendTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationFaultBackendTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationFaultBackend.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameFailure.cpp)
  target_compile_features(Chromaspace_MetalQualificationFaultBackendTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_MetalQualificationFaultBackendTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(
    Chromaspace_MetalQualificationFaultBackendTests)
  add_test(NAME Chromaspace.MetalQualificationFaultBackend
           COMMAND Chromaspace_MetalQualificationFaultBackendTests)
  add_executable(Chromaspace_MetalQualificationSourceAdapterTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationSourceAdapterTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalQualificationSourceAdapter.cpp
    tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSession.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_MetalQualificationSourceAdapterTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_MetalQualificationSourceAdapterTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(
    Chromaspace_MetalQualificationSourceAdapterTests)
  add_test(NAME Chromaspace.MetalQualificationSourceAdapter
           COMMAND Chromaspace_MetalQualificationSourceAdapterTests)
  add_executable(Chromaspace_MetalPlotRendererTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRendererTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_MetalPlotRendererTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_MetalPlotRendererTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(Chromaspace_MetalPlotRendererTests)
  add_test(NAME Chromaspace.MetalPlotRenderer
           COMMAND Chromaspace_MetalPlotRendererTests)
  add_executable(Chromaspace_MetalDerivedCacheTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCacheTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp)
  target_compile_features(Chromaspace_MetalDerivedCacheTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_MetalDerivedCacheTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(Chromaspace_MetalDerivedCacheTests)
  add_test(NAME Chromaspace.MetalDerivedCache
           COMMAND Chromaspace_MetalDerivedCacheTests)
  add_executable(Chromaspace_MetalTransientArenaTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalTransientArenaTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalTransientArena.cpp)
  target_compile_features(Chromaspace_MetalTransientArenaTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_MetalTransientArenaTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(Chromaspace_MetalTransientArenaTests)
  add_test(NAME Chromaspace.MetalTransientArena
           COMMAND Chromaspace_MetalTransientArenaTests)
  add_executable(Chromaspace_MetalSubmissionRetentionTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalSubmissionRetentionTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalSubmissionRetention.cpp)
  target_compile_features(Chromaspace_MetalSubmissionRetentionTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_MetalSubmissionRetentionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(
    Chromaspace_MetalSubmissionRetentionTests)
  add_test(NAME Chromaspace.MetalSubmissionRetention
           COMMAND Chromaspace_MetalSubmissionRetentionTests)
  add_executable(Chromaspace_MetalPlotCompilerTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotCompilerTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotCompiler.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_MetalPlotCompilerTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_MetalPlotCompilerTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(Chromaspace_MetalPlotCompilerTests)
  add_test(NAME Chromaspace.MetalPlotCompiler
           COMMAND Chromaspace_MetalPlotCompilerTests)
  add_executable(Chromaspace_MetalWorkspaceFrameTests
    tools/cube_viewer_stub/metal/ChromaspaceMetalWorkspaceFrameTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalWorkspaceFrame.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotCompiler.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalPlotRenderer.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalDerivedCache.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp
    tools/cube_viewer_stub/metal/ChromaspaceMetalFrameFailure.cpp
    src/ChromaspaceViewerCamera.cpp
    src/ChromaspaceViewerDomain.cpp
    src/ChromaspaceViewerFramePlan.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerLayout.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp
    ${WORKSHOP_COLOR_SOURCES})
  target_compile_features(Chromaspace_MetalWorkspaceFrameTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_MetalWorkspaceFrameTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal
    ${WORKSHOP_COMMON_DIR})
  chromaspace_enable_test_assertions(Chromaspace_MetalWorkspaceFrameTests)
  add_test(NAME Chromaspace.MetalWorkspaceFrame
           COMMAND Chromaspace_MetalWorkspaceFrameTests)
  add_executable(Chromaspace_ViewerFramePlanTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerFramePlanTests.cpp
    src/ChromaspaceViewerFramePlan.cpp)
  target_compile_features(Chromaspace_ViewerFramePlanTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerFramePlanTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerFramePlanTests)
  add_test(NAME Chromaspace.ViewerFramePlan
           COMMAND Chromaspace_ViewerFramePlanTests)
  add_executable(Chromaspace_ViewerFrameCoordinatorTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerFrameCoordinatorTests.cpp
    src/ChromaspaceViewerFrameCoordinator.cpp)
  target_compile_features(Chromaspace_ViewerFrameCoordinatorTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerFrameCoordinatorTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerFrameCoordinatorTests)
  add_test(NAME Chromaspace.ViewerFrameCoordinator
           COMMAND Chromaspace_ViewerFrameCoordinatorTests)
  add_executable(Chromaspace_ViewerDomainTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerDomainTests.cpp
    src/ChromaspaceViewerDomain.cpp)
  target_compile_features(Chromaspace_ViewerDomainTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerDomainTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerDomainTests)
  add_test(NAME Chromaspace.ViewerDomain
           COMMAND Chromaspace_ViewerDomainTests)
  add_executable(Chromaspace_ViewerCameraTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerCameraTests.cpp
    src/ChromaspaceViewerCamera.cpp
    src/ChromaspaceViewerDomain.cpp)
  target_compile_features(Chromaspace_ViewerCameraTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_ViewerCameraTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(Chromaspace_ViewerCameraTests)
  add_test(NAME Chromaspace.ViewerCamera
           COMMAND Chromaspace_ViewerCameraTests)
  add_executable(Chromaspace_ViewerCameraInteractionTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerCameraInteractionTests.cpp
    src/ChromaspaceViewerCameraInteraction.cpp
    src/ChromaspaceViewerUiScene.cpp
    src/ChromaspaceViewerFramePlan.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerDomain.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerCameraInteractionTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_ViewerCameraInteractionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(
    Chromaspace_ViewerCameraInteractionTests)
  add_test(NAME Chromaspace.ViewerCameraInteraction
           COMMAND Chromaspace_ViewerCameraInteractionTests)
  add_executable(Chromaspace_ViewerLassoInteractionTests
    tools/cube_viewer_stub/metal/ChromaspaceViewerLassoInteractionTests.cpp
    src/ChromaspaceViewerLassoInteraction.cpp
    src/ChromaspaceViewerUiScene.cpp
    src/ChromaspaceViewerFramePlan.cpp
    src/ChromaspaceViewerWorkspace.cpp
    src/ChromaspaceViewerDomain.cpp
    src/ChromaspaceViewerLayout.cpp)
  target_compile_features(Chromaspace_ViewerLassoInteractionTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_ViewerLassoInteractionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  chromaspace_enable_test_assertions(
    Chromaspace_ViewerLassoInteractionTests)
  add_test(NAME Chromaspace.ViewerLassoInteraction
           COMMAND Chromaspace_ViewerLassoInteractionTests)
  add_executable(Chromaspace_SourceExchangeStateTests
    tools/cube_viewer_stub/metal/ChromaspaceSourceExchangeStateTests.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_SourceExchangeStateTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_SourceExchangeStateTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  chromaspace_enable_test_assertions(Chromaspace_SourceExchangeStateTests)
  add_test(NAME Chromaspace.SourceExchangeV2.State
           COMMAND Chromaspace_SourceExchangeStateTests)
  add_executable(Chromaspace_SourceProducerStateTests
    tools/cube_viewer_stub/metal/ChromaspaceSourceProducerStateTests.cpp
    src/metal/ChromaspaceSourceProducerState.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_SourceProducerStateTests PRIVATE cxx_std_17)
  target_include_directories(Chromaspace_SourceProducerStateTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  chromaspace_enable_test_assertions(Chromaspace_SourceProducerStateTests)
  add_test(NAME Chromaspace.SourceExchangeV2.ProducerState
           COMMAND Chromaspace_SourceProducerStateTests)
  add_executable(Chromaspace_SourceProducerClientStateTests
    tools/cube_viewer_stub/metal/ChromaspaceSourceProducerClientStateTests.cpp
    src/metal/ChromaspaceSourceProducerClientState.cpp
    src/metal/ChromaspaceSourceProducerState.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_SourceProducerClientStateTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_SourceProducerClientStateTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  chromaspace_enable_test_assertions(
    Chromaspace_SourceProducerClientStateTests)
  add_test(NAME Chromaspace.SourceExchangeV2.ProducerClientState
           COMMAND Chromaspace_SourceProducerClientStateTests)
  add_executable(Chromaspace_SourceViewerClientStateTests
    tools/cube_viewer_stub/metal/ChromaspaceSourceViewerClientStateTests.cpp
    src/metal/ChromaspaceSourceViewerClientState.cpp
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_SourceViewerClientStateTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_SourceViewerClientStateTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  chromaspace_enable_test_assertions(Chromaspace_SourceViewerClientStateTests)
  add_test(NAME Chromaspace.SourceExchangeV2.ViewerClientState
           COMMAND Chromaspace_SourceViewerClientStateTests)
  add_executable(Chromaspace_ResidentSourceSessionTests
    tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSessionTests.cpp
    tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSession.cpp)
  target_compile_features(Chromaspace_ResidentSourceSessionTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_ResidentSourceSessionTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cube_viewer_stub/metal)
  chromaspace_enable_test_assertions(
    Chromaspace_ResidentSourceSessionTests)
  add_test(NAME Chromaspace.ResidentSourceSession
           COMMAND Chromaspace_ResidentSourceSessionTests)
  add_executable(Chromaspace_SourceExchangeManagerCoreTests
    tools/macos/ChromaspaceSourceExchangeManagerCoreTests.cpp
    tools/macos/ChromaspaceSourceExchangeManagerCore.cpp)
  target_compile_features(Chromaspace_SourceExchangeManagerCoreTests PRIVATE
    cxx_std_17)
  target_include_directories(Chromaspace_SourceExchangeManagerCoreTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/macos)
  chromaspace_enable_test_assertions(
    Chromaspace_SourceExchangeManagerCoreTests)
  add_test(NAME Chromaspace.SourceExchangeV2.ManagerCore
           COMMAND Chromaspace_SourceExchangeManagerCoreTests)
  if(Python3_Interpreter_FOUND)
    add_test(NAME Chromaspace.MacRuntimeV2.QualificationPolicy
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation/test_validate_chromaspace_macos_runtime.py")
    add_test(NAME Chromaspace.MacRuntimeV2.NativeViewerCutoverPolicy
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation/verify_macos_metal_viewer_cutover.py"
        --root "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME Chromaspace.MacRuntimeV2.NativeViewerCutoverPolicyUnit
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation/test_macos_metal_viewer_cutover_policy.py")
    set_tests_properties(
      Chromaspace.MacRuntimeV2.QualificationPolicy
      Chromaspace.MacRuntimeV2.NativeViewerCutoverPolicy
      Chromaspace.MacRuntimeV2.NativeViewerCutoverPolicyUnit
      PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  endif()
endif()
if(APPLE AND BUILD_TESTING)
  add_executable(Chromaspace_SourceExchangeV2Tests
    tools/cube_viewer_stub/metal/ChromaspaceSourceExchangeV2Tests.mm
    src/metal/ChromaspaceSourceExchangeV2.mm)
  target_compile_features(Chromaspace_SourceExchangeV2Tests PRIVATE cxx_std_17)
  target_compile_options(Chromaspace_SourceExchangeV2Tests PRIVATE -fobjc-arc)
  target_include_directories(Chromaspace_SourceExchangeV2Tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  target_link_libraries(Chromaspace_SourceExchangeV2Tests PRIVATE
    "-framework Foundation"
    "-framework Metal")
  add_test(NAME Chromaspace.SourceExchangeV2.Contracts
           COMMAND Chromaspace_SourceExchangeV2Tests)
endif()
if(APPLE)
  set(CHROMASPACE_MANAGER_AGENT_PLIST
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/com.chromaspace.SourceExchangeBroker.plist.template")
  add_executable(Chromaspace_SourceExchangeBroker
    tools/ChromaspaceSourceExchangeBroker.mm
    src/metal/ChromaspaceSourceExchangeV2.mm
    src/metal/ChromaspaceSourceExchangeState.cpp)
  target_compile_features(Chromaspace_SourceExchangeBroker PRIVATE cxx_std_17)
  target_compile_options(Chromaspace_SourceExchangeBroker PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_include_directories(Chromaspace_SourceExchangeBroker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  target_link_libraries(Chromaspace_SourceExchangeBroker PRIVATE
    "-framework Foundation"
    "-framework Metal")
  set_target_properties(Chromaspace_SourceExchangeBroker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BUILD_VIEWER_DIR}"
    OUTPUT_NAME "com.chromaspace.SourceExchangeBroker"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
      "com.chromaspace.SourceExchangeBroker"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
  add_executable(Chromaspace_SourceExchangeProducerRelay
    tools/ChromaspaceSourceExchangeProducerRelay.mm
    src/metal/ChromaspaceSourceExchangeV2.mm)
  target_compile_features(Chromaspace_SourceExchangeProducerRelay PRIVATE
    cxx_std_17)
  target_compile_options(Chromaspace_SourceExchangeProducerRelay PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_include_directories(Chromaspace_SourceExchangeProducerRelay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  target_link_libraries(Chromaspace_SourceExchangeProducerRelay PRIVATE
    "-framework Foundation"
    "-framework Metal"
    ${CHROMASPACE_SECURITY_FRAMEWORK})
  set_target_properties(Chromaspace_SourceExchangeProducerRelay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BUILD_VIEWER_DIR}"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
      "com.chromaspace.SourceExchangeProducerRelay"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
  set_target_properties(Chromaspace_CubeViewer PROPERTIES
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
      "com.moazelgabry.chromaspace.viewer"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
  add_executable(Chromaspace_RuntimeV2MetalProbe
    tools/validation/ChromaspaceRuntimeV2MetalProbe.mm)
  target_compile_options(Chromaspace_RuntimeV2MetalProbe PRIVATE
    -fobjc-arc)
  target_link_libraries(Chromaspace_RuntimeV2MetalProbe PRIVATE
    "-framework Foundation"
    "-framework Metal")
  set_target_properties(Chromaspace_RuntimeV2MetalProbe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${BUILD_ARTIFACT_DIR}/validation"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
  add_executable(Chromaspace_SourceExchangeManager MACOSX_BUNDLE
    tools/macos/ChromaspaceSourceExchangeManager.mm
    tools/macos/ChromaspaceSourceExchangeManagerCore.cpp)
  target_compile_features(Chromaspace_SourceExchangeManager PRIVATE cxx_std_17)
  target_compile_options(Chromaspace_SourceExchangeManager PRIVATE
    $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-mmacosx-version-min=13.0>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_options(Chromaspace_SourceExchangeManager PRIVATE
    -mmacosx-version-min=13.0)
  target_include_directories(Chromaspace_SourceExchangeManager PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/macos)
  target_link_libraries(Chromaspace_SourceExchangeManager PRIVATE
    "-framework Foundation")
  set_target_properties(Chromaspace_SourceExchangeManager PROPERTIES
    OUTPUT_NAME "Chromaspace SourceExchange Manager"
    MACOSX_BUNDLE_INFO_PLIST
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ChromaspaceSourceExchangeManager.Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER
      "com.chromaspace.SourceExchangeManager"
    XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET "13.0"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
    RUNTIME_OUTPUT_DIRECTORY "${BUILD_ARTIFACT_DIR}/source-exchange")
  add_custom_target(Chromaspace_SourceExchangeManagerPackagePayload
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_BUNDLE_DIR:Chromaspace_SourceExchangeManager>/Contents/Library/LaunchAgents"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CHROMASPACE_MANAGER_AGENT_PLIST}"
      "$<TARGET_BUNDLE_DIR:Chromaspace_SourceExchangeManager>/Contents/Library/LaunchAgents/com.chromaspace.SourceExchangeBroker.plist"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:Chromaspace_SourceExchangeBroker>"
      "$<TARGET_BUNDLE_DIR:Chromaspace_SourceExchangeManager>/Contents/MacOS/com.chromaspace.SourceExchangeBroker"
    DEPENDS
      Chromaspace_SourceExchangeManager
      Chromaspace_SourceExchangeBroker
      "${CHROMASPACE_MANAGER_AGENT_PLIST}"
    COMMENT "Assembling the SourceExchange manager package component"
    VERBATIM)
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

if(WIN32)
  add_executable(Chromaspace_ResolveBridge WIN32 tools/resolve_bridge/ChromaspaceResolveBridge.cpp)
else()
  add_executable(Chromaspace_ResolveBridge tools/resolve_bridge/ChromaspaceResolveBridge.cpp)
endif()
target_compile_features(Chromaspace_ResolveBridge PRIVATE cxx_std_17)
if(MSVC)
  target_link_options(Chromaspace_ResolveBridge PRIVATE $<HOST_LINK:/ENTRY:mainCRTStartup>)
endif()
set_target_properties(Chromaspace_ResolveBridge PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${VIEWER_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${VIEWER_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${VIEWER_OUTPUT_DIR}"
  PDB_OUTPUT_DIRECTORY "${VIEWER_OUTPUT_DIR}"
  PDB_OUTPUT_DIRECTORY_DEBUG "${VIEWER_OUTPUT_DIR}"
  PDB_OUTPUT_DIRECTORY_RELEASE "${VIEWER_OUTPUT_DIR}")
if(APPLE)
  set_target_properties(Chromaspace_ResolveBridge PROPERTIES
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
      "com.chromaspace.ResolveBridge"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO")
endif()
add_custom_target(Chromaspace_Bundle ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${OFX_STAGE_ROOT}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_OUTPUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${OFX_RESOURCES_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${PLUGIN_BASENAME}>" "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:${PLUGIN_BASENAME}>"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:Chromaspace_CubeViewer>" "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_CubeViewer>"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:Chromaspace_ResolveBridge>" "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_ResolveBridge>"
  COMMAND ${CMAKE_COMMAND} -E copy "${WORKSHOP_TEXT_ASSETS}" "${OFX_OUTPUT_DIR}/OpenSans-Regular.ttf"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/tools/resolve_bridge/chromaspace_resolve_bridge.py" "${OFX_RESOURCES_DIR}/chromaspace_resolve_bridge.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${PLUGIN_ICON_PNG}" "${OFX_RESOURCES_DIR}/${PLUGIN_IDENTIFIER}.png"
  DEPENDS ${PLUGIN_BASENAME} Chromaspace_CubeViewer Chromaspace_ResolveBridge
  COMMENT "Creating clean install-ready Chromaspace OFX bundle"
  VERBATIM)

if(APPLE)
  add_dependencies(Chromaspace_Bundle
    Chromaspace_ViewerMetallib
    Chromaspace_SourceExchangeProducerRelay
    Chromaspace_SourceExchangeManagerPackagePayload)
  add_custom_command(TARGET Chromaspace_Bundle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" "${OFX_BUNDLE_CONTENTS_DIR}/Info.plist"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CHROMASPACE_VIEWER_METALLIB}"
            "${OFX_OUTPUT_DIR}/${CHROMASPACE_VIEWER_METALLIB_NAME}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:Chromaspace_SourceExchangeProducerRelay>"
            "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_SourceExchangeProducerRelay>"
    VERBATIM)
  if(CHROMASPACE_PACKAGE_MACOS_EXCHANGE_HELPERS)
    add_dependencies(Chromaspace_Bundle Chromaspace_SourceExchangeBroker)
    add_custom_command(TARGET Chromaspace_Bundle POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:Chromaspace_SourceExchangeBroker>"
              "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_SourceExchangeBroker>"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CMAKE_CURRENT_SOURCE_DIR}/cmake/com.chromaspace.SourceExchangeBroker.plist.template"
              "${OFX_RESOURCES_DIR}/com.chromaspace.SourceExchangeBroker.plist.template"
      VERBATIM)
  endif()
  add_custom_target(Chromaspace_MacRuntimeV2Structure
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation/validate_chromaspace_macos_runtime.py"
      --mode structure
      --manager-app
        "$<TARGET_BUNDLE_DIR:Chromaspace_SourceExchangeManager>"
      --ofx-bundle "${OFX_STAGE_ROOT}"
      --metal-probe "$<TARGET_FILE:Chromaspace_RuntimeV2MetalProbe>"
    DEPENDS
      Chromaspace_SourceExchangeManagerPackagePayload
      Chromaspace_Bundle
      Chromaspace_RuntimeV2MetalProbe
    COMMENT "Validating staged macOS Runtime V2 structure"
    VERBATIM)
endif()

if(UNIX AND NOT APPLE)
  # Linux hosts launch the bundled viewer directly, so keep the execute bit on
  # both the built helper and the clean packaged copy.
  add_custom_command(TARGET Chromaspace_CubeViewer POST_BUILD
    COMMAND /bin/chmod +x "$<TARGET_FILE:Chromaspace_CubeViewer>"
    VERBATIM)
  add_custom_command(TARGET Chromaspace_ResolveBridge POST_BUILD
    COMMAND /bin/chmod +x "$<TARGET_FILE:Chromaspace_ResolveBridge>"
    VERBATIM)
  add_custom_command(TARGET Chromaspace_Bundle POST_BUILD
    COMMAND /bin/chmod +x "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_CubeViewer>"
    COMMAND /bin/chmod +x "${OFX_OUTPUT_DIR}/$<TARGET_FILE_NAME:Chromaspace_ResolveBridge>"
    VERBATIM)
endif()
