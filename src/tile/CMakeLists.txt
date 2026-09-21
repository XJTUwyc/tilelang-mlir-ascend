# Tile backend: lightweight .o load/launch runtime.
#
# The core launcher resolves ACL entry points through dlopen/dlsym.
# Optional profiler metadata support uses CANN headers when available, while
# libmsprofiler.so is still resolved dynamically at runtime.  dlopen/dlsym are
# POSIX interfaces; the Ascend NPU runtime is Linux-only (matches the Ascend
# backend), so this backend is a no-op on non-UNIX hosts.
if(NOT UNIX)
  return()
endif()

set(TILELANG_CANN_PROFILER_AVAILABLE OFF)

if(DEFINED ENV{ASCEND_HOME_PATH}
   AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  find_path(
    TILELANG_CANN_PROFILER_INCLUDE_DIR
    NAMES acl/acl_prof.h
    PATHS
      "$ENV{ASCEND_HOME_PATH}/include"
      "$ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    NO_DEFAULT_PATH
    NO_CACHE
  )

  if(TILELANG_CANN_PROFILER_INCLUDE_DIR)
    list(APPEND
      TILE_LANG_INCLUDES
      "${TILELANG_CANN_PROFILER_INCLUDE_DIR}"
    )
    set(TILELANG_CANN_PROFILER_AVAILABLE ON)
    message(STATUS "Tile CANN profiler metadata support enabled")
  else()
    message(STATUS "Tile CANN profiler metadata support disabled: acl/acl_prof.h not found")
  endif()
endif()

file(GLOB TILE_LANG_TILE_SRCS
  src/tile/runtime/tile_obj_launcher.cc
)

if(TILELANG_CANN_PROFILER_AVAILABLE)
  list(APPEND
    TILE_LANG_TILE_SRCS
    src/tile/runtime/tile_npu_profiler.cc
  )
endif()

list(APPEND TILE_LANG_SRCS ${TILE_LANG_TILE_SRCS})
