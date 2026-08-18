# Tile backend: lightweight .o load/launch runtime.
#
# The launcher resolves ACL entry points at runtime via dlopen/dlsym, so it
# needs no CANN headers or libraries at build time.  dlopen/dlsym are POSIX
# interfaces; the Ascend NPU runtime is Linux-only (matches the Ascend
# backend), so this backend is a no-op on non-UNIX hosts.
if(NOT UNIX)
  return()
endif()

file(GLOB TILE_LANG_TILE_SRCS
  src/tile/runtime/tile_obj_launcher.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_TILE_SRCS})
