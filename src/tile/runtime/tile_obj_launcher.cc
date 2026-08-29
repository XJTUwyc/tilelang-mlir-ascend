/*!
 * \file tile_obj_launcher.cc
 * \brief Lightweight load + launch for Tile backend objects (.o / aibin bytes).
 *
 * Device compilation happens outside the repository (manual bisheng/ccec
 * artifacts).  This file only implements the runtime half:
 *
 *   1. Lazily dlopen("libascendcl.so") and dlsym the ACL entry points,
 *      so TileLang keeps no CANN build-time dependency.
 *   2. Lazily load the object bytes via aclrtBinaryLoadFromData and resolve
 *      the kernel symbol via aclrtBinaryGetFunction (cached per device).
 *   3. Pack arguments (int64-encoded on the Python side) into a contiguous
 *      aligned buffer.  The pack layout is computed once per kernel and
 *      cached; small kernels use a stack-backed buffer so steady-state
 *      launches do not allocate.
 *   4. Submit with aclrtLaunchKernelWithHostArgs.  The grid (logical core
 *      count) and dynamic UBUF size are computed by the Python side.
 *
 * Python-side entry points:
 *   - tl.tile.LaunchKernel(..., args) uses ACL's automatic binary kind.
 *   - tl.tile.LaunchKernelWithBinaryKind(..., args, binary_kind) selects the
 *     AIV/AIC/mixed ELF magic explicitly.
 *
 *   - arg_types[i] is one of:
 *       "handle", "int8", "int16", "int32", "int64",
 *       "uint8", "uint16", "uint32", "uint64", "float32", "float64"
 *   - args[i] is one int64 per argument:
 *       handle  -> raw device pointer value
 *       integer -> the value
 *       float32 -> IEEE-754 bit pattern in the low 32 bits
 *       float64 -> IEEE-754 bit pattern
 *   - stream is the raw NPU stream handle (0 = ACL default stream).
 */
#include <dlfcn.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tile {
namespace {

using AclError = int32_t;
using AclBinHandle = void *;
using AclFuncHandle = void *;
using AclStream = void *;

constexpr AclError kAclSuccess = 0;
constexpr int32_t kAclBinaryLoadOptMagic = 2;
constexpr uint32_t kAclBinaryMagicElfAiCore = 0x43554245U;
constexpr uint32_t kAclBinaryMagicElfVectorCore = 0x41415246U;
constexpr uint32_t kAclBinaryMagicElfCubeCore = 0x41494343U;
constexpr int32_t kAclLaunchKernelAttrDynUbufSize = 2;
// Mirrors tilelang-ascend-cce's runtime: every scalar kernel argument is
// aligned to at least 4 bytes, and the whole packed buffer to 8 bytes.
constexpr size_t kAclArgMinAlignment = 4;
constexpr size_t kAclArgBufferAlignment = 8;

uint32_t ParseBinaryMagic(const std::string &binary_kind) {
  if (binary_kind == "auto") return 0;
  if (binary_kind == "aiv" || binary_kind == "vector") {
    return kAclBinaryMagicElfVectorCore;
  }
  if (binary_kind == "aic" || binary_kind == "aicore" ||
      binary_kind == "mix") {
    return kAclBinaryMagicElfAiCore;
  }
  if (binary_kind == "aicube" || binary_kind == "cube") {
    return kAclBinaryMagicElfCubeCore;
  }
  TVM_FFI_THROW(ValueError)
      << "Unsupported Tile binary kind `" << binary_kind
      << "`; expected auto, aiv, aic, aicore, mix, aicube, or cube";
  TVM_FFI_UNREACHABLE();
}

size_t AlignUp(size_t value, size_t alignment) {
  TVM_FFI_ICHECK_NE(alignment, 0U);
  TVM_FFI_ICHECK_EQ(alignment & (alignment - 1), 0U)
      << "Alignment must be a power of two, got " << alignment;
  return (value + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Argument kinds and packing
// ---------------------------------------------------------------------------

enum class AclArgKind {
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
  kFloat32,
  kFloat64,
  kHandle,
};

AclArgKind ParseArgKind(const std::string &type, size_t index) {
  if (type == "handle") return AclArgKind::kHandle;
  if (type == "int8") return AclArgKind::kInt8;
  if (type == "int16") return AclArgKind::kInt16;
  if (type == "int32") return AclArgKind::kInt32;
  if (type == "int64") return AclArgKind::kInt64;
  if (type == "uint8") return AclArgKind::kUInt8;
  if (type == "uint16") return AclArgKind::kUInt16;
  if (type == "uint32") return AclArgKind::kUInt32;
  if (type == "uint64") return AclArgKind::kUInt64;
  if (type == "float32") return AclArgKind::kFloat32;
  if (type == "float64") return AclArgKind::kFloat64;
  TVM_FFI_THROW(RuntimeError) << "Unsupported tile arg type `" << type
                              << "` at argument " << index;
  TVM_FFI_UNREACHABLE();
}

size_t GetAclArgSize(AclArgKind kind) {
  switch (kind) {
  case AclArgKind::kInt8:
  case AclArgKind::kUInt8:
    return 1;
  case AclArgKind::kInt16:
  case AclArgKind::kUInt16:
    return 2;
  case AclArgKind::kInt32:
  case AclArgKind::kUInt32:
  case AclArgKind::kFloat32:
    return 4;
  case AclArgKind::kInt64:
  case AclArgKind::kUInt64:
  case AclArgKind::kFloat64:
  case AclArgKind::kHandle:
    return 8;
  }
  TVM_FFI_UNREACHABLE();
}

size_t GetAclArgAlignment(AclArgKind kind) {
  size_t alignment = 0;
  switch (kind) {
  case AclArgKind::kInt8:
    alignment = alignof(int8_t);
    break;
  case AclArgKind::kInt16:
    alignment = alignof(int16_t);
    break;
  case AclArgKind::kInt32:
    alignment = alignof(int32_t);
    break;
  case AclArgKind::kInt64:
    alignment = alignof(int64_t);
    break;
  case AclArgKind::kUInt8:
    alignment = alignof(uint8_t);
    break;
  case AclArgKind::kUInt16:
    alignment = alignof(uint16_t);
    break;
  case AclArgKind::kUInt32:
    alignment = alignof(uint32_t);
    break;
  case AclArgKind::kUInt64:
    alignment = alignof(uint64_t);
    break;
  case AclArgKind::kFloat32:
    alignment = alignof(float);
    break;
  case AclArgKind::kFloat64:
    alignment = alignof(double);
    break;
  case AclArgKind::kHandle:
    alignment = alignof(void *);
    break;
  }
  return std::max(alignment, kAclArgMinAlignment);
}

struct AclArgLayout {
  AclArgKind kind;
  size_t offset;
};

struct AclArgPackPlan {
  std::vector<AclArgLayout> args;
  size_t buffer_size{0};
};

// Computed once per (function, arg_types) and cached; each launch only fills
// the buffer.  Mirrors MakeAclArgPackPlan in tilelang-ascend-cce's
// ascend_module.cc.
AclArgPackPlan MakeAclArgPackPlan(const ffi::Array<ffi::String> &arg_types) {
  AclArgPackPlan plan;
  plan.args.reserve(arg_types.size());
  size_t offset = 0;
  for (size_t i = 0; i < arg_types.size(); ++i) {
    AclArgKind kind = ParseArgKind(arg_types[i].operator std::string(), i);
    offset = AlignUp(offset, GetAclArgAlignment(kind));
    plan.args.push_back({kind, offset});
    offset += GetAclArgSize(kind);
  }
  plan.buffer_size = AlignUp(offset, kAclArgBufferAlignment);
  return plan;
}

void PackArg(AclArgKind kind, int64_t value, uint8_t *dst) {
  switch (kind) {
  case AclArgKind::kInt8:
  case AclArgKind::kUInt8: {
    uint8_t v = static_cast<uint8_t>(value);
    std::memcpy(dst, &v, sizeof(v));
    break;
  }
  case AclArgKind::kInt16:
  case AclArgKind::kUInt16: {
    uint16_t v = static_cast<uint16_t>(value);
    std::memcpy(dst, &v, sizeof(v));
    break;
  }
  case AclArgKind::kInt32:
  case AclArgKind::kUInt32:
  case AclArgKind::kFloat32: {
    uint32_t v = static_cast<uint32_t>(value);
    std::memcpy(dst, &v, sizeof(v));
    break;
  }
  case AclArgKind::kInt64:
  case AclArgKind::kUInt64:
  case AclArgKind::kFloat64:
  case AclArgKind::kHandle: {
    std::memcpy(dst, &value, sizeof(value));
    break;
  }
  }
}

// Stack-backed packing buffer for small kernels, heap-backed beyond
// kNumWords == 0.  Mirrors AclArgBuffer in tilelang-ascend-cce so the common
// case (<= 16 words, i.e. <= 128 bytes of arguments) never allocates.
template <size_t kNumWords> class AclArgBuffer {
public:
  explicit AclArgBuffer(size_t num_words) {
    TVM_FFI_ICHECK_LE(num_words, kNumWords)
        << "AclArgBuffer capacity exceeded";
  }

  uint8_t *data() { return reinterpret_cast<uint8_t *>(storage_.data()); }

private:
  alignas(kAclArgBufferAlignment) std::array<uint64_t, kNumWords> storage_{};
};

template <> class AclArgBuffer<0> {
public:
  explicit AclArgBuffer(size_t num_words) : storage_(num_words, 0) {}

  uint8_t *data() { return reinterpret_cast<uint8_t *>(storage_.data()); }

private:
  std::vector<uint64_t> storage_;
};

// ---------------------------------------------------------------------------
// ACL driver: lazy dlopen + dlsym
// ---------------------------------------------------------------------------

union AclLaunchKernelAttrValue {
  uint8_t schem_mode;
  uint32_t dyn_ubuf_size;
  uint32_t engine_type;
  uint32_t block_dim_offset;
  uint8_t is_block_task_prefetch;
  uint8_t is_data_dump;
  uint16_t timeout;
  uint32_t reserved[4];
};

struct AclLaunchKernelAttr {
  int32_t id;
  AclLaunchKernelAttrValue value;
};

struct AclLaunchKernelCfg {
  AclLaunchKernelAttr *attrs;
  size_t num_attrs;
};

// Local mirrors of the public ACL binary-load option ABI.  Keeping these
// declarations here preservers the runtime's no-CANN-header build contract.
union AclBinaryLoadOptionValue {
  uint32_t is_lazy_load;
  uint32_t magic;
  int32_t cpu_kernel_mode;
  uint32_t reserved[4];
};

struct AclBinaryLoadOption {
  int32_t type;
  AclBinaryLoadOptionValue value;
};

struct AclBinaryLoadOptions {
  AclBinaryLoadOption *options;
  size_t num_options;
};

class TileDriver {
public:
  static TileDriver *Global() {
    static auto *driver = new TileDriver();
    return driver;
  }

  AclError BinaryLoadFromData(const void *data, size_t size,
                              uint32_t binary_magic,
                              AclBinHandle *handle) const {
    if (binary_magic == 0) {
      return binary_load_from_data_(data, size, nullptr, handle);
    }
    AclBinaryLoadOption option{};
    option.type = kAclBinaryLoadOptMagic;
    option.value.magic = binary_magic;
    AclBinaryLoadOptions options{&option, 1};
    return binary_load_from_data_(data, size, &options, handle);
  }

  AclError BinaryGetFunction(AclBinHandle binary, const char *name,
                             AclFuncHandle *function) const {
    return binary_get_function_(binary, name, function);
  }

  AclError GetDevice(int32_t *device_id) const { return get_device_(device_id); }

  AclError LaunchKernelWithHostArgs(AclFuncHandle function, uint32_t num_blocks,
                                    AclStream stream,
                                    AclLaunchKernelCfg *config, void *args,
                                    size_t args_size) const {
    return launch_kernel_with_host_args_(function, num_blocks, stream, config,
                                         args, args_size, nullptr, 0);
  }

  const char *GetRecentErrorMessage() const {
    return get_recent_error_message_ == nullptr
               ? nullptr
               : get_recent_error_message_();
  }

private:
  using BinaryLoadFromDataFn = AclError (*)(const void *, size_t, const void *,
                                            AclBinHandle *);
  using BinaryGetFunctionFn = AclError (*)(AclBinHandle, const char *,
                                           AclFuncHandle *);
  using GetDeviceFn = AclError (*)(int32_t *);
  using LaunchKernelWithHostArgsFn = AclError (*)(AclFuncHandle, uint32_t,
                                                  AclStream,
                                                  AclLaunchKernelCfg *, void *,
                                                  size_t, void *, size_t);
  using GetRecentErrorMessageFn = const char *(*)();

  TileDriver() {
    library_ = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
    TVM_FFI_CHECK(library_ != nullptr, RuntimeError)
        << "Tile runtime could not load libascendcl.so: " << dlerror();
    binary_load_from_data_ =
        LoadSymbol<BinaryLoadFromDataFn>("aclrtBinaryLoadFromData");
    binary_get_function_ =
        LoadSymbol<BinaryGetFunctionFn>("aclrtBinaryGetFunction");
    get_device_ = LoadSymbol<GetDeviceFn>("aclrtGetDevice");
    launch_kernel_with_host_args_ =
        LoadSymbol<LaunchKernelWithHostArgsFn>("aclrtLaunchKernelWithHostArgs");
    get_recent_error_message_ =
        LoadSymbol<GetRecentErrorMessageFn>("aclGetRecentErrMsg");
  }

  template <typename FunctionType> FunctionType LoadSymbol(const char *name) {
    dlerror();
    void *symbol = dlsym(library_, name);
    const char *error = dlerror();
    TVM_FFI_CHECK(symbol != nullptr && error == nullptr, RuntimeError)
        << "Tile runtime could not resolve " << name
        << " from libascendcl.so: "
        << (error == nullptr ? "symbol not found" : error);
    return reinterpret_cast<FunctionType>(symbol);
  }

  void *library_{nullptr};
  BinaryLoadFromDataFn binary_load_from_data_{nullptr};
  BinaryGetFunctionFn binary_get_function_{nullptr};
  GetDeviceFn get_device_{nullptr};
  LaunchKernelWithHostArgsFn launch_kernel_with_host_args_{nullptr};
  GetRecentErrorMessageFn get_recent_error_message_{nullptr};
};

void CheckAcl(AclError result, const char *operation) {
  if (result == kAclSuccess) return;
  const char *message = TileDriver::Global()->GetRecentErrorMessage();
  TVM_FFI_THROW(RuntimeError)
      << operation << " failed with ACL error " << result
      << (message == nullptr ? "" : std::string(": ") + message);
}

// ---------------------------------------------------------------------------
// Binary / function handle cache (lazy load, process-lifetime)
// ---------------------------------------------------------------------------

class BinaryRegistry {
public:
  static BinaryRegistry *Global() {
    static auto *registry = new BinaryRegistry();
    return registry;
  }

  // Loads the object bytes on first use (per device) and resolves the kernel
  // symbol.  Handles are cached for the process lifetime; the OS reclaims
  // them at exit (matches the Ascend backend's driver handles).
  AclFuncHandle GetFunction(const std::string &obj_bytes,
                            const std::string &kernel_name, int32_t device_id,
                            uint32_t binary_magic) {
    std::lock_guard<std::mutex> lock(mutex_);
    TileDriver *driver = TileDriver::Global();
    // Key on the object *contents* rather than a std::hash digest: hashing
    // would let two different objects with colliding digests share one
    // device binary, silently launching the wrong kernel.
    BinaryKey key{obj_bytes, device_id, binary_magic};
    AclBinHandle &binary = binaries_[key];
    if (binary == nullptr) {
      CheckAcl(driver->BinaryLoadFromData(obj_bytes.data(), obj_bytes.size(),
                                          binary_magic, &binary),
               "aclrtBinaryLoadFromData");
    }
    FuncKey fkey{binary, kernel_name};
    AclFuncHandle &function = functions_[fkey];
    if (function == nullptr) {
      CheckAcl(driver->BinaryGetFunction(binary, kernel_name.c_str(), &function),
               "aclrtBinaryGetFunction");
    }
    return function;
  }

  // Returns the cached argument-pack layout for (function, arg_types), or
  // builds and caches it on first use.  The returned pointer stays valid for
  // the registry lifetime: unordered_map nodes keep their address across
  // rehashes.
  const AclArgPackPlan *GetPlan(AclFuncHandle function,
                                const std::string &plan_key,
                                const ffi::Array<ffi::String> &arg_types) {
    std::lock_guard<std::mutex> lock(mutex_);
    PlanKey key{function, plan_key};
    auto found = plans_.find(key);
    if (found != plans_.end()) {
      return &found->second;
    }
    auto inserted = plans_.emplace(std::move(key), MakeAclArgPackPlan(arg_types));
    return &inserted.first->second;
  }

private:
  struct BinaryKey {
    std::string obj_bytes;
    int32_t device_id;
    uint32_t binary_magic;

    bool operator==(const BinaryKey &other) const {
      return obj_bytes == other.obj_bytes &&device_id == other.device_id &&
             binary_magic == other.binary_magic;
    }
  };

  using FuncKey = std::pair<AclBinHandle, std::string>;
  using PlanKey = std::pair<AclFuncHandle, std::string>;

  struct BinaryKeyHash {
    size_t operator()(const BinaryKey &key) const {
      size_t hash = std::hash<std::string>{}(key.obj_bytes);
      hash ^= std::hash<int32_t>{}(key.device_id) << 1;
      hash ^= std::hash<uint32_t>{}(key.binary_magic) << 2;
      return hash;
    }
  };

  struct FuncKeyHash {
    size_t operator()(const FuncKey &key) const {
      return std::hash<const void *>{}(key.first) ^
             (std::hash<std::string>{}(key.second) << 1);
    }
  };

  struct PlanKeyHash {
    size_t operator()(const PlanKey &key) const {
      return std::hash<const void *>{}(key.first) ^
             (std::hash<std::string>{}(key.second) << 1);
    }
  };

  std::mutex mutex_;
  std::unordered_map<BinaryKey, AclBinHandle, BinaryKeyHash> binaries_;
  std::unordered_map<FuncKey, AclFuncHandle, FuncKeyHash> functions_;
  std::unordered_map<PlanKey, AclArgPackPlan, PlanKeyHash> plans_;
};

// ---------------------------------------------------------------------------
// The launch entry point
// ---------------------------------------------------------------------------

// Fills a stack- or heap-backed argument buffer and submits the kernel.
// Returns the raw ACL error code; error reporting stays in LaunchKernelImpl
// so the grid/ubuf context is attached in exactly one place.
template <size_t kNumWords>
AclError LaunchPacked(TileDriver *driver, AclFuncHandle function,
                      uint32_t num_blocks, AclStream stream,
                      AclLaunchKernelCfg *config,
                      const AclArgPackPlan &plan,
                      const ffi::Array<int64_t> &args) {
  AclArgBuffer<kNumWords> buffer(plan.buffer_size / sizeof(uint64_t));
  uint8_t *base = buffer.data();
  for (size_t i = 0; i < plan.args.size(); ++i) {
    PackArg(plan.args[i].kind, args[i], base + plan.args[i].offset);
  }
  return driver->LaunchKernelWithHostArgs(function, num_blocks, stream,
                                          config, base, plan.buffer_size);
}

void LaunchKernelImpl(ffi::Bytes o_bytes, ffi::String kernel_name, int64_t grid,
                      int64_t ubuf_size, uint64_t stream,
                      ffi::Array<ffi::String> arg_types,
                      ffi::Array<int64_t> args, ffi::String binary_kind) {
  TVM_FFI_ICHECK(grid > 0)
      << "Tile launch grid must be positive, got " << grid;
  TVM_FFI_ICHECK(grid <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "Tile launch grid exceeds uint32 range: " << grid;
  TVM_FFI_ICHECK(ubuf_size >= 0)
      << "Tile dynamic UBUF size must be non-negative, got " << ubuf_size;
  TVM_FFI_ICHECK(ubuf_size <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "Tile dynamic UBUF size exceeds uint32 range: " << ubuf_size;
  TVM_FFI_ICHECK_EQ(args.size(), arg_types.size())
      << "Tile kernel `" << kernel_name.operator std::string() << "` expects "
      << arg_types.size() << " arguments but got " << args.size();

  TileDriver *driver = TileDriver::Global();
  int32_t device_id = 0;
  CheckAcl(driver->GetDevice(&device_id), "aclrtGetDevice");
  std::string symbol = kernel_name.operator std::string();
  uint32_t binary_magic =
      ParseBinaryMagic(binary_kind.operator std::string());
  AclFuncHandle function = BinaryRegistry::Global()->GetFunction(
      std::string(o_bytes.data(), o_bytes.size()), symbol, device_id,
      binary_magic);

  // The pack layout depends only on arg_types: cache it per function so
  // repeated launches neither re-parse nor re-allocate.
  std::string plan_key;
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (i != 0) {
      plan_key.push_back(',');
    }
    plan_key += arg_types[i].operator std::string();
  }
  const AclArgPackPlan *plan = BinaryRegistry::Global()->GetPlan(
      function, plan_key, arg_types);

  AclLaunchKernelAttr attribute{};
  AclLaunchKernelCfg config{};
  AclLaunchKernelCfg *config_ptr = nullptr;
  if (ubuf_size > 0) {
    attribute.id = kAclLaunchKernelAttrDynUbufSize;
    attribute.value.dyn_ubuf_size = static_cast<uint32_t>(ubuf_size);
    config.attrs = &attribute;
    config.num_attrs = 1;
    config_ptr = &config;
  }

  AclError result;
  size_t num_words = plan->buffer_size / sizeof(uint64_t);
  if (num_words <= 4) {
    result = LaunchPacked<4>(driver, function, static_cast<uint32_t>(grid),
                             reinterpret_cast<AclStream>(stream), config_ptr,
                             *plan, args);
  } else if (num_words <= 8) {
    result = LaunchPacked<8>(driver, function, static_cast<uint32_t>(grid),
                             reinterpret_cast<AclStream>(stream), config_ptr,
                             *plan, args);
  } else if (num_words <= 16) {
    result = LaunchPacked<16>(driver, function, static_cast<uint32_t>(grid),
                              reinterpret_cast<AclStream>(stream), config_ptr,
                              *plan, args);
  } else {
    result = LaunchPacked<0>(driver, function, static_cast<uint32_t>(grid),
                             reinterpret_cast<AclStream>(stream), config_ptr,
                             *plan, args);
  }
  if (result != kAclSuccess) {
    const char *message = driver->GetRecentErrorMessage();
    std::ostringstream error;
    error << "aclrtLaunchKernelWithHostArgs failed for " << symbol
          << " with ACL error " << result << ", grid=" << grid
          << ", dyn_ubuf_bytes=" << ubuf_size;
    if (message != nullptr) {
      error << ": " << message;
    }
    TVM_FFI_THROW(RuntimeError) << error.str();
  }
}

void LaunchKernelAutoImpl(ffi::Bytes o_bytes, ffi::String kernel_name,
                          int64_t grid, int64_t ubuf_size, uint64_t stream,
                          ffi::Array<ffi::String> arg_types,
                          ffi::Array<int64_t> args) {
  LaunchKernelImpl(o_bytes, kernel_name, grid, ubuf_size, stream, arg_types,
                   args, ffi::String("auto"));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.tile.LaunchKernel", &LaunchKernelAutoImpl)
      .def("tl.tile.LaunchKernelWithBinaryKind", &LaunchKernelImpl);
}

} // namespace
} // namespace tile
} // namespace tvm
