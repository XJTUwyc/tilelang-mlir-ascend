#include "tile_npu_profiler.h"

#include <acl/acl_prof.h>
#include <dlfcn.h>
#include <tvm/runtime/logging.h>

#include <cstdlib>
#include <cstring>
#include <new>

namespace tvm {
namespace tile {

namespace {

bool IsNpuProfilerEnabled() {
  const char *value = std::getenv("TILELANG_NPU_PROFILE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

class NpuProfilerDriver {
public:
  static NpuProfilerDriver *Global() {
    static auto *driver = new NpuProfilerDriver();
    return driver;
  }

  bool available() const {
    return str_to_id_ != nullptr && range_push_ex_ != nullptr &&
           range_pop_ != nullptr;
  }

  uint64_t StrToId(const char *value) const { return str_to_id_(value); }

  aclError RangePush(aclprofEventAttributes *attributes) const {
    return range_push_ex_(attributes);
  }

  aclError RangePop() const { return range_pop_(); }

private:
  using StrToIdFn = uint64_t (*)(const char *);
  using RangePushExFn = aclError (*)(aclprofEventAttributes *);
  using RangePopFn = aclError (*)();

  NpuProfilerDriver() {
    library_ = dlopen("libmsprofiler.so", RTLD_LAZY | RTLD_LOCAL);
    if (library_ == nullptr) {
      const char *error = dlerror();
      LOG(WARNING)
          << "Tile NPU profiler could not load libmsprofiler.so; "
             "kernel launch will continue without profiler metadata: "
          << (error == nullptr ? "unknown error" : error);
      return;
    }

    str_to_id_ = LoadSymbol<StrToIdFn>("aclprofStr2Id");
    range_push_ex_ = LoadSymbol<RangePushExFn>("aclprofRangePushEx");
    range_pop_ = LoadSymbol<RangePopFn>("aclprofRangePop");

    if (!available()) {
      LOG(WARNING) << "Tile NPU profiler APIs are incomplete; "
                      "kernel launch will continue without profiler metadata";
    }
  }

  template <typename FunctionType>
  FunctionType LoadSymbol(const char *name) {
    dlerror();
    void *symbol = dlsym(library_, name);
    const char *error = dlerror();

    if (symbol == nullptr || error != nullptr) {
      return nullptr;
    }

    return reinterpret_cast<FunctionType>(symbol);
  }

  void *library_{nullptr};
  StrToIdFn str_to_id_{nullptr};
  RangePushExFn range_push_ex_{nullptr};
  RangePopFn range_pop_{nullptr};
};

} // namespace

struct NpuProfilerRange::State {
  aclprofTensorInfo tensor_info{};
  aclprofEventAttributes attributes{};
};

NpuProfilerRange::NpuProfilerRange(const char *op_name, uint32_t block_nums,
                                   uint64_t stream) {
  if (!IsNpuProfilerEnabled()) {
    return;
  }

  NpuProfilerDriver *driver = NpuProfilerDriver::Global();
  if (!driver->available()) {
    return;
  }

  State *state = new (std::nothrow) State();
  if (state == nullptr) {
    return;
  }

  state->tensor_info.opNameId = driver->StrToId(op_name);
  state->tensor_info.opTypeId = driver->StrToId("TileLang");
  state->tensor_info.tensorNum = 0;
  state->tensor_info.kernelType = 0;
  state->tensor_info.blockNums = block_nums;
  state->tensor_info.stream = reinterpret_cast<void *>(stream);
  state->tensor_info.tensors = nullptr;

  state->attributes.version = ACL_PROF_EVENT_ATTR_VERSION;
  state->attributes.size =
      static_cast<uint16_t>(sizeof(state->attributes.message));
  state->attributes.messageType = ACL_PROF_MESSAGE_TYPE_TENSOR_INFO;
  state->attributes.message.tensorInfo = &state->tensor_info;

  if (driver->RangePush(&state->attributes) != ACL_SUCCESS) {
    delete state;
    return;
  }

  state_ = state;
}

NpuProfilerRange::~NpuProfilerRange() {
  if (state_ == nullptr) {
    return;
  }

  (void)NpuProfilerDriver::Global()->RangePop();
  delete state_;
}

} // namespace tile
} // namespace tvm
