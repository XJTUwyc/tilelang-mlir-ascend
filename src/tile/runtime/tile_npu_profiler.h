#ifndef TVM_TL_TILE_RUNTIME_NPU_PROFILER_H_
#define TVM_TL_TILE_RUNTIME_NPU_PROFILER_H_

#include <cstdint>

namespace tvm {
namespace tile {

class NpuProfilerRange {
public:
    NpuProfilerRange(const char *op_name, uint32_t block_nums, uint64_t stream);
    ~NpuProfilerRange();

    NpuProfilerRange(const NpuProfilerRange &) = delete;
    NpuProfilerRange &operator=(const NpuProfilerRange &) = delete;

private:
    struct State;
    State *state_{nullptr};
};

} // namespace tile
} // namespace tvm

#endif // TVM_TL_TILE_RUNTIME_NPU_PROFILER_H_
