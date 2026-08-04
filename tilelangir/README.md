# TileLangIR Codegen

## 转换对应方案

转换分为两大类：优先转换为 MLIR 已有方言；已有方言无法完整表达 TileLang
语义时，转换为 `tilelang` 自定义方言。

### 1. 转换为已有 MLIR 方言

| TIRX / TileLang 语义 | 目标 MLIR 表示 | 说明 |
| --- | --- | --- |
| `IRModule` | `builtin.module` | 整个 Codegen 结果的顶层容器。 |
| `PrimFunc` | `func.func` | 函数参数、返回值和符号信息保持可追踪。 |
| `Buffer`、函数 Buffer 参数 | `memref` 类型 | 保留 shape、dtype 和动态维度。 |
| Buffer 分配 | `memref.alloc` 或 `memref.alloca` | 根据分配生命周期选择；内存 scope 作为内建 StringAttr 附在分配操作上。 |
| `BufferLoad` | `memref.load` | 索引表达式先转换为对应的 MLIR SSA Value。 |
| `BufferStore` | `memref.store` | 保持原始写入位置和执行顺序。 |
| Buffer Region、切片 | `memref.subview` | 保留 offset、size 和 stride。 |
| 整数、浮点数和布尔常量 | `arith.constant` | 保留原始 dtype。 |
| 加减乘除、比较、选择和类型转换 | `arith` 方言 | 根据整数、浮点数和有无符号语义选择对应 Operation。 |
| `exp` 等数学表达式 | `math` 方言 | 使用已有数学 Operation。 |
| Serial `For` | `scf.for` | 保留 min、extent、step 和循环体。 |
| Parallel `For` | `scf.parallel` | 保留并行维度，不在 Codegen 中改变调度。 |
| 条件语句 | `scf.if` | 保留 then/else Region。 |
| `SBlock` | `scf.execute_region` | 在 Operation 上增加 `tilelang.sblock_name` 等普通 Attribute；不转换自动推导的 `reads`/`writes`。 |
| `SimdVF` | `scf.execute_region` | 增加表示 SimdVF scope 的普通 Attribute。 |
| `vmuls`、`vadd`、`vmul`、`vmax` | `linalg` 结构化逐元素操作 | 标量计算体使用 `arith` Operation；不在 Codegen 中融合这些操作。 |
| `vreduce_sum`、`vreduce_max` | `linalg.reduce` | reduction body 分别使用加法或最大值 Operation。 |
| `vexp`、`vexpdif` | `linalg` 结构化逐元素操作 | 计算体使用 `math.exp`，`vexpdif` 同时保留减法。 |
| `vcvt` | `linalg` 结构化逐元素操作 | 计算体使用相应的 `arith` cast Operation。 |

`shared.dyn`、`local.fragment` 等 scope 当前不设计成自定义 MLIR 类型或自定义
Attribute 类型，而是使用普通字符串 Attribute 保留在对应 Buffer 分配 Operation
上。后续如果下游对 memory space 类型提出严格要求，再单独扩展类型系统。

### 2. 转换为 `tilelang` 自定义方言

以下语义目前没有能够完整、一一承接的标准 MLIR Operation，因此保留为自定义
Operation，不在 Codegen 阶段提前展开或硬件化：

| TIRX / TileLang 语义 | 自定义 MLIR Operation | 需要保留的信息 |
| --- | --- | --- |
| `T.copy` / `tl.tileop.copy` | `tilelang.copy` | 源和目标 Region，以及 `split_dim`、`transpose` 等 Attribute。 |
| `T.gemm` / `tl.tileop.gemm` | `tilelang.gemm` | A/B/C Region、transpose、M/N/K、clear_accum 和其他原始参数。 |
| Kernel launch / thread binding | `tilelang.launch_thread` | thread tag、extent、绑定变量和嵌套 Region。 |

自定义 Operation 的目标是无损保存前端语义，而不是在这里决定最终硬件指令。
后续真正面向某个后端时，可以再对 TileLangIR 运行独立的 lowering 或转换流程。

## 方案原理

整体链路如下：

```text
TileLang Python DSL
    -> TVM Script Parser / TileLang language helpers
    -> TIRX IRModule
    -> OpenTile Pass Pipeline
    -> Python DeviceCodegen
    -> TileLangIR Translator
    -> MLIR Python Builder API
    -> MLIR Module 文本
    -> TVM SourceModule 包装
```

### 1. Codegen 入口

`target="tile"` 被标准化为带有 `tile` key 的 Target。OpenTile Codegen 将
Python 函数注册进现有 `DeviceCodegen` 注册框架，因此 `lower()` 最终选择的是
Python Codegen，而不是已删除的 `src/opentile` C++ Codegen。

### 2. 遍历 TIRX 数据结构

Translator 接收 `tvm.IRModule`，直接访问其中的 `PrimFunc`、`Stmt`、`PrimExpr`、
`Buffer` 和 `Call` 等 TIRX Python ObjectRef。

Translator 根据节点的 Python 类型分派普通 TIRX 节点，并根据 `Call.op.name`
分派 `tl.tileop.copy`、`tl.tileop.gemm` 等 TileLang Operation。转换过程中维护：

- TIRX `Var` 到 MLIR SSA Value 的映射;TIRX `Buffer` 到 MLIR memref Value 的映射；
- TIRX Region 到 MLIR Region/Block 的嵌套关系；
- dtype、shape、scope、annotation 和符号信息的转换结果。

### 3. 使用 MLIR Python bindings 构造结果

TileLangIR Codegen 在运行时导入 `mlir.ir`，通过 Python Builder API 创建 Context、
Location、Module、Operation、Region、Block 和 SSA Value。LLVM/MLIR 已经编译在
依赖 wheel 提供的共享库中，因此 TileLang 自身的 CMake 编译过程不需要包含或
链接 LLVM Project。

MLIR Python bindings 采用 LLVM 官方 `llvm/eudsl` 发布的 wheel。`pyproject.toml`
以不限定版本的 `mlir-python-bindings` 声明该依赖，因此生成的 TileLang wheel
会包含对应的 `Requires-Dist` 元数据。

### 4. Codegen无优化、逐节点转换

Codegen 只负责表示转换，不改变程序语义和执行顺序：

- 不融合相邻计算；
- 不消除看似冗余的节点；
- 不改变循环顺序或内存布局；
- 不选择具体硬件指令；
- 不运行 MLIR optimization pipeline。

标准方言和自定义方言的选择，只取决于目标 Operation 能否完整表达源 TIRX
节点，而不取决于哪一种表示运行得更快。
