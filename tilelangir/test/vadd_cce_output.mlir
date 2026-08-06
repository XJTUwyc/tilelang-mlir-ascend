module attributes {cce.target = "dav-351x"} {
  func.func @vadd_kernel(%arg0: memref<?xf32, 1>, %arg1: memref<?xf32, 1>, %arg2: memref<?xf32, 1>, %arg3: i32) attributes {BlockIdx = 3 : i64, cce.core = #cce.core} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.muli %arg3, %c32_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %c60_i64 = arith.constant 60 : i64
    %c48_i64 = arith.constant 48 : i64
    %2 = cce.get.ctrl -> i64
    %3 = cce.sbitset0(%2, %c60_i64) : (i64, i64) -> i64
    cce.set.ctrl(%3) : i64
    %4 = cce.get.ctrl -> i64
    %5 = cce.sbitset1(%4, %c48_i64) : (i64, i64) -> i64
    cce.set.ctrl(%5) : i64
    %alloc = memref.alloc() : memref<32768xi8, 6>
    %alloc_0 = memref.alloc() : memref<32768xi8, 6>
    %alloc_1 = memref.alloc() : memref<32768xi8, 6>
    %view = memref.view %alloc[%c0][] : memref<32768xi8, 6> to memref<8192xf32, 6>
    %view_2 = memref.view %alloc_0[%c0][] : memref<32768xi8, 6> to memref<8192xf32, 6>
    %view_3 = memref.view %alloc_1[%c0][] : memref<32768xi8, 6> to memref<8192xf32, 6>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [8192], strides: [1] : memref<?xf32, 1> to memref<8192xf32, strided<[1], offset: ?>, 1>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [8192], strides: [1] : memref<?xf32, 1> to memref<8192xf32, strided<[1], offset: ?>, 1>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [8192], strides: [1] : memref<?xf32, 1> to memref<8192xf32, strided<[1], offset: ?>, 1>
    %c288230377225469952_i64 = arith.constant 288230377225469952 : i64
    %c35184372088864_i64 = arith.constant 35184372088864 : i64
    cce.mov.out.to.ub.align.v2.dv(%view, %reinterpret_cast, %c288230377225469952_i64, %c35184372088864_i64) : (memref<8192xf32, 6>, memref<8192xf32, strided<[1], offset: ?>, 1>, i64, i64)
    %c288230377225469952_i64_6 = arith.constant 288230377225469952 : i64
    %c35184372088864_i64_7 = arith.constant 35184372088864 : i64
    cce.mov.out.to.ub.align.v2.dv(%view_2, %reinterpret_cast_4, %c288230377225469952_i64_6, %c35184372088864_i64_7) : (memref<8192xf32, 6>, memref<8192xf32, strided<[1], offset: ?>, 1>, i64, i64)
    cce.vec_scope {
      %c0_10 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      scf.for %arg4 = %c0_10 to %c128 step %c1 {
        %6 = arith.muli %arg4, %c64 : index
        %reinterpret_cast_11 = memref.reinterpret_cast %view to offset: [%6], sizes: [64], strides: [1] : memref<8192xf32, 6> to memref<64xf32, strided<[1], offset: ?>, 6>
        %reinterpret_cast_12 = memref.reinterpret_cast %view_2 to offset: [%6], sizes: [64], strides: [1] : memref<8192xf32, 6> to memref<64xf32, strided<[1], offset: ?>, 6>
        %reinterpret_cast_13 = memref.reinterpret_cast %view_3 to offset: [%6], sizes: [64], strides: [1] : memref<8192xf32, 6> to memref<64xf32, strided<[1], offset: ?>, 6>
        %c0_14 = arith.constant 0 : index
        %c0_15 = arith.constant 0 : index
        %7 = arith.addi %c0_15, %c0_14 : index
        %8 = arith.index_cast %7 : index to i32
        %c4_i32 = arith.constant 4 : i32
        %9 = arith.muli %8, %c4_i32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_16 = arith.constant 0 : i32
        %10 = cce.vldsx1(%reinterpret_cast_11, %9, %c0_i32, %c0_i32_16) : (memref<64xf32, strided<[1], offset: ?>, 6>, i32, i32, i32) -> vector<64xf32>
        %c0_17 = arith.constant 0 : index
        %c0_18 = arith.constant 0 : index
        %11 = arith.addi %c0_18, %c0_17 : index
        %12 = arith.index_cast %11 : index to i32
        %c4_i32_19 = arith.constant 4 : i32
        %13 = arith.muli %12, %c4_i32_19 : i32
        %c0_i32_20 = arith.constant 0 : i32
        %c0_i32_21 = arith.constant 0 : i32
        %14 = cce.vldsx1(%reinterpret_cast_12, %13, %c0_i32_20, %c0_i32_21) : (memref<64xf32, strided<[1], offset: ?>, 6>, i32, i32, i32) -> vector<64xf32>
        %c0_i32_22 = arith.constant 0 : i32
        %15 = cce.pset(%c0_i32_22) {mask_bitwidth = 32 : i32} : (i32) -> vector<256xi1>
        %16 = cce.vadd(%10, %14, %15) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
        %c0_23 = arith.constant 0 : index
        %c0_24 = arith.constant 0 : index
        %17 = arith.addi %c0_24, %c0_23 : index
        %18 = arith.index_cast %17 : index to i32
        %c4_i32_25 = arith.constant 4 : i32
        %19 = arith.muli %18, %c4_i32_25 : i32
        %c0_i32_26 = arith.constant 0 : i32
        %20 = cce.pset(%c0_i32_26) {mask_bitwidth = 32 : i32} : (i32) -> vector<256xi1>
        %c2_i32 = arith.constant 2 : i32
        cce.vstsx1(%16, %reinterpret_cast_13, %19, %c2_i32, %c0_i32_26, %20) : (vector<64xf32>, memref<64xf32, strided<[1], offset: ?>, 6>, i32, i32, i32, vector<256xi1>)
      }
      cce.yield
    }
    %c288230377225469952_i64_8 = arith.constant 288230377225469952 : i64
    %c35184372088864_i64_9 = arith.constant 35184372088864 : i64
    cce.mov.ub.to.out.align.v2.dv(%reinterpret_cast_5, %view_3, %c288230377225469952_i64_8, %c35184372088864_i64_9) : (memref<8192xf32, strided<[1], offset: ?>, 1>, memref<8192xf32, 6>, i64, i64)
    return
  }
}

