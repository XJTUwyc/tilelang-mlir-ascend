module {
  func.func @vadd_kernel(%inputA: memref<?xf32>, %inputB: memref<?xf32>, %outputC: memref<?xf32>, %arg0 : i32) attributes {BlockIdx = 3: i64} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c32_i32 = arith.constant 32 : i32
    %1 = arith.muli %arg0, %c32_i32 : i32
    %2 = arith.index_cast %1 : i32 to index //gm offset

    // 1. 分配shared_memory-对应1; fragment-对应2
    %raw_A = memref.alloc() : memref<32768xi8, 1>
    %raw_B = memref.alloc() : memref<32768xi8, 1>
    %raw_C = memref.alloc() : memref<32768xi8, 1>

    // 2. 通过view转换为对应的shape
    %A_shared = memref.view %raw_A[%c0][] : memref<32768xi8, 1> to memref<32x256xf32, 1>
    %B_shared = memref.view %raw_B[%c0][] : memref<32768xi8, 1> to memref<32x256xf32, 1>
    %C_shared = memref.view %raw_C[%c0][] : memref<32768xi8, 1> to memref<32x256xf32, 1>

    // 3/ 解析GM-对应0
    %A = memref.reinterpret_cast %inputA to offset: [%2], sizes: [32, 256], strides: [256, 1] : memref<?xf32, 0> to memref<32x256xf32, strided<[256,1]>, 0>
    %B = memref.reinterpret_cast %inputB to offset: [%2], sizes: [32, 256], strides: [256, 1] : memref<?xf32, 0> to memref<32x256xf32, strided<[256,1]>, 0>
    %C = memref.reinterpret_cast %outputC to offset: [%2], sizes: [32, 256], strides: [256, 1] : memref<?xf32, 0> to memref<32x256xf32, strided<[256,1]>, 0>
    
    // 4. 用 tilelang.copy 将GM拷贝到UB
    tilelang.copy %A, %A_shared : memref<32x256xf32, 0> to memref<32x256xf32, 1>
    tilelang.copy %B, %B_shared : memref<32x256xf32, 0> to memref<32x256xf32, 1>

    // 5. SimdVF
    tilelang.scope {
        %c0_1 = arith.constant 0 : index
        %c0_2 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c256 = arith.constant 256 : index
        scf.for %arg1 = %c0_1 to %c32 step %c1 {
            scf.for %arg2 = %c0_2 to %c4 step %c1 {
                %offset_col = arith.muli %arg2, %c64 : index
                %row_offset = arith.muli %arg1, %c256 : index
                %linear_offset = arith.addi %row_offset, %offset_col : index
                %frag_A = memref.alloc() : memref<64xf32, 2>
                %frag_B = memref.alloc() : memref<64xf32, 2>
                %frag_C = memref.alloc() : memref<64xf32, 2>
                %subview_A_1d = memref.reinterpret_cast %A_shared to offset: [%linear_offset], sizes: [64], strides: [1] : memref<32x256xf32, 1> to memref<64xf32, strided<[1], offset: ?>>
                %subview_B_1d = memref.reinterpret_cast %B_shared to offset: [%linear_offset], sizes: [64], strides: [1] : memref<32x256xf32, 1> to memref<64xf32, strided<[1], offset: ?>>
                %subview_C_1d = memref.reinterpret_cast %C_shared to offset: [%linear_offset], sizes: [64], strides: [1] : memref<32x256xf32, 1> to memref<64xf32, strided<[1], offset: ?>>
                tilelang.copy %subview_A_1d, %frag_A : memref<64xf32, 1> to memref<64xf32, 2>
                tilelang.copy %subview_B_1d, %frag_B : memref<64xf32, 1> to memref<64xf32, 2>
                linalg.add ins(%frag_A, %frag_B : memref<64xf32, 2>, memref<64xf32, 2>) outs(%frag_C : memref<64xf32, 2>)
                tilelang.copy %frag_C, %subview_C_1d : memref<64xf32, 2> to memref<64xf32, 1>
            }
        }
    } {mode = #tilelang.scope_mode<simd>}
    tilelang.copy %C_shared, %C : memref<32x256xf32, 1> to memref<32x256xf32, 0>
    return
  }
}
