module {
  func.func @vadd_kernel(%inputA: memref<?xf32, #npu.address_space<gm>>, %inputB: memref<?xf32, #npu.address_space<gm>>, %outputC: memref<?xf32, #npu.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %bid64 = npu.get_block_idx() : i64
    %bid = arith.trunci %bid64 : i64 to i32
    %1 = arith.muli %bid, %c32_i32 : i32
    %2 = arith.index_cast %1 : i32 to index // gm offset

    // 1. Allocate UB shared memory (32*256*4 = 32768 bytes)
    %raw_A = memref.alloc() : memref<32768xi8, #npu.address_space<ub>>
    %raw_B = memref.alloc() : memref<32768xi8, #npu.address_space<ub>>
    %raw_C = memref.alloc() : memref<32768xi8, #npu.address_space<ub>>

    // 2. View as flat f32 memrefs (32*256 = 8192 f32 elements)
    %A_shared = memref.view %raw_A[%c0][] : memref<32768xi8, #npu.address_space<ub>> to memref<8192xf32, #npu.address_space<ub>>
    %B_shared = memref.view %raw_B[%c0][] : memref<32768xi8, #npu.address_space<ub>> to memref<8192xf32, #npu.address_space<ub>>
    %C_shared = memref.view %raw_C[%c0][] : memref<32768xi8, #npu.address_space<ub>> to memref<8192xf32, #npu.address_space<ub>>

    // 3. Reinterpret GM buffers with dynamic offset, flat 1D
    %A = memref.reinterpret_cast %inputA to offset: [%2], sizes: [8192], strides: [1] : memref<?xf32, #npu.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #npu.address_space<gm>>
    %B = memref.reinterpret_cast %inputB to offset: [%2], sizes: [8192], strides: [1] : memref<?xf32, #npu.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #npu.address_space<gm>>
    %C = memref.reinterpret_cast %outputC to offset: [%2], sizes: [8192], strides: [1] : memref<?xf32, #npu.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #npu.address_space<gm>>

    // 4. Copy GM -> UB
    npu.copy ins(%A : memref<8192xf32, strided<[1], offset: ?>, #npu.address_space<gm>>) outs(%A_shared : memref<8192xf32, #npu.address_space<ub>>)
    npu.copy ins(%B : memref<8192xf32, strided<[1], offset: ?>, #npu.address_space<gm>>) outs(%B_shared : memref<8192xf32, #npu.address_space<ub>>)

    // 5. Vector add in SIMD scope (8192 elements = 128 vectors of 64)
    npu.scope() {
      %c0_1 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      scf.for %tile = %c0_1 to %c128 step %c1 {
        %tile_offset = arith.muli %tile, %c64 : index
        %tile_A = memref.reinterpret_cast %A_shared to offset: [%tile_offset], sizes: [64], strides: [1] : memref<8192xf32, #npu.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #npu.address_space<ub>>
        %tile_B = memref.reinterpret_cast %B_shared to offset: [%tile_offset], sizes: [64], strides: [1] : memref<8192xf32, #npu.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #npu.address_space<ub>>
        %tile_C = memref.reinterpret_cast %C_shared to offset: [%tile_offset], sizes: [64], strides: [1] : memref<8192xf32, #npu.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #npu.address_space<ub>>
        %frag_A = npu.vload %tile_A : (memref<64xf32, strided<[1], offset: ?>, #npu.address_space<ub>>) -> vector<64xf32>
        %frag_B = npu.vload %tile_B : (memref<64xf32, strided<[1], offset: ?>, #npu.address_space<ub>>) -> vector<64xf32>
        %frag_C = npu.add %frag_A, %frag_B : vector<64xf32>, vector<64xf32> -> vector<64xf32>
        npu.vstore %frag_C, %tile_C : (vector<64xf32>, memref<64xf32, strided<[1], offset: ?>, #npu.address_space<ub>>) -> ()
      }
      npu.yield
    }  {mode = #npu.scope_mode<simd>} : () -> ()

    // 6. Copy UB -> GM
    npu.copy ins(%C_shared : memref<8192xf32, #npu.address_space<ub>>) outs(%C : memref<8192xf32, strided<[1], offset: ?>, #npu.address_space<gm>>)
    return
  }
}
