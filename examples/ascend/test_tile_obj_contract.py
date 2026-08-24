"""Contract-validation checks for TODO 3 (artifact <-> .o binding checks).

What is verified
----------------
The bind-time checks added to ``tilelang/opentile/tile_obj.py``:

- parsing an ELF symbol table without external dependencies,
- rejecting a wrong (artifact, .o) pairing at bind time,
- rejecting non-ELF / stripped / non-function / empty-kernel objects,
- graceful warn (not crash) when the parameter signature cannot be
  verified statically.

Workflow
--------
1. Run all checks on synthetic ELF objects (no NPU / torch / pyelftools
   needed)::

       python examples/ascend/test_tile_obj_contract.py

2. Validate a specific hand-compiled object file instead::

       python examples/ascend/test_tile_obj_contract.py /path/to/kernel.o

3. List the available checks::

       python examples/ascend/test_tile_obj_contract.py --list

Shared helpers live in ``tile_obj_test_common.py``; the TODO 2 launch-time
dtype checks live in ``test_tile_obj_dtype.py``.  Run from the repo root
with the project venv (``/home/wuyuchao/dev/.venv``).
"""

from __future__ import annotations

import os
import struct
import sys
import warnings
from pathlib import Path

from tile_obj_test_common import _expect_raises, _info, run_module, tile_obj


# ---------------------------------------------------------------------------
# Minimal ELF64 builder (little-endian), used to produce synthetic .o files.
# ---------------------------------------------------------------------------
def _make_elf64(symbols, with_symtab=True):
    """Return bytes of a minimal valid ELF64 relocatable object.

    ``symbols`` is a list of (name, st_type, st_size).  ``st_type`` uses the
    generic STT_* values (1 = object, 2 = func).
    """
    strtab = b"\x00"
    entries = []
    for name, st_type, st_size in symbols:
        if name:
            off = len(strtab)
            strtab += name.encode("utf-8") + b"\x00"
        else:
            off = 0
        entries.append(struct.pack("<IBBHQQ", off, st_type & 0xF, 0, 0, 0, st_size))
    symtab = b"".join(entries)

    hdr_len = 64
    strtab_off = hdr_len
    symtab_off = strtab_off + len(strtab)
    shoff = (symtab_off + len(symtab) + 7) & ~7
    e_shnum = 3 if with_symtab else 2

    blob = bytearray(shoff + e_shnum * 64)
    blob[0:4] = b"\x7fELF"
    blob[4] = 2  # ELFCLASS64
    blob[5] = 1  # ELFDATA2LSB
    blob[6] = 1  # EV_CURRENT
    struct.pack_into("<H", blob, 0x10, 1)  # e_type = ET_REL
    struct.pack_into("<H", blob, 0x12, 62)  # e_machine (arbitrary)
    struct.pack_into("<I", blob, 0x14, 1)  # e_version
    struct.pack_into("<Q", blob, 0x28, shoff)  # e_shoff
    struct.pack_into("<H", blob, 0x34, 64)  # e_ehsize
    struct.pack_into("<H", blob, 0x3A, 64)  # e_shentsize
    struct.pack_into("<H", blob, 0x3C, e_shnum)

    blob[strtab_off : strtab_off + len(strtab)] = strtab
    # section 1: .strtab
    struct.pack_into("<I", blob, shoff + 64 + 4, 3)  # sh_type = SHT_STRTAB
    struct.pack_into("<Q", blob, shoff + 64 + 24, strtab_off)
    struct.pack_into("<Q", blob, shoff + 64 + 32, len(strtab))
    struct.pack_into("<Q", blob, shoff + 64 + 56, 1)  # sh_entsize
    if with_symtab:
        blob[symtab_off : symtab_off + len(symtab)] = symtab
        # section 2: .symtab (sh_link -> section 1)
        struct.pack_into("<I", blob, shoff + 128 + 4, 2)  # sh_type = SHT_SYMTAB
        struct.pack_into("<Q", blob, shoff + 128 + 24, symtab_off)
        struct.pack_into("<Q", blob, shoff + 128 + 32, len(symtab))
        struct.pack_into("<I", blob, shoff + 128 + 40, 1)  # sh_link
        struct.pack_into("<Q", blob, shoff + 128 + 56, 24)  # sh_entsize
    return bytes(blob)


def _make_elf32(symbols, with_symtab=True):
    """Return bytes of a minimal valid ELF32 relocatable object (little-endian).

    ``symbols`` is a list of (name, st_type, st_size), same as ``_make_elf64``.
    Note the Elf32_Sym field order differs from Elf64_Sym: st_value comes
    before st_size and st_info sits at offset 12 (not 4).
    """
    strtab = b"\x00"
    entries = []
    for name, st_type, st_size in symbols:
        if name:
            off = len(strtab)
            strtab += name.encode("utf-8") + b"\x00"
        else:
            off = 0
        # Elf32_Sym: st_name(4) st_value(4) st_size(4) st_info(1) st_other(1) st_shndx(2)
        entries.append(struct.pack("<IIIBBH", off, 0, st_size, st_type & 0xF, 0, 0))
    symtab = b"".join(entries)

    hdr_len = 52
    strtab_off = hdr_len
    symtab_off = strtab_off + len(strtab)
    shoff = (symtab_off + len(symtab) + 3) & ~3
    e_shnum = 3 if with_symtab else 2

    blob = bytearray(shoff + e_shnum * 40)
    blob[0:4] = b"\x7fELF"
    blob[4] = 1  # ELFCLASS32
    blob[5] = 1  # ELFDATA2LSB
    blob[6] = 1  # EV_CURRENT
    struct.pack_into("<H", blob, 0x10, 1)  # e_type = ET_REL
    struct.pack_into("<H", blob, 0x12, 3)  # e_machine (arbitrary, EM_386)
    struct.pack_into("<I", blob, 0x14, 1)  # e_version
    struct.pack_into("<I", blob, 0x20, shoff)  # e_shoff
    struct.pack_into("<H", blob, 0x28, 52)  # e_ehsize
    struct.pack_into("<H", blob, 0x2E, 40)  # e_shentsize
    struct.pack_into("<H", blob, 0x30, e_shnum)

    blob[strtab_off : strtab_off + len(strtab)] = strtab
    # section 1: .strtab
    struct.pack_into("<I", blob, shoff + 40 + 4, 3)  # sh_type = SHT_STRTAB
    struct.pack_into("<I", blob, shoff + 40 + 16, strtab_off)  # sh_offset
    struct.pack_into("<I", blob, shoff + 40 + 20, len(strtab))  # sh_size
    if with_symtab:
        blob[symtab_off : symtab_off + len(symtab)] = symtab
        # section 2: .symtab (sh_link -> section 1)
        struct.pack_into("<I", blob, shoff + 80 + 4, 2)  # sh_type = SHT_SYMTAB
        struct.pack_into("<I", blob, shoff + 80 + 16, symtab_off)  # sh_offset
        struct.pack_into("<I", blob, shoff + 80 + 20, len(symtab))  # sh_size
        struct.pack_into("<I", blob, shoff + 80 + 24, 1)  # sh_link
        struct.pack_into("<I", blob, shoff + 80 + 36, 16)  # sh_entsize
    return bytes(blob)


# ---------------------------------------------------------------------------
# Symbol-table parsing
# ---------------------------------------------------------------------------
# 测试点：合成一个 ELF64 目标文件，内含 main/helper 两个函数符号、
#         data_obj 一个数据符号，以及索引 0 的空符号（未定义）。
# 验证功能：_read_elf_symbols 应能从 .symtab + .strtab 中解析出全部具名符号，
#           并正确还原每个符号的类型（STT_FUNC=2 / STT_OBJECT=1）和大小。
def check_symbol_parsing():
    """Parse the symbol table of a synthetic ELF64 object."""
    raw = _make_elf64(
        [
            ("", 0, 0),
            ("main", tile_obj._STT_FUNC, 128),
            ("helper", tile_obj._STT_FUNC, 64),
            ("data_obj", tile_obj._STT_OBJECT, 8),
        ]
    )
    syms = tile_obj._read_elf_symbols(raw)
    assert set(syms) == {"main", "helper", "data_obj"}, syms
    assert syms["main"].st_type == tile_obj._STT_FUNC
    assert syms["main"].st_size == 128
    assert syms["helper"].st_size == 64
    assert syms["data_obj"].st_type == tile_obj._STT_OBJECT


# 测试点：合成一个 ELF32 目标文件。Elf32_Sym 与 Elf64_Sym 字段顺序不同
#         （st_value 在 st_size 之前、st_info 位于 offset 12 而非 4）。
# 验证功能：_read_elf_symbols 的 ELF32 分支应使用按位宽索引的正确偏移
#           （st_info@12、st_size@8）解析符号类型与大小；若 st_info 固定
#           取 offset 4（Elf64 布局），会把 st_value 首字节当作类型、
#           把 st_info/st_other/st_shndx 拼成的整数当作大小。
def check_symbol_parsing_elf32():
    """Parse the symbol table of a synthetic ELF32 object (Elf32_Sym layout)."""
    raw = _make_elf32(
        [
            ("", 0, 0),
            ("main", tile_obj._STT_FUNC, 96),
            ("data_obj", tile_obj._STT_OBJECT, 4),
        ]
    )
    syms = tile_obj._read_elf_symbols(raw)
    assert set(syms) == {"main", "data_obj"}, syms
    assert syms["main"].st_type == tile_obj._STT_FUNC, syms["main"]
    assert syms["main"].st_size == 96, syms["main"]
    assert syms["data_obj"].st_type == tile_obj._STT_OBJECT, syms["data_obj"]
    assert syms["data_obj"].st_size == 4, syms["data_obj"]
    # 绑定期契约校验在 ELF32 上也应端到端可用（符号=main、函数、size>0）。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DWARF best-effort -> warn when unavailable
        tile_obj._validate_tile_obj(_info("main"), "obj32.o", raw)


# 测试点：输入数据一为非 ELF 魔数开头，一为只有魔数但其余内容全空的伪 ELF。
# 验证功能：_read_elf_symbols 对非法 ELF 输入应抛出 ValueError（边界防护）。
def check_non_elf_rejected():
    """Reject inputs that are not valid ELF images."""
    _expect_raises(ValueError, tile_obj._read_elf_symbols, b"this is not an elf at all")
    _expect_raises(ValueError, tile_obj._read_elf_symbols, b"\x7fELF" + b"\x00" * 16)


# ---------------------------------------------------------------------------
# Bind-time contract validation
# ---------------------------------------------------------------------------
# 测试点：.o 中的 kernel 符号与 IR 的 global_symbol（main）一致，
#         且为带实际代码的函数符号。
# 验证功能：_validate_tile_obj 的符号层契约校验应通过、不抛异常；
#           DWARF 层无法静态验证时仅降级为警告（此处忽略警告）。
def check_matching_symbol_ok():
    """A kernel symbol matching the IR global_symbol passes the contract."""
    raw = _make_elf64([("main", tile_obj._STT_FUNC, 128)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DWARF best-effort -> warn when unavailable
        tile_obj._validate_tile_obj(_info("main"), "obj.o", raw)


# 测试点：artifact 与 .o 错配——.o 只导出别的 kernel 符号
#         （simd_simt_gather_aiv），而 IR 期望的符号是 main。
# 验证功能：绑定期契约校验应发现缺失目标符号并抛出 ValueError，
#           从而把“传错配对”的问题暴露在绑定阶段，而不是拖到运行时。
def check_wrong_pairing_rejected():
    """A .o exposing a different kernel is rejected at bind time."""
    raw = _make_elf64([("simd_simt_gather_aiv", tile_obj._STT_FUNC, 256)])
    _expect_raises(ValueError, tile_obj._validate_tile_obj, _info("main"), "wrong.o", raw)


# 测试点：完全 stripped 的 .o——段表中不存在符号表段。
# 验证功能：没有符号表就无法验证 kernel 符号契约，_validate_tile_obj 应抛 ValueError。
def check_stripped_object_rejected():
    """A fully stripped object cannot satisfy the symbol contract."""
    raw = _make_elf64([], with_symtab=False)
    _expect_raises(ValueError, tile_obj._validate_tile_obj, _info("main"), "stripped.o", raw)


# 测试点：存在 .symtab 段，但其中没有任何符号条目。
# 验证功能：空符号表同样无法满足契约，_validate_tile_obj 应抛 ValueError。
def check_empty_symbol_table_rejected():
    """A present but empty .symtab is still not enough."""
    raw = _make_elf64([])  # symtab present but empty
    _expect_raises(ValueError, tile_obj._validate_tile_obj, _info("main"), "empty.o", raw)


# 测试点：同名符号存在，但是数据符号（STT_OBJECT）而不是函数符号。
# 验证功能：契约要求 kernel 必须是函数符号，类型不符时应抛 ValueError。
def check_symbol_not_function_rejected():
    """A data symbol named like the kernel is rejected."""
    raw = _make_elf64([("main", tile_obj._STT_OBJECT, 16)])
    _expect_raises(ValueError, tile_obj._validate_tile_obj, _info("main"), "data.o", raw)


# 测试点：函数符号存在，但 st_size 为 0（没有实际代码的空函数）。
# 验证功能：判定为无效 kernel，_validate_tile_obj 应抛 ValueError。
def check_zero_size_kernel_rejected():
    """A function symbol with zero size is rejected (kernel has no code)."""
    raw = _make_elf64([("main", tile_obj._STT_FUNC, 0)])
    _expect_raises(ValueError, tile_obj._validate_tile_obj, _info("main"), "empty_kernel.o", raw)


# ---------------------------------------------------------------------------
# DWARF best-effort layer
# ---------------------------------------------------------------------------
# 测试点：.o 不含 DWARF 调试信息（或环境未安装 pyelftools），无法静态核对参数签名。
# 验证功能：参数签名尽力校验应降级为警告（"warn"）而不是崩溃，
#           并返回说明信息（缺少 pyelftools 或找不到 DWARF subprogram）。
def check_dwarf_warns_when_not_verifiable():
    """Without DWARF the signature check downgrades to a warning."""
    raw = _make_elf64([("main", tile_obj._STT_FUNC, 128)])
    kind, msg = tile_obj._check_dwarf_signature(_info("main"), raw)
    assert kind == "warn", (kind, msg)
    assert msg  # either "pyelftools not installed" or "no DWARF subprogram"


# ---------------------------------------------------------------------------
# Integration against real hand-compiled .o files
# ---------------------------------------------------------------------------
# 测试点：对单个真实手编 .o 文件做完整契约校验，覆盖四条路径：
#         (1) 符号表可解析且非空；(2) 存在函数符号；
#         (3) 匹配的 kernel 名通过校验；(4) 不存在的 kernel 名被拒绝。
# 验证功能：真实文件应满足符号层契约；错误/缺失的 kernel 名应抛 ValueError。
def check_real_onboard_object(obj_path):
    """Validate a single real hand-compiled .o against the contract."""
    p = Path(obj_path)
    raw = p.read_bytes()
    syms = tile_obj._read_elf_symbols(raw)
    assert syms, f"{p.name}: no symbols parsed"
    funcs = [n for n, s in syms.items() if s.st_type == tile_obj._STT_FUNC]
    assert funcs, f"{p.name}: no function symbols"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tile_obj._validate_tile_obj(_info(funcs[0], n_handles=1), str(p), raw)
    _expect_raises(
        ValueError,
        tile_obj._validate_tile_obj,
        _info("__no_such_kernel__", n_handles=1),
        str(p),
        raw,
    )


# 测试点：批量校验 ~/dev/onboard_objects 目录下的所有 .o；
#         目录不存在时打印 SKIP 后跳过（不视为失败）。
# 验证功能：每个真实 .o 都应满足符号层契约，并正确拒绝不存在的 kernel 名。
def check_real_onboard_objects():
    """Validate every .o under ~/dev/onboard_objects, if present."""
    objs = sorted(Path(os.path.expanduser("~/dev/onboard_objects")).glob("*.o"))
    if not objs:
        print("SKIP: ~/dev/onboard_objects not present")
        return
    for p in objs:
        check_real_onboard_object(p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--list":
        # Validate a specific hand-compiled object file.
        try:
            check_real_onboard_object(sys.argv[1])
            print(f"PASS  check_real_onboard_object({sys.argv[1]})")
        except Exception as exc:
            print(f"FAIL  check_real_onboard_object({sys.argv[1]}): {type(exc).__name__}: {exc}")
            sys.exit(1)
        sys.exit(0)
    sys.exit(run_module(globals()))
