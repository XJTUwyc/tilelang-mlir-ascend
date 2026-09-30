"""Device compilation result consumed by the Tile execution adapter.

Like CUDA's binary plus FunctionInfo handoff, the object and final launch
metadata travel together. This module also owns automatic device compilation.
Logical parameters and sources remain in CompiledArtifact; out_idx remains
an adapter concern. Automatic compilation assumes order-preserving lowering and checks the final
LLVM signature; it cannot detect reordering between same-typed parameters.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .manifest import TileLaunchSpec


@dataclass(frozen=True)
class TileCompilationResult:
    """Owned device bytes and their matching, final launch contract.

    This structural validation does not prove object/metadata provenance or
    target compatibility. Those are the producer's responsibility. Bytes
    remain usable after a compiler's temporary directory has been removed.
    """

    object_bytes: bytes
    launch_spec: TileLaunchSpec

    def __post_init__(self):
        if not isinstance(self.object_bytes, bytes):
            raise TypeError("object_bytes must be bytes")
        if not self.object_bytes:
            raise ValueError("object_bytes must not be empty")
        if not isinstance(self.launch_spec, TileLaunchSpec):
            raise TypeError("launch_spec must be TileLaunchSpec")

# A deliberately bounded, order-preserving ABI contract. This is not an LLVM parser
# or a general reconstruction of compiler transformations.
def _parse_source_launch_contract(source):
    import json

    prefix = '// tilelang.launch.v1 '
    records = [json.loads(line[len(prefix):]) for line in source.splitlines()
               if line.startswith(prefix)]
    if len(records) != 1:
        raise ValueError('automatic Tile compilation requires one launch record from the current translator')
    record = records[0]
    if len(record['functions']) != 1:
        raise NotImplementedError('automatic Tile compilation supports one logical kernel')
    return record['functions'][0]


def _materialize_cv12_split_attr(module_path, contract):
    """Add the CCE marker for an explicitly partitioned 1:2 MIX kernel."""
    import re

    axes = contract['axes']
    uses_cv12_split = (
        [axis['tag'] for axis in axes] == ['blockIdx.x', 'blockIdx.y']
        and axes[1]['extent'] == 2
    )
    if not uses_cv12_split:
        return

    mlir = module_path.read_text()
    if re.search(r'\bnpu\.cv12_split\b', mlir):
        return
    if re.search(
        r'npu\.module_core_type\s*=\s*'
        r'#npu\.module_core_type\s*<\s*MIX\s*>',
        mlir,
    ) is None:
        raise ValueError(
            'blockIdx.y extent 2 requires an NPU MIX module before CCE lowering'
        )
    module_header = re.search(
        r'(?m)^(\s*module\s+attributes\s*\{)',
        mlir,
    )
    if module_header is None:
        raise ValueError('cannot locate the NPU module attribute dictionary')
    mlir = (
        mlir[:module_header.end()]
        + 'npu.cv12_split, '
        + mlir[module_header.end():]
    )
    module_path.write_text(mlir)


class _NPUCoreMode(Enum):
    """Core-mode decisions derived from the NPU split result."""

    AIC = 'aic'
    AIV = 'aiv'
    MIX = 'mix'

    @classmethod
    def from_split_ir(cls, module_ir):
        """Read the authoritative core mode produced by NPU splitting."""
        import re

        matches = re.findall(
            r'npu\.module_core_type\s*=\s*'
            r'#npu\.module_core_type\s*<\s*(AIC|AIV|MIX)\s*>',
            module_ir,
        )
        if len(matches) != 1:
            raise ValueError(
                'expected exactly one '
                'npu.module_core_type<AIC|AIV|MIX> after NPU splitting'
            )
        return cls[matches[0]]

    @property
    def is_mix(self):
        return self is _NPUCoreMode.MIX

    @property
    def entry_suffix(self):
        return None if self.is_mix else '_mix_' + self.value

    @property
    def ccec_arch(self):
        return 'dav-c310-vec' if self is _NPUCoreMode.AIV else 'dav-c310-cube'

    @property
    def requires_ccec_mix_flag(self):
        return self in (_NPUCoreMode.AIC, _NPUCoreMode.MIX)

    def expected_llvm_entries(self, root):
        if self.is_mix:
            return {root + '_mix_aic', root + '_mix_aiv'}
        return {root}

    def __str__(self):
        return self.value


def _rewrite_pure_core_entry_symbol(module_path, contract, mode):
    """Make a pure-core object self-describing to ACL's data loader.

    On a CV-separated device, aclrtBinaryLoadFromData defaults an untyped ELF
    entry to Cube. Runtime recognizes ``_mix_aic`` and ``_mix_aiv`` as Cube
    and Vector entries, respectively, and strips the suffix while registering
    the logical function, so callers still look up the original kernel name.
    """
    if mode.entry_suffix is None:
        return

    import re

    root = contract['name']
    tagged = root + mode.entry_suffix
    llvm_ir = module_path.read_text()
    bare = re.compile(r'@' + re.escape(root) + r'(?![-\w.$])')
    quoted = re.compile(r'@"' + re.escape(root) + r'"')
    if bare.search(llvm_ir) and quoted.search(llvm_ir):
        raise ValueError(f'inconsistent quoted LLVM symbol for {mode.value.upper()} kernel {root}')
    if quoted.search(llvm_ir):
        llvm_ir, replacements = quoted.subn('@"' + tagged + '"', llvm_ir)
    else:
        llvm_ir, replacements = bare.subn('@' + tagged, llvm_ir)
    if replacements == 0:
        raise ValueError(f'cannot locate LLVM symbol for {mode.value.upper()} kernel {root}')
    module_path.write_text(llvm_ir)


def _validate_and_build_launch_spec(llvm_ir, contract, mode):
    import re
    from .manifest import ConstantBinding, DeviceArgument, ParameterBinding

    # Match only the one-line signatures produced by tile-translate.
    # Reject unsupported syntax instead of accepting a partial interpretation.
    pattern = re.compile(
        r'^define\s+([^@\n]+)@("[^"]+"|[-\w.$]+)\(([^\n]*)\)\s*([^\n{]*)\{\n(.*?)^}',
        re.MULTILINE | re.DOTALL,
    )
    definitions = []
    for match in pattern.finditer(llvm_ir):
        prefix, name, arguments, attrs, body = match.groups()
        if 'ptc_kernel' not in prefix.split():
            continue
        if not prefix.rstrip().endswith('void'):
            raise NotImplementedError('Tile kernels must return void and write tensor outputs')
        name = name.strip('"')
        attr_id = re.search(r'#(\d+)', attrs)
        if attr_id:
            group = re.search(r'^attributes #' + attr_id[1] + r' = \{([^\n]*)\}', llvm_ir, re.MULTILINE)
            if group:
                attrs += group[1]
        cpu = re.search(r'"target-cpu"\s*=\s*"([^"]+)"', attrs)
        if cpu is None:
            raise ValueError(f"no target-cpu attribute for kernel {name}")
        parsed = []
        for argument in arguments.split(',') if arguments.strip() else []:
            parameter = re.fullmatch(
                r'\s*(ptr(?:\s+addrspace\(1\))?|i(?:8|16|32|64)|float|double)\s+(%[-\w.$]+)\s*',
                argument,
            )
            if parameter is None:
                raise NotImplementedError(f"unsupported final ABI argument: {argument}")
            parsed.append(parameter.groups())
        # Comments are not SSA uses. This conservative scan includes debug
        # instructions; an unproven dead argument is not silently zeroed.
        body = '\n'.join(line.split(';', 1)[0] for line in body.splitlines())
        definitions.append((name, parsed, body, cpu[1]))
    if not definitions:
        raise ValueError('no supported ptc_kernel definition in final LLVM IR')
    root = contract['name']
    names = {entry[0] for entry in definitions}
    expected_names = mode.expected_llvm_entries(root)
    if len(definitions) != len(names) or names != expected_names:
        raise NotImplementedError(f"expected {mode} kernel entries {sorted(expected_names)}, got {sorted(names)}")
    axes = contract['axes']
    if [axis['tag'] for axis in axes] not in (['blockIdx.x'], ['blockIdx.x', 'blockIdx.y']):
        raise NotImplementedError('supported axes are x, or x/y with y the MIX sub-block')
    if len(axes) == 2 and (not mode.is_mix or axes[1]['extent'] != 2):
        raise NotImplementedError('a second axis is supported only for MIX with extent 2')
    if mode.is_mix and len(axes) != 2:
        raise NotImplementedError('automatic MIX launch currently requires explicit x/y axes')
    logical = contract['kinds']
    scalar_types = {
        **{kind + str(bits): 'i' + str(bits)
           for kind in ('int', 'uint') for bits in (8, 16, 32, 64)},
        'float32': 'float', 'float64': 'double', 'handle': 'ptr',
    }
    logical_bindings = tuple(
        DeviceArgument(kind, ParameterBinding(i))
        for i, kind in enumerate(logical)
    )
    expected_types = [scalar_types[kind] for kind in logical]
    bindings = None

    supported_cpus = {'dav-c310-cube', 'dav-c310-vec'}
    for name, arguments, body, cpu in definitions:
        if cpu not in supported_cpus:
            raise NotImplementedError(f"{name}: unsupported target-cpu {cpu}")
        if not len(logical) <= len(arguments) <= len(logical) + len(axes):
            raise ValueError(f"{name}: final ABI does not preserve the logical argument prefix")
        for i, (expected, (actual, _)) in enumerate(zip(expected_types, arguments)):
            if (actual.startswith('ptr') if expected == 'ptr' else actual == expected):
                continue
            raise ValueError(f"{name}: logical parameter {i} requires {expected}, got {actual}")
        current = list(logical_bindings)
        for actual, parameter in arguments[len(logical):]:
            if actual != 'i32':
                raise ValueError(f"{name}: unexpected trailing {actual}; workspace/extra scalars unsupported")
            if re.search(re.escape(parameter) + r'(?![-\w.$])', body):
                raise ValueError(
                    f"{name}: axis parameter {parameter} is still used; the installed OpenTileAS "
                    "must hardwareize BlockIdx/SubBlockIdx before it can be replaced by zero"
                )
            current.append(DeviceArgument('int32', ConstantBinding(0)))
        if bindings is not None and tuple(current) != bindings:
            raise ValueError('MIX entries have different final argument layouts')
        bindings = tuple(current)

    # The supported pipeline statically plans local memory. No extra dynamic
    # UBUF or hidden workspace is supplied; other pipelines are out of scope.
    return TileLaunchSpec(root, bindings, axes[0]['extent'], dynamic_ubuf_bytes=0)


def compile_tile_obj(artifact, *, pass_configs=None, verbose=False):
    """Compile fixed-shape Tile MLIR into owned CCE object bytes.

    Uses the same source -> device compilation separation as GPU source
    adapters. Returns owned bytes; never runs from the kernel's __call__.
    The translator record is a candidate ABI, validated against final LLVM.
    """
    import logging
    import os
    import shlex
    import shutil
    import subprocess
    import tempfile
    from tilelang.jit.diagnostics import jit_phase
    from tilelang.transform import PassConfigKey

    if artifact.target is None or 'tile' not in artifact.target.keys:
        raise ValueError('compile_tile_obj requires target="tile"')
    contract = _parse_source_launch_contract(artifact.kernel_source)
    if artifact.params is None or len(artifact.params) != len(contract['kinds']):
        raise ValueError('logical artifact parameters do not match translator metadata')
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
           for param in artifact.params for dim in param.shape):
        raise NotImplementedError('automatic Tile compilation requires static tensor shapes')
    pass_configs = pass_configs or {}
    flags = pass_configs.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, []) or []
    flags = [flags] if isinstance(flags, str) else list(flags)
    if not all(isinstance(flag, str) for flag in flags):
        raise TypeError('compile flags must be strings')
    if any(flag.startswith(('-o', '--output', '--cce-aicore-arch', '-cce-aicore-arch',
                            '--cce-aicore-only', '-cce-enable-mix', '--cce-enable-mix'))
                            for flag in flags):
        raise ValueError('output and core-mode flags are selected by the Tile compiler')
    repo = Path(__file__).resolve().parents[2]
    tools_root = Path(os.environ.get('OPENTILEAS_ROOT', repo.parent / 'OpenTileAS')).expanduser()

    def tool(env_name, candidate, command):
        selected = os.environ.get(env_name)
        if selected:
            resolved = shutil.which(os.path.expanduser(selected))
        else:
            resolved = shutil.which(str(candidate)) if candidate else None
            resolved = resolved or shutil.which(command)
        if resolved is None:
            raise FileNotFoundError(f"{command} not found; set {env_name} to its executable path")
        return resolved

    opt = tool('TILE_OPT', tools_root / 'build/bin/tile-opt', 'tile-opt')
    translate = tool('TILE_TRANSLATE', tools_root / 'build/bin/tile-translate', 'tile-translate')
    ccec = tool('CCEC', None, 'ccec')
    keep_root = os.environ.get('TILELANG_TILE_BUILD_DIR')
    temporary = None
    if keep_root:
        Path(keep_root).expanduser().mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='tile-', dir=Path(keep_root).expanduser()))
    else:
        temporary = tempfile.TemporaryDirectory(prefix='tile-')
        directory = Path(temporary.name)
    logger = logging.getLogger(__name__)

    def run(phase, command):
        if verbose:
            logger.warning('Tile %s: %s', phase, shlex.join(command))
        with jit_phase(phase, verbose=verbose, kernel=contract['name']):
            process = subprocess.run(command, capture_output=True, text=True)
        if keep_root:
            (directory / (phase + '.log')).write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(
                f"Tile {phase} failed ({process.returncode})\n{shlex.join(command)}\n"
                f"{process.stdout}\n{process.stderr}\n"
                "Set TILELANG_TILE_BUILD_DIR to preserve intermediate files."
            )

    try:
        source, npu, split_ir, cce, llvm, obj = [directory / name for name in (
            '01_tile.mlir', '02_npu.mlir', '03_split.mlir', '04_cce.mlir', '05_kernel.ll', 'kernel.o'
        )]
        source.write_text(artifact.kernel_source)
        run('tile_to_npu', [opt, str(source), '--convert-tilelang-to-npu', '--npu-normalize', '-o', str(npu)])
        run('npu_split', [opt, str(npu), '--npu-split-scope', '--npu-plan-memory',
                          '--npu-split-mix-kernel', '--npu-sync-pipeline', '-o', str(split_ir)])
        _materialize_cv12_split_attr(split_ir, contract)
        mode = _NPUCoreMode.from_split_ir(split_ir.read_text())
        run('npu_to_cce', [opt, '--cce-pipeline=target=dav-351x', str(split_ir), '-o', str(cce)])
        run('cce_to_llvm', [translate, '--cce-to-backend', '-allow-unregistered-dialect', str(cce), '-o', str(llvm)])
        launch_spec = _validate_and_build_launch_spec(llvm.read_text(), contract, mode)
        _rewrite_pure_core_entry_symbol(llvm, contract, mode)
        if keep_root:
            import json
            from dataclasses import asdict
            (directory / 'launch.json').write_text(json.dumps(asdict(launch_spec), indent=2))
        options = [f'--cce-aicore-arch={mode.ccec_arch}', '--cce-aicore-only']
        if mode.requires_ccec_mix_flag:
            options.append('-cce-enable-mix')
        options += ['-O2', '-cce-bitcode-is-aicore', '-mllvm', '--cce-vf-auto-sync=global',
                    '--cce-simd-vf-fusion=false']
        run('ccec', [ccec, *options, *flags, '-c', str(llvm), '-o', str(obj)])
        result = TileCompilationResult(obj.read_bytes(), launch_spec)
        if verbose or keep_root:
            logger.warning('Tile object built: %s (%s)', obj, mode)
        return result
    finally:
        if temporary is not None:
            temporary.cleanup()
