#include "opentile/codegen/codegen_tile.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/target/target.h>

#include "support/check.h"
#include "target/source/codegen_source_base.h"

namespace tvm {
namespace codegen {

namespace {

std::string TileCodeGen(IRModule mod) {
  CodeGenTileLangTile codegen;

  for (const auto &item : mod->functions) {
    ICHECK(item.second->IsInstance<tirx::PrimFuncNode>())
        << "CodeGenTileLangTile: Can only take PrimFunc";
    GlobalVar global_var = Downcast<GlobalVar>(item.first);
    tirx::PrimFunc function = Downcast<tirx::PrimFunc>(item.second);
    codegen.AddFunction(global_var, function);
  }

  return codegen.Finish();
}

} // namespace

ffi::Module BuildTileLangTile(IRModule mod, Target target) {
  (void)target;
  std::string code = TileCodeGen(mod);
  return CSourceModuleCreate(code, "mlir", ffi::Array<ffi::String>());
}

ffi::Module BuildTileLangTileWithoutCompile(IRModule mod, Target target) {
  (void)target;
  std::string code = TileCodeGen(mod);
  return CSourceModuleCreate(code, "mlir", ffi::Array<ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_tile", BuildTileLangTile)
      .def("target.build.tilelang_tile_without_compile",
           BuildTileLangTileWithoutCompile);
}

} // namespace codegen
} // namespace tvm
