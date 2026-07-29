#include "opentile/codegen/codegen_tile.h"

namespace tvm {
namespace codegen {

void CodeGenTileLangTile::AddFunction(const GlobalVar &global_var,
                                      const tirx::PrimFunc &function) {
  (void)global_var;
  (void)function;
}

std::string CodeGenTileLangTile::Finish() { return ""; }

} // namespace codegen
} // namespace tvm
