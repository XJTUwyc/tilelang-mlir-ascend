/*!
 * \file opentile/codegen/codegen_tile.h
 * \brief Minimal Tile codegen interface.
 */
#ifndef TVM_TL_OPENTILE_CODEGEN_CODEGEN_TILE_H_
#define TVM_TL_OPENTILE_CODEGEN_CODEGEN_TILE_H_

#include <string>

#include <tvm/ir/expr.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace codegen {

/*! \brief Minimal code generator interface for the Tile backend. */
class CodeGenTileLangTile {
public:
  /*! \brief Accept a TIRX PrimFunc without generating code yet. */
  void AddFunction(const GlobalVar &global_var,
                   const tirx::PrimFunc &function);

  /*! \brief Return the generated MLIR source. */
  std::string Finish();
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_OPENTILE_CODEGEN_CODEGEN_TILE_H_
