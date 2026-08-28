//===- TileLangAttrs.td - TileLang attribute definitions -----*- tablegen -*-===//

#ifndef TILELANG_DIALECT_IR_TILELANGATTRS_TD
#define TILELANG_DIALECT_IR_TILELANGATTRS_TD

include "TileLang/IR/TileLangDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class TileLang_I32Enum<string name, string description,
                       list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tilelang";
}

class TileLang_I32EnumAttr<string mnemonic, TileLang_I32Enum enumInfo>
    : EnumAttr<TileLang_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def TileLang_SCOPE_DEFAULT
    : I32EnumAttrCase<"DEFAULT", 0, "default">;
def TileLang_SCOPE_SIMD : I32EnumAttrCase<"SIMD", 1, "simd">;
def TileLang_SCOPE_SIMT : I32EnumAttrCase<"SIMT", 2, "simt">;

def TileLang_ScopeModeEnum
    : TileLang_I32Enum<"ScopeMode", "TileLang execution scope mode", [
        TileLang_SCOPE_DEFAULT,
        TileLang_SCOPE_SIMD,
        TileLang_SCOPE_SIMT
      ]>;

def TileLang_ScopeModeAttr
    : TileLang_I32EnumAttr<"scope_mode", TileLang_ScopeModeEnum> {
  let description = [{
    Selects default, SIMD, or SIMT execution semantics for a TileLang scope.
  }];
}

#endif // TILELANG_DIALECT_IR_TILELANGATTRS_TD
