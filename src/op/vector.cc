/*!
 * \file tl/op/vector.cc
 * \brief Register vector operations preserved for direct backend codegen.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;

TVM_REGISTER_OP("tl.tileop.vmuls")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vmuls")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vadd")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vadd")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vmul")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vmul")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vsub")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vsub")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vdiv")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vdiv")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vmax")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vmax")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vreduce_max")
    .set_num_inputs(2)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vreduce_max")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vreduce_sum")
    .set_num_inputs(2)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vreduce_sum")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vexp")
    .set_num_inputs(2)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vexp")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vexpdif")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vexpdif")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.vcvt")
    .set_num_inputs(3)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "vcvt")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
