//===- LegalizeForLLVMExport.cpp - Prepare Gemmini for LLVM translation ---===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"
#include "Gemmini/Transform.h"

using namespace mlir;
using namespace buddy::gemmini;

namespace {

int64_t getNumberFromValue(Value &value) {
  return value.getDefiningOp()
      ->getAttr("value")
      .dyn_cast<IntegerAttr>()
      .getInt();
}

acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
  union {
    acc_scale_t_bits b;
    acc_scale_t f;
  } un;

  un.f = x;
  return un.b;
}

scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
  union {
    scale_t_bits b;
    scale_t f;
  } un;

  un.f = x;
  return un.b;
}
}; // namespace

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct GemminiFlushLowering : public ConvertOpToLLVMPattern<FlushOp> {
  using ConvertOpToLLVMPattern<FlushOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FlushOp flushOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = flushOp.getLoc();
    Value skip = flushOp.getSkip();
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(0);
    Value rs2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rs2Attr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(flushOp, skip, rs2);
    return success();
  }
};

struct GemminiConfigStLowering : public ConvertOpToLLVMPattern<ConfigStOp> {
  using ConvertOpToLLVMPattern<ConfigStOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigStOp configStOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value strideValue = configStOp.getStride();
    int stride = getNumberFromValue(strideValue);
    float scale = configStOp.getScale().convertToFloat();
    Location loc = configStOp.getLoc();
    uint64_t arg = (uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)scale)
                       << 32 |
                   (uint32_t)stride;
    Value value1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(CONFIG_ST));
    Value value2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(arg));
    rewriter.replaceOpWithNewOp<ConfigSt_IntrOp>(configStOp, value1, value2);
    return success();
  }
};

struct GemminiConfigLdLowering : public ConvertOpToLLVMPattern<ConfigLdOp> {
  using ConvertOpToLLVMPattern<ConfigLdOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigLdOp configLdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value rs2Value = configLdOp.getStride();
    float scale = configLdOp.getScale().convertToFloat();
    uint64_t rs1 = (uint64_t)scale_t_to_scale_t_bits(scale) << 32 |
                   ((uint64_t)16 << 16) | (uint64_t)1 << 8 |
                   configLdOp.getId() << 3 | configLdOp.getShrunk() << 2 |
                   CONFIG_LD;
    Location loc = configLdOp.getLoc();
    Value rs1value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    rewriter.replaceOpWithNewOp<ConifgLd_IntrOp>(configLdOp, rs1value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiConfigExLowering : public ConvertOpToLLVMPattern<ConfigExOp> {
  using ConvertOpToLLVMPattern<ConfigExOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigExOp configExOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = configExOp.getLoc();
    float scale = configExOp.getSysAccScale().convertToFloat();
    uint64_t rs1 =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
        configExOp.getAStride() << 16 | configExOp.getBTranspose() << 9 |
        configExOp.getATranspose() << 8 | configExOp.getSetOnlyStrides() << 7 |
        configExOp.getSysAct() << 3 | configExOp.getDataflow() << 2 | CONFIG_EX;

    uint64_t rs2 = configExOp.getCStride() << 48 | configExOp.getSysShift();
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.replaceOpWithNewOp<ConfigEX_IntrOp>(configExOp, rs1Value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiConfigNormOpLowering : public ConvertOpToLLVMPattern<ConfigNormOp> {
  using ConvertOpToLLVMPattern<ConfigNormOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigNormOp configNormOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = configNormOp.getLoc();
    // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 
    // (((uint64_t) ((uint32_t) q_const)) << 32) | ((q_const_type & 1) << 18) | ((set_stats_id_only & 1) << 17) | ((act_msb & 1) << 16) | ((uint64_t)stat_id << 8) | CONFIG_BERT, ((uint64_t)((uint32_t)(igelu_qc)) << 32) | ((uint64_t)((uint32_t)(igelu_qb))), k_CONFIG)
    uint64_t rs1 = (((uint64_t) (configNormOp.getQConst())) << 32) |
                   (configNormOp.getQConstType() & 1) << 18 |
                   (configNormOp.getSetStatsIdOnly() & 1) << 17 |
                   (configNormOp.getActMsb() & 1) << 16 |
                   configNormOp.getStatsId() << 8 | CONFIG_BERT;
    uint64_t rs2 = (((uint64_t) configNormOp.getIguluQc()) << 32) | ((uint64_t) ((uint32_t)configNormOp.getIguluQb()));
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ConfigNorm_IntrOp>(configNormOp, rs1Value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
  explicit GemminiMvinLowering(LLVMTypeConverter &typeConverter,
                               int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = mvinOp.getInput();
    Location loc = input.getLoc();
    MemRefType memRefType =
        mvinOp.getOperandTypes().front().dyn_cast<MemRefType>();
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, input);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddrValue = mvinOp.getAddr();
    uint64_t number = getNumberFromValue(spadAddrValue);
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, indexCastOp, spad);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
  explicit GemminiMvoutLowering(LLVMTypeConverter &typeConverter,
                                int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value output = mvoutOp.getOutput();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Location loc = mvoutOp.getLoc();
    Value extractOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, resultType, output);
    IntegerType i64Type = rewriter.getI64Type();
    Value indexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
    Value spadAddr = mvoutOp.getAddr();
    uint64_t number = getNumberFromValue(spadAddr);
    MemRefType memRefType =
        mvoutOp.getOperandTypes().front().dyn_cast<MemRefType>();
    llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
                           (uint64_t)memRefShape[1] << addrLen | number;
    Value newSpad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, indexCastOp, newSpad);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiPreloadZerosLowering
    : public ConvertOpToLLVMPattern<PreloadZerosOp> {
  using ConvertOpToLLVMPattern<PreloadZerosOp>::ConvertOpToLLVMPattern;
  explicit GemminiPreloadZerosLowering(LLVMTypeConverter &typeConverter,
                                       int64_t dim, int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(PreloadZerosOp preloadZerosOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value addr = preloadZerosOp.getAddr();
    Value cRows = preloadZerosOp.getCRows();
    Value cCols = preloadZerosOp.getCCols();
    Location loc = preloadZerosOp.getLoc();
    uint64_t addrInt = getNumberFromValue(addr);
    uint64_t cRowsInt = getNumberFromValue(cRows);
    uint64_t cColsInt = getNumberFromValue(cCols);
    uint64_t rs1 = (uint64_t)dim << (addrLen + 16) | (uint64_t)dim << addrLen |
                   (uint64_t)-1;
    uint64_t rs2 = cRowsInt << (addrLen + 16) | cColsInt << (addrLen) | addrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadZerosOp, rs1Value,
                                                rs2Value);
    return success();
  }

private:
  int64_t dim;
  int64_t addrLen;
};

struct GemminiPreloadLowering : public ConvertOpToLLVMPattern<PreloadOp> {
  using ConvertOpToLLVMPattern<PreloadOp>::ConvertOpToLLVMPattern;
  explicit GemminiPreloadLowering(LLVMTypeConverter &typeConverter,
                                  int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(PreloadOp preloadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value bdAddr = preloadOp.getBdAddr();
    Value cAddr = preloadOp.getCAddr();
    Value bdCols = preloadOp.getBdCols();
    Value bdRows = preloadOp.getBdRows();
    Value cCols = preloadOp.getCCols();
    Value cRows = preloadOp.getBdRows();
    Location loc = preloadOp.getLoc();
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t cAddrInt = getNumberFromValue(cAddr);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t cColsInt = getNumberFromValue(cCols);
    uint64_t cRowsInt = getNumberFromValue(cRows);
    uint64_t rs1 = bdRowsInt << (addrLen + 16) | bdColsInt << addrLen |
                   (uint64_t)bdAddrInt;
    uint64_t rs2 =
        cRowsInt << (addrLen + 16) | cColsInt << addrLen | (uint64_t)cAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadOp, rs1Value, rs2Value);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiComputePreloadedLowering
    : public ConvertOpToLLVMPattern<ComputePreloadedOp> {
  using ConvertOpToLLVMPattern<ComputePreloadedOp>::ConvertOpToLLVMPattern;
  explicit GemminiComputePreloadedLowering(LLVMTypeConverter &typeConverter,
                                           int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(ComputePreloadedOp computePreloadedOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value aAddr = computePreloadedOp.getAAddr();
    Value bdAddr = computePreloadedOp.getBdAddr();
    Value aRows = computePreloadedOp.getARows();
    Value aCols = computePreloadedOp.getACols();
    Value bdRows = computePreloadedOp.getBdRows();
    Value bdCols = computePreloadedOp.getBdCols();
    Location loc = computePreloadedOp.getLoc();
    uint64_t aAddrInt = getNumberFromValue(aAddr);
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t aRowsInt = getNumberFromValue(aRows);
    uint64_t aColsInt = getNumberFromValue(aCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t rs1 = aRowsInt << (addrLen + 16) | aColsInt << addrLen | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (addrLen + 16) | bdColsInt << addrLen | bdAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ComputePreloaded_IntrOp>(computePreloadedOp,
                                                         rs1Value, rs2Value);
    return success();
  }

private:
  int64_t addrLen;
};

struct GemminiComputeAccumulatedLowering
    : public ConvertOpToLLVMPattern<ComputeAccumulatedOp> {
  using ConvertOpToLLVMPattern<ComputeAccumulatedOp>::ConvertOpToLLVMPattern;
  explicit GemminiComputeAccumulatedLowering(LLVMTypeConverter &typeConverter,
                                             int64_t addrLen)
      : ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
  LogicalResult
  matchAndRewrite(ComputeAccumulatedOp computeAccumulatedOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value aAddr = computeAccumulatedOp.getAAddr();
    Value bdAddr = computeAccumulatedOp.getBdAddr();
    Value aRows = computeAccumulatedOp.getARows();
    Value aCols = computeAccumulatedOp.getACols();
    Value bdRows = computeAccumulatedOp.getBdRows();
    Value bdCols = computeAccumulatedOp.getBdCols();
    Location loc = computeAccumulatedOp.getLoc();
    uint64_t aAddrInt = getNumberFromValue(aAddr);
    uint64_t bdAddrInt = getNumberFromValue(bdAddr);
    uint64_t aRowsInt = getNumberFromValue(aRows);
    uint64_t aColsInt = getNumberFromValue(aCols);
    uint64_t bdRowsInt = getNumberFromValue(bdRows);
    uint64_t bdColsInt = getNumberFromValue(bdCols);
    uint64_t rs1 = aRowsInt << (addrLen + 16) | aColsInt << addrLen | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (addrLen + 16) | bdColsInt << addrLen | bdAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ComputeAccumulated_IntrOp>(computeAccumulatedOp,
                                                           rs1Value, rs2Value);

    return success();
  }

private:
  int64_t addrLen;
};

class GemminiTileMatMulLowering : public ConvertOpToLLVMPattern<TileMatMulOp> {
  void gemminiLoopWs(size_t i, size_t j, size_t k, size_t padI, size_t padJ,
                     size_t padK, Value &a, Value &b, Value &d, Value &c,
                     size_t aRowStride, size_t bRowStride, size_t dRowStride,
                     size_t cRowStride, bool aTranspose, bool bTranspose,
                     bool fullC, bool lowD, bool exAccumulate, int act,
                     TileMatMulOp &tileMatMulOp,
                     ConversionPatternRewriter &rewriter) const {
    // loopWsConfigBounds instruction.
    uint64_t rs1 = (uint64_t)padK << 32 | (uint64_t)padJ << 16 | (uint64_t)padI;
    uint64_t rs2 = (uint64_t)k << 32 | (uint64_t)j << 16 | (uint64_t)i;
    IntegerType i64Type = rewriter.getI64Type();
    Location loc = a.getLoc();
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.create<LoopWsConfigBounds_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigAddrsAB instruction.
    rewriter.create<LoopWsConfigAddrsAB_IntrOp>(loc, a, b);
    // loopWsConfigAddrsDC instruction
    rewriter.create<LoopWsConfigAddrsDC_IntrOp>(loc, d, c);
    // loopWsConfigStridesAB instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(aRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(bRowStride));
    rewriter.create<LoopWsConfigStridesAB_IntrOp>(loc, rs1Value, rs2Value);
    // loopWsConfigStrideDC instruction
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(dRowStride));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(cRowStride));
    rewriter.create<LoopWsConfigStridesDC_IntrOp>(loc, rs1Value, rs2Value);
    rs1 = (uint64_t)act << 8 | lowD << 2 | (fullC) << 1 | exAccumulate;
    rs2 = bTranspose << 1 | aTranspose;
    rs1Value = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(rs1));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(rs2));
    rewriter.create<LoopWs_IntrOp>(loc, rs1Value, rs2Value);
  }

  void spTiledMatmulWs(Value &a, Value &b, Value &d, Value &c,
                       scale_t aScaleFactor, scale_t bScaleFactor,
                       scale_acc_t dScaleFactor, size_t i, size_t j, size_t k,
                       size_t padI, size_t padJ, size_t padK, size_t strideA,
                       size_t strideB, size_t strideD, size_t strideC,
                       bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                       bool noBias, bool repeatingBias, int act,
                       TileMatMulOp &tileMatMulOp,
                       ConversionPatternRewriter &rewriter) const {

    gemminiLoopWs(i, j, k, padI, padJ, padK, a, b, d, c, strideA, strideB,
                  repeatingBias ? 0 : strideD, strideC, aTranspose, bTranspose,
                  fullC, lowD, !noBias, act, tileMatMulOp, rewriter);
  }

  // Tiling functions
  void spTiledMatmulOs(Value &a, Value &b, Value &d, Value &c,
                       scale_t aScaleFactor, scale_t bScaleFactor,
                       scale_acc_t dScaleFactor, size_t i, size_t j, size_t k,
                       size_t padI, size_t padJ, size_t padK, size_t strideA,
                       size_t strideB, size_t strideD, size_t strideC,
                       bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                       bool noBias, bool repeatingBias, int act,
                       TileMatMulOp &tileMatMulOp,
                       ConversionPatternRewriter &rewriter) const {
    const uint32_t aSpAddrStart = 0;
    const uint32_t bSpAddrStart = BANK_NUM * BANK_ROWS - k * j * DIM;
    const uint32_t dSpAddrStart = 1 << (ADDR_LEN - 1);
    const uint32_t cSpAddrStart =
        (3 << (ADDR_LEN - 2)) | (fullC << (ADDR_LEN - 3));

    const int aBlocks = k <= MAX_BLOCK_LEN ? k : MAX_BLOCK_LEN;
    const int bBlocks = j <= MAX_BLOCK_LEN ? j : MAX_BLOCK_LEN;
    const int dBlocks = j <= MAX_BLOCK_LEN_ACC ? j : MAX_BLOCK_LEN_ACC;

    Location loc = a.getLoc();
    bool dAddrNull = llvm::dyn_cast<arith::ConstantOp>(d.getDefiningOp()) && getNumberFromValue(d) == 0;
    bool cAddrNull = llvm::dyn_cast<arith::ConstantOp>(c.getDefiningOp()) && getNumberFromValue(c) == 0;

    // Move-in D
    if (!dAddrNull && !noBias) {
      const size_t dStride = repeatingBias ? 0 : strideD * sizeof(acc_t);
      Value strideValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(dStride));
      rewriter.create<ConfigLdOp>(loc, strideValue,
                                  llvm::APFloat((float)dScaleFactor));

      for (size_t i0 = 0; i0 < i; i0++) {
        for (size_t j0 = 0; j0 < j; j0 += dBlocks) {
          const size_t biasRow = repeatingBias ? 0 : i0;
          const size_t offset = (biasRow * strideD + j0) * DIM * sizeof (acc_t);
          const uint32_t dSpAddrAcc = dSpAddrStart + (i0 * j + j0) * DIM;
          const size_t blocks = j0 + dBlocks <= j ? dBlocks : j - j0;
          const size_t cols = blocks * DIM - (j0 + blocks >= j ? padJ : 0);
          const size_t rows = DIM - (i0 == i - 1 ? padI : 0);
          gemminiMvinOffset(d, offset, dSpAddrAcc, cols, rows, rewriter);
        }
      }
    }

    // Move-in B
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideB));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)bScaleFactor));
    for (size_t j0 = 0; j0 < j; j0 += bBlocks) {
      for (size_t k0 = 0; k0 < k; k0++) {
        const size_t offset = (k0 * strideB + j0) * DIM * sizeof (elem_t);
        const uint32_t bSpAddr = bSpAddrStart + (k0 * j + j0) * DIM;
        const size_t blocks = j0 + bBlocks <= j ? bBlocks : j - j0;
        const size_t cols = blocks * DIM - (j0 + blocks >= j ? padJ : 0);
        const size_t rows = DIM - (k0 == k - 1 ? padK : 0);
        gemminiMvinOffset(b, offset, bSpAddr, cols, rows, rewriter);
      }
    }

    // Move-in A
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideA));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)aScaleFactor));

    for (size_t i0 = 0; i0 < i; i0++) {
      for (size_t k0 = 0; k0 < k; k0 += aBlocks) {
        const size_t offset = (i0 * strideA + k0) * DIM * sizeof (elem_t);
        const uint32_t aSpAddr = aSpAddrStart + (i0 * k + k0) * DIM;
        const size_t blocks = k0 + aBlocks <= k ? aBlocks : k - k0;
        const size_t cols = blocks * DIM - (k0 + blocks >= k ? padK : 0);
        const size_t rows = DIM - (i0 == i - 1 ? padI : 0);
        gemminiMvinOffset(a, offset, aSpAddr, cols, rows, rewriter);
      }
    }

    for (size_t i0 = 0; i0 < i; i0++) {
      for (size_t j0 = 0; j0 < j; j0++) {
        const uint32_t cSpAddr = cSpAddrStart + (i0 * j + j0) * DIM;
        for (size_t k0 = 0; k0 < k; k0++) {

          const uint32_t aSpAddr = aSpAddrStart + (i0 * k + k0) * DIM;
          const uint32_t bSpAddr = bSpAddrStart + (k0 * j + j0) * DIM;

          uint32_t outSpAddr = k0 == k - 1 ? cSpAddr : GARBAGE_ADDR;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it

          int noBiasNewMatrix = noBias && !dAddrNull && k0 == k - 1;
          if (noBiasNewMatrix) {
            outSpAddr &= ~(1 << (ADDR_LEN - 2));
          }

          const size_t aCols = DIM - (k0 == k - 1 ? padK : 0);
          const size_t aRows = DIM - (i0 == i - 1 ? padI : 0);
          const size_t bCols = DIM - (j0 == j - 1 ? padJ : 0);
          const size_t bRows = DIM - (k0 == k - 1 ? padK : 0);
          const size_t cCols = DIM - (j0 == j - 1 ? padJ : 0);
          const size_t cRows = DIM - (i0 == i - 1 ? padI : 0);

          Value aColsOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(aCols));
          Value aRowsOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(aRows));
          Value bColsOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(bCols));
          Value bRowsOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(bRows));
          Value cColsOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(cCols));
          Value cRowsOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(cRows));

          Value aSpAddrOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(aSpAddr));
          Value bSpAddrOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(bSpAddr));
          Value outSpAddrOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(outSpAddr));

          Value garbageAddrOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(GARBAGE_ADDR));
          Value dimOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(DIM));

          rewriter.create<PreloadOp>(loc, garbageAddrOp, outSpAddrOp, dimOp,
                                     dimOp, cRowsOp, cColsOp);

          if (k0 == 0) { // First iteration
            rewriter.create<ComputePreloadedOp>(loc, aSpAddrOp, bSpAddrOp, aRowsOp, aColsOp, bRowsOp, bColsOp);

          } else { // All other iterations
            rewriter.create<ComputeAccumulatedOp>(loc, aSpAddrOp, bSpAddrOp, aRowsOp, aColsOp, bRowsOp, bColsOp);
          }
        }
      }
    }
    // Move-out C
    if (!cAddrNull) {
      const size_t sizeof_C = fullC ? sizeof(acc_t) : sizeof(elem_t);

      for (size_t i0 = 0; i0 < i; i0++) {
        for (size_t j0 = 0; j0 < j; j0++) {
          const size_t offset = (i0 *strideC + j0)*DIM*sizeof_C;
          const uint32_t cSpAddr = cSpAddrStart + (i0 *j + j0)*DIM;

          const size_t cCols = DIM - (j0 == j - 1 ? padJ : 0);
          const size_t cRows = DIM - (i0 == j - 1 ? padI : 0);

          gemminiMvoutOffset(c, offset, cSpAddr, cCols, cRows, rewriter);
        }
      }
    }
  }

  void gemminiMvinOffset(const Value &mem, const size_t offset, const uint32_t SpAddr,
           const size_t cols, const size_t rows,
           ConversionPatternRewriter &rewriter) const{
    Location loc = mem.getLoc();
    Value offsetOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(offset));
    IntegerType i64Type = rewriter.getI64Type();
    Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
    uint64_t spadAddrInt = (uint64_t)rows << (ADDR_LEN + 16) |
                           (uint64_t)cols << ADDR_LEN | (uint64_t) SpAddr;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
    rewriter.create<Mvin_IntrOp>(loc, configPtr, spad);
  }

  void gemminiMvoutOffset(const Value &mem, const size_t offset, const uint32_t SpAddr,
                         const size_t cols, const size_t rows,
                         ConversionPatternRewriter &rewriter) const{
    Location loc = mem.getLoc();
    Value offsetOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(offset));
    IntegerType i64Type = rewriter.getI64Type();
    Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
    uint64_t spadAddrInt = (uint64_t)rows << (ADDR_LEN + 16) |
                           (uint64_t)cols << ADDR_LEN | (uint64_t) SpAddr;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
    rewriter.create<Mvout_IntrOp>(loc, configPtr, spad);
  }

  void inner(Value &a, Value &b, Value &pre, Value &out, scale_t aScaleFactor, scale_t bScaleFactor, scale_acc_t dScaleFactor, size_t i, size_t j,
             size_t k, size_t padI, size_t padJ, size_t padK, size_t strideA,
             size_t strideB, size_t strideD, size_t strideC, bool aTranspose,
             bool bTranspose, bool fullC, bool lowD, bool noBias,
             bool repeatingBias, int act, TileMatMulOp &tileMatMulOp,
             ConversionPatternRewriter &rewriter) const {

    gemminiLoopWs(i, j, k, padI, padJ, padK, a, b, pre, out, strideA, strideB,
                  repeatingBias ? 0 : strideD, strideC, aTranspose, bTranspose,
                  fullC, lowD, !noBias, act, tileMatMulOp, rewriter);
  }

  void tiledMatmulOuter(size_t dimI, size_t dimJ, size_t dimK, Value &A,
                        Value &B, Value &D, Value &C, size_t strideA,
                        size_t strideB, size_t strideD, size_t strideC,
                        scale_t aScaleFactor, scale_t bScaleFactor,
                        scale_acc_t dScaleFactor, size_t tileI, size_t tileJ,
                        size_t tileK, int act, acc_scale_t scale,
                        acc_scale_t bertScale, bool repeatingBias,
                        bool aTranspose, bool bTranspose, bool fullC, bool lowD,
                        uint8_t weightA, int dataflow, TileMatMulOp &tileMatMulOp,
                        ConversionPatternRewriter &rewriter) const {
    const size_t dimIPadded = (dimI / dim + (dimI % dim != 0)) * dim;
    const size_t dimJPadded = (dimJ / dim + (dimJ % dim != 0)) * dim;
    const size_t dimKPadded = (dimK / dim + (dimK % dim != 0)) * dim;
    const size_t I0 =
        dimIPadded / (tileI * dim) + (dimIPadded % (tileI * dim) != 0);
    const size_t J0 =
        dimJPadded / (tileJ * dim) + (dimJPadded % (tileJ * dim) != 0);
    const size_t K0 =
        dimKPadded / (tileK * dim) + (dimKPadded % (tileK * dim) != 0);
    const size_t lastI =
        dimIPadded % (tileI * dim) == 0 ? tileI : (dimIPadded / dim) % tileI;
    const size_t lastJ =
        dimJPadded % (tileJ * dim) == 0 ? tileJ : (dimJPadded / dim) % tileJ;
    const size_t lastK =
        dimKPadded % (tileK * dim) == 0 ? tileK : (dimKPadded / dim) % tileK;
    const size_t paddingI = dimIPadded - dimI;
    const size_t paddingJ = dimJPadded - dimJ;
    const size_t paddingK = dimKPadded - dimK;
    const bool noBias = false;
    const size_t sizeofD = lowD ? sizeOfElemT : sizeOfAccT;
    const size_t sizeofC = fullC ? sizeOfAccT : sizeOfElemT;
    Location loc = tileMatMulOp.getLoc();
    llvm::APFloat accScaleIdentity((float)ACC_SCALE_IDENTITY);
    rewriter.create<ConfigExOp>(loc, /*dataflow = */ dataflow, /*sysAct = */ act & 3,
                                /* sysShift = */ 0, accScaleIdentity);
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideC * sizeofC));
    rewriter.create<ConfigStOp>(loc, strideValue, act & 3,
                                llvm::APFloat(scale));
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideA * sizeOfElemT));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(aScaleFactor),
                                false, 0);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideB * sizeOfElemT));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(bScaleFactor),
                                false, 1);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideD * sizeofD));
    rewriter.create<ConfigLdOp>(loc, strideValue,
                                llvm::APFloat((float)dScaleFactor), lowD, 2);
    
    /*
      Add config norm op
    */
    if (act == IGELU) {
      const acc_scale_t sqrt_2 = 1.41421356237;
      const acc_scale_t S = bertScale;
      const acc_scale_t S_erf = (-0.2888 * ((S*S) / 2));

      const acc_t qb = -1.769 / (S / sqrt_2);
      const acc_t qc = 1.0 / S_erf;
      rewriter.create<ConfigNormOp>(loc, 0, 0, 0, 0,0, qb, qc);
    }

    if (act == SOFTMAX) {
      const scale_t a = 0.3585;
      const scale_t b = 1.353;
      const scale_t c = 0.344;

      const acc_t qln2 = (int) (0.693147 / bertScale);
      const acc_t qln2_inv = 65536 / qln2;
      const acc_t qb = b / bertScale;
      const acc_t qc = c / (a*bertScale*bertScale);
      rewriter.create<ConfigNormOp>(loc, qln2, 0, 0, 1, 0, qb, qc);
      rewriter.create<ConfigNormOp>(loc, qln2_inv, 1, 0, 1, 0, qb, qc);
    }

    for (size_t i0 = 0; i0 < I0; i0++)
      for (size_t j0 = 0; j0 < J0; j0++)
        for (size_t k0 = 0; k0 < K0; k0++) {
          Value pre;
          Location loc = A.getLoc();
          if (k0 != 0) {
            IntegerAttr preAttr = rewriter.getI64IntegerAttr(0);
            pre = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                     preAttr);
          } else {
            size_t biasRow = repeatingBias ? 0 : i0 * tileI * dim;
            size_t offset =
                (biasRow * strideD + j0 * tileJ * dim) * sizeofD * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            pre = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), D,
                                                 offsetValue);
          }

          Value out;
          if (k0 == K0 - 1) {
            size_t offset = (i0 * tileI * dim * strideC + j0 * tileJ * dim) *
                            sizeofC * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            out = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), C,
                                                 offsetValue);
          } else {
            IntegerAttr outAttr = rewriter.getI64IntegerAttr(0);
            out = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(),
                                                     outAttr);
          }
          const size_t i = i0 < I0 - 1 ? tileI : lastI;
          const size_t j = j0 < J0 - 1 ? tileJ : lastJ;
          const size_t k = k0 < K0 - 1 ? tileK : lastK;
          const size_t padI = i0 == I0 - 1 ? paddingI : 0;
          const size_t padJ = j0 == J0 - 1 ? paddingJ : 0;
          const size_t padK = k0 == K0 - 1 ? paddingK : 0;
          Value a;
          if (aTranspose) {
            size_t offset =
                (k0 * tileK * dim * strideA + i0 * tileI * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          } else {
            size_t offset =
                (i0 * tileI * dim * strideA + k0 * tileK * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          }
          Value b;
          if (bTranspose) {
            size_t offset =
                (j0 * tileJ * dim * strideB + k0 * tileK * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          } else {
            size_t offset =
                (k0 * tileK * dim * strideB + j0 * tileJ * dim) * sizeOfElemT;
            IntegerAttr offsetAttr = rewriter.getI64IntegerAttr(offset);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64Type(), offsetAttr);
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          }
          if (dataflow == OUTPUT_STATIONARY) {
            spTiledMatmulOs(a, b, pre, out, aScaleFactor, bScaleFactor, dScaleFactor, i, j,
                            k, padI, padJ, padK, strideA, strideB, strideD, strideC,
                            aTranspose, bTranspose, fullC, lowD, noBias, repeatingBias, act,
                            tileMatMulOp, rewriter);
          } else { // WS
            spTiledMatmulWs(a, b, pre, out, aScaleFactor, bScaleFactor, dScaleFactor, i, j,
                            k, padI, padJ, padK, strideA, strideB, strideD, strideC,
                            aTranspose, bTranspose, fullC, lowD, noBias, repeatingBias, act,
                            tileMatMulOp, rewriter);
          }
        }
    IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
    Value flushValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), flushAttr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileMatMulOp, flushValue,
                                              flushValue);
    return;
  }

  size_t tiledMatmulTotalSpadRows(size_t I, size_t J, size_t K) const {
    return (I * K + K * J) * dim;
  }

  size_t tiledMatmulTotalAccRows(size_t I, size_t J) const {
    return (I * J) * dim;
  }

public:
  using ConvertOpToLLVMPattern<TileMatMulOp>::ConvertOpToLLVMPattern;
  explicit GemminiTileMatMulLowering(LLVMTypeConverter &typeConverter,
                                     int64_t dim, size_t sizeOfElemT,
                                     size_t sizeOfAccT)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim),
        sizeOfElemT(sizeOfElemT), sizeOfAccT(sizeOfAccT) {}

  LogicalResult
  matchAndRewrite(TileMatMulOp tileMatMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#define partitionRows (BANK_NUM * BANK_ROWS / 2)
#define matsInPartition (partition_rows / dim)
#define matsInAcc (ACC_ROWS / dim)
#define maxTileIJ ((size_t)sqrt(mats_in_acc))
#define maxTileK (matsInPartition / maxTileIJ)

#define dbPartitionRows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define dbMatsInPartition (dbPartitionRows / dim)
#define dbMatsInAcc ((ACC_ROWS / 2) / dim)
#define dbMaxTileIJ ((size_t)sqrt(dbMatsInAcc))
#define dbMaxTileK (dbMatsInPartition / dbMaxTileIJ)

    Value aArray = tileMatMulOp.getAArray();
    Value bArray = tileMatMulOp.getBArray();
    Value cArray = tileMatMulOp.getCArray();
    Value dArray = tileMatMulOp.getDArray();
    MemRefType aArrayType = aArray.getType().dyn_cast<MemRefType>();
    MemRefType bArrayType = bArray.getType().dyn_cast<MemRefType>();
    MemRefType cArrayType = cArray.getType().dyn_cast<MemRefType>();
    MemRefType dArrayType = dArray.getType().dyn_cast<MemRefType>();
    StridedLayoutAttr aArrayLayout =
        aArrayType.getLayout().dyn_cast<StridedLayoutAttr>();
    StridedLayoutAttr bArrayLayout =
        bArrayType.getLayout().dyn_cast<StridedLayoutAttr>();
    StridedLayoutAttr cArrayLayout =
        cArrayType.getLayout().dyn_cast<StridedLayoutAttr>();
    SmallVector<Type> resultType = {rewriter.getIndexType()};
    TypeRange typeRange(resultType);
    Location loc = tileMatMulOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value aArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                aArray);
    if (aArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, aArrayLayout.getOffset() * sizeOfElemT);
      aArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, aArrayExtractOp, offset);
    }
    Value aArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, aArrayExtractOp);
    Value bArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                bArray);
    if (bArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, bArrayLayout.getOffset() * sizeOfElemT);
      bArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, bArrayExtractOp, offset);
    }
    Value bArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, bArrayExtractOp);
    Value cArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                cArray);
    if (cArrayLayout) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, cArrayLayout.getOffset() * sizeOfElemT);
      cArrayExtractOp =
          rewriter.create<arith::AddIOp>(loc, cArrayExtractOp, offset);
    }
    Value cArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, cArrayExtractOp);
    Value dArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, typeRange,
                                                                dArray);
    Value dArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, dArrayExtractOp); 
    llvm::ArrayRef<int64_t> aArrayShape = aArrayType.getShape();
    llvm::ArrayRef<int64_t> bArrayShape = bArrayType.getShape();
    llvm::ArrayRef<int64_t> cArrayShape = cArrayType.getShape();
    llvm::ArrayRef<int64_t> dArrayShape = dArrayType.getShape();
    size_t dimI = aArrayShape[0];
    size_t dimK = aArrayShape[1];
    size_t dimJ = bArrayShape[1];
    size_t strideA = aArrayShape[1];
    size_t strideB = bArrayShape[1];
    size_t strideC = cArrayShape[1];
    size_t strideD = dArrayShape[1];
    scale_t aScaleFactor = tileMatMulOp.getAScaleFactor().convertToFloat();
    scale_t bScaleFactor = tileMatMulOp.getBScaleFactor().convertToFloat();
    scale_acc_t dScaleFactor = tileMatMulOp.getDScaleFactor().convertToFloat();
    int act = tileMatMulOp.getAct();
    acc_scale_t scale = tileMatMulOp.getAccScale().convertToFloat();
    acc_scale_t bertScale = tileMatMulOp.getBertScale().convertToFloat();
    bool repeatingBias = tileMatMulOp.getRepeatingBias();
    bool aTranspose = tileMatMulOp.getATranspose();
    bool bTranspose = tileMatMulOp.getBTranspose();
    bool fullC = tileMatMulOp.getFullC();
    bool lowD = tileMatMulOp.getLowD();
    uint8_t weightA = tileMatMulOp.getWeightA();
    size_t dimIPaded = (dimI / dim + (dimI % dim != 0)) * dim;
    size_t dimJPaded = (dimJ / dim + (dimJ % dim != 0)) * dim;
    size_t dimKPaded = (dimK / dim + (dimK % dim != 0)) * dim;
    size_t maxSpadRows = BANK_NUM * BANK_ROWS / 2;
    size_t maxAccRows = ACC_ROWS / 2;
    size_t tileI, tileJ, tileK;
    if (act == LAYERNORM || act == SOFTMAX) {
      tileI = 1;
      tileJ = dimJPaded | dim;
      tileK = 1;
    } else {
      tileI = dimIPaded / dim < dbMaxTileIJ ? dimIPaded / dim : dbMaxTileIJ;
      tileJ = dimJPaded / dim < dbMaxTileIJ ? dimJPaded / dim : dbMaxTileIJ;
      tileK = dimKPaded / dim < dbMaxTileK ? dimKPaded / dim : dbMaxTileK;
    }
    while (true) {
      bool increased = false;

      if (tiledMatmulTotalSpadRows(tileI, tileJ + 1, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI, tileJ + 1) <= maxAccRows &&
          (tileJ + 1) * dim <= dimJPaded) {
        tileJ++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI + 1, tileJ, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI + 1, tileJ) <= maxAccRows &&
          (tileI + 1) * dim <= dimIPaded) {
        tileI++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI, tileJ, tileK + 1) <= maxSpadRows &&
          (tileK + 1) * dim <= dimKPaded) {
        tileK++;
        increased = true;
      }
      if (!increased)
        break;
    }

#undef partitionRows
#undef matsInPartition
#undef matsInAcc
#undef maxTileIJ
#undef maxTileK

#undef dbPartitionRows
#undef dbMatsInPartition
#undef dbMatsInAcc
#undef dbMaxTileIJ
#undef dbMaxTileK
    int dataflow = tileMatMulOp.getDataflow();


    tiledMatmulOuter(dimI, dimJ, dimK, aArrayindexCastOp, bArrayindexCastOp,
                     dArrayindexCastOp, cArrayindexCastOp, strideA, strideB,
                     strideD, strideC, aScaleFactor, bScaleFactor, dScaleFactor,
                     tileI, tileJ, tileK, act, scale, bertScale, repeatingBias,
                     aTranspose, bTranspose, fullC, lowD, weightA, dataflow, tileMatMulOp,
                     rewriter);
    return success();
  };

private:
  int64_t dim;
  size_t sizeOfElemT;
  size_t sizeOfAccT;
};

class GemminiTileConvLowering : public ConvertOpToLLVMPattern<TileConvOp> {

  void gemminiLoopConvWs(
      int batchSize, int inDim, int inChannels, int outChannels, int outDim,
      int poolOutDim, int stride, int padding, int kernelDim,
      int kernelDilation, int poolSize, int poolStride, int poolPadding,
      int batches, int porows, int pocols, int pochs, int krows, int kcols,
      int kchs, int lpad, int rpad, int upad, int dpad, int plpad, int prpad,
      int pupad, int pdpad, int orows, int ocols, Value &weights, Value &output,
      Value &bias, Value &input, bool noBias, bool noPool, bool downsample,
      bool writ180, bool inputDilated, int act, bool transOutput1203,
      bool transWeight1203, bool transWeight0132, bool transInput3120,
      int maxPixelsPerRow, bool dw, TileConvOp &tileConvOp,
      ConversionPatternRewriter &rewriter) const {
    Location loc = tileConvOp.getLoc();
    // loopConvWsConfig1
    uint64_t rs1 = (uint64_t)outChannels << 48 | (uint64_t)inChannels << 32 |
                   (uint64_t)inDim << 16 | (uint64_t)batchSize;
    uint64_t rs2 = (uint64_t)padding << 48 | (uint64_t)stride << 32 |
                   (uint64_t)poolOutDim << 16 | (uint64_t)outDim;
    TypedAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    TypedAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig1_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig2
    rs1 = (uint64_t)kernelDim << 48 | (uint64_t)poolSize << 32 |
          (uint64_t)poolStride << 16 | (uint64_t)poolPadding;
    rs2 = (uint64_t)batches << 48 | (uint64_t)porows << 32 |
          (uint64_t)pocols << 16 | (uint64_t)pochs;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig2_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig3
    rs1 = (uint64_t)krows << 48 | (uint64_t)kcols << 32 | (uint64_t)kchs << 16 |
          (uint64_t)lpad;
    rs2 = (uint64_t)rpad << 48 | (uint64_t)upad << 32 | (uint64_t)dpad << 16 |
          (uint64_t)plpad;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig3_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig4
    rs1 = (uint64_t)orows << 48 | (uint64_t)prpad << 32 |
          (uint64_t)pupad << 16 | (uint64_t)pdpad;
    rs2 = (uint64_t)kernelDilation << 16 | (uint64_t)ocols;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWsConfig4_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsconfig5
    rewriter.create<LoopConvWsConfig5_IntrOp>(loc, weights, output);
    // loopConvWsconfig6
    rewriter.create<LoopConvWsConfig6_IntrOp>(loc, bias, input);
    // loopConvWs
    rs1 = (uint64_t)maxPixelsPerRow << 8 | dw << 6 | transInput3120 << 5 |
          transWeight0132 << 4 | transWeight1203 << 3 | transOutput1203 << 2 |
          writ180 << 1 | noBias;
    rs2 = act << 3 | inputDilated << 2 | downsample << 1 | noPool;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
    rewriter.create<LoopConvWs_IntrOp>(loc, rs1Value, rs2Value);
  }

  void spTiledConv(int batchSize, int inDim, int inChannels, int outChannels,
                   int outDim, int poolOutDim, int stride, int padding,
                   int kernelDim, int kernelDilation, int poolSize,
                   int poolStride, int poolPadding, int batches, int porows,
                   int pocols, int pochs, int krows, int kcols, int kchs,
                   int lpad, int rpad, int upad, int dpad, int plpad, int prpad,
                   int pupad, int pdpad, Value &input, Value &weights,
                   Value &output, Value &bias, int act, acc_scale_t scale,
                   bool wrot180, bool transOutput1203, bool transInput3120,
                   bool transWeight1203, bool transWeight0132, bool noBias,
                   bool noPool, bool downsample, bool inputDilated, bool dw,
                   TileConvOp &tileConvOp,
                   ConversionPatternRewriter &rewriter) const {
    if (dw) {
      kchs = 1;
      pochs = 1;
    }

    const int orows = porows * poolStride + poolSize - 1 - pupad - pdpad;
    const int ocols = pocols * poolStride + poolSize - 1 - plpad - prpad;
    const int ichs = kchs;

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
    const bool transposed =
        transOutput1203 || transInput3120 || transWeight1203 || transWeight0132;
    int maxPixelsPerRow = transposed || wrot180 || downsample || inputDilated ||
                                  kernelDilation > 1 || ichs > dim
                              ? 1
                              : dim / ichs;
    if (maxPixelsPerRow > kcols)
      maxPixelsPerRow = kcols;
#else
    const int maxPixelsPerRow = 1;
#endif
    gemminiLoopConvWs(
        batchSize, inDim, inChannels, outChannels, outDim, poolOutDim, stride,
        padding, kernelDim, kernelDilation, poolSize, poolStride, poolPadding,
        batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad,
        dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias,
        input, noBias, noPool, downsample, wrot180, inputDilated, act,
        transOutput1203, transWeight1203, transWeight0132, transInput3120,
        maxPixelsPerRow, dw, tileConvOp, rewriter);
  }

  void tiledConv(int batchSize, int inDim, int inChannels, int outChannels,
                 int outDim, int stride, int inputDilation, int kernelDilation,
                 int padding, int kernelDim, bool wrot180, bool transOutput1203,
                 bool transInput3120, bool transWeight1203,
                 bool transWeight0132, int batches, int porows, int pocols,
                 int pochs, int krows, int kcols, int kchs, const Value &input,
                 const Value &weights, const Value &bias, Value &output,
                 int act, acc_scale_t scale, int poolSize, int poolStride,
                 int poolPadding, TileConvOp &tileConvOp,
                 ConversionPatternRewriter &rewriter) const {
    bool noBias = false;
    bool noPool = poolStride == 0;
    if (noPool) {
      poolSize = 1;
      poolStride = 1;
      poolPadding = 0;
    }
    const bool downsample = stride == 2 && kernelDim == 1 && inDim % 2 == 0 &&
                            padding == 0 && noPool && inputDilation == 1 &&
                            !transInput3120;
    const int inputDilated = inputDilation == 2;
    int64_t stDramStride = transOutput1203
                               ? batchSize * outChannels * sizeOfElemT
                               : outChannels * sizeOfElemT;
    Location loc = tileConvOp.getLoc();
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(stDramStride));
    rewriter.create<ConfigStOp>(loc, strideValue, act, llvm::APFloat(scale));
    rewriter.create<ConfigExOp>(
        loc, /*dataflow = */ 1, /*act = */ 0, /*shift = */ 0,
        /*scale = */ llvm::APFloat((float)0), /*cStride = */ inputDilation,
        /*aStride = */ stride >> downsample,
        /*aTranspose = */ transInput3120, /*bTranspose*/ transWeight0132,
        /*setOnlyStrides = */ false);
    const int poolOutDim =
        (outDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const int dilatedInDim = inDim + (inputDilation - 1) * (inDim - 1);
    for (int b = 0; b < batchSize; b += batches) {
      for (int porow = 0; porow < poolOutDim; porow += porows) {
        const int orow = porow * poolStride - poolPadding;
        for (int pocol = 0; pocol < poolOutDim; pocol += pocols) {
          const int ocol = pocol * poolStride - poolPadding;
          for (int poch = 0; poch < outChannels; poch += pochs) {
            for (int krow = 0; krow < kernelDim; krow += krows) {
              const int orow_floored = orow < 0 ? 0 : orow;

              int irow =
                  orow_floored * stride + krow * kernelDilation - padding;
              for (int kcol = 0; kcol < kernelDim; kcol += kcols) {
                const int ocol_floored = ocol < 0 ? 0 : ocol;
                int icol =
                    ocol_floored * stride + kcol * kernelDilation - padding;

                for (int kch = 0; kch < inChannels; kch += kchs) {
                  TypedAttr offsetAttr =
                      rewriter.getI64IntegerAttr(((b * poolOutDim * poolOutDim +
                                                   porow * poolOutDim + pocol) *
                                                      outChannels +
                                                  poch) *
                                                 sizeOfElemT);
                  Value offsetValue =
                      rewriter.create<arith::ConstantOp>(loc, offsetAttr);
                  Value out = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), output,
                      offsetValue);
                  if (transOutput1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((porow * poolOutDim * batchSize + pocol * batchSize +
                          b) *
                             outChannels +
                         poch) *
                        sizeOfElemT);
                    offsetValue =
                        rewriter.create<arith::ConstantOp>(loc, offsetAttr);
                    out = rewriter.create<arith::AddIOp>(tileConvOp.getLoc(),
                                                         rewriter.getI64Type(),
                                                         output, offsetValue);
                  }

                  if (krow + krows < kernelDim || kcol + kcols < kernelDim ||
                      kch + kchs < inChannels) {
                    out = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), rewriter.getI64IntegerAttr(0));
                  }
                  Value pochValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(),
                      rewriter.getI64IntegerAttr(poch * sizeOfAccT));
                  Value bias_ = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), bias,
                      pochValue);
                  if (krow > 0 || kcol > 0 || kch > 0) {
                    bias_ = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), rewriter.getI64IntegerAttr(0));
                  }

                  const int batches_ =
                      batchSize - b > batches ? batches : batchSize - b;
                  const int porows_ =
                      poolOutDim - porow > porows ? porows : poolOutDim - porow;
                  const int pocols_ =
                      poolOutDim - pocol > pocols ? pocols : poolOutDim - pocol;
                  const int pochs_ =
                      outChannels - poch > pochs ? pochs : outChannels - poch;
                  const int krows_ =
                      kernelDim - krow > krows ? krows : kernelDim - krow;
                  const int kcols_ =
                      kernelDim - kcol > kcols ? kcols : kernelDim - kcol;
                  const int kchs_ =
                      inChannels - kch > kchs ? kchs : inChannels - kch;

                  const int ocols_ = pocols_ * poolStride + poolSize - 1;
                  const int orows_ = porows_ * poolStride + poolSize - 1;

                  const int plpad = ocol < 0 ? -ocol : 0;
                  const int prpad =
                      ocol + ocols_ > outDim ? ocol + ocols_ - outDim : 0;
                  const int pupad = orow < 0 ? -orow : 0;
                  const int pdpad =
                      orow + orows_ > outDim ? orow + orows_ - outDim : 0;

                  const int dilatedKrows_ =
                      krows_ + (kernelDilation - 1) * (krows_ - 1);
                  const int dilatedKcols_ =
                      kcols_ + (kernelDilation - 1) * (kcols_ - 1);

                  const int icols_ =
                      (ocols_ - plpad - prpad) * stride + dilatedKcols_ - 1;
                  const int irows_ =
                      (orows_ - pupad - pdpad) * stride + dilatedKrows_ - 1;

                  int lpad = icol < 0 ? -icol : 0;
                  int rpad = icol + icols_ > dilatedInDim
                                 ? icol + icols_ - dilatedInDim
                                 : 0;
                  int upad = irow < 0 ? -irow : 0;
                  int dpad = irow + irows_ > dilatedInDim
                                 ? irow + irows_ - dilatedInDim
                                 : 0;

                  if (inputDilated) {
                    lpad += lpad == 0 && icol % 2 != 0;
                    rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                    upad += upad == 0 && irow % 2 != 0;
                    dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                  }

                  int krow_ = krow;
                  int kcol_ = kcol;
                  if (wrot180) {
                    krow_ = kernelDim - krow - krows_;
                    kcol_ = kernelDim - kcol - kcols_;
                  }
                  offsetAttr = rewriter.getI64IntegerAttr(
                      ((krow_ * kernelDim * inChannels + kcol_ * inChannels +
                        kch) *
                           outChannels +
                       poch) *
                      sizeOfElemT);
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(), offsetAttr);
                  Value weightsSlice = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                      offsetValue);
                  if (transWeight1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((kch * kernelDim * kernelDim + krow_ * kernelDim +
                          kcol_) *
                             outChannels +
                         poch) *
                        sizeOfElemT);
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), offsetAttr);
                    weightsSlice = rewriter.create<arith::AddIOp>(
                        tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                        offsetValue);
                  } else if (transWeight0132) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((krow_ * kernelDim * outChannels +
                          kcol_ * outChannels + poch) *
                             inChannels +
                         kch) *
                        sizeOfElemT);
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(), offsetAttr);
                    weightsSlice = rewriter.create<arith::AddIOp>(
                        tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                        offsetValue);
                  }
                  offsetAttr = rewriter.getI64IntegerAttr(
                      ((b * inDim * inDim +
                        ((irow + upad) >> inputDilated) * inDim +
                        ((icol + lpad) >> inputDilated)) *
                           inChannels +
                       kch) *
                      sizeOfElemT);
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(), offsetAttr);
                  Value in = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), input,
                      offsetValue);
                  if (transInput3120) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((kch * inDim * inDim +
                          ((irow + upad) >> inputDilated) * inDim +
                          ((icol + lpad) >> inputDilated)) *
                             batchSize +
                         b) *
                        sizeOfElemT);
                    in = rewriter.create<arith::AddIOp>(tileConvOp.getLoc(),
                                                        rewriter.getI64Type(),
                                                        input, offsetValue);
                  }

                  spTiledConv(batchSize, inDim, inChannels, outChannels, outDim,
                              poolOutDim, stride, padding, kernelDim,
                              kernelDilation, poolSize, poolStride, poolPadding,
                              batches_, porows_, pocols_, pochs_, krows_,
                              kcols_, kchs_, lpad, rpad, upad, dpad, plpad,
                              prpad, pupad, pdpad, in, weightsSlice, out, bias_,
                              act, scale, wrot180, transOutput1203,
                              transInput3120, transWeight1203, transWeight0132,
                              noBias, noPool, downsample, inputDilated, false,
                              tileConvOp, rewriter);
                }
              }
            }
          }
        }
      }
    }
    IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
    Value flushValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), flushAttr);
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileConvOp, flushValue,
                                              flushValue);
  }

  int tiledConvTotalSpadRows(bool acc, int stride, int inputDilation,
                             int kernelDilation, bool downsample,
                             bool transWeight0132, bool transInput3120,
                             int batches, int porows, int pocols, int ochs,
                             int krows, int kcols, int kchs, int poolSize,
                             int poolStride) const {

    const int orows = porows * poolStride + poolSize - 1;
    const int ocols = pocols * poolStride + poolSize - 1;

    const int krowsDilated = krows + (kernelDilation - 1) * (krows - 1);
    const int kcolsDilated = kcols + (kernelDilation - 1) * (kcols - 1);

    int irows = orows * stride + krowsDilated - 1;
    int icols = ocols * stride + kcolsDilated - 1;
    const int ichs = kchs;

    irows = irows / inputDilation + (irows % inputDilation != 0);
    icols = icols / inputDilation + (icols % inputDilation != 0);

    const int inChannelsPerBank = ichs / dim + (ichs % dim != 0);
    const int outChannelsPerBank = ochs / dim + (ochs % dim != 0);
    const int batchesPerBank = batches / dim + (batches % dim != 0);

    const int aRows = transInput3120
                          ? (batchesPerBank * ichs * (irows >> downsample) *
                             (icols >> downsample))
                          : (inChannelsPerBank * batches *
                             (irows >> downsample) * (icols >> downsample));

    const int bRows = transWeight0132
                          ? inChannelsPerBank * kcols * krows * ochs
                          : outChannelsPerBank * kcols * krows * kchs;

    const int cRows = outChannelsPerBank * batches * orows * ocols;

    return acc ? cRows : aRows + bRows;
  }

public:
  using ConvertOpToLLVMPattern<TileConvOp>::ConvertOpToLLVMPattern;
  explicit GemminiTileConvLowering(LLVMTypeConverter &typeConverter,
                                   int64_t dim, size_t sizeOfElemT,
                                   size_t sizeOfAccT)
      : ConvertOpToLLVMPattern(typeConverter), dim(dim),
        sizeOfElemT(sizeOfElemT), sizeOfAccT(sizeOfAccT) {}
  LogicalResult
  matchAndRewrite(TileConvOp tileConvOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = tileConvOp.getInput();
    Value output = tileConvOp.getOutput();
    Value weights = tileConvOp.getWeights();
    Value bias = tileConvOp.getBias();
    MemRefType inputType = input.getType().dyn_cast<MemRefType>();
    MemRefType outputType = output.getType().dyn_cast<MemRefType>();
    MemRefType weightsType = weights.getType().dyn_cast<MemRefType>();
    MemRefType biasType = bias.getType().dyn_cast<MemRefType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    ArrayRef<int64_t> weightsShape = weightsType.getShape();
    ArrayRef<int64_t> biasShape = biasType.getShape();
    // inDim
    if (inputShape[1] != inputShape[2]) {
      llvm::outs() << "inDim error.\n";
      return failure();
    }
    // outChannels
    if (biasShape[0] != outputShape[1] || biasShape[0] != weightsShape[1]) {
      llvm::outs() << "outChannels error.\n";
      return failure();
    }
    Value outDimValue = tileConvOp.getOutDim();
    int outDim = getNumberFromValue(outDimValue);
    Value kernelDimValue = tileConvOp.getKernelDim();
    int kernelDim = getNumberFromValue(kernelDimValue);
    int batchSize = inputShape[0];
    int inDim = inputShape[1];
    int inChannels = inputShape[3];
    int outChannels = biasShape[0];
    int stride = tileConvOp.getStride();
    int inputDilation = tileConvOp.getInputDilation();
    int kernelDilation = tileConvOp.getKernelDilation();
    int padding = tileConvOp.getPadding();
    int act = tileConvOp.getAct();
    float scale = tileConvOp.getScale().convertToFloat();
    int poolSize = tileConvOp.getPoolSize();
    int poolStride = tileConvOp.getPoolStride();
    int poolPadding = tileConvOp.getPoolPadding();
    bool wrot180 = tileConvOp.getWrot180();
    bool transOutput1203 = tileConvOp.getTransOutput1203();
    bool transInput3120 = tileConvOp.getTransInput3120();
    bool transWeight1203 = tileConvOp.getTransWeight1203();
    bool transWeight0132 = tileConvOp.getTransWeight0132();
    Location loc = tileConvOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value inputExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input);
    Value inputIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, inputExtractOp);
    Value outputExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, output);
    Value outputIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, outputExtractOp);
    Value biasExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, bias);
    Value biasIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, biasExtractOp);
    Value weightsExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, weights);
    Value weightsIndexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, weightsExtractOp);
    const bool noPool = poolSize == 0;
    if (noPool) {
      poolSize = 1;
      poolStride = 1;
      poolPadding = 0;
    }
    const int poolOutDim =
        (outDim + 2 * poolPadding - poolSize) / poolStride + 1;
    const bool downsample = stride == 2 && kernelDim == 1 && padding == 0 &&
                            noPool && inDim % 2 == 0;
    int args[] = {batchSize, poolOutDim, poolOutDim, outChannels,
                  kernelDim, kernelDim,  inChannels};
    const int maxArgs[] = {batchSize, poolOutDim, poolOutDim, outChannels,
                           kernelDim, kernelDim,  inChannels};
    const int orowsIdx = 1;
    const int ocolsIdx = 2;
    const int outChannelsIdx = 3;
    const int inChannelsIdx = 6;
    const int maxSpadRows = (BANK_NUM * BANK_ROWS / 2);
    const int maxAccRows = (ACC_ROWS / 2);
    int spadRows = tiledConvTotalSpadRows(
        false, stride, inputDilation, kernelDilation, downsample,
        transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], poolSize, poolStride);
    int accRows = tiledConvTotalSpadRows(
        true, stride, inputDilation, kernelDilation, downsample,
        transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
        args[4], args[5], args[6], poolSize, poolStride);
    while (spadRows > maxSpadRows || accRows > maxAccRows) {
      int maxVal = -1;
      int maxIdx = -1;
      for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
        if (!(i == ocolsIdx && args[i] <= dim && args[orowsIdx] > 1) &&
            args[i] > maxVal) {
          maxVal = args[i];
          maxIdx = i;
        }
      }

      if (maxIdx == outChannelsIdx || maxIdx == inChannelsIdx) {
        if (args[maxIdx] % dim != 0) {
          args[maxIdx] = (args[maxIdx] / dim) * dim;
        } else {
          args[maxIdx] -= dim;
        }
        args[maxIdx] = args[maxIdx] == 0 ? 1 : args[maxIdx];
      } else {
        args[maxIdx]--;
      }
      spadRows = tiledConvTotalSpadRows(
          false, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
          args[4], args[5], args[6], poolSize, poolStride);
      accRows = tiledConvTotalSpadRows(
          true, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, args[0], args[1], args[2], args[3],
          args[4], args[5], args[6], poolSize, poolStride);
    }
    bool notIncreased = false;
    while (!notIncreased) {
      notIncreased = true;

      int argsCandidate[] = {args[0], args[1], args[2], args[3],
                             args[4], args[5], args[6]};
      argsCandidate[ocolsIdx]++;

      if (argsCandidate[ocolsIdx] > maxArgs[ocolsIdx])
        continue;

      spadRows = tiledConvTotalSpadRows(
          false, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
          argsCandidate[2], argsCandidate[3], argsCandidate[4],
          argsCandidate[5], argsCandidate[6], poolSize, poolStride);
      accRows = tiledConvTotalSpadRows(
          true, stride, inputDilation, kernelDilation, downsample,
          transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
          argsCandidate[2], argsCandidate[3], argsCandidate[4],
          argsCandidate[5], argsCandidate[6], poolSize, poolStride);

      if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
        args[ocolsIdx] = argsCandidate[ocolsIdx];
        notIncreased = false;
      }
    }

    bool nothingIncreased = false;
    while (!nothingIncreased) {
      nothingIncreased = true;
      for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
        int argsCandidate[] = {args[0], args[1], args[2], args[3],
                               args[4], args[5], args[6]};
        argsCandidate[i]++;

        if (argsCandidate[i] > maxArgs[i])
          continue;
        spadRows = tiledConvTotalSpadRows(
            false, stride, inputDilation, kernelDilation, downsample,
            transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
            argsCandidate[2], argsCandidate[3], argsCandidate[4],
            argsCandidate[5], argsCandidate[6], poolSize, poolStride);
        accRows = tiledConvTotalSpadRows(
            true, stride, inputDilation, kernelDilation, downsample,
            transWeight0132, transInput3120, argsCandidate[0], argsCandidate[1],
            argsCandidate[2], argsCandidate[3], argsCandidate[4],
            argsCandidate[5], argsCandidate[6], poolSize, poolStride);

        if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
          args[i] = argsCandidate[i];
          nothingIncreased = false;
        }
      }
    }
    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];
    tiledConv(batchSize, inDim, inChannels, outChannels, outDim, stride,
              inputDilation, kernelDilation, padding, kernelDim, wrot180,
              transOutput1203, transInput3120, transWeight1203, transWeight0132,
              batches, orows, ocols, ochs, krows, kcols, kchs, inputIndexCastOp,
              weightsIndexCastOp, biasIndexCastOp, outputIndexCastOp, act,
              scale, poolSize, noPool ? 0 : poolStride, poolPadding, tileConvOp,
              rewriter);
    return success();
  }

private:
  int64_t dim;
  size_t sizeOfElemT;
  size_t sizeOfAccT;
};

void mlir::populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t addrLen, size_t sizeOfElemT, size_t sizeOfAccT) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<GemminiFlushLowering>(converter);
  patterns.add<GemminiConfigStLowering>(converter);
  patterns.add<GemminiConfigLdLowering>(converter);
  patterns.add<GemminiMvinLowering>(converter, addrLen);
  patterns.add<GemminiMvoutLowering>(converter, addrLen);
  patterns.add<GemminiConfigExLowering>(converter);
  patterns.add<GemminiConfigNormOpLowering>(converter);
  patterns.add<GemminiPreloadZerosLowering>(converter, dim, addrLen);
  patterns.add<GemminiPreloadLowering>(converter, addrLen);
  patterns.add<GemminiComputePreloadedLowering>(converter, addrLen);
  patterns.add<GemminiComputeAccumulatedLowering>(converter, addrLen);
  patterns.add<GemminiTileMatMulLowering>(converter, dim, sizeOfElemT,
                                          sizeOfAccT);
  patterns.add<GemminiTileConvLowering>(converter, dim, sizeOfElemT,
                                        sizeOfAccT);
}

void mlir::configureGemminiegalizeForExportTarget(
    LLVMConversionTarget &target) {
  target.addLegalOp<
      Flush_IntrOp, ConfigSt_IntrOp, ConifgLd_IntrOp, ConfigEX_IntrOp,
      Mvin_IntrOp, Mvout_IntrOp, Preload_IntrOp, ComputePreloaded_IntrOp,
      ComputeAccumulated_IntrOp, LoopWsConfigBounds_IntrOp,
      LoopWsConfigAddrsAB_IntrOp, LoopWsConfigAddrsDC_IntrOp,
      LoopWsConfigStridesAB_IntrOp, LoopWsConfigStridesDC_IntrOp, LoopWs_IntrOp,
      LoopConvWsConfig1_IntrOp, LoopConvWsConfig2_IntrOp,
      LoopConvWsConfig3_IntrOp, LoopConvWsConfig4_IntrOp,
      LoopConvWsConfig5_IntrOp, LoopConvWsConfig6_IntrOp, LoopConvWs_IntrOp, ConfigNorm_IntrOp>();
  target.addIllegalOp<FlushOp, ConfigStOp, ConfigLdOp, ConfigExOp, MvinOp,
                      MvoutOp, PrintOp, PreloadZerosOp, PreloadOp,
                      ComputePreloadedOp, ComputeAccumulatedOp, TileMatMulOp,
                      TileConvOp, ConfigNormOp>();
}
