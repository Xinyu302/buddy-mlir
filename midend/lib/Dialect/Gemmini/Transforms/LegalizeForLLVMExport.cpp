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
#include <stdio.h>

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

void gemminiMvinOffset(const Value &mem, const size_t offset, const uint32_t SpAddr,
                       const size_t cols, const size_t rows,
                       ConversionPatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  Value spadAddrValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(SpAddr));
  uint64_t spadAddrInt = (uint64_t)rows << (ADDR_LEN + 16) |
                         (uint64_t)cols << ADDR_LEN | (uint64_t) SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  rewriter.create<Mvin_IntrOp>(loc, configPtr, spad);
}

void gemminiMvoutOffset(const Value &mem, const size_t offset, const uint32_t SpAddr,
                        const size_t cols, const size_t rows,
                        ConversionPatternRewriter &rewriter) {
  Location loc = mem.getLoc();
  Value offsetOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  IntegerType i64Type = rewriter.getI64Type();
  Value configPtr = rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
  Value spadAddrValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(SpAddr));
  uint64_t spadAddrInt = (uint64_t)rows << (ADDR_LEN + 16) |
                         (uint64_t)cols << ADDR_LEN | (uint64_t) SpAddr;
  Value spad = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(spadAddrInt));
  rewriter.create<Mvout_IntrOp>(loc, configPtr, spad);
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

struct GemminiFlushOpLowering : public ConvertOpToLLVMPattern<FlushOp> {
  using ConvertOpToLLVMPattern<FlushOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FlushOp flushOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = flushOp.getLoc();
    Value skip = flushOp.getSkip();
    Value rs2 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(flushOp, skip, rs2);
    return success();
  }
};

struct GemminiConfigStOpLowering : public ConvertOpToLLVMPattern<ConfigStOp> {
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

struct GemminiConfigLdOpLowering : public ConvertOpToLLVMPattern<ConfigLdOp> {
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

struct GemminiConfigExOpLowering : public ConvertOpToLLVMPattern<ConfigExOp> {
  using ConvertOpToLLVMPattern<ConfigExOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ConfigExOp configExOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = configExOp.getLoc();
    float scale = configExOp.getSysAccScale().convertToFloat();
    uint64_t rs1 =
        (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
        configExOp.getAStride() << 16 | configExOp.getBTranspose() << 9 |
        configExOp.getATranspose() << 8 | configExOp.getSetOnlyStrides() << 7 |
        configExOp.getSysAct() << 3 | configExOp.getDataflow() << 2 | CONFIG_EX;

    uint64_t rs2 = configExOp.getCStride() << 48 | configExOp.getSysShift();
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
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
    uint64_t rs1 = (uint64_t )((uint32_t )configNormOp.getQConst() << 32) |
                   (configNormOp.getQConstType() & 1) << 18 |
                   (configNormOp.getSetStatsIdOnly() & 1) << 17 |
                   (configNormOp.getActMsb() & 1) << 16 |
                   configNormOp.getStatsId() << 8 | CONFIG_BERT;
    uint64_t rs2 = ((uint64_t)((uint32_t)(configNormOp.getIguluQc())) << 32) | ((uint64_t)((uint32_t)(configNormOp.getIguluQb())));
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ConfigNorm_IntrOp>(configNormOp, rs1Value,
                                                 rs2Value);
    return success();
  }
};

struct GemminiMvinOpLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
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
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (ADDR_LEN + 16) |
                           (uint64_t)memRefShape[1] << ADDR_LEN | number;
    Value spad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
    rewriter.replaceOpWithNewOp<Mvin_IntrOp>(mvinOp, indexCastOp, spad);
    return success();
  }
};

struct GemminiMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
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
    uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (ADDR_LEN + 16) |
                           (uint64_t)memRefShape[1] << ADDR_LEN | number;
    Value newSpad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(spadAddrInt));
    rewriter.replaceOpWithNewOp<Mvout_IntrOp>(mvoutOp, indexCastOp, newSpad);
    return success();
  }
};

struct GemminiPreloadZerosLowering
    : public ConvertOpToLLVMPattern<PreloadZerosOp> {
  using ConvertOpToLLVMPattern<PreloadZerosOp>::ConvertOpToLLVMPattern;
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
    uint64_t rs1 = (uint64_t)16 << (ADDR_LEN + 16) | (uint64_t)16 << ADDR_LEN |
                   (uint64_t)-1;
    uint64_t rs2 =
        cRowsInt << (ADDR_LEN + 16) | cColsInt << (ADDR_LEN) | addrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadZerosOp, rs1Value,
                                                rs2Value);
    return success();
  }
};

struct GemminiPreloadLowering : public ConvertOpToLLVMPattern<PreloadOp> {
  using ConvertOpToLLVMPattern<PreloadOp>::ConvertOpToLLVMPattern;
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
    uint64_t rs1 = bdRowsInt << (ADDR_LEN + 16) | bdColsInt << ADDR_LEN |
                   (uint64_t)bdAddrInt;
    uint64_t rs2 =
        cRowsInt << (ADDR_LEN + 16) | cColsInt << ADDR_LEN | (uint64_t)cAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<Preload_IntrOp>(preloadOp, rs1Value, rs2Value);
    return success();
  }
};

struct GemminiComputePreloadedLowering
    : public ConvertOpToLLVMPattern<ComputePreloadedOp> {
  using ConvertOpToLLVMPattern<ComputePreloadedOp>::ConvertOpToLLVMPattern;
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
    uint64_t rs1 =
        aRowsInt << (ADDR_LEN + 16) | aColsInt << ADDR_LEN | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (ADDR_LEN + 16) | bdColsInt << ADDR_LEN | bdAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ComputePreloaded_IntrOp>(computePreloadedOp,
                                                         rs1Value, rs2Value);
    return success();
  }
};

struct GemminiComputeAccumulatedLowering
    : public ConvertOpToLLVMPattern<ComputeAccumulatedOp> {
  using ConvertOpToLLVMPattern<ComputeAccumulatedOp>::ConvertOpToLLVMPattern;
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
    uint64_t rs1 =
        aRowsInt << (ADDR_LEN + 16) | aColsInt << ADDR_LEN | aAddrInt;
    uint64_t rs2 =
        bdRowsInt << (ADDR_LEN + 16) | bdColsInt << ADDR_LEN | bdAddrInt;
    Value rs1Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs1));
    Value rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
    rewriter.replaceOpWithNewOp<ComputeAccumulated_IntrOp>(computeAccumulatedOp,
                                                           rs1Value, rs2Value);

    return success();
  }
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
        loc, rewriter.getI64IntegerAttr(rs1));
    rs2Value = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rs2));
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
    const size_t dimIPadded = (dimI / DIM + (dimI % DIM != 0)) * DIM;
    const size_t dimJPadded = (dimJ / DIM + (dimJ % DIM != 0)) * DIM;
    const size_t dimKPadded = (dimK / DIM + (dimK % DIM != 0)) * DIM;
    const size_t I0 =
        dimIPadded / (tileI * DIM) + (dimIPadded % (tileI * DIM) != 0);
    const size_t J0 =
        dimJPadded / (tileJ * DIM) + (dimJPadded % (tileJ * DIM) != 0);
    const size_t K0 =
        dimKPadded / (tileK * DIM) + (dimKPadded % (tileK * DIM) != 0);
    const size_t lastI =
        dimIPadded % (tileI * DIM) == 0 ? tileI : (dimIPadded / DIM) % tileI;
    const size_t lastJ =
        dimJPadded % (tileJ * DIM) == 0 ? tileJ : (dimJPadded / DIM) % tileJ;
    const size_t lastK =
        dimKPadded % (tileK * DIM) == 0 ? tileK : (dimKPadded / DIM) % tileK;
    const size_t paddingI = dimIPadded - dimI;
    const size_t paddingJ = dimJPadded - dimJ;
    const size_t paddingK = dimKPadded - dimK;
    const bool noBias = false;
    const size_t sizeofD = lowD ? sizeof(elem_t) : sizeof(acc_t);
    const size_t sizeofC = fullC ? sizeof(acc_t) : sizeof(elem_t);
    Location loc = tileMatMulOp.getLoc();
    llvm::APFloat accScaleIdentity((float)ACC_SCALE_IDENTITY);
    rewriter.create<ConfigExOp>(loc, /*dataflow = */ dataflow, /*sysAct = */ act & 3,
                                /* sysShift = */ 0, accScaleIdentity);
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideC * sizeofC));
    rewriter.create<ConfigStOp>(loc, strideValue, act & 3,
                                llvm::APFloat(scale));
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideA * sizeof(elem_t)));
    rewriter.create<ConfigLdOp>(loc, strideValue, llvm::APFloat(aScaleFactor),
                                false, 0);
    strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(strideB * sizeof(elem_t)));
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
            pre = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(0));
          } else {
            size_t biasRow = repeatingBias ? 0 : i0 * tileI * DIM;
            size_t offset = (biasRow * strideD + j0 * tileJ * DIM) * sizeofD *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            pre = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), D,
                                                 offsetValue);
          }

          Value out;
          if (k0 == K0 - 1) {
            size_t offset = (i0 * tileI * DIM * strideC + j0 * tileJ * DIM) *
                            sizeofC * sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            out = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), C,
                                                 offsetValue);
          } else {
            out = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(0));
          }
          const size_t i = i0 < I0 - 1 ? tileI : lastI;
          const size_t j = j0 < J0 - 1 ? tileJ : lastJ;
          const size_t k = k0 < K0 - 1 ? tileK : lastK;
          const size_t padI = i0 == I0 - 1 ? paddingI : 0;
          const size_t padJ = j0 == J0 - 1 ? paddingJ : 0;
          const size_t padK = k0 == K0 - 1 ? paddingK : 0;
          Value a;
          if (aTranspose) {
            size_t offset = (k0 * tileK * DIM * strideA + i0 * tileI * DIM) *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          } else {
            size_t offset = (i0 * tileI * DIM * strideA + k0 * tileK * DIM) *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            a = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), A,
                                               offsetValue);
          }
          Value b;
          if (bTranspose) {
            size_t offset = (j0 * tileJ * DIM * strideB + k0 * tileK * DIM) *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
            b = rewriter.create<arith::AddIOp>(loc, rewriter.getI64Type(), B,
                                               offsetValue);
          } else {
            size_t offset = (k0 * tileK * DIM * strideB + j0 * tileJ * DIM) *
                            sizeof(elem_t);
            Value offsetValue = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr(offset));
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
    Value flushValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<Flush_IntrOp>(tileMatMulOp, flushValue,
                                              flushValue);
    return;
  }

  size_t tiledMatmulTotalSpadRows(size_t I, size_t J, size_t K) const {
    return (I * K + K * J) * DIM;
  }

  size_t tiledMatmulTotalAccRows(size_t I, size_t J) const {
    return (I * J) * DIM;
  }

public:
  using ConvertOpToLLVMPattern<TileMatMulOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileMatMulOp tileMatMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#define partitionRows (BANK_NUM * BANK_ROWS / 2)
#define matsInPartition (partition_rows / DIM)
#define matsInAcc (ACC_ROWS / DIM)
#define maxTileIJ ((size_t)sqrt(mats_in_acc))
#define maxTileK (matsInPartition / maxTileIJ)

#define dbPartitionRows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define dbMatsInPartition (dbPartitionRows / DIM)
#define dbMatsInAcc ((ACC_ROWS / 2) / DIM)
#define dbMaxTileIJ ((size_t)sqrt(dbMatsInAcc))
#define dbMaxTileK (dbMatsInPartition / dbMaxTileIJ)
    Value aArray = tileMatMulOp.getAArray();
    Value bArray = tileMatMulOp.getBArray();
    Value cArray = tileMatMulOp.getCArray();
    Value dArray = tileMatMulOp.getDArray();
    TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
    Location loc = tileMatMulOp.getLoc();
    IntegerType i64Type = rewriter.getI64Type();
    Value aArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                aArray);
    Value aArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, aArrayExtractOp);
    Value bArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                bArray);
    Value bArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, bArrayExtractOp);
    
    Value cArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                cArray);
    Value cArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, cArrayExtractOp);
    Value dArrayExtractOp =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, resultType,
                                                                dArray);
    Value dArrayindexCastOp =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, dArrayExtractOp);
    MemRefType aArrayType = aArray.getType().dyn_cast<MemRefType>();
    MemRefType bArrayType = bArray.getType().dyn_cast<MemRefType>();
    MemRefType cArrayType = cArray.getType().dyn_cast<MemRefType>();
    MemRefType dArrayType = dArray.getType().dyn_cast<MemRefType>();
    llvm::ArrayRef<long> aArrayShape = aArrayType.getShape();
    llvm::ArrayRef<long> bArrayShape = bArrayType.getShape();
    llvm::ArrayRef<long> cArrayShape = cArrayType.getShape();
    llvm::ArrayRef<long> dArrayShape = dArrayType.getShape();
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
    size_t dimIPaded = (dimI / DIM + (dimI % DIM != 0)) * DIM;
    size_t dimJPaded = (dimJ / DIM + (dimJ % DIM != 0)) * DIM;
    size_t dimKPaded = (dimK / DIM + (dimK % DIM != 0)) * DIM;
    size_t maxSpadRows = BANK_NUM * BANK_ROWS / 2;
    size_t maxAccRows = ACC_ROWS / 2;
    size_t tileI, tileJ, tileK;
    if (act == LAYERNORM || act == SOFTMAX) {
      tileI = 1;
      tileJ = dimJPaded | DIM;
      tileK = 1;
    } else {
      tileI = dimIPaded / DIM < dbMaxTileIJ ? dimIPaded / DIM : dbMaxTileIJ;
      tileJ = dimJPaded / DIM < dbMaxTileIJ ? dimJPaded / DIM : dbMaxTileIJ;
      tileK = dimKPaded / DIM < dbMaxTileK ? dimKPaded / DIM : dbMaxTileK;
    }
    while (true) {
      bool increased = false;

      if (tiledMatmulTotalSpadRows(tileI, tileJ + 1, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI, tileJ + 1) <= maxAccRows &&
          (tileJ + 1) * DIM <= dimJPaded) {
        tileJ++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI + 1, tileJ, tileK) <= maxSpadRows &&
          tiledMatmulTotalAccRows(tileI + 1, tileJ) <= maxAccRows &&
          (tileI + 1) * DIM <= dimIPaded) {
        tileI++;
        increased = true;
      }

      if (tiledMatmulTotalSpadRows(tileI, tileJ, tileK + 1) <= maxSpadRows &&
          (tileK + 1) * DIM <= dimKPaded) {
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
};

class GemminiTileConvOpLowering : public ConvertOpToLLVMPattern<TileConvOp> {

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
    IntegerType i64Type = rewriter.getI64Type();
    // loopConvWsConfig1
    uint64_t rs1 = (uint64_t)outChannels << 48 | (uint64_t)inChannels << 32 |
                   (uint64_t)inDim << 16 | (uint64_t)batchSize;
    uint64_t rs2 = (uint64_t)padding << 48 | (uint64_t)stride << 32 |
                   (uint64_t)poolOutDim << 16 | (uint64_t)outDim;
    IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
    IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
    Value rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    Value rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
    rewriter.create<LoopConvWsConfig1_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig2
    rs1 = (uint64_t)kernelDim << 48 | (uint64_t)poolSize << 32 |
          (uint64_t)poolStride << 16 | (uint64_t)poolPadding;
    rs2 = (uint64_t)batches << 48 | (uint64_t)porows << 32 |
          (uint64_t)pocols << 16 | (uint64_t)pochs;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs2Attr);
    rewriter.create<LoopConvWsConfig2_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig3
    rs1 = (uint64_t)krows << 48 | (uint64_t)kcols << 32 | (uint64_t)kchs << 16 |
          (uint64_t)lpad;
    rs2 = (uint64_t)rpad << 48 | (uint64_t)upad << 32 | (uint64_t)dpad << 16 |
          (uint64_t)plpad;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc,i64Type, rs2Attr);
    rewriter.create<LoopConvWsConfig3_IntrOp>(loc, rs1Value, rs2Value);
    // loopConvWsConfig4
    rs1 = (uint64_t)orows << 48 | (uint64_t)prpad << 32 |
          (uint64_t)pupad << 16 | (uint64_t)pdpad;
    rs2 = (uint64_t)kernelDilation << 16 | (uint64_t)ocols;
    rs1Attr = rewriter.getI64IntegerAttr(rs1);
    rs2Attr = rewriter.getI64IntegerAttr(rs2);
    rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
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
    rs1Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
    rs2Value = rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
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
    Location loc = tileConvOp.getLoc();

    if (dw) {
      kchs = 1;
      pochs = 1;
    }

    const int orows = porows * poolStride + poolSize - 1 - pupad - pdpad;
    const int ocols = pocols * poolStride + poolSize - 1 - plpad - prpad;
    const int ochs = pochs;

    // Calculate image dimensions
    // Note: "irows" and "icols" includes padding
    const int dilated_krows = krows + (kernelDilation - 1)*(krows - 1);
    const int dilated_kcols = kcols + (kernelDilation - 1)*(kcols - 1);
    int irows = orows * stride + dilated_krows - 1;
    int icols = ocols * stride + dilated_kcols - 1;
    int irows_unpadded = irows - upad - dpad;
    int icols_unpadded = icols - lpad - rpad;
    const int ichs = kchs;

#define UNDILATED(x) ((inputDilated) ? (((x)+1)/2) : (x))

    if (inputDilated) {
      irows_unpadded = (irows_unpadded+1)/2;
      icols_unpadded = (icols_unpadded+1)/2;

      irows = irows_unpadded + UNDILATED(upad) + UNDILATED(dpad);
      icols = icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad);
    }


#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
    const bool transposed =
        transOutput1203 || transInput3120 || transWeight1203 || transWeight0132;
    int maxPixelsPerRow = transposed || wrot180 || downsample || inputDilated ||
                                  kernelDilation > 1 || ichs > DIM
                              ? 1
                              : DIM / ichs;
    if (maxPixelsPerRow > kcols)
      maxPixelsPerRow = kcols;
#else
    const int maxPixelsPerRow = 1;
#endif

    // Calculate spad address offsets
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
    const int bRows = transWeight0132 ?
                                         in_channels_per_bank * kcols * krows * ochs :
                                         out_channels_per_bank * kcols * krows * kchs;

    static uint32_t dSpAddrRow = 0;
    static uint32_t cSpAddrRow = 0;

    const uint32_t A_sp_addr_start = 0;
    const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - bRows;
    const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + dSpAddrRow;
    const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + cSpAddrRow;

    if (bias != 0) {
      dSpAddrRow = (dSpAddrRow + ACC_ROWS / 2) % ACC_ROWS;
    }

    if (output != 0) {
      cSpAddrRow = (cSpAddrRow + ACC_ROWS / 2) % ACC_ROWS;
    }

    gemminiLoopConvWs(
        batchSize, inDim, inChannels, outChannels, outDim, poolOutDim, stride,
        padding, kernelDim, kernelDilation, poolSize, poolStride, poolPadding,
        batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad,
        dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias,
        input, noBias, noPool, downsample, wrot180, inputDilated, act,
        transOutput1203, transWeight1203, transWeight0132, transInput3120,
        maxPixelsPerRow, dw, tileConvOp, rewriter);
/*
    if (!noPool) {
        // TODO: Exit, but now I don't known how to do
//      printf("Pooling with rectangular convolutions is currently not supported.\n");
//      exit(1);
    }

    // Only rectangular convolutions will use the following C code
    // mvin bias
    if (bias != NULL) {
      // TODO we probably don't need quite this many nested loops for this part

      const int maxOchsPerMvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
                                                                   MAX_BLOCK_LEN_ACC * DIM;
      Value zeroValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
//      rewriter.create<ConfigLdOp>(loc, zeroValue, llvm::APFloat(MVIN_SCALE_IDENTITY), false, )
      // TODO: configLd op 这里不够用，需要加block_mvin_stride和pixel_repeats，应该不难
      gemmini_extended4_config_ld(0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

      for (int b = 0; b < batches; b++)
        for (int orow = 0; orow < orows; orow++)
          for (int ocol = 0; ocol < ocols; ocol += DIM) {
            const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

            for (int och = 0; och < ochs; och += maxOchsPerMvin) {
              const int J = ochs - och > maxOchsPerMvin ? maxOchsPerMvin : ochs - och;

              const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

              const acc_t * bias_dram_addr = noBias ? NULL : bias + och;

              gemmini_extended_mvin3(bias_dram_addr,
                                     D_sp_addr,
                                     J, I);
            }
          }
    }

    // mvin input
    if (input != NULL){
      int max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
                                                        MAX_BLOCK_LEN * DIM;
      if (transInput3120) {
        max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
                                                         MAX_BLOCK_LEN * DIM;
      }

      const int dramStride = transInput3120 ?
                                               batchSize * sizeof(elem_t) :
                                               inChannels * sizeof(elem_t);

      const int spad_stride = transInput3120 ?
                                               ichs * (irows >> downsample) * (icols >> downsample) :
                                               batches * (irows >> downsample) * (icols >> downsample);

      gemmini_extended5_config_ld(dramStride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, maxPixelsPerRow, 0);

      const int b_it = transInput3120 ? max_chs_per_mvin : 1;
      const int ich_it = transInput3120 ? 1 : max_chs_per_mvin;

      for (int b = 0; b < batches; b += b_it)
        for (int irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += 1 + downsample) {
          const int irowPadded = irow + UNDILATED(upad);

          for (int icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
            // TODO There might be some unnecessary mvins here at the edge of the image

            int I = icols_unpadded - icol > (DIM << downsample) ?
                                                                (DIM << downsample) : icols_unpadded - icol;

            if (icol < 0) {
              I = -icol > DIM ? DIM : -icol;
            } else if (icol >= icols_unpadded) {
              I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
            }

            const int icol_padded = icol + UNDILATED(lpad);

            for (int ich = 0; ich < ichs; ich += ich_it) {
              int K = ichs - ich > max_chs_per_mvin ?
                                                    max_chs_per_mvin : ichs - ich;
              if (transInput3120) {
                K = batches - b > max_chs_per_mvin ?
                                                   max_chs_per_mvin : batches - b;
              }

#define DS(x) ((x) >> (downsample))

              uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irowPadded) * DS(icols) + DS(icol_padded);
              if (transInput3120) {
                A_sp_addr = A_sp_addr_start + (b / DIM) * ichs * DS(irows) * DS(icols) + ich * DS(irows) * DS(icols) + DS(irowPadded) * DS(icols) + DS(icol_padded);
              }

              const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

              const elem_t * in = input + (b*in_row_dim*in_col_dim + irow*in_col_dim + icol) * in_stride + ich;
              if (is_zeros) {
                in = NULL;
              } else if (transInput3120) {
                in = input + (ich*in_row_dim*in_col_dim + irow*in_col_dim + icol) * batch_size + b;
              }

              gemmini_extended_mvin(in,
                                    A_sp_addr,
                                    K, I >> downsample);
            }

            icol += I;
          }
        }
    }

    // mvin weights
    if (weights != NULL) {
      int max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
                                                        MAX_BLOCK_LEN * DIM;
      if (transWeight0132) {
        max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
                                                      MAX_BLOCK_LEN * DIM;
      }

      size_t dram_stride = weight_stride * sizeof(elem_t);
      if (dw) {
        dram_stride = sizeof(elem_t);
      } else if (transWeight1203) {
        dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
      } else if (transWeight0132) {
        dram_stride = in_channels * sizeof(elem_t);
      }

      const size_t spad_block_stride = transWeight0132 ?
                                                         krows * kcols * ochs : krows * kcols * kchs;

      gemmini_extended4_config_ld(dram_stride, MVIN_SCALE_IDENTITY, false, spad_block_stride, 1);

      const size_t och_it = transWeight0132 ? DIM : max_chs_per_mvin;
      const size_t kch_it = transWeight0132 ? max_chs_per_mvin : DIM;

      for (int och = 0; och < ochs; och += och_it) {
        for (int krow = 0; krow < krows; krow++)
          for (int kcol = 0; kcol < kcols; kcol++)
            for (int kch = 0; kch < kchs; kch += kch_it) {
              int K = kchs - kch > DIM ? DIM : kchs - kch;
              int J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
              if (transWeight0132) {
                K = ochs - och > DIM ? DIM : ochs - och;
                J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
              }

              uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
              if (transWeight0132) {
                B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
              }

              const elem_t * w = weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * weight_stride + och;
              if (dw) {
                w = weights + krow * kernel_dim + kcol;
              } else if (trans_weight_1203) {
                w = weights + (kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och;
              } else if (transWeight0132) {
                w = weights + (krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch;
              }

              gemmini_extended_mvin2(w, B_sp_addr, J, K);
            }
      }
    }

    // Compute
    {
      const int b_it = transInput3120 ? DIM : 1;
      const int ocol_it = transInput3120 ? 1 : (DIM << inputDilated);

      if (transInput3120) {
        gemmini_extended3_config_ex(0, 0, 0, 0, orows * ocols, irows * icols, 0,
                                    0, true);
      }

      for (int och = 0; och < ochs; och += DIM) {
        for (int krow = 0; krow < krows; krow++) {
          for (int kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
            for (int kch = 0; kch < kchs; kch += DIM) {
              bool new_weights = true;

              for (int b = 0; b < batches; b += b_it) {
                for (int orow = 0; orow < orows; orow++) {
                  // Skip some kernel rows due to input-dilation
                  if (inputDilated &&
                      ((krow * kernelDilation + orow * stride - upad) % 2 !=
                       0)) {
                    continue;
                  }

                  for (int ocol = 0; ocol < ocols;) {
                    // Skip some cols dimensions due to input-dilation
                    if (inputDilated &&
                        ((kcol + ocol * stride - lpad) % 2 != 0)) {
                      ocol++;
                      continue;
                    }

                    int irow = orow * stride + krow * kernelDilation;
                    int icol = ocol * stride + kcol * kernelDilation;

                    if (inputDilated) {
                      irow = (irow + 1) / 2;
                      icol = (icol + 1) / 2;
                    }

                    const int pixels = kcols - kcol > max_pixels_per_row
                                           ? max_pixels_per_row
                                           : kcols - kcol;

                    const uint32_t C_sp_addr =
                        C_sp_addr_start +
                        (och / DIM) * batches * orows * ocols +
                        b * orows * ocols + orow * ocols + ocol;

                    // Over here, construct a new matrix
                    //
                    // Let us assume that we only ever operate on
                    // one pixel in one row.
                    // Thus, krows == kcols == 1
                    //
                    // Then, for every set of I, J, and K values
                    //     - I = ocols
                    //     - J = ochs
                    //     - K = kchs

                    int I = UNDILATED(ocols - ocol > (DIM << inputDilated)
                                          ? (DIM << inputDilated)
                                          : ocols - ocol);
                    const int J = ochs - och > DIM ? DIM : ochs - och;
                    const int K =
                        pixels * (kchs - kch > DIM ? DIM : kchs - kch);

                    if (transInput3120) {
                      I = batches - b > DIM ? DIM : batches - b;
                    }

                    uint32_t A_sp_addr =
                        A_sp_addr_start +
                        (kch / DIM) * batches * DS(irows) * DS(icols) +
                        b * DS(irows) * DS(icols) + DS(irow) * DS(icols) +
                        DS(icol);
                    if (transInput3120) {
                      A_sp_addr = A_sp_addr_start +
                                  (b / DIM) * kchs * DS(irows) * DS(icols) +
                                  kch * DS(irows) * DS(icols) +
                                  DS(irow) * DS(icols) + DS(icol);
                    }

                    const int krow_ = wrot180 ? krows - krow - 1 : krow;
                    const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

                    uint32_t B_sp_addr =
                        B_sp_addr_start + (och / DIM) * krows * kcols * kchs +
                        krow_ * kcols * kchs + kcol_ * kchs + kch;
                    if (transWeight0132) {
                      B_sp_addr = B_sp_addr_start +
                                  (kch / DIM) * krows * kcols * ochs +
                                  krow_ * kcols * ochs + kcol_ * ochs + och;
                    }

                    const uint32_t pre_sp_addr =
                        new_weights ? B_sp_addr : GARBAGE_ADDR;

                    // perform matmul
                    gemmini_extended_preload(pre_sp_addr, C_sp_addr, J, K, J,
                                             I);

                    if (new_weights) {
                      gemmini_extended_compute_preloaded(
                          A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                    } else {
                      gemmini_extended_compute_accumulated(
                          A_sp_addr, GARBAGE_ADDR, K, I, J, I);
                    }

                    ocol += ocol_it;
                    new_weights = false;
                  }
                }
              }
            }
          }
        }
      }
    }
#undef DS
#undef UNDILATED

    // mvout output
    if (output != NULL) {
      if (noPool) {
        for (int b = 0; b < batches; b++)
          for (int orow = 0; orow < orows; orow++)
            for (int ocol = 0; ocol < ocols; ocol += DIM) {
              const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

              for (int och = 0; och < ochs; och += DIM) {
                const int J = ochs - och > DIM ? DIM : ochs - och;

                const uint32_t C_sp_addr =
                    C_sp_addr_start + (och / DIM) * batches * orows * ocols +
                    b * orows * ocols + orow * ocols + ocol;

                elem_t *out = output +
                              (b * out_row_dim * out_col_dim +
                               orow * out_col_dim + ocol) *
                                  out_stride +
                              och;
                if (trans_output_1203) {
                  out = output +
                        (orow * out_col_dim * batch_size + ocol * batch_size +
                         b) *
                            out_channels +
                        och;
                }

                gemmini_extended_mvout(out, C_sp_addr, J, I);
              }
            }
      } else {
        printf("Pooling with rectangular convolutions is currently not supported.\n");
        exit(1);
      }
    }
    */
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
                               ? batchSize * outChannels * sizeof(elem_t)
                               : outChannels * sizeof(elem_t);
    IntegerAttr strideAttr = rewriter.getI64IntegerAttr(stDramStride);
    Location loc = tileConvOp.getLoc();
    Value strideValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), strideAttr);
    rewriter.create<ConfigStOp>(loc, strideValue, act, llvm::APFloat(scale));
    rewriter.create<ConfigExOp>(
        loc, /*dataflow = */ WEIGHT_STATIONARY, /*act = */ 0, /*shift = */ 0,
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
                  IntegerAttr offsetAttr =
                      rewriter.getI64IntegerAttr(((b * poolOutDim * poolOutDim +
                                                   porow * poolOutDim + pocol) *
                                                      outChannels +
                                                  poch) *
                                                 sizeof(elem_t));
                  Value offsetValue = rewriter.create<arith::ConstantOp>(
                      loc, rewriter.getI64Type(), offsetAttr);
                  Value out = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), output,
                      offsetValue);
                  if (transOutput1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((porow * poolOutDim * batchSize + pocol * batchSize +
                          b) *
                             outChannels +
                         poch) *
                        sizeof(elem_t));
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        loc,rewriter.getI64Type(), offsetAttr);
                    out = rewriter.create<arith::AddIOp>(tileConvOp.getLoc(),
                                                         rewriter.getI64Type(),
                                                         output, offsetValue);
                  }

                  if (krow + krows < kernelDim || kcol + kcols < kernelDim ||
                      kch + kchs < inChannels) {
                    IntegerAttr attr = rewriter.getI16IntegerAttr(0);
                    out = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), attr);
                  }
                  IntegerAttr pochAttr =
                      rewriter.getI64IntegerAttr(poch * sizeof(acc_t));
                  Value pochValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), pochAttr);
                  Value bias_ = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), bias,
                      pochValue);
                  if (krow > 0 || kcol > 0 || kch > 0) {
                    IntegerAttr attr = rewriter.getI64IntegerAttr(0);
                    bias_ = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), attr);
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
                      sizeof(elem_t));
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
                  Value weightsSlice = rewriter.create<arith::AddIOp>(
                      tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                      offsetValue);
                  if (transWeight1203) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((kch * kernelDim * kernelDim + krow_ * kernelDim +
                          kcol_) *
                             outChannels +
                         poch) *
                        sizeof(elem_t));
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
                    weightsSlice = rewriter.create<arith::AddIOp>(
                        tileConvOp.getLoc(), rewriter.getI64Type(), weights,
                        offsetValue);
                  } else if (transWeight0132) {
                    offsetAttr = rewriter.getI64IntegerAttr(
                        ((krow_ * kernelDim * outChannels +
                          kcol_ * outChannels + poch) *
                             inChannels +
                         kch) *
                        sizeof(elem_t));
                    offsetValue = rewriter.create<arith::ConstantOp>(
                        tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
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
                      sizeof(elem_t));
                  offsetValue = rewriter.create<arith::ConstantOp>(
                      tileConvOp.getLoc(),rewriter.getI64Type(), offsetAttr);
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
                        sizeof(elem_t));
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

    const int inChannelsPerBank = ichs / DIM + (ichs % DIM != 0);
    const int outChannelsPerBank = ochs / DIM + (ochs % DIM != 0);
    const int batchesPerBank = batches / DIM + (batches % DIM != 0);

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
        if (!(i == ocolsIdx && args[i] <= DIM && args[orowsIdx] > 1) &&
            args[i] > maxVal) {
          maxVal = args[i];
          maxIdx = i;
        }
      }

      if (maxIdx == outChannelsIdx || maxIdx == inChannelsIdx) {
        if (args[maxIdx] % DIM != 0) {
          args[maxIdx] = (args[maxIdx] / DIM) * DIM;
        } else {
          args[maxIdx] -= DIM;
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
};

void mlir::populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns
      .add<ForwardOperands<func::CallOp>, ForwardOperands<func::CallIndirectOp>,
           ForwardOperands<func::ReturnOp>>(converter, &converter.getContext());
  patterns.add<GemminiFlushOpLowering>(converter);
  patterns.add<GemminiConfigStOpLowering>(converter);
  patterns.add<GemminiConfigLdOpLowering>(converter);
  patterns.add<GemminiMvinOpLowering>(converter);
  patterns.add<GemminiMvoutLowering>(converter);
  patterns.add<GemminiConfigExOpLowering>(converter);
  patterns.add<GemminiConfigNormOpLowering>(converter);
  patterns.add<GemminiPreloadZerosLowering>(converter);
  patterns.add<GemminiPreloadLowering>(converter);
  patterns.add<GemminiComputePreloadedLowering>(converter);
  patterns.add<GemminiComputeAccumulatedLowering>(converter);
  patterns.add<GemminiTileMatMulLowering>(converter);
  patterns.add<GemminiTileConvOpLowering>(converter);
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
