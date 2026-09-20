#ifndef LLM_HPP
#define LLM_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network.hpp"

enum LlmHeaderKey {
    VERSION = 0,
    ARCH_TYPE = 1,
    DIM = 2,
    HIDDEN_DIM = 3,
    N_LAYERS = 4,
    N_HEADS = 5,
    N_KV_HEADS = 6,
    N_EXPERTS = 7,
    N_ACTIVE_EXPERTS = 8,
    VOCAB_SIZE = 9,
    SEQ_LEN = 10,
    HIDDEN_ACT = 11,
    ROPE_THETA = 12,
    WEIGHT_FLOAT_TYPE = 13,
    ROPE_SCALING_FACTOR = 14,
    ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    ROPE_SCALING_HIGH_FREQ_FACTORY = 16,
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    ROPE_TYPE = 18,
    HEAD_DIM = 19,
    NORM_EPSILON = 20,
    MOE_HIDDEN_DIM = 21,
    SLIDING_WINDOW = 22, // Gemma 4: window of the sliding_attention layers (0 = none)
    FULL_ATT_INTERVAL = 23, // Gemma 4: layer `l` is full_attention iff (l + 1) % interval == 0 (0 = all layers are full attention)
    ROPE_THETA_SWA = 24, // Gemma 4: rope theta of the sliding layers (ROPE_THETA is the full-attention theta)
    ROPE_DIMS_FULL = 25, // Gemma 4: rotated dims per head on full-attention layers (partial rotary), 0 = all
    HEAD_DIM_FULL = 26, // Gemma 4: head dim of the full-attention layers (0 = HEAD_DIM)
    N_KV_HEADS_FULL = 27, // Gemma 4: kv heads of the full-attention layers (0 = N_KV_HEADS)
    FINAL_LOGIT_SOFTCAP = 28, // Gemma 4: logits = cap * tanh(logits / cap) (0 = none)
};

enum LlmHiddenAct {
    HIDDEN_ACT_GELU,
    HIDDEN_ACT_SILU,
};

enum LlmArchType {
    LLAMA = 0xABCD00,
    QWEN3 = 0xABCD01,
    QWEN3_MOE = 0xABCD02,
    GEMMA4 = 0xABCD03,
};

typedef struct {
    NnSize headerSize;
    NnSize fileSize;
    int version;
    LlmArchType archType;
    NnUint dim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headDim;
    NnUint nKvHeads;
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint origSeqLen; // Original model context length
    NnUint seqLen; // Limited context length by the `--max-seq-len` argument
    NnUint hiddenDim;
    NnUint moeHiddenDim;
    LlmHiddenAct hiddenAct;
    NnUint qDim;
    NnUint kvDim;
    NnUint vocabSize;
    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnUint ropeScalingOrigMaxSeqLen;
    float normEpsilon;

    // Gemma 4 (sliding/full attention layer types)
    NnUint slidingWindow;
    NnUint fullAttInterval;
    float ropeThetaSwa;
    NnUint ropeDimsFull;
    NnUint headDimFull;
    NnUint nKvHeadsFull;
    NnUint qDimFull;
    NnUint kvDimFull;
    float finalLogitSoftcap;

    NnFloatType weightType;
    NnFloatType syncType;
} LlmHeader;

typedef struct {
    LlmHeader *header;
    NnNetConfig netConfig;
    NnNodeConfig *nodeConfigs;
    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnColMatmulSlice woSlice;
    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;
    NnRowMatmulSlice wclsSlice;
    // Gemma 4 full-attention layers (head dim / kv heads differ from the sliding layers)
    NnRowMatmulSlice qSliceFull;
    NnRowMatmulSlice kSliceFull;
    NnColMatmulSlice woSliceFull;
    NnSize3D qkRmsNormSizeFull;
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;
    NnSize3D tokenEmbeddingSize;
    NnSize3D rmsNormSize;
    NnSize3D qkRmsNormSize;
    NnSize3D moeGateSize;
} LlmNet;

LlmHeader loadLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);
void printLlmHeader(LlmHeader *header);
bool isLlmFullAttLayer(const LlmHeader *header, NnUint layerIndex);
LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches);
void releaseLlmNet(LlmNet *net);
void loadLlmNetWeight(const char* path, LlmNet *net, NnRootWeightLoader *loader);

#endif