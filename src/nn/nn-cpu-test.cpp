#include "nn-core.hpp"
#include "nn-config-builder.hpp"
#include "nn-cpu.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define DIM 32
#define N_BATCHES 2

static void printOk(const char *name) {
    printf("✅ %24s passed\n", name);
}

static void assertClose(const char *name, float actual, float expected, float tolerance) {
    if (std::fabs(actual - expected) > tolerance) {
        printf("❌ %s: %f != %f (tolerance %f)\n", name, actual, expected, tolerance);
        exit(1);
    }
}

// Runs a single-node net with one segment and returns the executor pieces needed to inspect it
class OpRunner {
public:
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    NnNetExecution *execution;
    NnCpuDevice *device;
    NnExecutor *executor;
    std::vector<NnExecutorDevice> devices;
    NnFakeNodeSynchronizer synchronizer;

    OpRunner(NnNetConfigBuilder &netBuilder, NnNodeConfigBuilder &nodeBuilder, NnSegmentConfigBuilder &segmentBuilder, NnUint nThreads) {
        nodeBuilder.addSegment(segmentBuilder.build());
        netConfig = netBuilder.build();
        nodeConfig = nodeBuilder.build();
        execution = new NnNetExecution(nThreads, &netConfig);
        device = new NnCpuDevice(&netConfig, &nodeConfig, execution);
        devices.push_back(NnExecutorDevice(device, -1, -1));
        executor = new NnExecutor(&netConfig, &nodeConfig, &devices, execution, &synchronizer, false);
    }

    ~OpRunner() {
        delete executor;
        delete execution;
        releaseNetConfig(&netConfig);
        releaseNodeConfig(&nodeConfig);
    }
};

static void print2D(const char *name, NnUint x, NnUint y, float *w) {
    for (NnUint i = 0; i < y; i++) {
        printf("%s[%d] = ", name, i);
        for (NnUint j = 0; j < x; j++)
            printf("%f ", w[i * x + j]);
        printf("\n");
    }
}

static void testRmsNorm() {
    NnNetConfigBuilder netBuilder(1, N_BATCHES);
    NnUint xPipeIndex = netBuilder.addPipe("X", size2D(F_32, N_BATCHES, DIM));

    NnNodeConfigBuilder nodeBuilder(0);
    NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1));
    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addSync(xPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);
    segmentBuilder.addOp(OP_INV_RMS, "inv_rms", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{1e-5f, 1});
    segmentBuilder.addOp(OP_RMS_NORM, "rms_norm", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        size1D(F_32, DIM),
        NnRmsNormOpConfig{invRmsBufferIndex, 1});

    OpRunner runner(netBuilder, nodeBuilder, segmentBuilder, 2);
    float *x = (float *)runner.execution->pipes[0];
    std::vector<float> input(N_BATCHES * DIM);
    for (NnUint b = 0; b < N_BATCHES; b++) {
        for (NnUint i = 0; i < DIM; i++) {
            input[b * DIM + i] = i / (float)DIM + (float)b;
            x[b * DIM + i] = input[b * DIM + i];
        }
    }
    float rmsNormWeight[DIM];
    for (NnUint i = 0; i < DIM; i++)
        rmsNormWeight[i] = 0.5 + i / (float)DIM;
    runner.executor->loadWeight("rms_norm", 0u, 0u, sizeof(rmsNormWeight), (NnByte *)rmsNormWeight);

    runner.execution->setBatchSize(N_BATCHES);
    runner.executor->forward();

    float *rms = (float *)runner.device->buffers[0];
    print2D("rms", N_BATCHES, 1, rms);
    for (NnUint b = 0; b < N_BATCHES; b++) {
        float ss = 0.0f;
        for (NnUint i = 0; i < DIM; i++)
            ss += input[b * DIM + i] * input[b * DIM + i];
        const float invRms = 1.0f / std::sqrt(ss / DIM + 1e-5f);
        assertClose("inv_rms", rms[b], invRms, 1e-5f);
        for (NnUint i = 0; i < DIM; i++)
            assertClose("rms_norm", x[b * DIM + i], rmsNormWeight[i] * invRms * input[b * DIM + i], 1e-4f);
    }
    printOk("rms_norm");
}

static float pseudoRandom(NnUint seed) {
    // deterministic values in [-1, 1)
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

// Windowed multi-head attention against a straightforward reference:
// position `pos` attends to `t` iff `pos - t < window` (or every `t <= pos` when window == 0)
static void testMultiHeadAttWindow(const NnUint window, const float scale, const NnUint qHeadOffset, const NnUint nThreads) {
    const NnUint nHeads = 4;
    const NnUint nHeads0 = 2; // this node owns 2 of the 4 q heads
    const NnUint nKvHeads = 2;
    const NnUint headDim = 8;
    const NnUint seqLen = 16;
    const NnUint kvDim = nKvHeads * headDim; // the KV cache holds every kv head (replicated) when qHeadOffset > 0
    const NnUint qDim0 = nHeads0 * headDim;
    const NnUint pos = 11;
    const NnUint nBatches = 1;

    NnNetConfigBuilder netBuilder(1, nBatches);
    const NnUint posPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));

    NnNodeConfigBuilder nodeBuilder(0);
    const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, qDim0));
    const NnUint kBufferIndex = nodeBuilder.addBuffer("k", size2D(F_32, seqLen, kvDim));
    const NnUint vBufferIndex = nodeBuilder.addBuffer("v", size2D(F_32, seqLen, kvDim));
    const NnUint attBufferIndex = nodeBuilder.addBuffer("att", size2D(F_32, nBatches, nHeads0 * seqLen));
    const NnUint zBufferIndex = nodeBuilder.addBuffer("z", size2D(F_32, nBatches, qDim0));

    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addOp(OP_MULTIHEAD_ATT, "att", 0,
        pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
        pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
        size0(),
        NnMultiHeadAttOpConfig{
            nHeads, nHeads0, nKvHeads, headDim, seqLen, qDim0, kvDim,
            posPipeIndex, qBufferIndex, kBufferIndex, vBufferIndex, attBufferIndex,
            window, scale, qHeadOffset});

    OpRunner runner(netBuilder, nodeBuilder, segmentBuilder, nThreads);
    float *position = (float *)runner.execution->pipes[posPipeIndex];
    float *q = (float *)runner.device->buffers[qBufferIndex];
    float *k = (float *)runner.device->buffers[kBufferIndex];
    float *v = (float *)runner.device->buffers[vBufferIndex];
    float *z = (float *)runner.device->buffers[zBufferIndex];

    position[0] = (float)pos;
    for (NnUint i = 0; i < qDim0; i++)
        q[i] = pseudoRandom(i + 1);
    for (NnUint i = 0; i < seqLen * kvDim; i++) {
        k[i] = pseudoRandom(1000 + i);
        v[i] = pseudoRandom(5000 + i);
    }

    runner.execution->setBatchSize(nBatches);
    runner.executor->forward();

    // Reference: masked scores, the engine's own softmax (its NEON expf is a polynomial approximation,
    // so a libm reference would only test that), weighted sum of the values inside the window
    const NnUint kvMul = nHeads / nKvHeads;
    const NnUint tStart = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
    const NnUint nAtt = pos + 1 - tStart;
    for (NnUint h0 = 0; h0 < nHeads0; h0++) {
        const NnUint kvHead = (qHeadOffset + h0) / kvMul;
        std::vector<float> probs(nAtt, 0.0f);
        for (NnUint t = tStart; t <= pos; t++) {
            float dot = 0.0f;
            for (NnUint i = 0; i < headDim; i++)
                dot += q[h0 * headDim + i] * k[t * kvDim + kvHead * headDim + i];
            probs[t - tStart] = dot * scale;
        }
        softmax_F32(probs.data(), nAtt);
        for (NnUint i = 0; i < headDim; i++) {
            float expected = 0.0f;
            for (NnUint t = tStart; t <= pos; t++)
                expected += probs[t - tStart] * v[t * kvDim + kvHead * headDim + i];
            assertClose("multihead_att", z[h0 * headDim + i], expected, 1e-5f);
        }
    }

    char name[64];
    snprintf(name, sizeof(name), "att w=%u s=%.2f o=%u t=%u", window, scale, qHeadOffset, nThreads);
    printOk(name);
}

// NeoX/Falcon rope with a partial rotary dim against the HF "proportional" reference:
// pair (j, j + headDim/2) is rotated by pos * theta^(-2j/headDim) for j < ropeDims/2 and left untouched otherwise
static void testRopePartial(const NnUint ropeDims, const NnUint nThreads) {
    const NnUint nHeads = 2;
    const NnUint nKvHeads = 2;
    const NnUint headDim = 16;
    const NnUint qDim = nHeads * headDim;
    const NnUint kvDim = nKvHeads * headDim;
    const NnUint seqLen = 32;
    const float theta = 1000000.0f;
    const NnUint pos = 7;
    const NnUint nBatches = 1;

    NnRopeSlice slice = sliceRope(ROPE_FALCON, qDim, kvDim, nKvHeads, 1, seqLen, headDim, theta, 0, ropeDims);

    NnNetConfigBuilder netBuilder(1, nBatches);
    const NnUint posPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));

    NnNodeConfigBuilder nodeBuilder(0);
    const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, qDim));
    const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", slice.cacheSize);

    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addOp(OP_ROPE, "rope", 0,
        pointerBatchConfig(SRC_BUFFER, qBufferIndex),
        pointerBatchConfig(SRC_BUFFER, qBufferIndex),
        size0(),
        NnRopeOpConfig{ROPE_FALCON, 1, posPipeIndex, ropeCacheBufferIndex, 1.0f, 0.0f, 0.0f, 0u, slice});

    OpRunner runner(netBuilder, nodeBuilder, segmentBuilder, nThreads);
    float *position = (float *)runner.execution->pipes[posPipeIndex];
    float *q = (float *)runner.device->buffers[qBufferIndex];
    std::vector<float> input(qDim);

    position[0] = (float)pos;
    for (NnUint i = 0; i < qDim; i++) {
        input[i] = pseudoRandom(77 + i);
        q[i] = input[i];
    }

    runner.execution->setBatchSize(nBatches);
    runner.executor->forward();

    const NnUint half = headDim / 2;
    for (NnUint h = 0; h < nHeads; h++) {
        const float *x = &input[h * headDim];
        const float *y = &q[h * headDim];
        for (NnUint j = 0; j < half; j++) {
            float expected0 = x[j];
            float expected1 = x[j + half];
            if (j < ropeDims / 2) {
                const float invFreq = 1.0f / std::pow(theta, (2.0f * j) / (float)headDim);
                const float angle = pos * invFreq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                // rotate_half: (x1, x2) -> (x1 * cos - x2 * sin, x2 * cos + x1 * sin)
                expected0 = x[j] * c - x[j + half] * s;
                expected1 = x[j + half] * c + x[j] * s;
            }
            assertClose("rope", y[j], expected0, 1e-5f);
            assertClose("rope", y[j + half], expected1, 1e-5f);
        }
    }

    char name[64];
    snprintf(name, sizeof(name), "rope dims=%u/%u t=%u", ropeDims, headDim, nThreads);
    printOk(name);
}

static void testSoftcap() {
    const float cap = 30.0f;
    NnNetConfigBuilder netBuilder(1, N_BATCHES);
    const NnUint xPipeIndex = netBuilder.addPipe("X", size2D(F_32, N_BATCHES, DIM));
    NnNodeConfigBuilder nodeBuilder(0);
    const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, N_BATCHES, DIM));
    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addOp(OP_SOFTCAP, "softcap", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
        size0(),
        NnSoftcapOpCodeConfig{cap});

    OpRunner runner(netBuilder, nodeBuilder, segmentBuilder, 2);
    float *x = (float *)runner.execution->pipes[xPipeIndex];
    float *y = (float *)runner.device->buffers[yBufferIndex];
    for (NnUint i = 0; i < N_BATCHES * DIM; i++)
        x[i] = (float)i * 5.0f - 100.0f;

    runner.execution->setBatchSize(N_BATCHES);
    runner.executor->forward();

    for (NnUint i = 0; i < N_BATCHES * DIM; i++)
        assertClose("softcap", y[i], cap * std::tanh(x[i] / cap), 1e-5f);
    printOk("softcap");
}

static void testScalarMul() {
    const float scale = 0.05444f;
    NnNetConfigBuilder netBuilder(1, N_BATCHES);
    const NnUint xPipeIndex = netBuilder.addPipe("X", size2D(F_32, N_BATCHES, DIM));
    NnNodeConfigBuilder nodeBuilder(0);
    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addOp(OP_SCALAR_MUL, "scalar_mul", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        size1D(F_32, 1),
        NnScalarMulOpCodeConfig{});

    OpRunner runner(netBuilder, nodeBuilder, segmentBuilder, 2);
    runner.executor->loadWeight("scalar_mul", 0u, 0u, sizeof(float), (NnByte *)&scale);
    float *x = (float *)runner.execution->pipes[xPipeIndex];
    std::vector<float> input(N_BATCHES * DIM);
    for (NnUint i = 0; i < N_BATCHES * DIM; i++) {
        input[i] = pseudoRandom(900 + i);
        x[i] = input[i];
    }

    runner.execution->setBatchSize(N_BATCHES);
    runner.executor->forward();

    for (NnUint i = 0; i < N_BATCHES * DIM; i++)
        assertClose("scalar_mul", x[i], input[i] * scale, 1e-6f);
    printOk("scalar_mul");
}

// OP_MERGE_SET: output = sum of the node slices (no residual), for F32 and Q80 inputs
static void testMergeSet(NnFloatType inputType) {
    const NnUint nSlices = 3;
    NnNetConfigBuilder netBuilder(1, N_BATCHES);
    const NnUint xPipeIndex = netBuilder.addPipe("X", size2D(inputType, N_BATCHES, DIM * nSlices));
    NnNodeConfigBuilder nodeBuilder(0);
    const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, N_BATCHES, DIM));
    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addOp(OP_MERGE_SET, "merge_set", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
        size0(),
        NnMergeSetOpCodeConfig{});

    OpRunner runner(netBuilder, nodeBuilder, segmentBuilder, 2);
    float *y = (float *)runner.device->buffers[yBufferIndex];
    std::vector<float> input(N_BATCHES * DIM * nSlices);
    for (NnUint i = 0; i < input.size(); i++)
        input[i] = pseudoRandom(300 + i);
    for (NnUint i = 0; i < N_BATCHES * DIM; i++)
        y[i] = 1234.0f; // must be overwritten, not accumulated

    float tolerance;
    if (inputType == F_32) {
        std::memcpy(runner.execution->pipes[xPipeIndex], input.data(), input.size() * sizeof(float));
        tolerance = 1e-5f;
    } else {
        NnBlockQ80 *x = (NnBlockQ80 *)runner.execution->pipes[xPipeIndex];
        for (NnUint b = 0; b < N_BATCHES; b++)
            quantizeF32toQ80(&input[b * DIM * nSlices], &x[b * DIM * nSlices / Q80_BLOCK_SIZE], DIM * nSlices, 1, 0);
        tolerance = 3.0f / 127.0f; // q80 rounding of 3 slices in [-1, 1)
    }

    runner.execution->setBatchSize(N_BATCHES);
    runner.executor->forward();

    for (NnUint b = 0; b < N_BATCHES; b++) {
        for (NnUint i = 0; i < DIM; i++) {
            float expected = 0.0f;
            for (NnUint s = 0; s < nSlices; s++)
                expected += input[b * DIM * nSlices + s * DIM + i];
            assertClose("merge_set", y[b * DIM + i], expected, tolerance);
        }
    }
    printOk(inputType == F_32 ? "merge_set f32" : "merge_set q80");
}

int main() {
    initQuants();

    testRmsNorm();

    testMultiHeadAttWindow(0, 1.0f / std::sqrt(8.0f), 0, 1); // Llama-style causal attention
    testMultiHeadAttWindow(0, 1.0f, 0, 2);
    testMultiHeadAttWindow(4, 1.0f, 0, 1); // sliding window
    testMultiHeadAttWindow(4, 1.0f, 0, 2);
    testMultiHeadAttWindow(12, 1.0f, 0, 2); // window == pos + 1 must equal full attention
    testMultiHeadAttWindow(1, 1.0f, 0, 2); // attends to itself only
    testMultiHeadAttWindow(4, 1.0f, 2, 2); // replicated KV cache: this node owns global heads 2..3

    testRopePartial(16, 1); // full rotary
    testRopePartial(16, 2);
    testRopePartial(4, 1); // partial rotary (Gemma 4 full-attention layers: 0.25 * headDim)
    testRopePartial(4, 2);

    testSoftcap();
    testScalarMul();
    testMergeSet(F_32);
    testMergeSet(F_Q80);
    return 0;
}
