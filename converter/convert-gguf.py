import sys
import os
import struct
import time
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

# Converts a Gemma 4 (dense text) GGUF file, e.g. google/gemma-4-12B-it-qat-q4_0-gguf, into the
# Distributed Llama model format. Q4_0 tensors are passed through untouched: the ggml `block_q4_0`
# layout (fp16 scale + 16 bytes of nibbles, low nibble = element j, high nibble = element j + 16,
# value = (q - 8) * d) is identical to Distributed Llama's `NnBlockQ40`. Every other quantized
# tensor (the Q6_K token embedding in Google's QAT files) is dequantized with gguf-py.
#
# This script deliberately does not depend on torch, so it also runs on a Raspberry Pi.

ARCH_TYPE_GEMMA4 = 0xABCD03
HIDDEN_ACT_GELU = 0
FLOAT_TYPE_F32 = 0
FLOAT_TYPE_Q40 = 2

Q40_BLOCK_SIZE = 32
Q40_BLOCK_DTYPE = np.dtype([('d', '<f2'), ('qs', 'u1', (Q40_BLOCK_SIZE // 2,))])

HEADER_KEYS = {
    'version': 0,
    'arch_type': 1,
    'dim': 2,
    'hidden_dim': 3,
    'n_layers': 4,
    'n_heads': 5,
    'n_kv_heads': 6,
    'n_experts': 7,
    'n_active_experts': 8,
    'vocab_size': 9,
    'max_seq_len': 10,
    'hidden_act': 11,
    'rope_theta': 12,
    'weights_float_type': 13,
    'head_dim': 19,
    'norm_epsilon': 20,
    'sliding_window': 22,
    'full_att_interval': 23,
    'rope_theta_swa': 24,
    'rope_dims_full': 25,
    'head_dim_full': 26,
    'n_kv_heads_full': 27,
    'final_logit_softcap': 28,
}

def quantizeQ40(x: np.ndarray) -> np.ndarray:
    # Same algorithm as converter/writer.py (and ggml's quantize_row_q4_0_ref)
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.size % Q40_BLOCK_SIZE == 0
    groups = x.reshape(-1, Q40_BLOCK_SIZE)
    gmax = np.max(groups, axis=1)
    gmin = np.min(groups, axis=1)
    deltas = np.divide(np.where(-gmin > gmax, gmin, gmax), -8)
    deltas16 = deltas.astype(np.float16)
    ids = np.where(deltas != 0, 1.0 / deltas, 0)
    q = np.add(groups * ids[:, np.newaxis], 8.5)
    q = np.clip(q, 0, 15).astype(np.uint8)
    blocks = np.empty(groups.shape[0], dtype=Q40_BLOCK_DTYPE)
    blocks['d'] = deltas16
    blocks['qs'] = (q[:, :Q40_BLOCK_SIZE // 2] & 0xF) | ((q[:, Q40_BLOCK_SIZE // 2:] & 0xF) << 4)
    return blocks

def writeHeader(file, params):
    header = struct.pack('i', 0xA00ABCD)
    data = b''
    for key, value in params.items():
        if key not in HEADER_KEYS:
            raise Exception(f'Unknown header key: {key}')
        data += struct.pack('ii', HEADER_KEYS[key], int(value))
    header += struct.pack('i', len(header) * 2 + len(data))
    file.write(header)
    file.write(data)
    for key, value in params.items():
        print(f'🎓 {key}: {value}')
    print()

def parseRmsNormEpsilon(epsilon: float):
    if abs(epsilon - 1e-05) < 1e-12:
        return 5
    if abs(epsilon - 1e-06) < 1e-13:
        return 6
    raise Exception(f'Unsupported epsilon: {epsilon}')

class Gemma4Gguf:
    def __init__(self, path: str):
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}

    def field(self, name: str, default=None):
        f = self.reader.fields.get(name)
        if f is None:
            if default is not None:
                return default
            raise Exception(f'Missing GGUF field: {name}')
        return f.contents()

    def tensor(self, name: str):
        t = self.tensors.get(name)
        if t is None:
            raise Exception(f'Missing GGUF tensor: {name}')
        return t

    def has(self, name: str) -> bool:
        return name in self.tensors

def loadConfig(gguf: Gemma4Gguf):
    arch = gguf.field('general.architecture')
    if arch != 'gemma4':
        raise Exception(f'Unsupported GGUF architecture: {arch}')
    p = 'gemma4.'

    nLayers = int(gguf.field(p + 'block_count'))
    nHeads = int(gguf.field(p + 'attention.head_count'))
    headCountKv = gguf.field(p + 'attention.head_count_kv')
    if not isinstance(headCountKv, list):
        headCountKv = [int(headCountKv)] * nLayers
    swaPattern = [bool(v) for v in gguf.field(p + 'attention.sliding_window_pattern')]
    if len(swaPattern) != nLayers or len(headCountKv) != nLayers:
        raise Exception('Unexpected per-layer array lengths')

    fullLayers = [i for i, isSwa in enumerate(swaPattern) if not isSwa]
    if len(fullLayers) == 0:
        raise Exception('Expected at least one full_attention layer')
    interval = fullLayers[0] + 1
    for i in range(nLayers):
        if ((i + 1) % interval == 0) != (not swaPattern[i]):
            raise Exception(f'Sliding window pattern is not periodic with interval {interval}: {swaPattern}')

    nKvHeadsSwa = set(headCountKv[i] for i in range(nLayers) if swaPattern[i])
    nKvHeadsFull = set(headCountKv[i] for i in fullLayers)
    if len(nKvHeadsSwa) != 1 or len(nKvHeadsFull) != 1:
        raise Exception(f'Expected one kv head count per layer type, got {headCountKv}')

    if int(gguf.field(p + 'attention.shared_kv_layers', 0)) != 0:
        raise Exception('Shared kv layers are not supported')
    if int(gguf.field(p + 'embedding_length_per_layer_input', 0)) != 0:
        raise Exception('Per-layer input embeddings are not supported')
    if gguf.has('blk.0.ffn_gate_inp.weight'):
        raise Exception('MoE variants are not supported')

    headDimFull = int(gguf.field(p + 'attention.key_length'))
    headDimSwa = int(gguf.field(p + 'attention.key_length_swa'))
    if int(gguf.field(p + 'attention.value_length')) != headDimFull or int(gguf.field(p + 'attention.value_length_swa')) != headDimSwa:
        raise Exception('Expected key_length == value_length')
    if int(gguf.field(p + 'rope.dimension_count')) != headDimFull:
        raise Exception('Expected rope.dimension_count == key_length (proportional rope is encoded by rope_freqs)')

    # Proportional rope: rope_freqs.weight holds a factor per rotation pair (1.0 = rotated, huge = frozen)
    ropeFreqs = np.array(gguf.tensor('rope_freqs.weight').data, dtype=np.float32).reshape(-1)
    if ropeFreqs.shape[0] != headDimFull // 2:
        raise Exception(f'Unexpected rope_freqs size {ropeFreqs.shape}')
    rotated = ropeFreqs == 1.0
    frozen = ropeFreqs >= 1e20
    nRotated = int(np.sum(rotated))
    if not (np.all(rotated[:nRotated]) and np.all(frozen[nRotated:])):
        raise Exception('rope_freqs is not a [1.0]*k + [huge]*m pattern, cannot express it with ROPE_DIMS_FULL')

    softcap = float(gguf.field(p + 'final_logit_softcapping', 0.0))
    if softcap != int(softcap):
        raise Exception(f'final_logit_softcapping must be an integer for the header: {softcap}')

    return {
        'version': 0,
        'arch_type': ARCH_TYPE_GEMMA4,
        'hidden_act': HIDDEN_ACT_GELU,
        'dim': int(gguf.field(p + 'embedding_length')),
        'hidden_dim': int(gguf.field(p + 'feed_forward_length')),
        'n_layers': nLayers,
        'n_heads': nHeads,
        'n_kv_heads': int(next(iter(nKvHeadsSwa))),
        'n_experts': 0,
        'n_active_experts': 0,
        'vocab_size': int(gguf.field(p + 'vocab_size', len(gguf.field('tokenizer.ggml.tokens')))),
        'max_seq_len': int(gguf.field(p + 'context_length')),
        'weights_float_type': FLOAT_TYPE_Q40,
        'rope_theta': int(gguf.field(p + 'rope.freq_base')),
        'head_dim': headDimSwa,
        'norm_epsilon': parseRmsNormEpsilon(float(gguf.field(p + 'attention.layer_norm_rms_epsilon'))),
        'sliding_window': int(gguf.field(p + 'attention.sliding_window')),
        'full_att_interval': interval,
        'rope_theta_swa': int(gguf.field(p + 'rope.freq_base_swa')),
        'rope_dims_full': nRotated * 2,
        'head_dim_full': headDimFull,
        'n_kv_heads_full': int(next(iter(nKvHeadsFull))),
        'final_logit_softcap': int(softcap),
    }

class Gemma4Writer:
    def __init__(self, gguf: Gemma4Gguf, config, outputFile):
        self.gguf = gguf
        self.config = config
        self.file = outputFile

    def __writeBytes(self, name: str, data: bytes):
        self.file.write(data)
        print(f'🔶 {name}: {len(data)} bytes')

    def __writeF32(self, name: str, array: np.ndarray):
        array = np.ascontiguousarray(array, dtype=np.float32)
        self.__writeBytes(name, array.tobytes())

    def __writeQ40Tensor(self, name: str, shape):
        t = self.gguf.tensor(name)
        if t.tensor_type != GGMLQuantizationType.Q4_0:
            raise Exception(f'{name}: expected Q4_0, got {t.tensor_type.name}')
        dims = tuple(int(d) for d in t.shape)
        if dims != tuple(shape):
            raise Exception(f'{name}: expected shape {shape}, got {dims}')
        self.__writeBytes(f'{name} {dims} Q4_0', np.ascontiguousarray(t.data).tobytes())

    def __writeF32Tensor(self, name: str, shape):
        t = self.gguf.tensor(name)
        if t.tensor_type != GGMLQuantizationType.F32:
            raise Exception(f'{name}: expected F32, got {t.tensor_type.name}')
        dims = tuple(int(d) for d in t.shape)
        if dims != tuple(shape):
            raise Exception(f'{name}: expected shape {shape}, got {dims}')
        self.__writeF32(f'{name} {dims} F32', np.array(t.data, dtype=np.float32))

    def __embeddingRows(self, chunkRows: int):
        t = self.gguf.tensor('token_embd.weight')
        dim = self.config['dim']
        vocabSize = self.config['vocab_size']
        dims = tuple(int(d) for d in t.shape)
        if dims != (dim, vocabSize):
            raise Exception(f'token_embd.weight: expected shape {(dim, vocabSize)}, got {dims}')
        data = t.data
        for start in range(0, vocabSize, chunkRows):
            end = min(start + chunkRows, vocabSize)
            if t.tensor_type == GGMLQuantizationType.F32:
                rows = np.array(data[start:end], dtype=np.float32)
            elif t.tensor_type == GGMLQuantizationType.F16:
                rows = np.array(data[start:end], dtype=np.float16).astype(np.float32)
            else:
                rows = dequantize(np.ascontiguousarray(data[start:end]), t.tensor_type)
            rows = rows.reshape(end - start, dim)
            yield rows

    def __writeEmbedding(self):
        # Gemma scales the input embeddings by sqrt(dim) (f32 like llama.cpp; HF rounds the scale to bf16)
        scale = np.float32(np.sqrt(np.float32(self.config['dim'])))
        t = self.gguf.tensor('token_embd.weight')
        t0 = time.time()
        nBytes = 0
        for rows in self.__embeddingRows(8192):
            b = (rows * scale).astype(np.float32).tobytes()
            self.file.write(b)
            nBytes += len(b)
        print(f'🔶 token_embd.weight ({t.tensor_type.name}) -> embedding * sqrt(dim) F32: {nBytes} bytes in {time.time() - t0:.1f}s')

    def __writeTiedLmHead(self):
        t = self.gguf.tensor('token_embd.weight')
        t0 = time.time()
        nBytes = 0
        if self.gguf.has('output.weight'):
            raise Exception('Untied output.weight is not expected for Gemma 4')
        for rows in self.__embeddingRows(8192):
            b = quantizeQ40(rows).tobytes()
            self.file.write(b)
            nBytes += len(b)
        print(f'🔶 token_embd.weight ({t.tensor_type.name}) -> tied lm_head Q4_0: {nBytes} bytes in {time.time() - t0:.1f}s')

    def write(self):
        c = self.config
        dim = c['dim']
        ff = c['hidden_dim']
        nLayers = c['n_layers']
        nHeads = c['n_heads']

        self.__writeEmbedding()

        for l in range(nLayers):
            isFull = (l + 1) % c['full_att_interval'] == 0
            headDim = c['head_dim_full'] if isFull else c['head_dim']
            nKvHeads = c['n_kv_heads_full'] if isFull else c['n_kv_heads']
            qDim = nHeads * headDim
            kvDim = nKvHeads * headDim
            b = f'blk.{l}.'

            # Must match loadGemma4LlmNetWeight() in src/llm.cpp
            self.__writeQ40Tensor(b + 'attn_q.weight', (dim, qDim))
            self.__writeQ40Tensor(b + 'attn_k.weight', (dim, kvDim))
            if isFull:
                if self.gguf.has(b + 'attn_v.weight'):
                    raise Exception(f'Layer {l}: full_attention layer with a v_proj is not supported (expected attention_k_eq_v)')
            else:
                self.__writeQ40Tensor(b + 'attn_v.weight', (dim, kvDim))
            self.__writeQ40Tensor(b + 'attn_output.weight', (qDim, dim))

            self.__writeQ40Tensor(b + 'ffn_gate.weight', (dim, ff))
            self.__writeQ40Tensor(b + 'ffn_down.weight', (ff, dim))
            self.__writeQ40Tensor(b + 'ffn_up.weight', (dim, ff))

            # Google's GGUF stores the norm weights already shifted by +1 (Gemma RMSNorm scales by 1 + w)
            self.__writeF32Tensor(b + 'attn_q_norm.weight', (headDim,))
            self.__writeF32Tensor(b + 'attn_k_norm.weight', (headDim,))
            self.__writeF32(b + 'v_norm (ones)', np.ones(headDim, dtype=np.float32))

            self.__writeF32Tensor(b + 'attn_norm.weight', (dim,))
            self.__writeF32Tensor(b + 'post_attention_norm.weight', (dim,))
            self.__writeF32Tensor(b + 'ffn_norm.weight', (dim,))
            self.__writeF32Tensor(b + 'post_ffw_norm.weight', (dim,))

            if self.gguf.has(b + 'layer_output_scale.weight'):
                scale = np.array(self.gguf.tensor(b + 'layer_output_scale.weight').data, dtype=np.float32).reshape(-1)
                if not np.all(scale == 1.0):
                    raise Exception(f'Layer {l}: layer_output_scale != 1.0 is not supported: {scale}')

        self.__writeF32Tensor('output_norm.weight', (dim,))
        self.__writeTiedLmHead()

def printUsage():
    print('Usage: python convert-gguf.py <ggufPath> <name>')
    print()
    print('Options:')
    print('  <ggufPath> The path to a Gemma 4 GGUF file with Q4_0 weights (e.g. google/gemma-4-12B-it-qat-q4_0-gguf)')
    print('  <name>     The name of the model (e.g. "gemma4_12b_it")')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        printUsage()
        exit(1)

    ggufPath = sys.argv[1]
    name = sys.argv[2]
    outputFileName = f'dllama_model_{name}_q40.m'
    print(f'Output file: {outputFileName}')

    gguf = Gemma4Gguf(ggufPath)
    config = loadConfig(gguf)

    with open(outputFileName, 'wb') as outputFile:
        writeHeader(outputFile, config)
        Gemma4Writer(gguf, config, outputFile).write()

    print(f'✅ {outputFileName} created successfully ({os.path.getsize(outputFileName)} bytes)')
