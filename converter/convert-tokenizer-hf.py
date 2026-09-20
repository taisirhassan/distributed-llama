import sys
import json
import os
writer = __import__('tokenizer-writer')

def openJson(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def unicodeToBytes():
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))

class TokensResolver:
    def __init__(self, dirPath, tokenizerConfig):
        self.dirPath = dirPath
        self.tokenizerConfig = tokenizerConfig
        self.bosId = None
        self.eosIds = None
        self.tokens = []
        self.scores = []
        self.specialIds = None

    def resolvePreTrainedTokenizerFast(self):
        from transformers import PreTrainedTokenizerFast
        utb = unicodeToBytes()
        tokenizer = PreTrainedTokenizerFast(tokenizer_file = os.path.join(self.dirPath, 'tokenizer.json'))
        config = openJson(os.path.join(self.dirPath, 'config.json'))
        vocabLen = len(tokenizer.get_vocab())
        for i in range(vocabLen):
            tokenChars = list(tokenizer.convert_ids_to_tokens([i])[0])
            tokenBytes = []
            for chr in tokenChars:
                if (chr in utb):
                    tokenBytes.append(utb[chr])
                else:
                    tokenBytes += list(chr.encode('utf-8'))
            self.tokens.append(bytes(tokenBytes))
            self.scores.append(-float(i))

        # Pad tokenizer vocab to match model vocab_size if needed
        targetVocabSize = config.get('vocab_size', vocabLen)
        if targetVocabSize > vocabLen:
            print(f'⚠️ Padding tokenizer vocab from {vocabLen} to {targetVocabSize}')
            for i in range(vocabLen, targetVocabSize):
                self.tokens.append(f'<|reserved_{i}|>'.encode('utf-8'))
                self.scores.append(-float(i))

        self.bosId = tokenizer.bos_token_id
        if (tokenizer.eos_token_id):
            self.eosIds = [tokenizer.eos_token_id]
        if (self.bosId is None or self.eosIds is None):
            if (self.bosId is None):
                self.bosId = config['bos_token_id']
            if (self.eosIds is None):
                self.eosIds = config['eos_token_id']
                if isinstance(self.eosIds, list):
                    self.eosIds = self.eosIds
                else:
                    self.eosIds = [self.eosIds]

    def resolveLlamaTokenizer(self):
        from sentencepiece import SentencePieceProcessor
        modelPath = os.path.join(self.dirPath, 'tokenizer.model')
        processor = SentencePieceProcessor(model_file=modelPath)

        assert processor.vocab_size() == processor.get_piece_size()
        self.bosId = processor.bos_id()
        self.eosIds = [processor.eos_id()]
        vocabSize = processor.vocab_size()
        for i in range(vocabSize):
            t = processor.id_to_piece(i)
            s = processor.get_score(i)
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            # Check for byte characters
            if len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
                # For example, "<0x0A>"" is a newline character
                b = bytearray.fromhex(t[3:-1])
            else:
                b = t.encode('utf-8')
            self.tokens.append(b)
            self.scores.append(s)

    def resolveGemmaTokenizer(self):
        # Gemma 3/4 tokenizer.json: BPE with byte fallback, sentencepiece-style "▁" for spaces,
        # <0xXX> byte tokens and special tokens interleaved with regular ones (bos = 2).
        tokenizerJson = openJson(os.path.join(self.dirPath, 'tokenizer.json'))
        model = tokenizerJson['model']
        assert model['type'] == 'BPE', f'Unsupported tokenizer model: {model["type"]}'
        assert model.get('byte_fallback', False), 'Expected byte_fallback'
        vocab = model['vocab']
        vocabLen = len(vocab)
        pieces = [None] * vocabLen
        for piece, tokenId in vocab.items():
            pieces[tokenId] = piece
        assert all(p is not None for p in pieces), 'Vocab ids are not contiguous'

        # The engine merges the adjacent pair whose concatenation has the highest score, so the score of
        # a token is minus the rank of the earliest BPE merge producing it. Tokens no merge produces
        # (single characters, bytes, special/added tokens) get a score that never wins a merge, and the
        # <0xXX> byte-fallback tokens get an even lower one so a text token with the same bytes is preferred.
        neverMerge = -1e11
        byteFallback = -1e12
        scores = [neverMerge] * vocabLen
        for rank, merge in enumerate(model['merges']):
            if isinstance(merge, str):
                merge = merge.split(' ', 1)
            merged = merge[0] + merge[1]
            tokenId = vocab.get(merged)
            if tokenId is not None and scores[tokenId] == neverMerge:
                scores[tokenId] = -float(rank)

        for i in range(vocabLen):
            piece = pieces[i]
            if len(piece) == 6 and piece.startswith('<0x') and piece.endswith('>'):
                b = bytearray.fromhex(piece[3:-1])
                scores[i] = byteFallback
            else:
                b = piece.replace('▁', ' ').encode('utf-8')
            self.tokens.append(bytes(b))
            self.scores.append(scores[i])

        config = openJson(os.path.join(self.dirPath, 'config.json'))
        targetVocabSize = config.get('vocab_size')
        if targetVocabSize is None and 'text_config' in config:
            targetVocabSize = config['text_config'].get('vocab_size')
        if targetVocabSize is not None and targetVocabSize > vocabLen:
            print(f'⚠️ Padding tokenizer vocab from {vocabLen} to {targetVocabSize}')
            for i in range(vocabLen, targetVocabSize):
                self.tokens.append(f'<|reserved_{i}|>'.encode('utf-8'))
                self.scores.append(neverMerge)

        self.specialIds = sorted(t['id'] for t in tokenizerJson.get('added_tokens', []) if t.get('special', False))

        bosToken = self.tokenizerConfig.get('bos_token')
        eosToken = self.tokenizerConfig.get('eos_token')
        self.bosId = vocab[bosToken] if bosToken is not None else config['bos_token_id']
        eosIds = config.get('eos_token_id')
        if eosIds is None and 'text_config' in config:
            eosIds = config['text_config'].get('eos_token_id')
        if eosIds is None:
            eosIds = vocab[eosToken]
        self.eosIds = list(eosIds) if isinstance(eosIds, list) else [eosIds]

    def resolve(self):
        cls = self.tokenizerConfig['tokenizer_class']
        if (cls == 'GemmaTokenizer' or cls == 'GemmaTokenizerFast'):
            return self.resolveGemmaTokenizer()
        if (cls == 'PreTrainedTokenizer' or
            cls == 'PreTrainedTokenizerFast' or
            cls == 'LlamaTokenizerFast' or
            cls == 'Qwen2Tokenizer'):
            return self.resolvePreTrainedTokenizerFast()
        if (cls == 'LlamaTokenizer'):
            return self.resolveLlamaTokenizer()
        raise Exception(f'Tokenizer {cls} is not supported')

def printUsage():
    print('Usage: python convert-tokenizer-hf.py <tokenizerFolderPath> <name>')
    print()
    print('Options:')
    print('  <tokenizerFolderPath> The path to the folder with tokenizer_config.json')
    print('  <name>                The name of the tokenizer (e.g. "llama3")')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        printUsage()
        exit(1)

    dirPath = sys.argv[1]
    name = sys.argv[2]
    tokenizerConfig = openJson(os.path.join(dirPath, 'tokenizer_config.json'))

    resolver = TokensResolver(dirPath, tokenizerConfig)
    resolver.resolve()

    if (resolver.bosId is None or resolver.eosIds is None):
        raise Exception('Cannot resolve bosId or eosIds')
    print(f'bosId: {resolver.bosId} ({resolver.tokens[resolver.bosId]})')
    for eosId in resolver.eosIds:
        print(f'eosId: {eosId} ({resolver.tokens[eosId]})')

    chatTemplate = None
    if ('chat_template' in tokenizerConfig):
        chatTemplate = tokenizerConfig['chat_template'].encode('utf-8')
    elif (os.path.isfile(os.path.join(dirPath, 'chat_template.jinja'))):
        with open(os.path.join(dirPath, 'chat_template.jinja'), 'r', encoding='utf-8') as file:
            chatTemplate = file.read().encode('utf-8')

    addBos = True
    if ('add_bos_token' in tokenizerConfig):
        addBos = tokenizerConfig['add_bos_token']

    outputFileName = f'dllama_tokenizer_{name}.t'
    with open(outputFileName, 'wb') as outputFile:
        writer.writeTokenizer(
            outputFile,
            resolver.tokens,
            resolver.scores,
            chatTemplate,
            resolver.bosId,
            addBos,
            resolver.eosIds,
            resolver.specialIds)
    print(f'✅ Created {outputFileName}')
