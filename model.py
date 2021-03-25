#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
import torch.nn.functional as F

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind, unkTokenIdx, padTokenIdx):
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w, unkTokenIdx) for w in s] for s in source]
        sents_padded = [s + (m - len(s)) * [padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=self.device))

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName, map_location=torch.device('cpu')))

    def __init__(self, embedSize, hidSize, dropout, layers, device, word2indENG, word2indBG, unkToken,
                 padToken, endToken, startToken):
        super(NMTmodel, self).__init__()
        self.embedSize = embedSize
        self.unkTokenENIdx = word2indENG[unkToken]
        self.padTokenENIdx = word2indENG[padToken]
        self.endTokenENIdx = word2indENG[endToken]
        self.unkTokenBGIdx = word2indBG[unkToken]
        self.padTokenBGIdx = word2indBG[padToken]
        self.endTokenBGIdx = word2indBG[endToken]
        self.startTokenBGIdx = word2indBG[startToken]
        self.word2indENG = word2indENG
        self.word2indBG = word2indBG
        self.device = device
        self.encoder = torch.nn.LSTM(embedSize, hidSize, num_layers=layers, bidirectional=True)
        self.decoder = torch.nn.LSTM(embedSize, hidSize, num_layers=layers * 2)
        self.embedEN = torch.nn.Embedding(len(word2indENG), embedSize)
        self.embedBG = torch.nn.Embedding(len(word2indBG), embedSize)
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(hidSize, len(word2indBG))
        self.attention = torch.nn.Linear(hidSize * 3, hidSize)
        self.multiplicativeWeights = torch.nn.Parameter(torch.randn(hidSize, hidSize * 2, device=device))
    def forward(self, source, target):
        X = self.preparePaddedBatch(source, self.word2indENG, self.unkTokenENIdx, self.padTokenENIdx)
        XE = self.embedEN(X)
        Y = self.preparePaddedBatch(target, self.word2indBG, self.unkTokenBGIdx, self.padTokenBGIdx)
        YE = self.embedBG(Y[:-1])

        ###Encoder
        sourceLengths = [len(s) for s in source]
        outputPackedSource, (hiddenSource, stateSource) = self.encoder(
            torch.nn.utils.rnn.pack_padded_sequence(XE, sourceLengths, enforce_sorted=False))
        outputSource, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedSource)
        ###

        ###Decoder
        targetLengths = [len(t) - 1 for t in target]
        outputPackedTarget, (hiddenTarget, stateSource) = self.decoder(
            torch.nn.utils.rnn.pack_padded_sequence(YE, targetLengths, enforce_sorted=False),
            (hiddenSource, stateSource))
        outputTarget, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedTarget)
        ###

        ###Attention
        attentionWeights = F.softmax(outputTarget.permute(1, 0, 2) @ self.multiplicativeWeights @ outputSource.permute(1, 2, 0) , dim=2)
        contextVector = torch.bmm(outputSource.permute(1, 2, 0), attentionWeights.permute(0, 2, 1)).permute(2, 0, 1)
        outputTarget = self.attention(torch.cat((contextVector, outputTarget), dim=-1))
        ###
        outputTarget = self.projection(self.dropout(outputTarget.flatten(0, 1)))

        return F.cross_entropy(outputTarget, Y[1:].flatten(0, 1), ignore_index=self.padTokenBGIdx)

    def translateSentence(self, sentence, limit=1000):
        ind2word = dict(enumerate(self.word2indBG))
        X = self.preparePaddedBatch([sentence], self.word2indENG, self.unkTokenENIdx, self.padTokenENIdx)
        XE = self.embedEN(X)

        outputSource, (hiddenSource, stateSource) = self.encoder(XE)
        decoderOutputs = []
        decoderInput = torch.tensor([[self.startTokenBGIdx]], device=self.device)
        hiddenTarget = hiddenSource
        stateTarget = stateSource
        for _ in range(limit):
            outputTarget = self.embedBG(decoderInput)
            outputTarget, (hiddenTarget, stateTarget) = self.decoder(outputTarget, (hiddenTarget, stateTarget))

            attentionWeights = F.softmax(outputTarget.permute(1, 0, 2) @ self.multiplicativeWeights @ outputSource.permute(1, 2, 0) , dim=2)
            contextVector = torch.bmm(outputSource.permute(1, 2, 0), attentionWeights.permute(0, 2, 1)).permute(2, 0, 1)
            outputTarget = self.attention(torch.cat((contextVector, outputTarget), dim=-1))

            outputTarget = self.projection(self.dropout(outputTarget.flatten(0, 1)))
            topv, topi = outputTarget.data.topk(1)
            currentWordIndex = topi[0].item()

            if currentWordIndex == self.endTokenBGIdx:
                break
            else:
                decoderOutputs.append(ind2word[currentWordIndex])
                decoderInput = torch.tensor([[currentWordIndex]], device=self.device)
        return decoderOutputs


