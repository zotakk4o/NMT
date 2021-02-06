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
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w, unkTokenIdx) for w in s] for s in source]
        sents_padded = [s + (m - len(s)) * [padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName,  map_location=torch.device('cpu')))

    def __init__(self, embedSize, hidSize, dropout, encoderLayers, decoderLayers, word2indENG, word2indBG, unkToken,
                 padToken, endToken):
        super(NMTmodel, self).__init__()
        self.embedSize = embedSize
        self.hiddenSzie = hidSize
        self.unkTokenENIdx = word2indENG[unkToken]
        self.padTokenENIdx = word2indENG[padToken]
        self.endTokenENIdx = word2indENG[endToken]
        self.unkTokenBGIdx = word2indBG[unkToken]
        self.padTokenBGIdx = word2indBG[padToken]
        self.endTokenBGIdx = word2indBG[endToken]
        self.word2indENG = word2indENG
        self.word2indBG = word2indBG
        self.encoder = torch.nn.LSTM(embedSize, hidSize, encoderLayers)
        self.decoder = torch.nn.LSTM(embedSize, hidSize, decoderLayers)
        self.embedEN = torch.nn.Embedding(len(word2indENG), embedSize)
        self.embedBG = torch.nn.Embedding(len(word2indBG), embedSize)
        self.attn_combine = torch.nn.Linear(hidSize * 2, hidSize)
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(hidSize, len(word2indBG))

    def forward(self, source, target):
        X = self.preparePaddedBatch(source, self.word2indENG, self.unkTokenENIdx, self.padTokenENIdx)
        XE = self.embedEN(X[:-1])
        Y = self.preparePaddedBatch(target, self.word2indBG, self.unkTokenBGIdx, self.padTokenBGIdx)
        YE = self.embedBG(Y[:-1])

        source_lengths = [len(s) - 1 for s in source]
        outputPackedSource, (hiddenSource, _) = self.encoder(
            torch.nn.utils.rnn.pack_padded_sequence(XE, source_lengths, enforce_sorted=False))
        outputSource, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedSource)
        outputSource = outputSource.flatten(0, 1)

        target_lengths = [len(t) - 1 for t in target]
        outputPackedTarget, (hiddenTarget, _) = self.decoder(
            torch.nn.utils.rnn.pack_padded_sequence(YE, target_lengths, enforce_sorted=False),
            (hiddenSource, torch.zeros(hiddenSource.size()).to(next(self.parameters()).device)))
        outputTarget, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedTarget)
        outputTarget = outputTarget.flatten(0, 1)

        attnWeights = F.softmax(torch.matmul(outputTarget, outputSource.transpose(0, 1)), dim=1)
        outputTarget = outputTarget.add(torch.matmul(attnWeights, outputSource))
        outputTarget = self.projection(self.dropout(outputTarget))
        return torch.nn.functional.cross_entropy(outputTarget, Y[1:].flatten(0, 1), ignore_index=self.padTokenBGIdx)

    def translateSentence(self, sentence, limit=1000):
        ind2wordBG = dict(enumerate(self.word2indBG))
        device = next(self.parameters()).device
        X = self.preparePaddedBatch([[sentence[0]]], self.word2indENG, self.unkTokenENIdx, self.padTokenENIdx)
        XE = self.embedEN(X)
        outputSource, (hiddenSource, cSource) = self.encoder(XE)

        for i in range(1, len(sentence)):
            X = self.preparePaddedBatch([[sentence[i]]], self.word2indENG, self.unkTokenENIdx, self.padTokenENIdx)
            XE = self.embedEN(X)
            outputSource, (hiddenSource, cSource) = self.encoder(XE, (hiddenSource, cSource))

        decoder_input = [["<S>"]]
        decoded_words = []
        Y = self.preparePaddedBatch(decoder_input, self.word2indBG, self.unkTokenBGIdx, self.padTokenBGIdx)
        YE = self.embedBG(Y)
        outputTarget, (hiddenTarget, c) = self.decoder(YE, (
            hiddenSource, torch.zeros(hiddenSource.size()).to(device)))
        outputTarget = self.projection(self.dropout(outputTarget.flatten(0, 1)))
        topv, topi = outputTarget.data.topk(1)
        decoded_words.append(ind2wordBG[topi[0].item()])
        decoder_input = [[ind2wordBG[topi[0].item()]]]
        for di in range(limit):
            Y = self.preparePaddedBatch(decoder_input, self.word2indBG, self.unkTokenBGIdx, self.padTokenBGIdx)
            YE = self.embedBG(Y)
            outputTarget, (hiddenTarget, c) = self.decoder(YE, (hiddenTarget, c))
            outputTarget = self.projection(self.dropout(outputTarget.flatten(0, 1)))
            topv, topi = outputTarget.data.topk(1)
            if topi[0].item() == self.endTokenBGIdx:
                break
            else:
                decoded_words.append(ind2wordBG[topi[0].item()])

            decoder_input = [[ind2wordBG[topi[0].item()]]]

        return decoded_words
