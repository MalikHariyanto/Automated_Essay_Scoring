class PromptAwareHeads(IndependentTraitHeads):
    def __init__(self, dropout):
        super().__init__(dropout)
        self.question_query = nn.Linear(768, 4 * 64, bias=False)
        self.answer_key = nn.Linear(768, 64, bias=False)
        self.prompt_residual = nn.Linear(768, 64, bias=False)
        self.trait_residual = nn.Linear(64, 64, bias=False)
        nn.init.zeros_(self.prompt_residual.weight)
        nn.init.zeros_(self.trait_residual.weight)

    def forward(self, fusion, tokens, question_mask, answer_mask, mapping):
        prompt = (tokens * question_mask.unsqueeze(-1)).sum(dim=1) / question_mask.sum(1, keepdim=True)
        q = self.question_query(prompt).reshape(len(tokens), 4, 64)
        k = self.answer_key(tokens)
        logits = q.float() @ k.float().transpose(1, 2) / 8
        weights = logits.masked_fill(~answer_mask[:, None, :], float('-inf')).softmax(dim=-1)
        pooled = weights.to(tokens.dtype) @ tokens
        context = mean_by_essay(pooled.flatten(1), mapping, len(fusion)).reshape(len(fusion), 4, 768)
        signal = self.prompt_residual(context)
        hidden = torch.stack([
            self.heads[name][2](self.heads[name][1](self.heads[name][0](fusion) + signal[:, i]))
            for i, (name, *_) in enumerate(TRAITS)
        ], dim=1)
        other_logits = hidden.float() @ hidden.float().transpose(1, 2) / 8
        diagonal = torch.eye(4, dtype=torch.bool, device=hidden.device)
        other_weights = other_logits.masked_fill(diagonal, float('-inf')).softmax(dim=-1)
        updated = hidden + self.trait_residual(other_weights.to(hidden.dtype) @ hidden)
        return torch.cat([self.heads[name][3](updated[:, i]) for i, (name, *_) in enumerate(TRAITS)], dim=1)


def prompt_answer_masks(details, width, padding_side, device):
    question, answer = [], []
    for info in details:
        for segments, positions in zip(info['segments'], info['answer_positions']):
            prompt_positions = [i for i, label in enumerate(segments)
                                if label == 'Soal/referensi'][:info['question_kept']]
            if len(prompt_positions) != info['question_kept'] or not prompt_positions or not positions:
                raise ValueError('Token soal dan jawaban harus tersedia untuk attention.')
            offset = width - len(segments) if padding_side == 'left' else 0
            question.append([i - offset in prompt_positions for i in range(width)])
            answer.append([i - offset in positions for i in range(width)])
    return (torch.tensor(question, dtype=torch.bool, device=device),
            torch.tensor(answer, dtype=torch.bool, device=device))


def correlated_trait_pairs(labels, threshold):
    # Correlation uses only the training partition of the current fit.
    centered = labels.astype(np.float64) - labels.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    denominators = np.outer(norm, norm)
    corr = np.divide(centered.T @ centered, denominators,
                     out=np.zeros_like(denominators), where=denominators > 0)
    return np.nonzero(np.triu(corr >= threshold, k=1))


def relation_error(prediction, targets, pairs):
    left, right = pairs
    if left.numel() == 0:
        return prediction.new_zeros(())
    error = prediction - targets
    return (error[:, left] - error[:, right]).square().mean()
