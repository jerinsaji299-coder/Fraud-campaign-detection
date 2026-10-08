/** Plain-language definitions shown in tooltips on technical terms. */

export const GLOSSARY: Record<string, string> = {
  campaign:
    'One money-laundering attempt from the dataset’s ground-truth file: a group of transactions that belong to the same scheme.',
  visibility:
    'The largest share of a campaign’s transactions that any single institution can see. 100% means one bank sees the whole thing; 25% means the best-placed bank sees only a quarter.',
  'lead time':
    'How far ahead of a campaign’s true completion it was discovered. Measured in hours, as a share of the campaign’s duration, and as the share of its transactions seen so far.',
  'hub pattern':
    'A campaign shaped around one central account (fan-in, fan-out, gather-scatter). That account’s own bank always sees every transaction, so visibility cannot be reduced — these act as a control group.',
  fragmentable:
    'A campaign whose accounts can be spread across institutions so that no single one sees all of it — cycles, chains, bipartite and random patterns.',
  unfragmentable:
    'A fragmentable-shaped campaign that in practice still cannot be pushed below 90% visibility, usually because it is small. Grouped with the hub control group for analysis.',
  fedavg:
    'Federated averaging: institutions train locally and share only model weights, never raw data. Measures the benefit of shared knowledge.',
  'embedding exchange':
    'On top of shared weights, institutions also exchange compact numeric summaries of the accounts that sit on their boundary, at detection time. Measures the benefit of shared evidence.',
  institution:
    'A simulated bank group. The dataset’s 30k+ banks are divided into 8 institutions of roughly equal transaction volume.',
  'eval_ok':
    'A campaign large enough to evaluate: at least 3 transactions and at least 3 accounts. Smaller ones make lead time meaningless.',
  cutoff:
    'Sept 11 2022. Transactions from then on are excluded from model input because 59% of that tail is laundering, versus about 0.1% overall — a model could cheat on it.',
  censored:
    'A campaign that had not finished by the cutoff, so its later transactions are not available as evidence.',
  'shared account':
    'An account that appears in both a training campaign and a test campaign. A model might recognise it instead of generalising, so results are reported with and without these.',
}

export function glossaryLookup(term: string): string | undefined {
  return GLOSSARY[term.toLowerCase()]
}
