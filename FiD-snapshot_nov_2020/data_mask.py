import torch
import random
import json
import pdb

class QAExample():
    def __init__(self, id, question, answers, target=None, titles=None, contexts=None):
        self.id = id
        self.question = question
        self.answers = answers
        self.target = target
        self.titles = titles
        self.contexts = contexts
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, n_context, tokenizer, max_passage_length=250, no_title=False):
        self.data = data
        self.n_context = n_context
        self.tokenizer = tokenizer
        self.max_passage_length = max_passage_length
        self.no_title = no_title
        self.question_prefix = 'question:'
        self.title_prefix = 'title:'
        self.context_prefix = 'context:'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example.question
        if example.target is None:
            target = random.choice(example.answers)
        else:
            target = example.target

        titles = example.titles[:self.n_context]
        contexts = example.contexts[:self.n_context]

        passages = []
        for i in range(min(self.n_context, len(contexts))):
            c = contexts[i]
            t = titles[i]
            to_concatenate = [self.question_prefix, question]
            if c is not None:
                if not self.no_title:
                    to_concatenate += [self.title_prefix, t]
                to_concatenate += [self.context_prefix, c]
            text = ' '.join(to_concatenate)
            passages.append(text)
        # import pdb
        # pdb.set_trace()

        for _ in range(len(contexts), self.n_context):
            to_concatenate = [self.question_prefix, question] 
            text = ' '.join(to_concatenate)
            passages.append(text)

        return {'index':index, 'question':question, 'target':target, 'passages':passages}

    def get_example(self, index):
        return self.data[index]


class Collator(object):
    def __init__(self, opt, tokenizer):
        self.tokenizer = tokenizer
        self.max_passage_length = opt.max_passage_length
        self.model_type = opt.model_type

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])
        question = [ex['question'] for ex in batch]
        if self.model_type == 'bart':
            target = [ex['target'] for ex in batch]
        else:
            target = [ex['target'] + ' </s>' for ex in batch]
        target = self.tokenizer.batch_encode_plus(target, pad_to_max_length=True, return_tensors="pt")
        target_ids, target_mask = target["input_ids"], target["attention_mask"]

        batch_text_passages = [ex['passages'] for ex in batch]
        batch_encoded_passages = []
        # import pdb
        # pdb.set_trace()
        max_context_length = 0
        for k, text_passages in enumerate(batch_text_passages):
            encoded_passages = []
            for text_p in text_passages:
                # encoded_p = self.tokenizer.encode(text_p,padding='max_length')
                encoded_p = self.tokenizer.encode(text_p)
                if len(encoded_p) > self.max_passage_length:
                    encoded_p = encoded_p[:self.max_passage_length]
                max_context_length = max(max_context_length, len(encoded_p)) 
                encoded_passages.append(encoded_p)
            batch_encoded_passages.append(encoded_passages)
        max_context_length = min(max_context_length, self.max_passage_length)

        batch_passage_ids, batch_passage_masks = [], []
        for k, encoded_passages in enumerate(batch_encoded_passages):
            p_ids, p_masks = [], []
            for p in encoded_passages:
                plen = len(p)
                m = torch.ones(plen) 
                c = torch.cat((torch.tensor(p), torch.zeros(max_context_length-plen).long()), dim=0)
                m = torch.cat((m.bool(), torch.zeros(max_context_length-plen).bool()), dim=0) 
                p_ids.append(c[None])
                p_masks.append(m[None])
            p_ids = torch.cat(p_ids, dim=0)
            p_masks = torch.cat(p_masks, dim=0)
            batch_passage_ids.append(p_ids[None])
            batch_passage_masks.append(p_masks[None])
        
        batch_passage_ids = torch.cat(batch_passage_ids, dim=0)
        batch_passage_masks = torch.cat(batch_passage_masks, dim=0)

        # import pdb
        # pdb.set_trace()
        return (index, target_ids, target_mask, batch_passage_ids, batch_passage_masks)


def load_data(data_path, global_rank=-1, world_size=-1):
    with open(data_path, "r") as f:
        data = json.load(f)

    examples = [] 
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if 'id' in example:
            id = example['id']
        else:
            id = k
        if 'target' in example:
            target = example['target']
        else:
            target = None
        answers = example['answers']
        question = example['question']
        ctxs = example['ctxs']
        titles, contexts = [], []
        for i, c in enumerate(ctxs):
            titles.append(c['title'])
            contexts.append(c['text'])
        ex = QAExample(id=id, question=question, answers=answers, target=target, titles=titles, contexts=contexts)
        examples.append(ex)
        # pdb.set_trace()
    return examples
