import os
import csv
import pickle
from collections import defaultdict
import numpy as np
import faiss
from tqdm import tqdm
from graph_reader import Relation,read_condensed_relations_for_question,read_condensed_hyper_relations_for_question,convert_hyper_relation_to_text,BinaryStream



from collections import Counter

# read WebQSP
def get_answers(question):
    """extract unique answers from question parses."""
    answers = set()
    for parse in question["Parses"]:
        for answer in parse["Answers"]:
            answers.add((answer["AnswerArgument"],
                answer["EntityName"]))
    return answers

def get_entities(question):
    """extract oracle entities from question parses."""
    entities = set()
    for parse in question["Parses"]:
        if parse["TopicEntityMid"] is not None:
            entities.add((parse["TopicEntityMid"], parse["TopicEntityName"]))
    return entities

def get_fid(s):
    return '/' + s.replace('.', '/')

def convert_subgraph_to_passages(
    qid, subgraph, entity_names, MAX_TOKENS, MAX_PASSAGES, IGNORE_CVT
):
    passages = []
    tok_cnt = 0
    sentences = []
    seen_sentences = set()
    
    for relation, _ in subgraph:
        sentence = convert_hyper_relation_to_text(relation, entity_names, ignore_cvt=IGNORE_CVT)
        if sentence is None or sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        num_tokens = len(sentence.split(' '))
        if num_tokens > MAX_TOKENS:
            if " person quotations " in sentence:
                # skip the quotation relations
                continue
            print("sentence too long!", sentence, num_tokens)
        if tok_cnt + num_tokens > MAX_TOKENS:
            # create a new passage
            passage = {
                "text": ' '.join(sentences),
                "title": '',
            }
            passages.append(passage)
            if len(passages) >= MAX_PASSAGES:
                # got enough passages
                break
            sentences = [sentence]
            tok_cnt = num_tokens
        else:
            sentences.append(sentence)
            tok_cnt += num_tokens
    if sentences:
        passage = {
            "text": ' '.join(sentences),
            "title": '',
        }
        passages.append(passage)

    if len(passages) == 0:
        passages.append({"text":'', "title":''})
    # import pdb
    # pdb.set_trace()
    return passages

import json

def save_fid_input(
    pruned_graph_dir, 
    output_dir, 
    MAX_PASSAGES, 
    MAX_TOKENS, 
    IGNORE_CVT=False, 
    convert_subgraph_to_passages=convert_subgraph_to_passages
):
    results = defaultdict(list)
    for split in ['train', 'test']:
        for question in webqsp_questions[split]:
            out_split = split
            qid = question['QuestionId']
            # use WebQTrn-0 to WebQTrn-377 as dev set
            if qid.startswith("WebQTrn-"):
                num = int(qid[8:])
                if num <= 376:
                    out_split = 'dev'
            print("Processing", qid, "...")
            qtext = question['QuestionText']
            answers = [ans['text'] if ans['text'] is not None else ans['freebaseId'] for ans in question['Answers']]
            if not answers:
                answers = ["None"]

            pruned_graph_oufname = os.path.join(pruned_graph_dir, f"{qid}.pkl")
            if os.path.exists(pruned_graph_oufname):
                with open(pruned_graph_oufname, 'rb') as inf:
                    subgraph = pickle.load(inf)
                passages = convert_subgraph_to_passages(
                    qid, subgraph, entity_names, MAX_TOKENS, MAX_PASSAGES, IGNORE_CVT
                )
                ###passages里面村有一个个passage，每个passage里面有多个子图，且token数量小于150个
            else:
                passages = [{"text":'', "title":''}]
            # import pdb
            # pdb.set_trace()
            results[out_split].append({
                "id": qid,
                "question": qtext,
                "answers": answers,
                "ctxs": passages,
            })

    for split, samples in results.items():
        oufname = os.path.join(output_dir, f"webqsp_{split}.json")
        with open(oufname, 'w') as ouf:
            json.dump(samples, ouf)


# Answer Coverage
# check retrieval accuracy at top 10 and top 100

def ans_in_ctxs(answers, ctxs, topk=0):
    # import pdb
    # pdb.set_trace()
    if topk <= 0:
        topk = len(ctxs)
    for ans in answers:
        for ctx in ctxs[:topk]:
            if ans in ctx['text']:
                return True
    return False


def get_ans_recall(fid_input_dir):
    topks = [10,20,50,100,200]
    for split in ['train', 'dev', 'test']:
        infname = os.path.join(fid_input_dir, f"webqsp_{split}.json")
        with open(infname) as inf:
            samples = json.load(inf)
        hit = Counter()
        ttl = 0
        for question in samples:
            ttl += 1
            for topk in topks:
                if ans_in_ctxs(question['answers'], question['ctxs'], topk):
                    hit[topk] += 1
        print("Split:", split)
        for topk in topks:
            print(f"Top {topk} passages, answer found in {hit[topk]}/{ttl} (= {hit[topk]/ttl}) questions")


if __name__ == '__main__':
    '''一些各种各样的路径'''
    # change to your local path
    WEBQSP_DIR = os.path.dirname(os.path.realpath('.'))
    # DATA_DIR = os.path.join(WEBQSP_DIR, "data")
    DATA_DIR = os.path.join(WEBQSP_DIR, "UniK-QA-main","data")
    FREEBASE_DIR = os.path.join(DATA_DIR, "freebase_2hop")

    # read DPR embeddings for condensed relations
    CONDENSED_KB_DIR = os.path.join(FREEBASE_DIR, "condensed.stagg.neighborhoods", "condensed_edges_only")
    # condensed relations are n-ary relation broken into n binary relations with CVT entities removed
    CONDENSED_HYPER_KB_DIR = os.path.join(FREEBASE_DIR, "condensed_hyper.stagg.neighborhoods", "condensed_hyper_relations_only")

    print("---------------load FastRDFStore---------------")
    ALL_ENTITY_NAME_BIN = os.path.join(DATA_DIR,"FastRDFStore", "data", "namesTable.bin")
    entity_names = {}
    with open(ALL_ENTITY_NAME_BIN, 'rb') as inf:
        stream = BinaryStream(inf)
        dict_cnt = stream.readInt32()
        print("total entities:", dict_cnt)
        for _ in tqdm(range(dict_cnt)):
            key = stream.readString().decode()
            if key.startswith('m.') or key.startswith('g.'):
                key = '/' + key[0] + '/' + key[2:]
            value = stream.readString().decode()
            entity_names[key] = value


    print("---------------load webqsp---------------")
    webqsp_questions = defaultdict(list)
    for split in ['train', 'test']:
        infname = os.path.join(DATA_DIR, "webqsp", f"webqsp_{split}.json")
        data = json.load(open(infname))
        for question in tqdm(data["Questions"]):
            q_obj = {
                "QuestionId": question["QuestionId"],
                "QuestionText": question["RawQuestion"],
                "ProcessedQuestionText": question["ProcessedQuestion"],
                "OracleEntities": [
                    {"freebaseId": get_fid(entity[0]), "text": entity[1]}
                    for entity in get_entities(question)
                ],
                "Answers": [
                    {"freebaseId": get_fid(answer[0])
                    if answer[0].startswith("m.") or answer[0].startswith("g.") else answer[0],
                    "text": answer[1]}
                    for answer in get_answers(question)
                ]
            }
            webqsp_questions[split].append(q_obj)

    print(len(webqsp_questions['train']), len(webqsp_questions['test']))


    


    '''[需要修改] dpr输出的字图(子图,score) 中的路径'''
    SUBGRAPH_OUTPUT_DIR = os.path.join(WEBQSP_DIR, "DPR_Output_DATA", "dpr_outputs_adv")

    # output dir for top-2k relations + condensed relations + condensed hyper-relations
    REL_CONDHYPERREL_SUBGRAPH_DIR = os.path.join(SUBGRAPH_OUTPUT_DIR, "dpr_top_2k_relations_per_question")
    os.makedirs(REL_CONDHYPERREL_SUBGRAPH_DIR, exist_ok=True)

    '''[需要修改]fid 输入路径修改'''
    FID_INPUT_DIR = os.path.join(WEBQSP_DIR, "FID_Input_DATA","dpr_nq_adv")
    MAX_TOKENS = 150
    MAX_PASSAGES = 100

    # top 100 passages with relations + condensed hyper relations
    FID_INPUT_TOP_COND_HYPER_RELS_DIR = os.path.join(FID_INPUT_DIR)
    os.makedirs(FID_INPUT_TOP_COND_HYPER_RELS_DIR, exist_ok=True)


    #保存fid的输入
    print("---------------save fid input data----------------")
    save_fid_input(
        REL_CONDHYPERREL_SUBGRAPH_DIR, 
        FID_INPUT_TOP_COND_HYPER_RELS_DIR, 
        MAX_PASSAGES=MAX_PASSAGES, 
        MAX_TOKENS=MAX_TOKENS,
        IGNORE_CVT=True,
    )

    print("---------------output metric----------------")
    # The top-100 answer coverage is around 95-96%
    get_ans_recall(FID_INPUT_TOP_COND_HYPER_RELS_DIR)