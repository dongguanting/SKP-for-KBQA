import os
import csv
import pickle
from collections import defaultdict
import numpy as np
import faiss
from tqdm import tqdm
from graph_reader import Relation,read_condensed_relations_for_question,read_condensed_hyper_relations_for_question,BinaryStream




def read_relations_for_question(qid, ignore_rel=True):
    infname = os.path.join(FREEBASE_DIR, "stagg.neighborhoods", f"{qid}.nxhd")
    if not os.path.exists(infname):
        return None
    relations = []
    edge_map = defaultdict(list)
    with open(infname) as inf:
        for line in inf:
            rel = Relation(line)
            if ignore_rel and rel.should_ignore():
                continue
            relations.append(rel)
            edge_map[rel.subj].append(rel)
            edge_map[rel.obj].append(rel)
    return edge_map, relations





def save_results_to_file(docids, scores, idxs, docid2rel, oufname):
    topk_rels = []
    for score, idx in zip(scores, idxs):
        topk_rels.append((docid2rel[docids[idx]], score))       
    # save topk (hyper) relations to file
    with open(oufname, 'wb') as ouf:
        pickle.dump(topk_rels, ouf)

            
# save rels + cond hyper rels
def save_topk_hyper_relations(question, topk):
    qvec = question['question']
    qid = question['qid']
    print("Processing", qid, "...")
    # save docid2rel map
    docid2rel = {}
    # relations
    tup = read_relations_for_question(qid)
    if tup is None:
        raise ValueError(f"unknown qid {qid}")
    else:
        _, relations = tup
    for i, rel in enumerate(relations):
        docid = f"{qid}.relations.{i}"
        docid2rel[docid] = (rel.subj, rel.rel, rel.obj)
        
    # condensed relations
    cond_relations = read_condensed_relations_for_question(qid)
    if cond_relations:
        for docid, rel in cond_relations:
            docid2rel[docid] = rel
    # condensed hyper relations
    hyper_relations = read_condensed_hyper_relations_for_question(qid)    
    if hyper_relations:
        for docid, rel in hyper_relations:
            docid2rel[docid] = rel
    
    index = faiss.IndexFlatIP(vector_size)
    docids = []
    # index relations
    if question['relations']:
        # import pdb
        # pdb.set_trace()
        ids, vectors = zip(*question['relations'])
        docids += list(ids)
        vectors = np.array(vectors)
        assert len(docids) == len(vectors)
        index.add(vectors)

    # add condensed relations
    if question['condensed_relations']:
        # import pdb
        # pdb.set_trace()
        ids, condensed_relations = zip(*question['condensed_relations'])
        docids += list(ids)
        if condensed_relations:
            index.add(np.array(condensed_relations))
    
    # add condensed hyper-relations
    if question['condensed_hyper_relations']:
        ids, condensed_hyper_relations = zip(*question['condensed_hyper_relations'])
        docids += list(ids)
        if condensed_hyper_relations:
            index.add(np.array(condensed_hyper_relations))
    
    if len(docids) == 0:
        # no relations for this question
        print("no relations for", qid)
        return
    scores, idxs = index.search(np.expand_dims(qvec[1], 0), min(topk, len(docids)))
    scores = scores[0]
    idxs = idxs[0]
    # import pdb
    # pdb.set_trace()
    save_results_to_file(docids, scores, idxs, docid2rel, os.path.join(REL_CONDHYPERREL_SUBGRAPH_DIR, f"{qid}.pkl")) 








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

    '''读各种关系文件'''

    '''[需要修改]修改input路径'''
    print("-----------read DPR embeddings for condensed relations----------")
    COND_DPR_OUT_DIR = os.path.join(WEBQSP_DIR,"UniK-QA-main", "models", "mlm+cl+drop_122", "webqsp_condensed_relations")  ###如果要修改input的路径，在这里

    # read all condensed relations   
    print("-----------read all condensed relations----------")    
    NUM_SHARDS_CONDENSED = 16
    condensed_rels_vectors = defaultdict(list)
    for shard in tqdm(range(NUM_SHARDS_CONDENSED)):
        condensed_dpr_output_file = os.path.join(COND_DPR_OUT_DIR, f"all_relations.dpr_out_{shard}")
        with open(condensed_dpr_output_file, 'rb') as inf:
            dpr_output = pickle.load(inf)
            for docid, vec in dpr_output:
                parts = docid.split('.')
                qid, dtype = parts[0], parts[1]
                condensed_rels_vectors[qid].append((docid, vec))

    # read DPR embeddings for condensed hyper-relations
    
    '''[需要修改]修改input路径'''
    COND_HYPER_DPR_OUT_DIR = os.path.join(WEBQSP_DIR,"UniK-QA-main","models", "mlm+cl+drop_122", "webqsp_condensed_hyper_relations")

    # read all condensed relations  
    print("-----------read all condensed hyper-relations----------")   
    NUM_SHARDS_CONDENSED_HYPER = 16
    condensed_hyper_rels_vectors = defaultdict(list)
    for shard in tqdm(range(NUM_SHARDS_CONDENSED_HYPER)):
        condensed_dpr_output_file = os.path.join(COND_HYPER_DPR_OUT_DIR, f"all_relations.dpr_out_{shard}")
        with open(condensed_dpr_output_file, 'rb') as inf:
            dpr_output = pickle.load(inf)
            for docid, vec in dpr_output:
                parts = docid.split('.')
                qid, dtype = parts[0], parts[1]
                condensed_hyper_rels_vectors[qid].append((docid, vec))

    '''[需要修改]输出路径修改'''
    SUBGRAPH_OUTPUT_DIR = os.path.join(WEBQSP_DIR, "DPR_Output_DATA","mlm+cl+drop_122") 
    print("-----------output for top-2k 3 kinds of relations ----------")  
    # output dir for top-2k relations + condensed relations + condensed hyper-relations
    REL_CONDHYPERREL_SUBGRAPH_DIR = os.path.join(SUBGRAPH_OUTPUT_DIR, "dpr_top_2k_relations_per_question")
    os.makedirs(REL_CONDHYPERREL_SUBGRAPH_DIR, exist_ok=True)

    vector_size = 768


    # read DPR embeddings for relations
    # read and save one question after another to save memory
    '''[需要修改]输入路径修改'''
    DPR_OUT_DIR = os.path.join(WEBQSP_DIR, "UniK-QA-main", "models", "mlm+cl+drop_122", "webqsp_relations")

    NUM_SHARDS = 100 #分成了
    # 2000 relations should be more than enough to generate 100 150-token passages
    topk = 2000
    cur_qid = None
    questions = {}

    print("-----------start generating----------")  

    for shard in tqdm(range(NUM_SHARDS)):
        dpr_output_file = os.path.join(DPR_OUT_DIR, f"all_relations.dpr_out_{shard}")
        with open(dpr_output_file, 'rb') as inf:
            dpr_output = pickle.load(inf)
        for docid, vec in dpr_output:
            parts = docid.split('.')
            qid, dtype = parts[0], parts[1]
            if qid != cur_qid:
                # moving to a new question: save top-k
                if cur_qid is not None:
                    # add condensed relations
                    questions[cur_qid]['condensed_relations'] = condensed_rels_vectors[cur_qid]
                    # add condensed hyper-relations
                    questions[cur_qid]['condensed_hyper_relations'] = condensed_hyper_rels_vectors[cur_qid]
                    # import pdb
                    # pdb.set_trace()
                    save_topk_hyper_relations(questions[cur_qid], topk)
                    # optional to save memory
                    del questions[cur_qid]
                cur_qid = qid
            if qid not in questions:
                questions[qid] = {
                    'qid': qid,
                    'question': None,
                    'relations': [],
                }
            question = questions[qid]
            if dtype == 'question':
                question[dtype] = (docid, vec)
            elif dtype == 'relations':
                question[dtype].append((docid, vec))
            else:
                raise ValueError(f"Unknown dtype {dtype}")

    # save the last question
    # add condensed relations
    questions[cur_qid]['condensed_relations'] = condensed_rels_vectors[cur_qid]
    # add condensed hyper-relations
    questions[cur_qid]['condensed_hyper_relations'] = condensed_hyper_rels_vectors[cur_qid]

    save_topk_hyper_relations(questions[cur_qid], topk)