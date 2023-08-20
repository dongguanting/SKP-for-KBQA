import os
import csv

# Get entity names from FastRDFStore
# https://github.com/microsoft/FastRDFStore

from struct import *
from collections import defaultdict
import json
import faiss
from tqdm import tqdm
from graph_reader import Relation,read_condensed_relations_for_question,read_condensed_hyper_relations_for_question,BinaryStream,convert_relation_to_text,read_relations_for_question,convert_hyper_relation_to_text



class BinaryStream:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def readByte(self):
        return self.base_stream.read(1)

    def readBytes(self, length):
        return self.base_stream.read(length)

    def readChar(self):
        return self.unpack('b')

    def readUChar(self):
        return self.unpack('B')

    def readBool(self):
        return self.unpack('?')

    def readInt16(self):
        return self.unpack('h', 2)

    def readUInt16(self):
        return self.unpack('H', 2)

    def readInt32(self):
        return self.unpack('i', 4)

    def readUInt32(self):
        return self.unpack('I', 4)

    def readInt64(self):
        return self.unpack('q', 8)

    def readUInt64(self):
        return self.unpack('Q', 8)

    def readFloat(self):
        return self.unpack('f', 4)

    def readDouble(self):
        return self.unpack('d', 8)

    def decode_from_7bit(self):
        """
        Decode 7-bit encoded int from str data
        """
        result = 0
        index = 0
        while True:
            byte_value = self.readUChar()
            result |= (byte_value & 0x7f) << (7 * index)
            if byte_value & 0x80 == 0:
                break
            index += 1
        return result

    def readString(self):
        length = self.decode_from_7bit()
        return self.unpack(str(length) + 's', length)

    def writeBytes(self, value):
        self.base_stream.write(value)

    def writeChar(self, value):
        self.pack('c', value)

    def writeUChar(self, value):
        self.pack('C', value)

    def writeBool(self, value):
        self.pack('?', value)

    def writeInt16(self, value):
        self.pack('h', value)

    def writeUInt16(self, value):
        self.pack('H', value)

    def writeInt32(self, value):
        self.pack('i', value)

    def writeUInt32(self, value):
        self.pack('I', value)

    def writeInt64(self, value):
        self.pack('q', value)

    def writeUInt64(self, value):
        self.pack('Q', value)

    def writeFloat(self, value):
        self.pack('f', value)

    def writeDouble(self, value):
        self.pack('d', value)

    def writeString(self, value):
        length = len(value)
        self.writeUInt16(length)
        self.pack(str(length) + 's', value)

    def pack(self, fmt, data):
        return self.writeBytes(pack(fmt, data))

    def unpack(self, fmt, length = 1):
        return unpack(fmt, self.readBytes(length))[0]


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





if __name__ == '__main__':

    # change to your local path
    WEBQSP_DIR = os.path.dirname(os.path.realpath('.'))
    # DATA_DIR = os.path.join(WEBQSP_DIR, "data")
    DATA_DIR = os.path.join(WEBQSP_DIR, "UniK-QA-main","data")
    FREEBASE_DIR = os.path.join(DATA_DIR, "freebase_2hop")

    print("-----------read fastRDF----------") 
    '''读entity的id与自然语言之间的映射'''

    ALL_ENTITY_NAME_BIN = os.path.join(DATA_DIR,"FastRDFStore", "data","namesTable.bin")
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

    print("test")
    print(entity_names[key])
    print(key)

    print("-----------save data to Dict----------") 
    '''读webqsp的数据到dict中'''
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

    DPR_INPUT_DIR = os.path.join(WEBQSP_DIR, "DPR_Input_DATA","dpr_inputs_1211")
    IDV_REL_DIR = os.path.join(DPR_INPUT_DIR, "individual_relations")
    os.makedirs(IDV_REL_DIR, exist_ok=True)

    '''写所有的relation,到tsv中'''
    oufname = os.path.join(IDV_REL_DIR, "all_relations.tsv")

    # Note: We simply dump all relations from each question into a single file, which is sub-optimal.
    # There're many duplicate relations, but we keep it as is for better replication.
    # De-duplication can be done to significantly reduce the number of relations DPR need to encode,
    # while not affecting the accuracy at all.

    print("----------------------write all_relation to tsv---------------------")
    with open(oufname, 'w') as ouf:
        writer = csv.writer(ouf, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        
        for split in ['train', 'test']:
            for question in tqdm(webqsp_questions[split]):
                qid = question['QuestionId']
                print("Processing", qid, "...")
                tup = read_relations_for_question(qid)
                if tup is None:
                    continue
                else:
                    _, relations = tup
                # write question
                writer.writerow([
                    f"{qid}.question",
                    question['QuestionText'],
                    '',
                ])
                # write relations: each relation is a separate document
                for i, rel in enumerate(relations):
                    docid = f"{qid}.relations.{i}"
                    text = convert_relation_to_text(rel, entity_names)
                    writer.writerow([docid, text, ''])

    # condensed relations are n-ary relation broken into n binary relations with CVT entities removed
    CONDENSED_KB_DIR = os.path.join(FREEBASE_DIR, "condensed.stagg.neighborhoods", "condensed_edges_only")
    # A condensed hyper-relation is a n-ary relation with CVT entities removed
    CONDENSED_HYPER_KB_DIR = os.path.join(FREEBASE_DIR, "condensed_hyper.stagg.neighborhoods", "condensed_hyper_relations_only")
    # condense relations
    COND_REL_DIR = os.path.join(DPR_INPUT_DIR, "condensed_relations")
    os.makedirs(COND_REL_DIR, exist_ok=True)


    print("-------------------write all_condensed_relations to tsv---------------------")
    oufname = os.path.join(COND_REL_DIR, "all_condensed_relations.tsv")

    with open(oufname, 'w') as ouf:
        writer = csv.writer(ouf, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        
        for split in ['train', 'test']:
            for question in tqdm(webqsp_questions[split]):
                qid = question['QuestionId']
                print("Processing", qid, "...")
                relations = read_condensed_relations_for_question(qid)
                if relations is None:
                    print("No condensed relations found for", qid)
                    continue
                # write condensed relations: each relation is a separate document
                for i, tup in enumerate(relations):
                    docid1 = f"{qid}.condensed_relations.{i}"
                    docid, rel = tup
                    assert docid == docid1
                    text = convert_relation_to_text(rel, entity_names)
                    writer.writerow([docid, text, ''])    


        # condensed hyper relations

    print("-------------------write all_condensed_hyper_relations to tsv---------------------")
    COND_HYPERREL_DIR = os.path.join(DPR_INPUT_DIR, "condensed_hyper_relations")
    os.makedirs(COND_HYPERREL_DIR, exist_ok=True)
    oufname = os.path.join(COND_HYPERREL_DIR, "all_condensed_hyper_relations.tsv")

    with open(oufname, 'w') as ouf:
        writer = csv.writer(ouf, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        
        for split in ['train', 'test']:
            for question in tqdm(webqsp_questions[split]):
                qid = question['QuestionId']
                print("Processing", qid, "...")
                relations = read_condensed_hyper_relations_for_question(qid)
                if relations is None:
                    print("No condensed relations found for", qid)
                    continue
                # write condensed relations
                for docid, rel in relations:
                    text = convert_hyper_relation_to_text(rel, entity_names)
                    writer.writerow([docid, text, ''])