import argparse

import torch
from scipy import io
import pandas as pd

from lib.evaluation import Evaluator
from lib.utils import sort2query, csr2test, sort2query_vector

from recsys.PMF_model import *
import torch.optim as optim
import copy

docList = []
enList = []
questionsIndex=[]
releqrel=''


def MRR(pred, ground_truth):
    result = []
    for user, v_dict in pred.items():
        if user in ground_truth.keys():
            ground = ground_truth[user].keys()
            fit = [i for i in v_dict.keys()]
            tmp = 0.0
            for j in range(len(fit)):
                if fit[j] in ground:
                    tmp = 1.0 / (j + 1)
                    break
            result.append(tmp)
    return np.array(result).mean()

def Get_entity_doc_matrix(input_dir, items2):

    file_input = open(input_dir)

    str_list = file_input.readlines()
    count = 0
    for str_doc_en in str_list:

        entities = str_doc_en.split(",")
        if entities[0].strip().lower() in items2:
            docList.append(entities[0].strip().lower())#example: b00000jb91txt
            count=count+1
            count2 = 0
            for entity in entities:
                if entity.strip()!="" and count2!=0:
                    if entity.strip() not in enList:
                        enList.append(entity.strip())
                count2 = count2+1

    file_input.close()

    print('len(entity)：'+str(len(enList)))
    print('len(item)：' + str(len(docList)))

    entity_Doc_Matrix = np.zeros((len(enList), count), dtype=np.float64)
    for z in range(len(entity_Doc_Matrix)):
        entity_Doc_Matrix[z] = [-1 for j in range(count)]
    file_input1 = open(input_dir)
    str_list1 = file_input1.readlines()
    column1 = 0
    for line in str_list1:

        entities1 = line.split(",")
        if entities1[0].strip().lower() in items2:
            count3 = 0
            for entity1 in entities1:
                if entity1.strip()!="" and count3!=0:
                    entity_Doc_Matrix[enList.index(entity1.strip()), column1] = 1
                count3 = count3+1
            column1=column1+1

    file_input1.close()
    return entity_Doc_Matrix

def GBS_update(entity_Doc_Matrix, Alpha, updateAlpha, discardfile, user_idx, Y, item_id_index_txt):
    PreferencePi = np.zeros(len(docList), dtype=np.float64)
    countAlpha= np.sum(Alpha)
    for q in range(len(Alpha)):
        PreferencePi[q] = Alpha[q] / countAlpha
    min = 1000000000.0 #infinite
    minIndex = 0
    discard_mul=np.array([(1-discardfile[i]) for i in range(len(discardfile))])

    count2 = 0
    for z in range(len(entity_Doc_Matrix)):
        argMin = 0.0
        if z not in questionsIndex:
            argMin=np.dot(entity_Doc_Matrix[z], np.multiply(PreferencePi,discard_mul))
            judge = abs(argMin)
            if judge < min:
                min = judge
                minIndex = count2
        count2= count2+1
    questionsIndex.append(minIndex)

    if (releqrel+"txt") in docList:
        indexM = docList.index(releqrel+"txt")
        if entity_Doc_Matrix[minIndex,indexM]== -1:

            for m in range(len(entity_Doc_Matrix[minIndex])):
                if entity_Doc_Matrix[minIndex,m]== -1:
                    Alpha[m] = Alpha[m] + updateAlpha
                    Y_column=item_id_index_txt[docList[m]]
                    Y[user_idx,Y_column]=Y[user_idx,Y_column]+updateAlpha
                else:
                    discardfile[m] = 1
        elif entity_Doc_Matrix[minIndex,indexM]==1:

            for m1 in range(len(entity_Doc_Matrix[minIndex])):
                if entity_Doc_Matrix[minIndex,m1]==1:
                    Alpha[m1] = Alpha[m1] + updateAlpha
                    Y_column = item_id_index_txt[docList[m1]]
                    Y[user_idx, Y_column] = Y[user_idx, Y_column] + updateAlpha
                else:
                    discardfile[m1] = 1

    return PreferencePi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='running on gpu', action='store_true')
    parser.add_argument('--dir', help='dataset directory', type=str, default='DataExample')
    parser.add_argument('--data', help='specify dataset', type=str, default='data')
    parser.add_argument('--record', help='record the result', action='store_true', default=False)
    parser.add_argument('--epoches', help='number of epoches to train', type=int, default=100)
    parser.add_argument('--maxiter', help='max number of iteration', type=int, default=100)
    parser.add_argument('--N', help='number of recommended items', type=int, default=100)
    parser.add_argument('--lamb', help='parameter lambda', type=float, default=0.1)
    parser.add_argument('--fold', help='specify the fold', type=int, default=1)
    parser.add_argument('--seed', help=' random seed', type=int, default=5)
    parser.add_argument('--k', help='parameter k', type=int, default=3)
    parser.add_argument('--alpha', help='weight for online part (incorporating Y)', type=float, default=0.5)
    parser.add_argument('--NQ', help='number of questions', type=int, default=10)

    parser.add_argument('--category', help='category dir', type=str, default='pet')
    parser.add_argument('--inputdir', help='dataset directory',type=str, default='DataExample/TagMeResults/pet_supplies.txt')

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(args)

    print('dataset directory: ' + args.dir)
    directory = args.dir + '/' + args.data
    path = '{}/{}/train{}.txt'.format(directory, args.category, args.fold)
    print('train data path: ' + path)
    testfile = 'test'

    dataset = pickle.load(open('DataExample/data/dataset-'+args.category+'.txt', 'rb'))
    item_id_index=dataset['item_id_index']
    items2=dataset['items2']
    item_id_index_txt=dataset['item_id_index_txt']
    NUM_USERS=dataset['NUM_USERS']
    NUM_ITEMS=dataset['NUM_ITEMS']

    item_name = list(item_id_index.keys())
    item_index_i = list(item_id_index.values())

    R = io.mmread(path).A
    R = torch.from_numpy(R.astype('float32')).to(args.device)
    args.m, args.n = R.shape
    print('model: conversationalMF')
    print(R.shape)

    Y = np.zeros((args.m, args.n), dtype=np.float64)
    entity_Doc_Matrix = Get_entity_doc_matrix(args.inputdir, items2)

    path_test = '{}/{}/{}{}.txt'.format(directory, args.category, testfile, args.fold)
    print('{} path: {}'.format(testfile, path_test))
    T = io.mmread(path_test)
    test = csr2test(T.tocsr())

    # training
    model_py = PMF(args).to(args.device)
    loss_function = PMFLoss(lam_u=args.lamb, lam_v=args.lamb, lam_p=args.lamb, lam_q=args.lamb).to(args.device)
    optimizer = optim.Adam(model_py.parameters(), lr=1e-3)
    train_loss_list = []
    Y = torch.from_numpy(Y.astype('float32')).to(args.device)

    for epoch in range(1, args.epoches + 1):
        optimizer.zero_grad()
        loss = loss_function(R, model_py.user_embeddings.weight, model_py.item_embeddings.weight, model_py.alpha, Y, model_py.P_para.weight, model_py.Q_para.weight)
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss)

    score = model_py.recommend()
    _, idx = torch.sort(score, 1, True)
    run_mf = sort2query(idx[:, :])

    U_store = copy.deepcopy(model_py.user_embeddings)
    V_store = copy.deepcopy(model_py.item_embeddings)
    I_store = copy.deepcopy(model_py.I)
    P_store = copy.deepcopy(model_py.P_para)
    Q_store = copy.deepcopy(model_py.Q_para)

    file_test=open(path_test)
    next(file_test)
    next(file_test)
    str_test=file_test.readlines()
    run_cmf_online={}
    for str_doc_test in str_test:

        model_py.user_embeddings=copy.deepcopy(U_store)
        model_py.item_embeddings=copy.deepcopy(V_store)
        model_py.I=copy.deepcopy(I_store)
        model_py.P_para=copy.deepcopy(P_store)
        model_py.Q_para=copy.deepcopy(Q_store)

        line=str_doc_test.strip().split(' ')
        user_idx = int(line[0])-1
        item_idx = int(line[1])-1
        releqrel = item_name[item_index_i.index(item_idx)].strip().lower()
        print(str_doc_test.strip('\n'))

        PrefPi1 = np.zeros(len(docList), dtype=np.float64)
        updateAlpha = 5.0
        questionsIndex = []
        Alpha = np.zeros(len(docList), dtype=np.float64)
        discardfile = np.zeros(len(docList), dtype=np.int)
        Alpha_D = run_mf[str(user_idx)]
        for key, value in Alpha_D.items():
            tem_str = item_name[item_index_i.index(int(key))]
            id_prod = docList.index(tem_str.strip().lower() + 'txt')
            Alpha[id_prod] = value

        Y = np.zeros((args.m, args.n), dtype=np.float64)
        Y = torch.from_numpy(Y.astype('float32')).to(args.device)
        for k in range(args.NQ):
            PrefPi1 = GBS_update(entity_Doc_Matrix, Alpha, updateAlpha, discardfile, user_idx, Y, item_id_index_txt)
            model_py.update_U(R, Y, user_idx)
            model_py.update_V(R, Y)

        score = model_py.recommend()
        _, idx = torch.sort(score, 1, True)
        run = sort2query_vector(idx[user_idx,:])
        run_cmf_online[str(user_idx)]=copy.deepcopy(run) # no duplicate users

    print("CMF online resutls:")
    evaluator3 = Evaluator({'recall','map_cut','ndcg_cut'})
    evaluator3.evaluate(run_cmf_online, test)
    result3 = evaluator3.show_all()
    print(result3)
    mrr3=MRR(run_cmf_online, test)
    print("mrr:"+str(mrr3))

    if args.record:
        line = 'CMF\t{}'.format(args)
        for _, value in result3.items():
            line += '\t{:.5f}'.format(value)
        line += '\r\n'
        file = open('result', 'a')
        file.write(line)
        file.close()
