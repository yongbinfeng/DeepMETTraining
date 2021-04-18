'''
example to retrieve parameters saved in Keras h5 model file,
and run the preprocessing to project charge and pdgId information from
inputs to embeded space
'''
import h5py
import numpy as np

maxCands = 100

def retrieve_parameters(filename_model):
    """
    retrieve the embedding parameters saved in filename_model
    """
    hfile = h5py.File(filename_model, "r")
    
    # charge embedding
    embed0 = np.asarray(hfile['model_weights']['embedding0']['embedding0']['embeddings:0'])
    print(embed0)
    # pdgId embedding
    embed1 = np.asarray(hfile['model_weights']['embedding1']['embedding1']['embeddings:0'])
    print(embed1)

    return embed0, embed1

def preprocessing_example(filename_model, filename_data):
    encoding_charge = {-1: 0, 0: 1, 1: 2}
    encoding_pdgId = {-211: 0, -13: 1, -11: 2, 0: 3, 11: 4, 13: 5, 22: 6, 130: 7, 211: 8}

    from utils import get_features
    features_cands = ['L1PuppiCands_charge','L1PuppiCands_pdgId']

    # get the chg and pdgId information from data
    features_cands_array = get_features(filename_data, features_cands, maxCands)
    # encode them into values between 0 and n
    chg = np.vectorize(encoding_charge.get)(features_cands_array[:,:,0]) # charge: nparticle x maxCands
    pdgId = np.vectorize(encoding_pdgId.get)(features_cands_array[:,:,1]) # pdgId: nparticle x maxCands


    # retrieve embedding layer parameters in the model
    embed0, embed1 = retrieve_parameters(filename_model)
    # project chg and pdgId into embeded space
    embed_chg = embed0[chg] # chg embedded: nparticle x maxCands x Embed
    embed_pdgId = embed1[pdgId] # pdgId embedded: nparticle x maxCands x Embed

    print("chg", chg)
    print("pdgId", pdgId)
    print("embed_chg", embed_chg)
    print("embed_pdgId", embed_pdgId)
    print("embed_chg shape", embed_chg.shape)
    print("embed_pdgId shape", embed_pdgId.shape)

    return embed_chg, embed_pdgId


if __name__ == "__main__":
    # change this to the model directory
    filename_model = "models/2021-03-03_21-42/model_best.h5"
    # change this to the data directory
    filename_data = "/eos/cms/store/user/yofeng/L1METML/input_MET_PupCandi.h5"

    preprocessing_example(filename_model, filename_data)

