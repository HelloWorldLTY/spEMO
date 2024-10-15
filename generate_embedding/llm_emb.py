import pickle

'''
Here we simplify the process, and the embeddings can be downloaded from this website: https://sites.google.com/yale.edu/scelmolib

For more details of how to generate embeddings from different LLMs, please refer: https://github.com/HelloWorldLTY/scELMo/blob/main/Get%20outputs%20from%20LLMs/query_35.ipynb

'''

adata_s = sc.read_h5ad("../dplfc_data/adata_visium.h5ad")
with open("/home/tl688/scratch/ensem_emb_gpt3.5all_new.pickle", "rb") as fp:
    GPT_3_5_gene_embeddings = pickle.load(fp)
gene_names= list(adata_s.var.index)
count_missing = 0
EMBED_DIM = 1536 # embedding dim from GPT-3.5
lookup_embed_genept = np.zeros(shape=(len(gene_names),EMBED_DIM))
for i, gene in enumerate(gene_names):
    if gene in GPT_3_5_gene_embeddings:
        lookup_embed_genept[i,:] = GPT_3_5_gene_embeddings[gene].flatten()
    else:
        count_missing+=1

print("count_missing", count_missing)

try:
    adata_s.X = adata_s.X.toarray()
except:
    pass

genePT_w_emebed = adata_s.X / np.sum(adata_s.X, axis=1) [:,None]  @ lookup_embed_genept

