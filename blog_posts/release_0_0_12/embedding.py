import runhouse as rh
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    gpu = rh.ondemand_cluster(name='rh-a10x', instance_type='A10G:1').up()
    GPUModel = rh.module(SentenceTransformer).to(gpu, env=["torch", "transformers", "sentence-transformers"])
    model = GPUModel("sentence-transformers/all-mpnet-base-v2", device="cuda")
    text = ["You don't give 'em as a joke gift or wear them ironically or do pub crawls in "
            "'em like the snuggy. They're not like the snuggy."]
    print(model.encode(text))
