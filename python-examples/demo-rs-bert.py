import candle_bert
from time import perf_counter
import numpy as np
import random   
MODEL_DIR_PATH = "/mnt/c/Users/23174/Desktop/GitHub Project/LLMGaoKaoAdvisor/embedding-server-github/embedding_models/all-MiniLM-L6-v2"
model = candle_bert.CandleBert(model_dir_path=MODEL_DIR_PATH, use_cuda=False, use_pth=True, approximate_gelu=False)
sentence = "This is a demo sentence This is a demo sentenceThis is a demo sentenceThis is a demo sentence"
sentences = [random.choice(sentence.split()) for _ in range(100)]
start = perf_counter()
res = model.forward(sentences)
end = perf_counter()
print("Time per sentence: ", (end - start) / len(sentences))
res_tensor = np.array(res)
print("Res shape: ", res_tensor.shape)