import torch
from chromadb.utils import embedding_functions
from typing import Union
from time import perf_counter
import tiktoken
import numpy as np

def get_current_cuda_device():
    return torch.cuda.current_device()


def get_embedding(model, input_text: Union[list, str], use_gpu=False):
    _input_text = None
    if isinstance(input_text, list):
        _input_text = input_text
    elif isinstance(input_text, str):
        _input_text = [input_text]
    else:
        raise ValueError("Invalid input text type.")
    if use_gpu:
        return model._model.encode(  # type: ignore
            _input_text,
            convert_to_numpy=True,
            normalize_embeddings=model._normalize_embeddings,
            device=torch.device(torch.cuda.current_device()),
        ).tolist()
    else:
        return model(_input_text)


def load_model(model_path, use_gpu=False):
    device = "cuda" if use_gpu else "cpu"
    if use_gpu:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path, device=device
        )
        embedding_function._model = embedding_function._model.to(device)
        embedding_function._target_device = torch.device(device)
    else:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path, device=device
        )
    return embedding_function


def load_tokenizer(tokenizer_model: str = "cl100k_base"):
    return tiktoken.get_encoding(tokenizer_model)


def main():
    import random
    MODEL_DIR_PATH = "/mnt/c/Users/23174/Desktop/GitHub Project/LLMGaoKaoAdvisor/embedding-server-github/embedding_models/all-MiniLM-L6-v2"
    use_gput = False
    model = load_model(MODEL_DIR_PATH, use_gpu=use_gput)
    sentence = "This is a demo sentence This is a demo sentenceThis is a demo sentenceThis is a demo sentence"
    # 从上面的句子中随机抽取句子进行测试
    sentences = [random.choice(sentence.split()) for _ in range(100)]
    start = perf_counter()
    res = get_embedding(model, sentences, use_gpu=use_gput)
    # res = get_embedding(model, sentences, use_gpu=True)
    end = perf_counter()
    print("Time per sentence: ", (end - start) / len(sentences))
    res_tensor = np.array(res)
    print("Res shape: ", res_tensor.shape)


if __name__ == "__main__":
    main()