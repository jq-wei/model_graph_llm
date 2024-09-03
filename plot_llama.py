from torchview import draw_graph
import torchtext
from torch import nn
import torch
import graphviz

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "../llama3.1-8b/llama3.1-8b/"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))


model_graph = draw_graph(
    model, input_ids,
    graph_name='llama_graph_dep_3', expand_nested=True,
    depth=3,
    save_graph = True,
)

model_graph.visual_graph.render(format='pdf')
