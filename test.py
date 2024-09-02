from torchview import draw_graph
import torchtext
from torch import nn
import torch
import graphviz

from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

model_path = "../mamba"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MambaForCausalLM.from_pretrained(model_path)
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))

model_graph = draw_graph(
    model, input_ids,
    graph_name='Mamba_graph_dep3', expand_nested=True,
    depth=4,
    save_graph = True,
)

model_graph.visual_graph.render(format='pdf')