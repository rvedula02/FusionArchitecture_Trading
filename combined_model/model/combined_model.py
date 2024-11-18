# model/combined_model.py

from model.master import MASTERModel
from model.llm4ts import LLM4TSModel
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, master_config, llm_config):
        super(CombinedModel, self).__init__()
        self.master = MASTERModel(master_config)
        self.llm4ts = LLM4TSModel(llm_config)
        self.combined_layer = nn.Linear(master_config['output_dim'] + llm_config['output_dim'], 1)

    def forward(self, master_input, llm_input):
        master_output = self.master(master_input)
        llm_output = self.llm4ts(llm_input)
        combined = torch.cat((master_output, llm_output), dim=1)
        output = self.combined_layer(combined)
        return output