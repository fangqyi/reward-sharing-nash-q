from module.agents.dgn_agent import DGNAgent
from module.agents.rnn_agent import RNNAgent, RNNAgentImageVec

REGISTRY = {}

REGISTRY["rnn_agent"] = RNNAgent
REGISTRY["dgn_agent"] = DGNAgent
REGISTRY["rnn_agent_image_vec"] = RNNAgentImageVec
