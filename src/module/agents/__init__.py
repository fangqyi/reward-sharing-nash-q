from module.agents.nash_q_agent import NashQAgent
from module.agents.rnn_agent import RNNAgent

REGISTRY = {}

REGISTRY["nash_q_agent"] = NashQAgent
REGISTRY["rnn_agent"] = RNNAgent