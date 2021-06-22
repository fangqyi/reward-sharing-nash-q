from env.cleanup import Cleanup
from env.escaperoom import EscapeRoom
from env.ipd import IteratedPrisonerDilemma
from env.unfair_game.gridmaze import GridMaze
from env.unfair_game.surviving import Surviving


REGISTRY = {}

REGISTRY["gridmaze"] = GridMaze
REGISTRY["surviving"] = Surviving
REGISTRY["cleanup"] = Cleanup
REGISTRY["escaperoom"] = EscapeRoom
REGISTRY["ipd"] = IteratedPrisonerDilemma