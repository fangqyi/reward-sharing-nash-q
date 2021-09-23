from controller.latent_d_shared_controller import BasicLatentMAC
from controller.latent_d_separate_controller import SeparateLatentMAC
REGISTRY = {}

REGISTRY["latent_dist"] = BasicLatentMAC
REGISTRY["separate_dist"] = SeparateLatentMAC