from controller.basic_pq_shared_controller import BasicPQMAC
from controller.latent_d_shared_controller import BasicLatentMAC
from controller.shared_controller import BasicMAC
from controller.shared_controller_nash_q import SharedNashMAC
from controller.latent_d_separate_controller import SeparateLatentMAC
REGISTRY = {}

REGISTRY["shared_nash"] = SharedNashMAC
REGISTRY["shared"] = BasicMAC
REGISTRY["shared_pq"] = BasicPQMAC
REGISTRY["latent_dist"] = BasicLatentMAC
REGISTRY["separate_dist"] = SeparateLatentMAC