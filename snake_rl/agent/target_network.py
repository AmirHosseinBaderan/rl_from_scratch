from agent import QNetwork


class TargetNetwork:
    def __init__(
            self,
            state_size: int,
            action_size: int,
    ):
        self.network = QNetwork(
            state_size=state_size,
            action_size=action_size,
        )

    def copy_from(self, online_network: QNetwork) -> None:
        self.network.load_state_dict(
            online_network.state_dict()
        )