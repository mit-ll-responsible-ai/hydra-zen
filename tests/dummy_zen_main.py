from hydra.core.config_store import ConfigStore

from hydra_zen import make_config, zen

cs = ConfigStore.instance()
cs.store(name="my_app", node=make_config("x", "y", z="${y}"))


@zen
def f(x: int, y: int, z: int):
    ...


if __name__ == "__main__":
    f.hydra_main(config_name="my_app", config_path=None)
