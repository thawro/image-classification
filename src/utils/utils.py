from omegaconf import DictConfig, OmegaConf
import rich
import rich.syntax
import rich.tree
from pathlib import Path

STYLE = "dim"
ROOT = Path(__file__).parent.parent.parent


def print_config_tree(cfg: DictConfig, keys: list[str] | str = "all", style: str = "dim"):
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    if keys == "all":
        branch = tree.add("config", style=style, guide_style=style)
        branch_content = OmegaConf.to_yaml(cfg, resolve=True)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    else:
        for key, group in cfg.items():
            if key in keys:
                branch = tree.add(key, style=style, guide_style=style)
                branch_content = OmegaConf.to_yaml(group, resolve=True)
                branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
