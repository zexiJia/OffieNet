import torch

class checkpoint:
    def __init__(self, ckp, key) -> None:
        self.path = ckp
        self.ckp = torch.load(self.path)
        if key is not None:
            self.ckp = self.ckp[key]
    
    def debug(self):
        for k, v in self.ckp.items():
            print(k)
        return self
    
    def remove_prefix(self, prefix):
        ckp = {}
        for k, v in self.ckp.items():
            if k.startswith(f'{prefix}.') and f'{prefix}.' != k:
                ckp[k[len(prefix)+1:]] = v
            else:
                ckp[k] = v
        self.ckp = ckp
        return self
    
    def remove(self, name):
        ckp = {}
        for k, v in self.ckp.items():
            if not k.startswith(f'{name}.'):
                ckp[k] = v
        self.ckp = ckp
        return self

    def keep(self, name, remove_prefix=False):
        ckp = {}
        for k, v in self.ckp.items():
            if k.startswith(f'{name}.'):
                ckp[k] = v
        self.ckp = ckp
        if remove_prefix:
            self.remove_prefix(name)
        return self

    
    def load_ckp(self, model, freeze=False, strict=False):
        model.load_state_dict(self.ckp, strict=strict)
        if freeze:
            for i in model.parameters():
                i.requires_grad_(False)
        return model
    
    # def load_ckp_and_freeze(self, model:torch.nn.Module):
    #     model.load_state_dict(self.ckp)
    #     for i in model.parameters():
    #         i.requires_grad_(False)
    #     return model
    
    def items(self):
        return self.ckp.items()

    def __str__(self) -> str:
        s = ''
        for k, v in self.items():
            s += (k +' ')
        return s

