import torch
from datetime import datetime

context_globals = {}

class ct:
    def __init__(self, tname):
        self.tname = tname

class Context:
    def __init__(self, device, name=None, parts=None, file_name=None, include=None, patch=None, variables=None):
        assert parts is not None or file_name is not None
        self.device = device

        if parts is not None:
            assert name is not None
            self.name = name
            self.parts = parts
            for part in self.parts:
                part["constructor"] = part["constructor"].__name__
            self.epoch = 0
            self.iteration = 0
            self.creation_time = datetime.now().strftime("%y%m%d-%H%M%S")
            self.info = {}
            self.variables = variables if variables is not None else {}

        else:
            checkpoint = torch.load(file_name)
            self.name = checkpoint["name"]
            if include is None:
                self.parts = checkpoint["parts"]
            else:
                self.parts = tuple([part for part in checkpoint["parts"] if part['name'] in include])

            if patch is not None:
                for part in self.parts:
                    if part["name"] in patch:
                        part["constructor"] = patch[part["name"]]["constructor"].__name__
                        part["params"] = patch[part["name"]]["params"]

            self.epoch = checkpoint["epoch"]
            self.iteration = checkpoint["iteration"]
            self.creation_time = checkpoint["creation_time"]
            self.variables = {}
            if "variables" in checkpoint:
                self.variables = checkpoint["variables"]
            self.info = {}
            if "info" in checkpoint:
                self.info = checkpoint["info"]

        self.time_name = f'{self.creation_time}_{self.name}'

        self.state = {}
        for part in self.parts:
            self._init_part(part)

    def __getattr__(self, attr):
        if attr in self.state:
            return self.state[attr]
        else:
            return None


    def add_part(self, i, part):
        context_globals[part["constructor"].__name__] = part["constructor"]
        part["constructor"] = part["constructor"].__name__

        parts = list(self.parts)
        parts.insert(i, part.copy())

        self.parts = tuple(parts)
        self._init_part(part)

    def _init_part(self, part):
        name = part["name"]
        constructor = context_globals[part["constructor"]]
        params = part["params"].copy()

        for k, v in params.items():
            if isinstance(v, ct):
                params[k] = eval("self." + v.tname)
            if isinstance(v, str):
                for k2, v2 in self.variables.items():
                    if not isinstance(v2, str):
                        continue
                    params[k] = v.replace("$" + k2, v2)

        if "optimize_target" in part:
            target = self.state[part["optimize_target"]]
            new_obj = constructor(target.parameters(), **params)
        elif "target" in part:
            target = self.state[part["target"]]
            new_obj = constructor(target, **params)
        else:
            new_obj = constructor(**params)

        if "state_dict" in part:
            new_obj.load_state_dict(part["state_dict"])

        if isinstance(new_obj, torch.nn.Module):
            new_obj = new_obj.to(self.device)

        self.state[name] = new_obj

    def get_part(self, name):
        return self.state[name]

    def get_parts(self, names):
        out = []
        for name in names:
            if name in self.state:
                out.append(self.state[name])
            else:
                out.append(None)
        return out

    def save(self, filename):
        for part in self.parts:
            if hasattr(self.state[part["name"]], "state_dict"):
                part["state_dict"] = self.state[part["name"]].state_dict()

        if self.info is None:
            self.info = {}
        checkpoint = dict(name=self.name, parts=self.parts, epoch=self.epoch, iteration=self.iteration,
                          creation_time=self.creation_time, variables=self.variables, info=self.info)

        torch.save(checkpoint, filename)

    def __repr__(self):
        out = f'{self.name}\n'
        out += f'{self.creation_time}\n'
        out += f"epoch {self.epoch}\n"
        out += f"iteration {self.iteration}\n"
        for k, v in self.info.items():
            out += f"{k} {v}\n"
        out += "\n"
        for part in self.parts:
            out += f"{part['name']} {part['constructor']} {part['params']}\n"
        return out