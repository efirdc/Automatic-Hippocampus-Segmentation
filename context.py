import torch
from datetime import datetime

context_globals = {}

class ct:
    def __init__(self, tname):
        self.tname = tname


class Context:
    def __init__(self, device, name=None, parts=None, file_name=None, include=None, patch=None, variables=None,
                 globals=None):
        if file_name is None:
            assert name is not None, "name is required for a new context"
        self.globals = globals
        self.name = name
        self.device = device
        self.epoch = 0
        self.iteration = 0
        self.creation_time = datetime.now().strftime("%y%m%d-%H%M%S")
        self.info = {}
        self.variables = variables if variables is not None else {}
        self.parts = []

        if parts is not None:
            self.parts = list(parts)
            for part in self.parts:
                part["constructor"] = part["constructor"].__name__
        elif file_name is not None:
            checkpoint = torch.load(file_name)
            if "variables" in checkpoint:
                old_vars = checkpoint["variables"]
                if set(self.variables.keys()) != set(old_vars.keys()):
                    raise ValueError(f"Loading this context requires variables {set(old_vars.keys())}.")
            self.name = checkpoint["name"]
            if include is None:
                self.parts = list(checkpoint["parts"])
            else:
                self.parts = [part for part in checkpoint["parts"] if part['name'] in include]

            if patch is not None:
                for part in self.parts:
                    if part["name"] in patch:
                        part["constructor"] = patch[part["name"]]["constructor"].__name__
                        part["params"] = patch[part["name"]]["params"]

            self.epoch = checkpoint["epoch"]
            self.iteration = checkpoint["iteration"]
            self.creation_time = checkpoint["creation_time"]
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

    def add_part(self, name, constructor, **params):
        part = dict(name=name, constructor=constructor.__name__, params=params)
        self.parts.append(part)
        self._init_part(part)

    def insert_part(self, i, name, constructor, **params):
        part = dict(name=name, constructor=constructor.__name__, params=params)
        self.parts.insert(i, part)
        self._init_part(part)

    def _fix_params(self, params):
        if isinstance(params, dict):
            return {k: self._fix_params(v) for k, v in params.items()}
        if isinstance(params, list):
            return [self._fix_params(param) for param in params]
        if isinstance(params, tuple):
            return tuple(self._fix_params(param) for param in params)

        param = params
        if isinstance(param, ct):
            return eval("self." + param.tname)
        if isinstance(param, str):
            for variable, value in self.variables.items():
                if not isinstance(value, str):
                    continue
                param = param.replace("$" + variable, value)
            return param
        return param

    def _init_part(self, part):
        name = part["name"]
        if self.globals is None:
            constructor = context_globals[part["constructor"]]
        else:
            constructor = self.globals[part["constructor"]]
        params = self._fix_params(part["params"])

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