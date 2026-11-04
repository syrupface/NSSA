""" History of operations per channel with undo, and recipes (the operations as JSON) that can be applied to other
channels and files. """

import dataclasses
import json
import pathlib
from typing import Callable, Optional

import numpy as np

import lateral
import ops
from imgdata import Channel, ImageFile

RECIPE_FORMAT = "nssa-recipe"
OLD_RECIPE_FORMATS = ("nanosurfimg-recipe",)              # macros saved before the rename
RECIPE_VERSION = 2                  # 1: one list of operations; 2: a macro (steps of several channels, grain analysis)
CREATE = "channel.create"           # journal step that makes a new channel from two others (combine.py)
LINKED_OPS = ("geometry.crop", "geometry.flip", "geometry.rotate", "geometry.resample", "geometry.square_pixels", "rescale.xy")   # applied to all channels
ALL = "*"                           # channel key of a step that was applied to all channels
CURRENT = ""                        # channel key of a step for the channel that is shown when the macro is applied


def _plain(value):
    """ JSON compatible copy of a parameter value. """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    return value


@dataclasses.dataclass
class Op:
    name: str
    params: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "params": {k: _plain(v) for k, v in self.params.items()}}

    @staticmethod
    def from_dict(item: dict) -> "Op":
        if "name" not in item:
            raise ops.OpError("A recipe entry has no operation name.")
        return Op(item["name"], dict(item.get("params", {})))

    def describe(self) -> str:
        if self.name == CREATE and self.params.get("operation") == "copy":
            return f"Copy of {self.params.get('base', '?')}"
        spec = ops.OPS.get(self.name)
        if spec is None:
            return self.name
        if self.name == "geometry.rotate":
            return {"cw": "Rotate 90° clockwise", "ccw": "Rotate 90° counterclockwise",
                    "180": "Rotate 180°"}.get(self.params.get("turn", "cw"), spec.label)
        if self.name == "filter.basic":                 # one op for many filters: name the filter, its own settings
            kind = self.params.get("kind", "mean")
            used = {"gaussian": ("sigma",), "sharpen": ("sigma", "amount")}.get(kind, ("size", "shape"))
            parts = [f"{key}={self.params.get(key, spec.defaults[key]):.4g}" if key in ("sigma", "amount")
                     else f"{key}={self.params.get(key, spec.defaults[key])}" for key in used]
            region = self.params.get("apply_to", "all")
            if region != "all":
                parts.append(f"apply_to={region}")
            return f"{ops.FILTER_KINDS.get(kind, kind)} filter ({', '.join(parts)})"
        changed = []
        for key, value in self.params.items():
            if isinstance(value, (list, tuple)) and len(value) > 4:
                value = f"{len(value)} items"
            elif isinstance(value, float):
                value = f"{value:.4g}"
            elif isinstance(value, (list, tuple)):
                value = "(" + ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in value) + ")"
            if spec.defaults.get(key) != self.params[key]:
                changed.append(f"{key}={value}")
        return spec.label + (f" ({', '.join(changed)})" if changed else "")


class Pipeline:
    """ The original image of a channel plus the operations applied to it. """

    def __init__(self, base: ops.State, resolve: Optional[Callable] = None):
        self.base = base
        self.resolve = resolve
        self.ops: list = []
        self._states: list = [base]

    @property
    def current(self) -> ops.State:
        return self._states[-1]

    @property
    def can_undo(self) -> bool:
        return bool(self.ops)

    def add(self, name: str, params: dict) -> ops.State:
        """ Applies an operation to the current image; on an error nothing changes. """
        new = ops.apply_op(self.current, name, params, self.resolve)
        self.ops.append(Op(name, dict(params)))
        self._states.append(new)
        return new

    def extend(self, recipe: list):
        """ Appends the operations of a recipe; if one fails the whole recipe is rolled back and the error names it. """
        start = len(self.ops)
        for number, op in enumerate(recipe, start=1):
            try:
                self.add(op.name, op.params)
            except ops.OpError as e:
                while len(self.ops) > start:
                    self.undo()
                raise ops.OpError(f"Step {number} ({op.describe()}): {e}")

    def undo(self):
        if self.ops:
            self.ops.pop()
            self._states.pop()

    def reset(self):
        self.ops.clear()
        del self._states[1:]

    def replay(self, recipe: list):
        """ Replaces the history by the operations of a recipe, applied to the original image. If one fails nothing
            changes and the error names the step. """
        states, applied = [self._states[0]], []
        for number, op in enumerate(recipe, start=1):
            try:
                states.append(ops.apply_op(states[-1], op.name, op.params, self.resolve))
            except ops.OpError as e:
                raise ops.OpError(f"Step {number} ({op.describe()}): {e}")
            applied.append(op)
        self.ops, self._states = applied, states

    def recipe(self) -> list:
        return list(self.ops)

    def uses_other_channels(self, keys) -> bool:
        """ True if one of the operations reads another channel (a 'source' or 'reference' parameter). """
        return any(op.params.get(name) in keys for op in self.ops for name in ("source", "reference"))

    def replay_on(self, base: ops.State, resolve: Optional[Callable] = None) -> list:
        """ The states of the current operations applied to another original image (nothing is changed). """
        states = [base]
        for number, op in enumerate(self.ops, start=1):
            try:
                states.append(ops.apply_op(states[-1], op.name, op.params, resolve or self.resolve))
            except ops.OpError as e:
                raise ops.OpError(f"Step {number} ({op.describe()}): {e}")
        return states

    def set_states(self, states: list):
        """ Installs the result of replay_on. """
        self.base, self._states = states[0], states


class Document:
    """ An opened image file with one pipeline per channel. """

    def __init__(self, image_file: ImageFile):
        self.file = image_file
        self.layers: dict = {}
        self.journal: list = []                    # (channel key or ALL, Op) in the order they were applied
        self.created: dict = {}                    # key of a channel made from two others -> how (see create)
        self.redo_steps: list = []                 # (channel key or ALL, Op) taken back by undo, newest last
        self._redoing = False
        self.lateral = lateral.LateralSettings()
        self.lateral_info = ""                      # description of the current lateral correction
        for key, channel in image_file.channels.items():
            self.layers[key] = Pipeline(self._nominal_state(key), self._resolve)

    @property
    def keys(self) -> list:
        return list(self.layers)

    def _nominal_state(self, key: str) -> ops.State:
        channel = self.file.channels[key]
        mask = channel.mask.copy() if channel.mask is not None else np.zeros(channel.data.shape, bool)
        return ops.State(channel.data.copy(), mask, channel.size_x, channel.size_y, channel.unit)

    def position_channels(self) -> Optional[tuple]:
        """ (x name, y name) of the measured position channels found in the file, or None. """
        return lateral.find_position_channels(sorted({c.name for c in self.file.channels.values() if c.is_height}))

    def _key(self, name: str, direction: str) -> str:
        return f"{name} ({direction})" if direction else name

    def set_lateral(self, settings: "lateral.LateralSettings") -> str:
        """ Switches all channels between the recorded lateral scale (closed loop) and the one measured by the X and Y
            position channels, keeping every channel's operations (they are replayed on the new geometry). Returns a
            description. Nothing changes if that is not possible (the error says why). The position channels
            themselves keep their raw data in 'resample' mode so that the measured positions can still be inspected. """
        bases = {key: self._nominal_state(key) for key in self.layers if key not in self.created}
        info = "XY scale: as recorded"
        if settings.mode == "sensor":
            names = (settings.x_channel, settings.y_channel) if settings.x_channel and settings.y_channel \
                else self.position_channels()
            if names is None:
                raise ops.OpError("No X and Y position channels were found in this file: select the channels that "
                                  "hold the measured positions.")
            notes, sensors = [], set()
            for direction in sorted({c.direction for c in self.file.channels.values()}):
                keys = [self._key(name, direction) for name in names]
                missing = [k for k in keys if k not in self.file.channels]
                if missing:
                    raise ops.OpError(f"The channel '{missing[0]}' does not exist in this file.")
                x_channel, y_channel = (self.file.channels[k] for k in keys)
                if x_channel.unit != "m" or y_channel.unit != "m":
                    raise ops.OpError("The position channels must be given in m.")
                correction = lateral.Correction(x_channel.data, y_channel.data, x_channel.size_x, x_channel.size_y,
                                                settings.correction, settings.smoothing)
                notes.append(f"{direction or 'image'}: {correction.summary()}")
                for key, channel in self.file.channels.items():
                    if key not in bases or channel.direction != direction or \
                            channel.data.shape != x_channel.data.shape:      # removed channels stay removed
                        continue
                    if settings.correction == "resample" and key in keys:
                        continue
                    bases[key] = correction.apply_state(bases[key])
            info = f"XY scale: from sensors ({names[0]}, {names[1]}), {settings.correction}. " + "; ".join(notes)
        self._rebase(bases)
        self.lateral, self.lateral_info = settings, info
        return info

    def _rebase(self, bases: dict):
        """ Replays the operations of every channel on new original images; all or nothing. Operations that read
            other channels see the new results (a second pass settles chains of such operations). """
        latest = {key: layer.current.data for key, layer in self.layers.items()}

        def resolve(key):
            if key not in latest:
                raise ops.OpError(f"The channel '{key}' does not exist in this file.")
            return latest[key]

        passes = 2 if any(layer.uses_other_channels(self.layers) for layer in self.layers.values()) else 1
        for _ in range(passes):
            new = {key: layer.replay_on(bases[key], resolve) for key, layer in self.layers.items()
                   if key not in self.created}
            for key, info in self.created.items():         # in the order they were made: from the new states
                params = info["params"]
                a_states, b_states = new[params["base"]], new[params["source"]]
                base = self._combined(params, a_states[info["base_steps"]], b_states[info["source_steps"]])
                new[key] = self.layers[key].replay_on(base, resolve)
            latest = {key: states[-1].data for key, states in new.items()}
        for key, states in new.items():
            self.layers[key].set_states(states)

    def _resolve(self, key: str) -> np.ndarray:
        if key not in self.layers:
            raise ops.OpError(f"The channel '{key}' does not exist in this file.")
        return self.layers[key].current.data

    # ---- operations with the journal (the order of the steps over all channels, for macros) ----

    # ---- channels made from two others (lateral force: forward and backward combined) ----

    def channel_info(self, key: str) -> tuple:
        """ (name, direction, unit) of a channel of the file or of a created one. """
        if key in self.file.channels:
            channel = self.file.channels[key]
            return channel.name, channel.direction, self.layers[key].current.unit
        return key, "", self.layers[key].current.unit

    def _combined(self, params: dict, a_state: ops.State, b_state: ops.State) -> ops.State:
        if params.get("operation") == "copy":           # a duplicate: the processed state of the base, mask included
            return ops.State(a_state.data.copy(), a_state.mask.copy(), a_state.size_x, a_state.size_y, a_state.unit)
        import combine
        return combine.combine(a_state, b_state, params.get("operation", "diff"), float(params.get("shift_x", 0.0)),
                               float(params.get("shift_y", 0.0)))

    def _create(self, key: str, params: dict):
        """ A new channel `key` from the current states of params['base'] (A) and params['source'] (B), combined as
            params['operation'] with B shifted by params['shift_x'/'shift_y'] pixels. """
        if not key or key in self.layers:
            raise ops.OpError(f"A channel named '{key}' exists already: choose another name.")
        base, source = params.get("base"), params.get("source")
        for name in (base, source):
            if name not in self.layers:
                raise ops.OpError(f"The channel '{name}' does not exist in this file.")
        state = self._combined(params, self.layers[base].current, self.layers[source].current)
        self.layers[key] = Pipeline(state, self._resolve)
        self.created[key] = {"params": dict(params), "base_steps": len(self.layers[base].ops),
                             "source_steps": len(self.layers[source].ops)}

    @property
    def removed(self) -> list:
        """ Channels of the file that were removed (they can be restored), in file order. """
        return [key for key in self.file.channels if key not in self.layers]

    def duplicate_name(self, key: str) -> str:
        """ A free name for a copy of a channel: 'Name copy (Forward)', 'Name copy 2 (Forward)', ... """
        name, direction = (key.rsplit(" (", 1) + [""])[:2] if key.endswith(")") and " (" in key else (key, "")
        for number in range(1, 1000):
            label = f"{name} copy" + (f" {number}" if number > 1 else "")
            candidate = f"{label} ({direction}" if direction else label
            if candidate not in self.layers:
                return candidate
        raise ops.OpError("Too many copies of this channel.")

    def duplicate_channel(self, key: str, name: str = "") -> str:
        """ A new channel that starts from the current (processed) state of `key`, mask included, and is processed
            on its own from then on. A journal step, so it can be undone, removed and is part of macros. Returns the
            new key. """
        if key not in self.layers:
            raise ops.OpError(f"The channel '{key}' does not exist in this file.")
        name = name or self.duplicate_name(key)
        self.apply(name, CREATE, {"operation": "copy", "base": key, "source": key})
        return name

    def remove_channel(self, key: str):
        """ Removes a channel with its steps: a channel of the file (restore_channel brings it back) or a created
            one. Refused for the last channel and while something depends on it. """
        if key not in self.layers:
            raise ops.OpError(f"The channel '{key}' does not exist in this file.")
        if len(self.layers) == 1:
            raise ops.OpError("The last channel cannot be removed.")
        for other, info in self.created.items():
            if other != key and key in (info["params"].get("base"), info["params"].get("source")):
                raise ops.OpError(f"The channel '{other}' was made from it: remove that one first.")
        for other, op in self.journal:
            if other != key and key in (op.params.get("source"), op.params.get("reference")):
                raise ops.OpError(f"A step of '{other}' uses it ({op.describe()}): undo that step first.")
        del self.layers[key]
        self.created.pop(key, None)
        self.journal = [(k, op) for k, op in self.journal if k != key]
        self.redo_steps = [(k, op) for k, op in self.redo_steps if k != key]

    def restore_channel(self, key: str):
        """ Brings back a removed channel of the file: its original data with the steps that were applied to all
            channels (crop, flip, ...) and the current XY scale; its own steps are not restored. """
        if key not in self.removed:
            raise ops.OpError(f"The channel '{key}' was not removed.")
        layer = Pipeline(self._nominal_state(key), self._resolve)
        layer.extend([op for k, op in self.journal if k == ALL])
        order = list(self.file.channels) + [k for k in self.layers if k not in self.file.channels]
        self.layers[key] = layer
        self.layers = {k: self.layers[k] for k in order if k in self.layers}
        if self.lateral.mode != "nominal":
            try:
                self.set_lateral(self.lateral)
            except ops.OpError:
                del self.layers[key]
                raise

    def apply(self, key: str, name: str, params: dict):
        """ Applies an operation to a channel, or to all channels for the operations that change the geometry
            (LINKED_OPS); CREATE makes the new channel `key`. All or nothing; raises ops.OpError. """
        if not self._redoing:
            self.redo_steps.clear()                 # a new step: what was undone cannot be redone any more
        if name == CREATE:
            self._create(key, params)
            self.journal.append((key, Op(name, dict(params))))
            return
        linked = name in LINKED_OPS
        targets = list(self.layers.values()) if linked else [self.layers[key]]
        lengths = [len(layer.ops) for layer in targets]
        try:
            for layer in targets:
                layer.add(name, params)
        except Exception:
            for layer, length in zip(targets, lengths):
                while len(layer.ops) > length:
                    layer.undo()
            raise
        self.journal.append((ALL if linked else key, Op(name, dict(params))))

    def undo(self, key: str):
        """ Removes the last operation of a channel (of all channels if it was applied to all of them). A created
            channel without steps is removed. """
        layer = self.layers[key]
        if not layer.ops:
            if key in self.created:
                params = dict(self.created[key]["params"])
                self.remove_channel(key)
                self.redo_steps.append((key, Op(CREATE, params)))
            return
        last = layer.ops[-1]
        if last.name in LINKED_OPS:
            others = [other for other in self.layers.values() if other is not layer]
            if any(not (other.ops and other.ops[-1] == last) for other in others):
                raise ops.OpError(f"'{last.describe()}' was applied to all channels, and other channels have later "
                                  "operations (or were created after it). Undo or remove those first.")
            for other in others:
                other.undo()
        layer.undo()
        wanted = ALL if last.name in LINKED_OPS else key
        for index in range(len(self.journal) - 1, -1, -1):
            if self.journal[index][0] == wanted:
                del self.journal[index]
                break
        self.redo_steps.append((wanted, last))

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_steps)

    def redo(self) -> tuple:
        """ Applies again the step taken back last by undo; returns (channel key or ALL, Op). Raises ops.OpError (the
            step stays available) if it cannot be applied. """
        if not self.redo_steps:
            raise ops.OpError("Nothing to redo.")
        key, op = self.redo_steps[-1]
        target = next(iter(self.layers)) if key == ALL else key
        if op.name != CREATE and target not in self.layers:
            raise ops.OpError(f"The channel '{target}' does not exist any more.")
        self._redoing = True
        try:
            self.apply(target, op.name, dict(op.params))
        finally:
            self._redoing = False
        self.redo_steps.pop()
        return key, op

    def _journal_index(self, key: str, index: int) -> int:
        """ The position in the journal of step `index` (0 = oldest) in the history of channel `key`. """
        exists = key not in self.created                  # a created channel has the steps made after its creation
        count = -1
        for position, (k, op) in enumerate(self.journal):
            if k == key and op.name == CREATE:
                exists = True
                continue
            if (k == key or (k == ALL and exists)) and op.name != CREATE:
                count += 1
                if count == index:
                    return position
        raise ops.OpError(f"The channel '{key}' has no step {index + 1}.")

    def remove_step(self, key: str, index: int) -> Op:
        """ Removes one step (0 = oldest) from the history of a channel, also one in the middle: the original data is
            processed again with every other step of the file, in the order they were made, so steps that come
            later, steps of other channels that read this one, and channels made from it follow. A step applied to all
            channels (crop, flip, resample, rescale) is removed from all of them. All or nothing: if a later step
            fails without the removed one, nothing changes and the error names it. Returns the removed step. """
        position = self._journal_index(key, index)
        journal = self.journal[:position] + self.journal[position + 1:]
        removed = self.journal[position][1]
        saved = (dict(self.layers), dict(self.created), list(self.journal),
                 {k: (layer.ops[:], layer._states[:]) for k, layer in self.layers.items()}, list(self.redo_steps))
        try:
            self._reset()
            for number, (k, op) in enumerate(journal, start=1):
                target = next(iter(self.layers)) if k == ALL else k
                try:
                    self.apply(target, op.name, op.params)
                except ops.OpError as e:
                    where = "all channels" if k == ALL else f"'{k}'"
                    raise ops.OpError(f"Without that step, a later one fails: {op.describe()} on {where}: {e}")
        except Exception:
            self.layers, self.created, self.journal, self.redo_steps = saved[0], saved[1], saved[2], saved[4]
            for k, (ops_list, states) in saved[3].items():
                self.layers[k].ops, self.layers[k]._states = ops_list, states
            raise
        return removed

    def reset(self, key: Optional[str] = None):
        self.redo_steps.clear()
        self._reset(key)

    def _reset(self, key: Optional[str] = None):
        """ Back to the original data: one channel, or every channel (key None; created channels are removed). """
        if key is None:
            for created in list(self.created):
                del self.layers[created]
            self.created.clear()
        for k, layer in self.layers.items():
            if key is None or k == key:
                layer.reset()
        self.journal = [] if key is None else [(k, op) for k, op in self.journal if k != key]

    def extend(self, key: str, steps: list):
        """ Applies operations to one channel, all or nothing; the error names the step. """
        start = len(self.journal)
        for number, op in enumerate(steps, start=1):
            try:
                self.apply(key, op.name, op.params)
            except ops.OpError as e:
                self._undo_journal(start)
                raise ops.OpError(f"Step {number} ({op.describe()}): {e}")

    def _undo_journal(self, length: int):
        while len(self.journal) > length:
            key, _ = self.journal[-1]
            self.undo(next(iter(self.layers)) if key == ALL else key)

    def macro(self, grains: Optional[dict] = None) -> "Macro":
        """ The steps applied so far as a macro; grains: {'channel': key, 'settings': {...}} of a grain analysis. """
        return Macro(list(self.journal), grains)

    def apply_macro(self, macro: "Macro", current: str) -> list:
        """ Applies the steps of a macro, all or nothing. Steps of a channel this file does not have are skipped
            (returned as notes); a macro that uses a single channel is applied to `current` if that channel is
            missing. Returns the notes. Raises ops.OpError (nothing changed) if a step fails. """
        mapping, notes = self._channel_mapping(macro, current), []
        start = len(self.journal)
        for number, (key, op) in enumerate(macro.steps, start=1):
            if op.name == CREATE:                         # made here only if both sources exist
                params = self._mapped_params(op.params, mapping)
                if not all(params.get(n) in self.layers for n in ("base", "source")):
                    mapping[key] = None
                    notes.append(f"Skipped the channel '{key}' (made from channels that are not in this file).")
                    continue
                mapping[key] = key
            target = mapping.get(key)
            if target is None:
                if f"channel '{key}'" not in " ".join(notes):
                    notes.append(f"Skipped the steps of channel '{key}' (not in this file).")
                continue
            try:
                self.apply(target, op.name, self._mapped_params(op.params, mapping))
            except ops.OpError as e:
                self._undo_journal(start)
                raise ops.OpError(f"Step {number} ({op.describe()}{'' if key in (ALL, CURRENT) else ' on ' + key}): {e}")
        return notes

    def _channel_mapping(self, macro: "Macro", current: str) -> dict:
        keys = macro.channels()
        mapping = {ALL: current, CURRENT: current}
        for key in keys:
            if key in self.layers:
                mapping[key] = key
        missing = [k for k in keys if k not in self.layers]
        if len(keys) == 1 and missing:
            mapping[keys[0]] = current                    # a single-channel macro follows the shown channel
        return mapping

    @staticmethod
    def _mapped_params(params: dict, mapping: dict) -> dict:
        """ 'source' / 'reference' parameters that name a channel are mapped like the steps. """
        out = dict(params)
        for name in ("source", "reference", "base"):
            value = out.get(name)
            if isinstance(value, str) and value in mapping and value not in (ALL, CURRENT):
                out[name] = mapping[value]
        return out

    def channel(self, key: str) -> Channel:
        """ The channel with its current processed data. """
        state = self.layers[key].current
        name, direction, _ = self.channel_info(key)
        return Channel(name, direction, state.unit, state.data, state.size_x, state.size_y)


@dataclasses.dataclass
class Macro:
    """ Steps (channel key, Op) in the order they were applied, and optionally a grain analysis to run at the end
        ({'channel': key, 'settings': {...}}). A recipe of version 1 (one list of operations) is a macro whose steps
        all belong to CURRENT, the channel that is shown when it is applied. """
    steps: list
    grains: Optional[dict] = None

    def channels(self) -> list:
        """ The channel keys the steps belong to (without ALL and CURRENT), in order. """
        keys = []
        for key, _ in self.steps:
            if key not in (ALL, CURRENT) and key not in keys:
                keys.append(key)
        return keys

    def describe(self) -> str:
        lines = []
        for key, op in self.steps:
            where = "all channels" if key == ALL else "shown channel" if key == CURRENT else key
            lines.append(f"{op.describe()}  [{where}]")
        if self.grains:
            lines.append(f"Grain analysis  [{self.grains.get('channel') or 'shown channel'}]")
        return "\n".join(lines)


def macro_to_text(macro: Macro) -> str:
    content = {"format": RECIPE_FORMAT, "version": RECIPE_VERSION,
               "steps": [{"channel": key, **op.to_dict()} for key, op in macro.steps]}
    if macro.grains:
        content["grains"] = macro.grains
    return json.dumps(content, indent=1, ensure_ascii=False)


def macro_from_text(text: str) -> Macro:
    try:
        content = json.loads(text)
    except json.JSONDecodeError as e:
        raise ops.OpError(f"The macro is not valid JSON: {e}")
    if not isinstance(content, dict) or content.get("format") not in (RECIPE_FORMAT,) + OLD_RECIPE_FORMATS:
        raise ops.OpError("This is not a macro or recipe of NSSA (or its earlier names).")
    if content.get("version", 1) > RECIPE_VERSION:
        raise ops.OpError("The macro was written by a newer version of the program.")
    if "steps" in content:
        steps = [(str(item.get("channel", CURRENT)), Op.from_dict(item)) for item in content["steps"]]
    else:
        steps = [(CURRENT, Op.from_dict(item)) for item in content.get("ops", [])]
    grains = content.get("grains")
    if grains is not None and not isinstance(grains, dict):
        raise ops.OpError("The grain settings of the macro are invalid.")
    return Macro(steps, grains)


def save_macro(path, macro: Macro):
    pathlib.Path(path).write_text(macro_to_text(macro), encoding="utf-8")


def load_macro(path) -> Macro:
    return macro_from_text(pathlib.Path(path).read_text(encoding="utf-8"))


def recipe_to_text(recipe: list) -> str:
    return json.dumps({"format": RECIPE_FORMAT, "version": 1, "ops": [op.to_dict() for op in recipe]},
                      indent=1)


def recipe_from_text(text: str) -> list:
    """ The operations of a recipe or of a macro whose steps all belong to one channel. """
    macro = macro_from_text(text)
    if len({key for key, _ in macro.steps if key not in (ALL, CURRENT)}) > 1:
        raise ops.OpError("This macro has steps for several channels: apply it as a macro.")
    return [op for _, op in macro.steps]


def save_recipe(path, recipe: list):
    pathlib.Path(path).write_text(recipe_to_text(recipe), encoding="utf-8")


def load_recipe(path) -> list:
    return recipe_from_text(pathlib.Path(path).read_text(encoding="utf-8"))
