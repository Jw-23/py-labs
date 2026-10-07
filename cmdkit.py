"""cmdkit —— 用装饰器把 Python 类变成命令行程序。

设计目标只有一句话:写完一个普通类,命令行就自动有了 ——

* ``@command`` 装饰类,``@subcommand`` 装饰方法;
* 方法签名(参数名 / 类型标注 / 默认值)就是命令行参数的定义;
* 装饰器可以额外声明:是否必填、默认值、描述、别名、取值范围……
* 出错时抛出的异常里自带完整帮助文本,``main()`` 会自动打印帮助并返回退出码。

最小例子::

    from cmdkit import command, subcommand

    @command(description="我的小工具", version="0.1.0")
    class Tool:
        '''演示用命令行。'''

        @subcommand(help="打个招呼")
        def hello(self, name: str, *, times: int = 1):
            for _ in range(times):
                print(f"hello, {name}!")

    if __name__ == "__main__":
        raise SystemExit(Tool.main())

命令行行为::

    $ python tool.py hello world --times 3
    $ python tool.py --help
    $ python tool.py hello            # 缺参数 -> 打印错误 + 帮助, 退出码 2

类上会多出这些东西::

    Tool.parse(["hello", "world"])   # -> Tool 实例, 参数已经填好
    cmd = Tool.parse(sys.argv[1:])
    cmd.subcommand                   # "hello"(规范名,传别名也是它)
    cmd.command_line                 # 原始 argv
    cmd.execute()                    # 执行子命令, 返回值原样返回(叫 run() 也行)
    Tool.main()                      # 解析 + 执行 + 出错自动打印帮助, 返回退出码
    Tool.help("hello")               # 直接拿子命令的帮助文本

参数怎么描述(三种写法可以混用):

1. 只靠签名 —— 没有默认值就是必填,有默认值就是可选::

       def hello(self, name: str, times: int = 1): ...

2. 用 :func:`param` / :func:`flag` 写在默认值的位置::

       def hello(self, name: str = param(help="名字"),
                 times: int = param(1, "-n", "--times", help="次数"),
                 loud: bool = flag("-l", "--loud", help="喊出来")): ...

3. 用装饰器参数集中声明(适合"签名不动、只补元数据")::

       @subcommand(help="打招呼", params={
           "name":  {"required": True, "help": "名字"},
           "times": {"default": 1, "help": "次数", "flag": "-n --times"},
           "loud":  {"default": False, "help": "喊出来", "flag": "-l"},
       })
       def hello(self, name, times, loud): ...

规则速查:

=====================  =======================================================
``param(0, "-s")``     变成选项 ``-s``(写了名字就是选项)
不写名字               位置参数,顺序 = 签名顺序
keyword-only 参数      自动变成 ``--参数名``
``bool`` 标注          自动变成开关 ``store_true``,默认 False
没有默认值             必填(例外:``bool``、``Optional[...]``)
``list[int]``          自动 ``nargs``,可吃多个值
``tuple[int, int]``    固定两个值,结果是 tuple
``Literal[...]``/``Enum``  自动变成 ``choices``,写错会报错并带帮助
=====================  =======================================================

小陷阱也顺手兜住了:``param("-t", "--target")`` 里以 ``-`` 开头的一律当选项名,
不会被误读成"默认值是字符串 -t";子命令叫 ``run`` 也不会被调度器覆盖。
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import inspect
import keyword
import sys
import types
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

__all__ = [
    "command",
    "subcommand",
    "param",
    "option",
    "flag",
    "Param",
    "CommandExit",
    "CommandError",
    "HelpRequested",
    "VersionRequested",
]

_SUB_DEST = "__cmdkit_command__"
_NO_VALUE_ACTIONS = frozenset({"store_true", "store_false", "store_const", "count", "help", "version"})
_NAME_KEYS = ("flag", "flags", "names", "alias", "aliases", "option", "options")


class _Missing:
    """哨兵:表示"没有默认值",和 ``None`` 区分开。"""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - 只为调试
        return "<未设置>"

    def __bool__(self) -> bool:
        return False


_MISSING = _Missing()


# --------------------------------------------------------------------------- #
# 参数元数据
# --------------------------------------------------------------------------- #
class Param:
    """一个命令行参数的元数据。日常用 :func:`param` / :func:`flag` 构造。"""

    __slots__ = (
        "default",
        "names",
        "required",
        "help",
        "choices",
        "nargs",
        "action",
        "metavar",
        "type",
        "dest",
        "hidden",
    )

    def __init__(
        self,
        default: Any = _MISSING,
        *names: str,
        required: Optional[bool] = None,
        help: Optional[str] = None,
        choices: Optional[Sequence[Any]] = None,
        nargs: Any = None,
        action: Optional[str] = None,
        metavar: Optional[str] = None,
        type: Optional[Callable[[str], Any]] = None,
        dest: Optional[str] = None,
        hidden: bool = False,
    ) -> None:
        if isinstance(default, str) and default.startswith("-") and len(default) > 1:
            # 兜住 `param("-t", "--target")` 这种漏写默认值的写法:
            # 以短横线开头的一律当成选项名,而不是默认值。
            names = (default, *names)
            default = _MISSING
        self.default = default
        self.names: Tuple[str, ...] = tuple(_split_names(names))
        self.required = required
        self.help = help
        self.choices = tuple(choices) if choices is not None else None
        self.nargs = nargs
        self.action = action
        self.metavar = metavar
        self.type = type
        self.dest = dest
        self.hidden = hidden

    def replaced(self, other: "Param") -> "Param":
        """返回合并结果:``other`` 里显式给出的字段覆盖自己。"""
        new = Param(other.default if other.default is not _MISSING else self.default, *(other.names or self.names))
        for field in ("required", "help", "choices", "nargs", "action", "metavar", "type", "dest"):
            value = getattr(other, field)
            setattr(new, field, value if value is not None else getattr(self, field))
        new.hidden = self.hidden or other.hidden
        return new

    @classmethod
    def from_spec(cls, spec: Any) -> "Param":
        """把 ``{"required": True, "default": 1, "help": "..."}`` 变成 :class:`Param`。"""
        if isinstance(spec, Param):
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError(f"参数声明只能是 dict 或 Param,收到 {type(spec).__name__}")
        data: Dict[str, Any] = dict(spec)
        default = data.pop("default", _MISSING)
        names: List[str] = []
        for key in _NAME_KEYS:
            if key in data:
                names.extend(_split_names(data.pop(key)))
        if isinstance(data.get("type"), type) and data["type"] is str:
            data["type"] = None
        return cls(default, *names, **data)

    def __repr__(self) -> str:  # pragma: no cover - 只为调试
        bits = [name for name in ("default", "required", "help", "action") if getattr(self, name) is not None]
        shown = ", ".join(f"{b}={getattr(self, b)!r}" for b in bits)
        label = " ".join(self.names) or "位置参数"
        return f"Param({label}{', ' + shown if shown else ''})"


def _split_names(names: Iterable[Any]) -> List[str]:
    """``("-s --seed",)`` / ``["-s", "--seed"]`` / ``"-s,--seed"`` 都拆成列表。"""
    if isinstance(names, str):
        names = [names]
    out: List[str] = []
    for item in names:
        if item is None:
            continue
        if isinstance(item, str):
            out.extend(piece for piece in item.replace(",", " ").split() if piece)
        else:
            out.extend(_split_names(item))
    return out


def param(default: Any = _MISSING, *names: str, **kwargs: Any) -> Param:
    """声明一个参数。省略 ``names`` 时按签名决定是位置参数还是 ``--选项``。"""
    return Param(default, *names, **kwargs)


def option(default: Any = _MISSING, *names: str, **kwargs: Any) -> Param:
    """和 :func:`param` 一样,读起来更像"这是个选项"。"""
    return Param(default, *names, **kwargs)


def flag(*names: str, default: Any = False, **kwargs: Any) -> Param:
    """布尔开关。默认 ``action="store_true"``,需要取反就传 ``action="store_false"``。"""
    kwargs.setdefault("action", "store_true")
    return Param(default, *names, **kwargs)


# --------------------------------------------------------------------------- #
# 异常:报错时把帮助一起带出来
# --------------------------------------------------------------------------- #
class CommandExit(Exception):
    """解析过程中的"正常退出"或"报错退出",自带帮助文本。"""

    code = 0
    error_prefix = False

    def __init__(
        self,
        message: str = "",
        *,
        help_text: str = "",
        parser: Optional[argparse.ArgumentParser] = None,
        code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.help_text = help_text or (parser.format_help() if parser is not None else "")
        self.parser = parser
        if code is not None:
            self.code = code

    @property
    def prog(self) -> str:
        return self.parser.prog if self.parser is not None else ""

    def render(self) -> str:
        """错误信息 + 用法 + 完整帮助(调用方什么都不用拼)。"""
        parts: List[str] = []
        if self.message:
            if not self.error_prefix:
                parts.append(self.message)
            else:
                parts.append(f"{self.prog}: error: {self.message}" if self.prog else f"error: {self.message}")
        if self.help_text:
            parts.append(self.help_text.rstrip("\n"))
        return "\n\n".join(parts) + "\n"

    def __str__(self) -> str:
        # 让 print(exc) 直接吐出"错误 + 帮助"。
        return self.render()


class CommandError(CommandExit):
    """参数用错了。``.render()`` / ``str(exc)`` 会给出错误 + 帮助,退出码 2。"""

    code = 2
    error_prefix = True


class HelpRequested(CommandExit):
    """用户请求了 ``--help``(退出码 0)。"""

    code = 0


class VersionRequested(CommandExit):
    """用户请求了 ``--version``(退出码 0)。"""

    code = 0


class _HelpAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None, **kwargs):
        super().__init__(
            option_strings=option_strings,
            dest=argparse.SUPPRESS,
            default=argparse.SUPPRESS,
            nargs=0,
            help=help or "显示帮助信息并退出",
        )

    def __call__(self, parser, namespace, values, option_string=None):
        raise HelpRequested(parser=parser)


class _VersionAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, version="", help=None, **kwargs):
        self.version = version
        super().__init__(
            option_strings=option_strings,
            dest=argparse.SUPPRESS,
            default=argparse.SUPPRESS,
            nargs=0,
            help=help or "显示版本号并退出",
        )

    def __call__(self, parser, namespace, values, option_string=None):
        # 版本号是结果本身,不追加帮助。
        raise VersionRequested(f"{parser.prog} {self.version}")


class _Parser(argparse.ArgumentParser):
    """把 argparse 的"打印 + 退出"改成抛异常,好让调用方拿到帮助文本。"""

    def error(self, message):  # type: ignore[override]
        raise CommandError(message, parser=self)

    def exit(self, status=0, message=None):  # type: ignore[override]
        if status:
            raise CommandError(message or "", parser=self, code=status)
        raise HelpRequested(parser=self)

    def print_help(self, file=None):  # type: ignore[override]
        super().print_help(file or sys.stdout)


def _add_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-h", "--help", action=_HelpAction)


# --------------------------------------------------------------------------- #
# 类型标注 -> argparse 能力
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _TypeInfo:
    converter: Optional[Callable[[str], Any]] = None
    nargs: Any = None
    choices: Optional[Tuple[Any, ...]] = None
    metavar: Optional[str] = None
    post: Optional[Callable[[Any], Any]] = None
    is_bool: bool = False
    is_list: bool = False
    optional: bool = False


def _resolve_type(annotation: Any) -> _TypeInfo:
    if annotation in (inspect.Parameter.empty, None, Any, str):
        return _TypeInfo()

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1:
            info = _resolve_type(rest[0])
            info.optional = True
            return info
        return _TypeInfo()

    if origin is Literal:
        return _TypeInfo(choices=tuple(args))

    if origin in (list, List, Sequence, Iterable):
        inner = _resolve_type(args[0]) if args else _TypeInfo()
        return _TypeInfo(converter=inner.converter, choices=inner.choices, metavar=inner.metavar, is_list=True)

    if origin in (tuple, Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            inner = _resolve_type(args[0])
            return _TypeInfo(converter=inner.converter, choices=inner.choices, metavar=inner.metavar, is_list=True)
        if args:
            inner = _resolve_type(args[0]) if len(set(args)) == 1 else _TypeInfo()
            return _TypeInfo(
                converter=inner.converter,
                choices=inner.choices,
                metavar=inner.metavar,
                nargs=len(args),
                post=tuple,
            )
        return _TypeInfo(is_list=True)

    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            names = ", ".join(name.lower() for name in annotation.__members__)
            return _TypeInfo(converter=_enum_converter(annotation), metavar=f"{{{names}}}")
        if annotation is bool:
            return _TypeInfo(is_bool=True)
        if annotation is Path:
            return _TypeInfo(converter=Path)
        if annotation in (int, float, complex, str):
            return _TypeInfo(converter=annotation)
        if callable(annotation):
            return _TypeInfo(converter=annotation)
    return _TypeInfo()


def _enum_converter(enum_cls: type) -> Callable[[str], Any]:
    """Enum 参数:成员名和取值都能写,写错了给出可选清单。"""
    lookup: Dict[str, Any] = {name.lower(): member for name, member in enum_cls.__members__.items()}
    for member in enum_cls:
        lookup.setdefault(str(member.value).lower(), member)
    choices = ", ".join(name.lower() for name in enum_cls.__members__)

    def convert(text: str) -> Any:
        try:
            return lookup[text.lower()]
        except (KeyError, AttributeError):
            raise argparse.ArgumentTypeError(
                f"{text!r} 不是合法的 {enum_cls.__name__},可选:{choices}"
            ) from None

    return convert


# --------------------------------------------------------------------------- #
# 签名 -> 参数规格
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _ParamSpec:
    name: str
    default: Any = _MISSING
    required: bool = False
    help: Optional[str] = None
    names: Tuple[str, ...] = ()
    converter: Optional[Callable[[str], Any]] = None
    choices: Optional[Tuple[Any, ...]] = None
    nargs: Any = None
    action: Optional[str] = None
    metavar: Optional[str] = None
    post: Optional[Callable[[Any], Any]] = None
    hidden: bool = False
    is_bool: bool = False

    @property
    def positional(self) -> bool:
        return not self.names


def _cli_name(name: str) -> str:
    """``dry_run`` -> ``dry-run``,``class_`` -> ``class``。"""
    base = name
    if name.endswith("_"):
        trimmed = name[:-1]
        if keyword.iskeyword(trimmed):
            base = trimmed
    return base.replace("_", "-")


def _type_hints(func: Callable[..., Any]) -> Dict[str, Any]:
    try:
        return get_type_hints(func)
    except Exception:  # 前向引用解析不了就退化成"没有标注"
        return {}


def _build_param_spec(
    name: str,
    annotation: Any,
    python_default: Any,
    marker: Optional[Param],
    override: Any,
    *,
    keyword_only: bool,
) -> _ParamSpec:
    marker = marker or Param()
    if override is not None:
        marker = marker.replaced(Param.from_spec(override))

    if marker.default is not _MISSING:
        default = marker.default
    elif python_default is not inspect.Parameter.empty and not isinstance(python_default, Param):
        default = python_default
    else:
        default = _MISSING

    info = _resolve_type(annotation)
    names = tuple(marker.names)
    if not names and (keyword_only or info.is_bool):
        names = (f"--{_cli_name(name)}",)

    action = marker.action
    if action is None and info.is_bool:
        action = "store_true"
    if (
        action is None
        and default is False
        and info.converter is None
        and info.choices is None
        and marker.nargs is None
    ):
        # 只写了 default=False 的开关(常见于装饰器 params= 声明)当成 store_true。
        action = "store_true"

    if marker.required is not None:
        required = bool(marker.required)
    elif info.optional or info.is_bool:
        required = False
    else:
        required = default is _MISSING

    if info.is_bool and default is _MISSING:
        default = False

    if action in _NO_VALUE_ACTIONS:
        converter = None
        choices = None
        nargs = None
        if default is _MISSING:
            default = False
    else:
        converter = marker.type or info.converter
        choices = marker.choices if marker.choices is not None else info.choices
        if marker.nargs is not None:
            nargs = marker.nargs
        elif info.nargs is not None:
            nargs = info.nargs
        elif info.is_list:
            nargs = "+" if required else "*"
        else:
            nargs = None

    if not required and default is _MISSING:
        default = False if info.is_bool else None

    return _ParamSpec(
        name=marker.dest or name,
        default=default,
        required=required,
        help=marker.help,
        names=names,
        converter=converter,
        choices=choices,
        nargs=nargs,
        action=action,
        metavar=marker.metavar,
        post=info.post,
        hidden=marker.hidden,
        is_bool=info.is_bool,
    )


def _collect_specs(
    func: Callable[..., Any],
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    known: Iterable[str] = ("self", "cls"),
) -> List[_ParamSpec]:
    hints = _type_hints(func)
    overrides = dict(overrides or {})
    specs: List[_ParamSpec] = []
    for pname, p in inspect.signature(func).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if pname in known and not specs:
            continue
        override = overrides.pop(pname, None)
        marker = p.default if isinstance(p.default, Param) else None
        specs.append(
            _build_param_spec(
                pname,
                hints.get(pname, p.annotation),
                p.default,
                marker,
                override,
                keyword_only=p.kind is p.KEYWORD_ONLY,
            )
        )
    if overrides:
        raise TypeError(f"{func.__qualname__}(...) 的 params 里有签名中不存在的参数: {sorted(overrides)}")
    return specs


def _add_arguments(parser: argparse.ArgumentParser, specs: Iterable[_ParamSpec], *, suppress: bool = False) -> None:
    for spec in specs:
        kwargs: Dict[str, Any] = {}
        if spec.hidden:
            kwargs["help"] = argparse.SUPPRESS
        elif spec.help is not None:
            kwargs["help"] = spec.help
        if spec.metavar:
            kwargs["metavar"] = spec.metavar

        default = argparse.SUPPRESS if suppress else (None if spec.default is _MISSING else spec.default)
        has_value = spec.action not in _NO_VALUE_ACTIONS
        if has_value:
            if spec.converter is not None:
                kwargs["type"] = spec.converter
            if spec.choices is not None:
                kwargs["choices"] = spec.choices

        if spec.positional:
            if spec.nargs is not None:
                kwargs["nargs"] = spec.nargs
            elif not spec.required:
                kwargs["nargs"] = "?"
            if default is not argparse.SUPPRESS:
                kwargs["default"] = default
            parser.add_argument(spec.name, **kwargs)
        else:
            if spec.action is not None:
                kwargs["action"] = spec.action
            if spec.nargs is not None:
                kwargs["nargs"] = spec.nargs
            kwargs["default"] = default
            if spec.required and has_value:
                kwargs["required"] = True
            parser.add_argument(*spec.names, dest=spec.name, **kwargs)


def _doc_parts(obj: Any) -> Tuple[Optional[str], Optional[str]]:
    """docstring -> (第一行, 全文)。"""
    doc = inspect.getdoc(obj)
    if not doc:
        return None, None
    first = doc.strip().splitlines()[0].strip()
    return (first or None), doc.strip()


# --------------------------------------------------------------------------- #
# 装饰器
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _Subcommand:
    name: str
    aliases: Tuple[str, ...]
    func: Callable[..., Any]
    params: List[_ParamSpec]
    help: Optional[str] = None
    description: Optional[str] = None
    epilog: Optional[str] = None
    parser: Optional[argparse.ArgumentParser] = None


@dataclasses.dataclass
class _Conf:
    cls: type
    parser: argparse.ArgumentParser
    single: bool
    params: List[_ParamSpec] = dataclasses.field(default_factory=list)
    globals: List[_ParamSpec] = dataclasses.field(default_factory=list)
    subcommands: Dict[str, _Subcommand] = dataclasses.field(default_factory=dict)
    aliases: Dict[str, str] = dataclasses.field(default_factory=dict)
    order: List[str] = dataclasses.field(default_factory=list)


def subcommand(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    aliases: Union[str, Iterable[str], None] = None,
    help: Optional[str] = None,
    description: Optional[str] = None,
    epilog: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
):
    """把一个方法注册成子命令。

    :param name: 命令名,默认取方法名并把下划线换成短横线(``gen_image`` -> ``gen-image``)
    :param aliases: 别名,可以传 ``"g gi"`` 或 ``["g", "gi"]``
    :param help: 出现在父命令里的单行说明,默认取 docstring 第一行
    :param description: ``--help`` 里的详细说明,默认取整个 docstring
    :param params: 逐参数补充 / 覆盖元数据,例如
        ``{"count": {"required": True, "default": 1, "help": "数量"}}``
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        raw = aliases
        func.__cmdkit_subcommand__ = {  # type: ignore[attr-defined]
            "name": name,
            "aliases": tuple(_split_names([raw] if isinstance(raw, str) else (raw or ()))),
            "help": help,
            "description": description,
            "epilog": epilog,
            "params": dict(params or {}),
        }
        return func

    return decorate(_func) if _func is not None else decorate


def command(
    _cls: Optional[type] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    prog: Optional[str] = None,
    epilog: Optional[str] = None,
    params: Any = None,
    formatter_class: Any = None,
):
    """把一个类变成命令行程序。

    :param name: 程序名(显示在 ``usage`` 里)
    :param description: 整体描述,默认取类 docstring
    :param version: 给了就会有 ``--version``
    :param params: 所有子命令通用的全局选项,例如
        ``[param(False, "-v", "--verbose", action="store_true", help="输出细节")]``
    :param formatter_class: 帮助排版类,默认 :class:`argparse.RawDescriptionHelpFormatter`
    """

    def decorate(cls: type) -> type:
        return _apply_command(
            cls,
            name=name,
            description=description,
            version=version,
            prog=prog,
            epilog=epilog,
            params=params,
            formatter_class=formatter_class,
        )

    return decorate(_cls) if _cls is not None else decorate


def _apply_command(cls: type, **options: Any) -> type:
    formatter = options["formatter_class"] or argparse.RawDescriptionHelpFormatter
    prog = options["prog"] or _default_prog(cls, options["name"])
    _, long_doc = _doc_parts(cls)
    description = options["description"] or long_doc
    raw_params = options["params"]

    methods = _collect_subcommands(cls)
    if methods:
        globals_specs = _global_specs(raw_params)
        single_params: List[_ParamSpec] = []
    elif isinstance(raw_params, Mapping):
        globals_specs = []
        single_params = _collect_specs(cls.__init__, raw_params)
    else:
        globals_specs = _global_specs(raw_params)
        single_params = _collect_specs(cls.__init__)

    if methods:
        _zero_arg_init(cls)
        conf = _build_subcommand_conf(cls, methods, globals_specs, options, prog, formatter, description)
    else:
        conf = _build_single_conf(cls, single_params, globals_specs, options, prog, formatter, description)

    reserved = {"parse", "main", "help", "execute"} & set(conf.subcommands)
    if reserved:
        raise TypeError(
            f"{cls.__name__} 的子命令名 {sorted(reserved)} 和命令行 API 重名,"
            "请用 @subcommand(name=\"...\") 换个名字"
        )

    cls._cmdkit_conf = conf  # type: ignore[attr-defined]
    cls.parser = conf.parser  # type: ignore[attr-defined]
    cls.parse = classmethod(_parse_impl)  # type: ignore[attr-defined]
    cls.main = classmethod(_main_impl)  # type: ignore[attr-defined]
    cls.help = classmethod(_help_impl)  # type: ignore[attr-defined]
    cls.execute = _execute_impl  # type: ignore[attr-defined]
    if not any("run" in vars(klass) for klass in cls.__mro__ if klass is not object):
        cls.run = _execute_impl  # type: ignore[attr-defined]
    cls.__cmdkit_command__ = True  # type: ignore[attr-defined]
    if cls.__repr__ is object.__repr__:
        cls.__repr__ = _repr_impl  # type: ignore[method-assign]
    return cls


def _default_prog(cls: type, name: Optional[str]) -> str:
    if name:
        return name
    argv0 = sys.argv[0] if sys.argv else ""
    prog = Path(argv0).name if argv0 else ""
    if prog and not prog.startswith("-") and Path(prog).stem != "__main__":
        return Path(prog).stem
    return cls.__name__.replace("_", "-").lower()


def _global_specs(params: Any) -> List[_ParamSpec]:
    if not params:
        return []
    items = params.items() if isinstance(params, Mapping) else [(None, p) for p in params]
    specs: List[_ParamSpec] = []
    for key, raw in items:
        marker = Param.from_spec(raw)
        name = marker.dest or key or _name_from_flags(marker.names)
        if not name:
            raise TypeError("全局参数需要 dest= 或以 --长选项 命名")
        names = tuple(marker.names) or (f"--{_cli_name(name)}",)
        action = marker.action
        if action is None and marker.default is False and marker.type is None:
            action = "store_true"
        info = _resolve_type(marker.type)
        has_value = action not in _NO_VALUE_ACTIONS
        default = marker.default if marker.default is not _MISSING else (False if not has_value else None)
        specs.append(
            _ParamSpec(
                name=name,
                default=default,
                required=bool(marker.required),
                help=marker.help,
                names=names,
                converter=(marker.type or info.converter) if has_value else None,
                choices=marker.choices,
                nargs=marker.nargs,
                action=action,
                metavar=marker.metavar,
                hidden=marker.hidden,
            )
        )
    return specs


def _name_from_flags(names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name.startswith("--"):
            return _cli_name(name[2:]).replace("-", "_")
    return None


def _collect_subcommands(cls: type) -> List[_Subcommand]:
    found: Dict[str, _Subcommand] = {}
    for klass in reversed(cls.__mro__):
        for attr, value in vars(klass).items():
            meta = getattr(value, "__cmdkit_subcommand__", None)
            if meta is None:
                continue
            short, long = _doc_parts(value)
            found[meta["name"] or attr.replace("_", "-")] = _Subcommand(
                name=meta["name"] or attr.replace("_", "-"),
                aliases=meta["aliases"],
                func=value,
                params=_collect_specs(value, meta["params"]),
                help=meta["help"] or short,
                description=meta["description"] or long,
                epilog=meta["epilog"],
            )
    return list(found.values())


def _zero_arg_init(cls: type) -> None:
    """子命令模式下 ``__init__`` 必须无参可调用,否则快速失败。"""
    init = cls.__init__
    if init is object.__init__:
        return
    for pname, p in inspect.signature(init).parameters.items():
        if pname in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty:
            raise TypeError(
                f"{cls.__name__}.__init__ 必须能无参调用:子命令的参数写在 @subcommand 方法上,"
                f"不要放在 __init__ 里(现在需要 {pname!r})"
            )


def _build_subcommand_conf(cls, methods, globals_specs, options, prog, formatter, description) -> _Conf:
    root = _Parser(
        prog=prog,
        description=description,
        epilog=options["epilog"],
        formatter_class=formatter,
        add_help=False,
    )
    _add_help(root)
    if options["version"]:
        root.add_argument("--version", action=_VersionAction, version=options["version"])
    _add_arguments(root, globals_specs)

    # 全局选项要在子命令前后都能写:子命令里挂一份"只填自己出现过的"副本。
    inherited = _Parser(add_help=False, argument_default=argparse.SUPPRESS)
    _add_arguments(inherited, globals_specs, suppress=True)

    sub_parsers = root.add_subparsers(dest=_SUB_DEST, metavar="<command>", required=True, title="可用命令")
    conf = _Conf(cls=cls, parser=root, single=False, globals=globals_specs)
    global_names = {spec.name for spec in globals_specs}
    global_flags = {name for spec in globals_specs for name in spec.names}
    for sub in methods:
        clash = sorted(({spec.name for spec in sub.params} & global_names) or ({n for spec in sub.params for n in spec.names} & global_flags))
        if clash:
            raise TypeError(
                f"子命令 {sub.name!r} 的参数 {clash} 和 @command(params=...) 里的全局参数重名,"
                "子命令里直接读 self.<名字> 就行"
            )
        kwargs: Dict[str, Any] = {
            "description": sub.description,
            "epilog": sub.epilog,
            "formatter_class": formatter,
            "add_help": False,
            "parents": [inherited],
        }
        if sub.help:
            kwargs["help"] = sub.help
        parser = sub_parsers.add_parser(sub.name, aliases=list(sub.aliases), **kwargs)
        _add_help(parser)
        _add_arguments(parser, sub.params)
        sub.parser = parser
        conf.subcommands[sub.name] = sub
        conf.order.append(sub.name)
        conf.aliases[sub.name] = sub.name
        for alias in sub.aliases:
            conf.aliases[alias] = sub.name
    return conf


def _build_single_conf(cls, params, globals_specs, options, prog, formatter, description) -> _Conf:
    root = _Parser(
        prog=prog,
        description=description,
        epilog=options["epilog"],
        formatter_class=formatter,
        add_help=False,
    )
    _add_help(root)
    if options["version"]:
        root.add_argument("--version", action=_VersionAction, version=options["version"])
    _add_arguments(root, globals_specs)
    _add_arguments(root, params)
    return _Conf(cls=cls, parser=root, single=True, params=params, globals=globals_specs)


# --------------------------------------------------------------------------- #
# 挂到类上的方法
# --------------------------------------------------------------------------- #
def _value(namespace: argparse.Namespace, spec: _ParamSpec) -> Any:
    value = getattr(namespace, spec.name, None if spec.default is _MISSING else spec.default)
    if spec.post is not None and value is not None:
        value = spec.post(value)
    return value


def _remember(obj: Any, argv: Sequence[str], globals_specs: Iterable[_ParamSpec], namespace, *, subcommand) -> None:
    obj.command_line = list(argv)
    obj.subcommand = subcommand
    for spec in globals_specs:
        setattr(obj, spec.name, _value(namespace, spec))
    obj.options = namespace


def _parse_impl(cls, argv: Optional[Sequence[str]] = None):
    """解析命令行,返回一个填好参数的类实例。

    出错时抛 :class:`CommandError`,``str(exc)`` 就是"错误 + 帮助"。
    """
    conf: _Conf = cls._cmdkit_conf
    argv = list(sys.argv[1:] if argv is None else argv)
    namespace = conf.parser.parse_args(argv)

    if conf.single:
        obj = cls(**{spec.name: _value(namespace, spec) for spec in conf.params})
        _remember(obj, argv, conf.globals, namespace, subcommand=None)
        return obj

    invoked = getattr(namespace, _SUB_DEST, None)
    sub = conf.subcommands[conf.aliases.get(invoked, invoked)]
    obj = cls.__new__(cls)
    if cls.__init__ is not object.__init__:
        cls.__init__(obj)
    for spec in sub.params:
        setattr(obj, spec.name, _value(namespace, spec))
    _remember(obj, argv, conf.globals, namespace, subcommand=sub.name)
    return obj


def _execute_impl(self):
    """执行 ``parse()`` 选中的子命令(单命令模式下无事可做)。

    别名 ``run()`` —— 只有当类里没有 ``run`` 这个方法时才会挂上,
    所以把子命令叫 ``run`` 也不会被覆盖。
    """
    conf: _Conf = type(self)._cmdkit_conf
    if conf.single:
        return None
    sub = conf.subcommands[self.subcommand]
    return sub.func(self, **{spec.name: getattr(self, spec.name) for spec in sub.params})


def _help_impl(cls, command_name: Optional[str] = None) -> str:
    """取帮助文本。``Tool.help()`` 是总帮助,``Tool.help("build")`` 是子命令帮助。"""
    conf: _Conf = cls._cmdkit_conf
    if command_name is None:
        return conf.parser.format_help()
    name = conf.aliases.get(command_name, command_name)
    if name not in conf.subcommands:
        raise KeyError(f"没有子命令 {command_name!r},可选:{', '.join(conf.order)}")
    parser = conf.subcommands[name].parser
    return parser.format_help() if parser is not None else conf.parser.format_help()


def _main_impl(cls, argv: Optional[Sequence[str]] = None, *, exit: bool = False):
    """解析 + 执行。报错时自动打印帮助,返回退出码(``exit=True`` 则抛 SystemExit)。"""
    try:
        obj = cls.parse(argv)
    except HelpRequested as exc:
        print(exc.render(), end="")
        code = exc.code
    except VersionRequested as exc:
        print(exc.render(), end="")
        code = exc.code
    except CommandError as exc:
        print(exc.render(), file=sys.stderr, end="")
        code = exc.code
    else:
        result = obj.execute()
        code = result if isinstance(result, int) and not isinstance(result, bool) else 0
    if exit:
        raise SystemExit(code)
    return code


def _repr_impl(self) -> str:
    conf: _Conf = type(self)._cmdkit_conf
    if conf.single:
        body = ", ".join(f"{spec.name}={getattr(self, spec.name, None)!r}" for spec in conf.params)
    else:
        sub = conf.subcommands.get(self.subcommand)
        inner = ", ".join(f"{s.name}={getattr(self, s.name, None)!r}" for s in sub.params) if sub else ""
        body = f"{self.subcommand}({inner})"
    return f"<{type(self).__name__} {body}>"
