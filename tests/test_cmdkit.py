import enum
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Literal, Optional

from cmdkit import CommandError, HelpRequested, VersionRequested, command, flag, param, subcommand


@command(prog="calc", version="1.2.3")
class Calc:
    """计算器演示。

    类 docstring 会成为整体描述。
    """

    @subcommand(help="相加", aliases="a plus")
    def add(
        self,
        a: int,
        b: int = 0,
        *,
        scale: float = param(1.0, "-s", "--scale", help="缩放比例"),
    ):
        """把两个数相加(这段是子命令的详细说明)。"""
        return (a + b) * scale

    @subcommand
    def echo(self, text: str, upper: bool = flag("-u", "--upper", help="转大写")):
        return text.upper() if upper else text


@command(prog="multi", params=[param(False, "-v", "--verbose", action="store_true", help="细节")])
class Multi:
    @subcommand(aliases=["r", "go"])
    def run(
        self,
        *,
        target: str = param("-t", "--target", help="目标"),
        mode: Literal["fast", "slow"] = "fast",
    ):
        """跑一次。

        子命令恰好叫 run:execute() 仍然是调度入口,run 不会被覆盖。
        """
        return f"{target}:{mode}:verbose={self.verbose}"


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"


@command(prog="types")
class Types:
    @subcommand
    def go(
        self,
        nums: list[int],
        *,
        size: tuple[int, int] = param((1, 1), "--size", help="宽 高"),
        out: Path = Path("a.txt"),
        seed: Optional[int] = None,
        color: Color = Color.RED,
    ):
        return nums, size, out, seed, color

    @subcommand
    def pos(self, out: Path = Path("a.txt")):
        return out


@command(prog="declared")
class Declared:
    @subcommand(
        params={
            "count": {"required": True, "help": "必填项"},
            "level": {"default": 3, "help": "默认值来自装饰器", "flag": "-l --level", "type": int},
            "loud": {"default": False, "help": "开关", "flag": "--loud"},
        }
    )
    def go(self, count, level, loud):
        return count, level, loud


@command(prog="single")
class Single:
    def __init__(self, count: int, out: Path = Path("o.txt")):
        self.count = count
        self.out = out


class CmdKitTests(unittest.TestCase):
    # -- 基本解析 ---------------------------------------------------------- #
    def test_positional_required_and_default(self):
        cmd = Calc.parse(["add", "2"])
        self.assertEqual((cmd.a, cmd.b, cmd.scale), (2, 0, 1.0))
        self.assertEqual(cmd.run(), 2)

    def test_type_conversion_and_options(self):
        cmd = Calc.parse(["add", "2", "3", "-s", "2.5"])
        self.assertEqual(cmd.run(), 12.5)
        self.assertIsInstance(cmd.scale, float)

    def test_bool_flag(self):
        self.assertEqual(Calc.parse(["echo", "hi"]).run(), "hi")
        self.assertEqual(Calc.parse(["echo", "hi", "-u"]).run(), "HI")

    def test_parse_returns_class_instance(self):
        cmd = Calc.parse(["echo", "hi"])
        self.assertIsInstance(cmd, Calc)
        self.assertEqual(cmd.subcommand, "echo")
        self.assertEqual(cmd.command_line, ["echo", "hi"])

    def test_underscore_method_becomes_dashed_command(self):
        cmd = Types.parse(["pos"])
        self.assertEqual(cmd.subcommand, "pos")

    def test_aliases_resolve_to_canonical_name(self):
        for argv in (["plus", "1"], ["a", "1"], ["add", "1"]):
            cmd = Calc.parse(argv)
            self.assertEqual(cmd.subcommand, "add")
            self.assertEqual(cmd.a, 1)

    # -- 类型标注 ---------------------------------------------------------- #
    def test_list_tuple_optional_and_path(self):
        nums, size, out, seed, color = Types.parse(
            ["go", "1", "2", "3", "--size", "4", "5", "--seed", "9", "--color", "blue"]
        ).run()
        self.assertEqual(nums, [1, 2, 3])
        self.assertEqual(size, (4, 5))
        self.assertIsInstance(size, tuple)
        self.assertIsInstance(out, Path)
        self.assertEqual(seed, 9)
        self.assertIs(color, Color.BLUE)

    def test_enum_default_and_name_or_value(self):
        self.assertIs(Types.parse(["go", "1"]).run()[4], Color.RED)
        self.assertIs(Types.parse(["go", "1", "--color", "GREEN"]).run()[4], Color.GREEN)
        with self.assertRaises(CommandError) as ctx:
            Types.parse(["go", "1", "--color", "purple"])
        self.assertIn("可选", str(ctx.exception))

    def test_literal_choices_reported_in_help(self):
        self.assertIn("{fast,slow}", Multi.help("run"))
        with self.assertRaises(CommandError):
            Multi.parse(["run", "-t", "x", "--mode", "medium"])

    # -- 装饰器声明元数据 -------------------------------------------------- #
    def test_decorator_params_required_default_and_help(self):
        count, level, loud = Declared.parse(["go", "5"]).run()
        self.assertEqual((count, level, loud), ("5", 3, False))
        self.assertEqual((Declared.parse(["go", "5", "-l", "7", "--loud"]).run()), ("5", 7, True))
        text = Declared.help("go")
        self.assertIn("必填项", text)
        self.assertIn("默认值来自装饰器", text)
        self.assertIn("-l, --level", text)
        self.assertIn("--loud", text)

    def test_missing_required_option_reports_help(self):
        with self.assertRaises(CommandError) as ctx:
            Multi.parse(["run"])
        self.assertIn("--target", str(ctx.exception))

    def test_unknown_param_in_decorator_is_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            @command(prog="bad")
            class Bad:
                @subcommand(params={"nope": {"default": 1}})
                def go(self, yes):
                    ...

        self.assertIn("nope", str(ctx.exception))

    def test_subcommand_init_with_required_args_is_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            @command(prog="bad2")
            class Bad:
                def __init__(self, size):
                    self.size = size

                @subcommand
                def go(self):
                    ...

        self.assertIn("__init__", str(ctx.exception))

    # -- 全局参数 ---------------------------------------------------------- #
    def test_global_option_before_and_after_subcommand(self):
        before = Multi.parse(["-v", "run", "-t", "x"])
        self.assertEqual(before.execute(), "x:fast:verbose=True")
        self.assertIs(before.verbose, True)
        self.assertEqual(Multi.parse(["run", "-t", "x", "-v"]).execute(), "x:fast:verbose=True")
        self.assertEqual(Multi.parse(["run", "-t", "x"]).execute(), "x:fast:verbose=False")

    def test_subcommand_named_run_is_not_shadowed(self):
        self.assertTrue(callable(getattr(Multi, "execute")))
        self.assertEqual(Multi.parse(["run", "-t", "x"]).subcommand, "run")

    # -- 帮助与报错 -------------------------------------------------------- #
    def test_help_text_contains_subcommands_and_docstring(self):
        text = Calc.help()
        self.assertIn("usage: calc", text)
        self.assertIn("计算器演示。", text)
        self.assertIn("把两个数相加", Calc.help("add"))
        self.assertIn("{fast,slow}", Multi.help("go"))  # 别名也能取帮助
        with self.assertRaises(KeyError):
            Multi.help("nope")

    def test_help_requested_raises_with_text(self):
        with self.assertRaises(HelpRequested) as ctx:
            Calc.parse(["--help"])
        self.assertIn("usage: calc", str(ctx.exception))
        with self.assertRaises(HelpRequested):
            Calc.parse(["add", "--help"])

    def test_version(self):
        with self.assertRaises(VersionRequested) as ctx:
            Calc.parse(["--version"])
        self.assertEqual(str(ctx.exception).strip(), "calc 1.2.3")

    def test_error_is_self_describing(self):
        with self.assertRaises(CommandError) as ctx:
            Calc.parse(["add"])
        error = ctx.exception
        text = str(error)
        self.assertEqual(error.code, 2)
        self.assertIn("error:", text)
        self.assertIn("usage: calc add", text)  # 只给相关子命令的用法
        self.assertIn("-h, --help", text)  # 完整帮助
        self.assertIn("缩放比例", text)

    def test_missing_subcommand_shows_full_help(self):
        with self.assertRaises(CommandError) as ctx:
            Calc.parse([])
        text = str(ctx.exception)
        self.assertIn("可用命令", text)
        self.assertIn("add", text)

    def test_unknown_subcommand_and_extra_argument(self):
        with self.assertRaises(CommandError):
            Calc.parse(["nope"])
        with self.assertRaises(CommandError) as ctx:
            Calc.parse(["add", "1", "extra-positional"])
        self.assertIn("error:", str(ctx.exception))

    def test_parse_does_not_print_anything(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), self.assertRaises(CommandError):
            Calc.parse(["add"])
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))

    # -- main() ------------------------------------------------------------ #
    def test_main_runs_and_returns_zero(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = Calc.main(["echo", "hi"])
        self.assertEqual(code, 0)

    def test_main_prints_help_to_stdout(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = Calc.main(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("usage: calc", out.getvalue())

    def test_main_prints_error_plus_help_to_stderr(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = Calc.main(["add"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("error:", err.getvalue())
        self.assertIn("usage: calc add", err.getvalue())
        self.assertIn("缩放比例", err.getvalue())

    def test_main_uses_int_return_value_as_exit_code(self):
        @command(prog="code")
        class Code:
            @subcommand
            def fail(self) -> int:
                return 3

        self.assertEqual(Code.main(["fail"]), 3)
        with self.assertRaises(SystemExit) as ctx:
            Code.main(["fail"], exit=True)
        self.assertEqual(ctx.exception.code, 3)

    def test_global_and_subcommand_param_clash_is_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            @command(prog="clash", params=[param(False, "--verbose", action="store_true")])
            class Clash:
                @subcommand
                def go(self, *, verbose: bool = False):
                    ...

        self.assertIn("重名", str(ctx.exception))

    # -- 单命令模式 -------------------------------------------------------- #
    def test_single_command_mode_uses_init_signature(self):
        cmd = Single.parse(["7", "x.txt"])
        self.assertEqual(cmd.count, 7)
        self.assertEqual(cmd.out, Path("x.txt"))
        self.assertIsNone(cmd.subcommand)
        self.assertIsNone(cmd.run())

    def test_single_command_error_carries_help(self):
        with self.assertRaises(CommandError) as ctx:
            Single.parse([])
        self.assertIn("usage: single", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
