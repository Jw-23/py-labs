"""cmdkit 演示:一个"py-labs 实验工具箱"命令行。

整个文件里没有一行 argparse 代码 —— 参数来自方法签名 + 装饰器。

    .venv/bin/python demo_cmdkit.py --help
    .venv/bin/python demo_cmdkit.py resist --help
    .venv/bin/python demo_cmdkit.py cutline 3 --threshold 0.4 -v
    .venv/bin/python demo_cmdkit.py resist trapezoid -n 2 --seed 7
    .venv/bin/python demo_cmdkit.py cutline            # 缺参数 -> 错误 + 帮助
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from cmdkit import command, flag, param, subcommand


class Shape(Enum):
    """Enum 标注会自动变成 choices。"""

    LINE = "line"
    TRAPEZOID = "trapezoid"
    GOURAUD = "gouraud"


@command(
    name="lab",
    description="py-labs 实验工具箱 —— 演示 @command / @subcommand",
    version="0.1.0",
    params=[param(False, "-v", "--verbose", action="store_true", help="打印更多信息")],
    epilog="例子:\n"
    "  lab cutline 3 --threshold 0.4 -v\n"
    "  lab resist trapezoid -n 2 --seed 7\n"
    "  lab resist --help",
)
class Lab:
    """py-labs 实验工具箱。

    类 docstring 会自动成为 ``--help`` 顶部的描述文字。
    全局选项 ``-v/--verbose`` 在子命令前后都写得,*子命令自己不用声明它*。
    """

    @subcommand(help="生成合成切割线(位置参数 + 类型标注)", aliases="cut c")
    def cutline(
        self,
        count: int,  # 位置参数:没有默认值 -> 必填
        out: Path = Path("data/cutline_case"),  # 有默认值 -> 可选
        *,
        threshold: float = param(0.5, "-t", "--threshold", help="判据阈值"),
        seed: Optional[int] = param(None, help="随机种子,不给就随机"),
        sign: Literal["positive", "negative"] = "positive",
    ):
        """按给定次数生成合成切割线样例。

        这是 ``--help`` 里的详细描述(取整个 docstring)。
        """
        print(f"[cutline] count={count} out={out} threshold={threshold} seed={seed} sign={sign!r}")
        print(f"[cutline] verbose={self.verbose}  (全局选项,写在 self 上)")

    @subcommand(help="生成 resist 图像(Enum / 列表 / 元组)", name="resist")
    def resist_image(
        self,
        shape: Shape,  # Enum -> choices=line/trapezoid/gouraud
        *,
        counts: list[int] = param([1], "-n", "--count", help="可以有多个值"),
        size: tuple[int, int] = param((512, 512), "--size", help="宽 高"),
        seed: Optional[int] = param(None, help="随机种子"),
        dry_run: bool = flag("-d", "--dry-run", help="只打印不写盘"),
    ):
        """生成 resist 图像。列表参数可以重复给值,元组参数固定吃两个。"""
        print(f"[resist] shape={shape.value} counts={counts} size={size} seed={seed} dry_run={dry_run}")
        print(f"[resist] verbose={self.verbose}")

    @subcommand(
        help="用装饰器声明必填 / 默认值 / 描述(签名里什么都不写)",
        aliases=["bench", "b"],
        params={
            "rounds": {"required": True, "help": "重复轮数(必填,因为签名里没有默认值)"},
            "suite": {"default": "full", "help": "跑哪套基准", "choices": ["quick", "full"], "flag": "-s --suite"},
            "quiet": {"default": False, "help": "安静模式", "flag": "-q --quiet"},
            "tag": {"default": "baseline", "help": "结果标签", "flag": "-T --tag"},
        },
    )
    def benchmark(self, rounds, suite, quiet, tag):
        """这个方法故意把元数据全放在装饰器里。

        对应地,``params=`` 里给出 ``required`` / ``default`` / ``help``。
        """
        print(f"[bench] rounds={rounds} suite={suite!r} quiet={quiet} tag={tag!r}")

    @subcommand(help="演示 subcommand 的别名与 params 覆盖", aliases="ls")
    def list_cases(self, root: Path = Path("data"), pattern: str = param("*", "-p", "--pattern")):
        """列出 data 目录下的样例。"""
        base = Path(__file__).resolve().parent
        prefix = Path(root)
        found = sorted(p.name for p in (base / prefix).glob(pattern)) if (base / prefix).is_dir() else []
        print(f"[list] root={prefix} pattern={pattern!r} -> {found}")


if __name__ == "__main__":
    raise SystemExit(Lab.main())
