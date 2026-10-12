#!/usr/bin/env python3
"""Keep the gold answer visible before long text in TRL completion tables."""
import difflib
import importlib.util
from pathlib import Path


def patch_source(source):
    if 'TAPO_VISIBLE_GOLD_ANSWER' in source:
        return source
    anchor = '            df_base = pd.DataFrame(table)'
    replacement = '''            # TAPO_VISIBLE_GOLD_ANSWER: reference and score precede wide text.
            if "correct_answer" in table:
                leading = [key for key in ("step", "correct_answer", "correctness_reward") if key in table]
                table = {key: table[key] for key in leading + [key for key in table if key not in leading]}

''' + anchor
    if source.count(anchor) != 1:
        raise RuntimeError('Unexpected TRL completion table source; refusing to patch.')
    updated = source.replace(anchor, replacement, 1)
    compile(updated, 'grpo_trainer.py', 'exec')
    return updated


def main():
    path = Path(importlib.util.find_spec('trl').origin).parent / 'trainer' / 'grpo_trainer.py'
    old = path.read_text()
    new = patch_source(old)
    if new == old:
        print('Visible gold-answer columns already installed.')
        return
    backup = path.with_suffix('.py.before-tapo-columns-patch')
    if not backup.exists():
        backup.write_text(old)
    path.write_text(new)
    Path(__file__).with_name('trl-1.9.2-completion-columns.patch').write_text(''.join(
        difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                             fromfile='a/trainer/grpo_trainer.py', tofile='b/trainer/grpo_trainer.py')))
    print('Installed visible gold-answer column ordering:', path)


if __name__ == '__main__':
    main()
