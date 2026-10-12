"""Lazy command dispatcher; training starts its supervisor before ML imports."""
import importlib
import sys

COMMANDS = {
    'snapshot': 'indic_reasoning.checkpoints',
    'train': 'indic_reasoning.training.rl',
    'sft': 'indic_reasoning.training.sft',
    'evaluate': 'indic_reasoning.evaluation.runner',
    'prepare-test': 'indic_reasoning.evaluation.holdout',
    'translate': 'indic_reasoning.translation.translate_tapo',
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m indic_reasoning {' + ','.join(COMMANDS) + '} [options]')
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f'Unknown command: {command}')
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    return importlib.import_module(COMMANDS[command]).main()

if __name__ == '__main__':
    raise SystemExit(main())
