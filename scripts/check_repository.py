"""소스 구문, 로컬 링크, 공개용 노트북의 출력 상태를 검사합니다."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    for directory in ('src', 'tests', 'scripts', 'examples'):
        for path in (ROOT / directory).rglob('*.py'):
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    documents = [ROOT/'README.md', ROOT/'CONTRIBUTING.md', *sorted((ROOT/'docs').rglob('*.md'))]
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'{path.name}: 실행 출력을 지운 뒤 저장해 주세요.')
                compile(''.join(cell['source']), str(path), 'exec')
        documents.append(path)
    for path in documents:
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.ipynb':
            text = '\n'.join(''.join(c['source']) for c in json.loads(text)['cells'] if c['cell_type'] == 'markdown')
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if link.startswith(('http:', 'https:', 'mailto:', '#')):
                continue
            target = (path.parent / link.split('#')[0]).resolve()
            if not target.is_file():
                errors.append(f'{path.relative_to(ROOT)}: 연결 파일이 없습니다: {link}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Python 구문, 문서 링크, 노트북 출력 제거 상태: 통과')


if __name__ == '__main__':
    main()
