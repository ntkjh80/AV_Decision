"""기본 CPU 실습을 실제 커널로 실행합니다. 원본 노트북은 수정하지 않습니다."""
from pathlib import Path
import os
import tempfile


def main():
    import nbformat
    from nbclient import NotebookClient
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks/S1_Semantic_Video_Verification.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix='decision-kernel-') as tmp:
        # 커널 통신 파일과 IPython 설정은 실습 임시 폴더에 둡니다.
        previous = {key: os.environ.get(key) for key in ('IPYTHONDIR', 'JUPYTER_RUNTIME_DIR')}
        os.environ['IPYTHONDIR'] = str(Path(tmp) / 'ipython')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(Path(tmp) / 'runtime')
        try:
            client = NotebookClient(notebook, timeout=120, kernel_name='python3',
                                    resources={'metadata': {'path': str(root)}})
            client.execute()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    print(f'CPU 노트북 코드 셀 {count}개 실행 완료. 원본 출력은 비운 상태로 유지했습니다.')


if __name__ == '__main__':
    main()
