# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_data_files

hiddenimports = ['torch', 'chess', 'pygame', 'numpy']
datas = collect_data_files('chess_core') + [('move_vocab.pkl', '.'), ('v7p3r_chess_ai_model.pth', '.'), ('images/*', 'images/')]

a = Analysis(
    ['chess_game.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports
)