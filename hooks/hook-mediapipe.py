from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules('mediapipe')
datas = collect_data_files('mediapipe')
