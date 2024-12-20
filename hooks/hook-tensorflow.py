from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules('tensorflow')
datas = collect_data_files('tensorflow')

def pre_safe_import_module(api):
    # Previne o erro de registro duplo de gradientes
    api.add_runtime_package('tensorflow.python')
