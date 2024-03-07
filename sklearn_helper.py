"""
    Módulo auxiliar ao uso da biblioteca scikit learn.
    Possui ferramentas para auxiliar no download e tratamento de dados, formatação de datasets e avaliação de desempenho de modelos.
"""

from collections.abc import Iterable
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.base import clone

class DadosVibracao():
    """
    Classe para armazenar dados de vibração e identificar seus picos.
    Indexação do objeto acessa seus picos e fatias de dados nos arredores destes.

    Parameters
    ----------
    dados : pd.DataFrame
        Dataframe com os dados de vibração.
    coluna_picos : str
        Coluna do dataframe onde os picos serão procurados.
    altura_pico : int | tuple[int], optional
        Altura mínima dos picos. Pode ser fornecida uma lista no formato [altura_mínima, altura_máxima]. Valor padrão 0.
    distancia_entre_picos : int | tuple[int], optional
        Distância mínima em número de leituras entre dois picos. Pode ser fornecida uma lista no formato [distância_mínima, distância_máxima]. Valor padrão 0.

    Exemplos
    ----------
    Os objetos do tipo DadosVibracao (dados_vib) podem ser indexados seguindo o esquema:
    
        dados_vib[pico, comprimento, offset]
        
    Múltiplos picos podem ser acessados com slices:
    
        dados_vib[pico_inicial : pico_final : passo, comprimento, offset]
    
    Para encontrar o número de leituras a serem retornadas:
    
        comprimento - offset = leituras
    
    * dados_vib[1]
    
    Retorna os dados referentes ao segundo pico.
    
    * dados_vib[0:5]
    
    Retorna os dados referentes aos 5 primeiros picos.
    
    * dados_vib[1, 3]
    
    Retorna os dados referentes ao segundo pico e os próximos 2 registros (comprimento = o pico + 2 = 3).
    
    * dados_vib[3, 10, -5]
    
    Retorna o quarto pico, os próximos 9 registros (comprimento = 10) e 5 registros anteriores ao pico (offset = -5).
    
    * dados_vib[0, 5, 2]
    
    Retorna o segundo registro após o pico (offset = 2) e os próximos 3 (comprimento = 5 - 2 de offset = 3 registros).
    
    * dados_vib[:, 5, 2]
    
    Retorna uma lista contendo uma fatia de dados para cada pico. As fatias segem o formato do exemplo anterior.
    """
    
    def __init__(self, dados: pd.DataFrame, coluna_picos: str,
                 altura_pico: int|tuple[int] = 0, distancia_entre_picos: int|tuple[int] = 0) -> None:
        self._coluna_picos = coluna_picos
        self.dados = dados
        
        self.salvar_picos(altura_pico, distancia_entre_picos)
    
    def salvar_picos(self, altura_pico: int|tuple[int], distancia_entre_picos: int|tuple[int],
                     coluna_picos: str = None) -> None:
        """
        Encontra picos nos dados e os salva no atributo (self.picos).
        Pode ser utilizado para atualizar o conjunto de dados presente no atributo (self.picos) ao procurar picos novamente, alterando os parâmetros de busca.

        Parameters
        ----------
        altura_pico : int | tuple[int]
            Altura mínima dos picos. Pode ser fornecida uma lista no formato [altura_mínima, altura_máxima].
        distancia_entre_picos : int | tuple[int]
            Distância mínima em número de leituras entre dois picos. Pode ser fornecida uma lista no formato [distância_mínima, distância_máxima].
        coluna_picos : str, optional
            Coluna do dataframe onde os picos serão procurados.
        """        
        if coluna_picos is not None:
            self._coluna_picos = coluna_picos
        
        self.picos, _ = find_peaks(self.dados[self._coluna_picos], altura_pico, distance=distancia_entre_picos)
    
    def __getitem__(self, arg: int|tuple[int]) -> pd.DataFrame:
        if not isinstance(arg, tuple): # caso em que input é um único argumento
            return self.dados.iloc[self.picos[arg]]
        
        index = self.picos[arg[0]]
        comprimento = arg[1]
        offset = arg[2] if len(arg) == 3 else 0
        
        if isinstance(arg[0], slice):
            output = []
            for i in index:
                output.append(self.dados.iloc[i + offset : i + comprimento])
            return output
        
        return self.dados.iloc[index + offset : index + comprimento]



def baixar_dados(link: str, *, output_dir: str|os.PathLike = 'Dados') -> bool:
    """
    Baixa um arquivo do Google Drive para a pasta (out) se esta estiver vazia.
    
    Parameters
    ----------
    link : str
        Link do Google Drive onde o arquivo se encontra.
    output_dir : str | os.PathLike, optional
        Local onde os downloads seram armazenados. Por padrão, cria uma pasta chamada 'Dados'.

    Returns
    -------
    bool
        Indica se o download ocorreu (True = fez download, False = não fez download)
    """
    
    from gdown import download
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if os.listdir(output_dir) == []:
        raiz = os.getcwd()
        os.chdir(output_dir)
        
        download(link, fuzzy=True)
        
        os.chdir(raiz)
        
        return True

    return False
    


def unzip_arquivos_da_pasta(path: str, *, output_dir: str = None) -> None:
    """
    Descompacta todos os arquivos zip de uma pasta.

    Parameters
    ----------
    path : str
        Caminho da pasta contendo os arquivos.
    output_dir : str, optional
        Caminho da pasta em que os arquivos descompactados serão salvo. Por padrão, salva os arquivos na mesma pasta dos zips.
    """
    
    from zipfile import ZipFile
    
    zips = [zip for zip in os.listdir(path) if zip.endswith('.zip')]
    out_path = path if output_dir is None else output_dir

    for i in zips:
        with ZipFile(path + i) as zip:
            zip.extractall(out_path)



def achatar_dados(df: pd.DataFrame|Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Achata os dados de data frames para uma única linha.
    Caso múltiplos data frames sejam fornecidos, cada um será achatado para uma linha do data frame retornado, na ordem em que foram fornecidos.

    Parameters
    ----------
    df : pd.DataFrame | Iterable[pd.DataFrame]
        Data frame ou conjunto de data frames a serem achatados.
        
    Returns
    -------
    pd.DataFrame
        Data frame com cada linha representando todos os dados de um data frame fornecido.
    """
    
    if isinstance(df, pd.DataFrame):
        nomes_colunas = []
        for i in range(len(df)):
            nomes_colunas += [f'{nome}_{i}' for nome in df.columns]
    
        dados_achatados = np.array(df).flatten()
        out_df = pd.Series(dados_achatados, index=nomes_colunas).to_frame().T
    
        return out_df
    
    dfs = []
    for d in df:
        dfs.append(achatar_dados(d))
    out_df = pd.concat(dfs).reset_index().drop('index', axis= 1)
    
    return out_df



def testar_precisao(modelo, x: pd.DataFrame, y: pd.DataFrame, rodadas: int = 10) -> None:
    """
    Testa a precisão média do modelo em rodadas de treino e teste.
    Os grupos de treino e teste são sorteados novamente no ínicio de cada rodada, dentre os dados x e y fornecidas.

    Parameters
    ----------
    modelo
        Modelo de machine learning a ser testado.
    x : pd.DataFrame
        Amostras de dados para análise.
    y : pd.DataFrame
        Valores alvo referentes a cada amostra de dados.
    rodadas : int, optional
        Número de rodadas treino-teste a serem executadas. Valor padrão 10.
    """
    
    copia_modelo = clone(modelo)
    
    scores = []
    for _ in range(rodadas):
        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
        copia_modelo.fit(x_treino, y_treino)
    
        scores.append(copia_modelo.score(x_teste, y_teste))
    
    print(f"""
    Precisão Média:   {np.mean(scores):.2%}
    Desvio Padrão:    {np.std(scores):.2f}
    Variância:        {np.var(scores):.2e}
    """
    )



def testar_precisao_dados_fixos(modelo, treino:tuple[pd.DataFrame], teste:tuple[pd.DataFrame],
                                rodadas:int= 10) -> None:
    """
    Testa a precisão média do modelo em rodadas de treino e teste.
    O modelo é treinado e testado sobre os dados fornecidos, em todas as rodadas.

    Parameters
    ----------
    modelo
        Modelo de machine learning a ser testado.
    treino : tuple[pd.DataFrame]
        Dados a serem utilizados para treinar o modelo. Deve seguir o formato (x_treino, y_treino).
    teste : tuple[pd.DataFrame]
        Dados a serem utilizados para testar o modelo. Deve seguir o formato (x_teste, y_teste).
    rodadas : int, optional
        Número de rodadas treino-teste a serem executadas. Valor padrão 10.
    """    
    
    copia_modelo = clone(modelo)
    
    scores = []
    for _ in range(rodadas):
        copia_modelo.fit(treino[0], treino[1])
    
        scores.append(copia_modelo.score(teste[0], teste[1]))
    
    print(f"""  
    Precisão Média:   {np.mean(scores):.2%}
    Desvio Padrão:    {np.std(scores):.2f}
    Variância:        {np.var(scores):.2e}
    """
    )