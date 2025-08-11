from fpdf import FPDF
import os
import streamlit as st
import pandas as pd
import tempfile
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import numpy as np


# ==============================
# CLASSE PDF BÁSICA
# ==============================
# ==============================
# CLASSE PDF BÁSICA CORRIGIDA
# ==============================
from fpdf import FPDF
import os

class PDF(FPDF):
    def __init__(self, capa_fundo=None, fundo_paginas=None):
        super().__init__('L', 'mm', 'A4')
        self.capa_fundo = capa_fundo
        self.fundo_paginas = fundo_paginas
        self.organizacao = ''
        self.periodo = ''
        self.tipo_equipamento = ''

    def set_dados_capa(self, organizacao, periodo, tipo_equipamento):
        self.organizacao = organizacao
        self.periodo = periodo
        self.tipo_equipamento = tipo_equipamento

    def header(self):
        if self.page_no() == 1:
            if self.capa_fundo and os.path.exists(self.capa_fundo):
                self.image(self.capa_fundo, x=0, y=0, w=self.w, h=self.h)

            # Organização
            self.set_font('Arial', '', 11)
            self.set_text_color(0, 0, 0)
            self.set_x(10)
            self.cell(0, 10, f"Organização: {self.organizacao}", ln=True, align='L')

            # Período
            self.set_x(10)
            self.cell(0, 10, f"Período: {self.periodo}", ln=True, align='L')

        else:
            if self.fundo_paginas and os.path.exists(self.fundo_paginas):
                self.image(self.fundo_paginas, x=0, y=0, w=self.w, h=self.h)

class PDFConsolidado(PDF):
    def add_capa(self, total_orgs, total_tratores, total_pulvs, total_colhs, data_inicio, data_fim, tatica=None):
        self.add_page()
        if self.capa_fundo and os.path.exists(self.capa_fundo):
            self.image(self.capa_fundo, x=0, y=0, w=self.w, h=self.h)

        self.set_font("Arial", "B", 20)
        self.set_text_color(0, 70, 0)
        ##self.cell(0, 20, "Relatório Consolidado de Máquinas", ln=True, align="C")

        self.ln(100)  # espaço vertical maior para baixar as informações

        self.set_font("Arial", "", 14)
        self.set_text_color(0, 0, 0)

        x_inicio = 15  # distância da margem esquerda

        self.set_x(x_inicio)
        self.cell(0, 10, f"Período: {data_inicio} até {data_fim}", ln=True, align="L")

        self.set_x(x_inicio)
        self.cell(0, 10, f"Total de Organizações: {total_orgs}", ln=True, align="L")

        self.set_x(x_inicio)
        self.cell(0, 10, f"Total de Tratores: {total_tratores}", ln=True, align="L")

        self.set_x(x_inicio)
        self.cell(0, 10, f"Total de Pulverizadores: {total_pulvs}", ln=True, align="L")

        self.set_x(x_inicio)
        self.cell(0, 10, f"Total de Colheitadeiras: {total_colhs}", ln=True, align="L")

        if tatica and tatica != "Tática não definida":
            self.set_x(x_inicio)
            self.cell(0, 10, f"Tática: {tatica}", ln=True, align="L")


    def add_tabela(self, caminho_imagem):
        self.add_page()
        if self.fundo_paginas and os.path.exists(self.fundo_paginas):
            self.image(self.fundo_paginas, x=0, y=0, w=self.w, h=self.h)
        self.image(caminho_imagem, x=10, y=20, w=self.w - 20)

    def add_graficos_com_titulo(self, graficos_dict):
        self.add_page()
        if self.fundo_paginas and os.path.exists(self.fundo_paginas):
            self.image(self.fundo_paginas, x=0, y=0, w=self.w, h=self.h)

        self.set_font("Arial", "B", 12)
        self.set_text_color(0, 0, 0)

        # Layout 2x2 (mesmo usando só 3 posições)
        largura = (self.w / 2) - 20  # margem lateral
        altura = (self.h / 2) - 25   # margem vertical
        positions = [
            (10, 30),               # Subiu 5 unidades na coordenada Y
            (self.w / 2 + 5, 30),
            (self.w / 4 + 5, self.h / 2 + 15),  # Também desceu 5 unidades
        ]
        
        tipos = ["Trator", "Pulverizador", "Colheitadeira"]

        for i, tipo in enumerate(tipos):
            caminho_grafico = graficos_dict.get(tipo)
            if caminho_grafico and os.path.exists(caminho_grafico):
                x, y = positions[i]
                self.set_xy(x, y - 10)
                self.cell(largura, 10, tipo, align="C")
                self.image(caminho_grafico, x=x, y=y, w=largura, h=altura)

# =====================
# FUNÇÕES AUXILIARES
# =====================
def limpar_texto(texto):
    return texto.replace("\u2122", "(TM)")

def quebrar_linha(texto, largura=100):
    texto = limpar_texto(texto)
    return "\n".join(textwrap.wrap(texto, width=largura))

def salvar_grafico(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def gerar_pdf_completo(caminho_tabela, graficos_paths, capa_fundo, fundo_paginas, organizacao, periodo, tipo_equipamento, grafico_extra_path=None):
    pdf = PDF(capa_fundo=capa_fundo, fundo_paginas=fundo_paginas)
    pdf.set_dados_capa(organizacao=organizacao, periodo=periodo, tipo_equipamento=tipo_equipamento)

    # Página 1: Capa
    pdf.add_page()

    # Página 2: Tabela centralizada e sem título
    pdf.add_page()
    pdf.image(caminho_tabela, x=10, y=12, w=pdf.w - 20)

    # Página 3: Gráficos até 4 imagens em layout 2x2
    if graficos_paths:
        pdf.add_page()
        positions = [(10, 20), (pdf.w / 2 + 5, 20), (10, pdf.h / 2 + 5), (pdf.w / 2 + 5, pdf.h / 2 + 5)]
        size = (pdf.w / 2 - 15, (pdf.h / 2) - 25)
        for i, path in enumerate(graficos_paths[:4]):
            if os.path.exists(path):
                x, y = positions[i]
                pdf.image(path, x=x, y=y, w=size[0], h=size[1])

    return pdf

def salvar_tabela_com_matplotlib(df, caminho_imagem, organizacao, data_inicio, data_final):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    num_linhas = len(df)
    altura_por_linha = 0.16
    altura_total = min(7.2, num_linhas * altura_por_linha)

    fig, ax = plt.subplots(figsize=(12.5, altura_total))  # mais espaço horizontal
    ax.axis('tight')
    ax.axis('off')

    def get_color(val):
        try:
            if isinstance(val, str) and val.endswith('%'):
                val = float(val.replace('%', '').strip())
            cmap = plt.get_cmap('RdYlGn')
            norm = mcolors.Normalize(vmin=0, vmax=100)
            rgba = cmap(norm(val))
            return mcolors.to_hex(rgba)
        except:
            return 'white'

    table_data = df.values.tolist()
    col_labels = df.columns.tolist()

    # colWidths com ajuste leve com base no nome
    colWidths = [max(0.05, min(0.25, len(str(c)) / 100)) for c in col_labels]

    tabela = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        colWidths=colWidths,
        bbox=[0, -0.18, 1, 1]  # posição vertical ideal
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(7.0)
    tabela.scale(1.0, 0.9)  # mais seguro para evitar espremimento horizontal

    media_col_index = df.columns.get_loc("Média (%)")

    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = tabela[(i + 1, j)]
            if df.iloc[i]["Máquina"] == "Média Geral":
                cell.set_facecolor("#d3d3d3")
            elif j == media_col_index:
                cell.set_facecolor(get_color(df.iloc[i, media_col_index]))

    plt.savefig(
        caminho_imagem,
        bbox_inches='tight',
        dpi=300,
        pad_inches=0.0,
        transparent=True
    )
    plt.close()


def quebrar_texto_em_linhas(texto, max_len=12, max_linhas=5):
    palavras = texto.split()
    linhas = []
    linha_atual = ''

    for palavra in palavras:
        if len(linha_atual) + len(palavra) + (1 if linha_atual else 0) <= max_len:
            if linha_atual:
                linha_atual += ' ' + palavra
            else:
                linha_atual = palavra
        else:
            linhas.append(linha_atual)
            linha_atual = palavra
            if len(linhas) == max_linhas - 1:
                break

    if linha_atual:
        linhas.append(linha_atual)

    # Se sobrar texto, junta tudo restante na última linha (limita o total a max_linhas)
    resto = palavras[len(' '.join(linhas).split()):]
    if resto:
        linhas[-1] += ' ' + ' '.join(resto)

    return '\n'.join(linhas)



def salvar_tabela_com_matplotlib_cons(df, caminho_imagem):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Corrigir nomes longos das organizações (antes do parêntese)
    df.index = df.index.to_series().str.extract(r'^(.+?)\s*\(')[0].fillna(df.index.to_series())
    df = df.reset_index()

    num_linhas = len(df)
    altura_por_linha = 0.16
    altura_total = max(7.5, num_linhas * altura_por_linha * 0.2)  # mais altura

    fig, ax = plt.subplots(figsize=(14.5, altura_total))
    ax.axis('tight')
    ax.axis('off')

    def get_color(val):
        try:
            if isinstance(val, str) and val.endswith('%'):
                val = float(val.replace('%', '').strip())
            cmap = plt.get_cmap('RdYlGn')  # verde = bom, vermelho = ruim
            norm = mcolors.Normalize(vmin=0, vmax=100)
            rgba = cmap(norm(val))
            return mcolors.to_hex(rgba)
        except:
            return 'white'

    table_data = df.values.tolist()
    col_labels = df.columns.tolist()
    n_cols = len(col_labels)

    # Largura personalizada: 1ª coluna maior
    colWidths = [0.25] + [(0.75 / (n_cols - 1))] * (n_cols - 1)

    tabela = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        colWidths=colWidths,
        bbox=[0, -0.08, 1, 1.3]  # mais espaço superior para os títulos
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(6)
    tabela.scale(1.0, 1.4)  # altura das linhas

    # Localizar coluna da média
    media_col_index = df.columns.get_loc("MÉDIA POR ORGANIZAÇÃO (%)")

    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = tabela[(i + 1, j)]
            if df.iloc[i, 0] == "TOTAL":
                cell.set_facecolor("#d3d3d3")  # linha final em cinza
            elif j == media_col_index:
                val_str = df.iloc[i, media_col_index]
                try:
                    val = float(val_str.replace('%', '')) if isinstance(val_str, str) else float(val_str)
                    cell.set_facecolor(get_color(val))
                except:
                    pass

    for j, label in enumerate(col_labels):
        cell = tabela[(0, j)]
        cell.get_text().set_ha('center')
        cell.get_text().set_va('center')
        cell.get_text().set_fontsize(4)
        cell.get_text().set_weight('bold')
        cell.set_facecolor('#e6e6e6')
        cell.set_height(0.15)

        texto_quebrado = quebrar_texto_em_linhas(str(label), max_len=12, max_linhas=5)
        cell.get_text().set_text(texto_quebrado)
        cell.get_text().set_rotation(0)


    plt.savefig(
        caminho_imagem,
        bbox_inches='tight',
        dpi=300,
        pad_inches=0.0,
        transparent=True
    )
    plt.close()

def add_graficos_com_titulo(self, graficos_dict):
    self.add_page()
    self.image(self.fundo_paginas, x=0, y=0, w=297, h=210)  # A4 paisagem

    titulos = {
        "Trator": "Médias das Tecnologias - Trator",
        "Pulverizador": "Médias das Tecnologias - Pulverizador",
        "Colheitadeira": "Médias das Tecnologias - Colheitadeira"
    }

    posicoes = {
        "Trator": (15, 40),
        "Pulverizador": (150, 40),
        "Colheitadeira": (15, 120)
    }

    tamanho_w, tamanho_h = 120, 70  # Tamanho das imagens no PDF

    for tipo, caminho_imagem in graficos_dict.items():
        if os.path.exists(caminho_imagem):
            x, y = posicoes[tipo]
            self.set_xy(x, y - 8)
            self.set_font("Arial", "B", 11)
            self.cell(w=tamanho_w, h=6, txt=titulos[tipo], ln=1, align="C")
            self.image(caminho_imagem, x=x, y=y, w=tamanho_w, h=tamanho_h)
        else:
            print(f"[AVISO] Imagem não encontrada: {caminho_imagem}")


def gerar_pdf_completo_cons(
    caminho_tabela,
    graficos_dict,
    capa_fundo,
    fundo_paginas,
    total_orgs,
    total_tratores,
    total_pulvs,
    total_colhs,
    data_inicio,
    data_fim,
    tatica=None,  # adicione este parâmetro
    caminho_saida="relatorio_consolidado.pdf"
):
    # seu código aqui

    pdf = PDF(capa_fundo=capa_fundo, fundo_paginas=fundo_paginas)
    
    # Página 1 – Capa
    pdf.add_capa(
        total_orgs=total_orgs,
        total_tratores=total_tratores,
        total_pulvs=total_pulvs,
        total_colhs=total_colhs,
        data_inicio=data_inicio,
        data_fim=data_fim,
        tatica=tatica
    )

    # Página 2 – Tabela
    pdf.add_tabela(caminho_tabela)

    # Página 3 – Gráficos
    pdf.add_graficos_com_titulo(graficos_dict)

    pdf.output(caminho_saida)


# =========================
# INTERFACE STREAMLIT
# =========================
# CONFIG INICIAL
st.set_page_config(layout="wide")
st.title("📊 Geração de Relatório PDF - Máquinas")

# 1. Carregar a planilha de organizações com tática (cache para otimizar)
@st.cache_data
def carregar_organizacoes():
    df_org = pd.read_excel("tabelas/organizacoes_referentes.xlsx")
    # Explode a coluna Tática para ter uma linha por tática
    df_org["Tática"] = df_org["Tática"].str.split(";")
    df_org = df_org.explode("Tática")
    df_org["Tática"] = df_org["Tática"].str.strip()
    return df_org

organizacoes_df = carregar_organizacoes()

# Lista de táticas disponíveis para filtro no sidebar, incluindo opção padrão
taticas_disponiveis = ["Tática não definida"] + sorted(organizacoes_df["Tática"].dropna().unique().tolist())
tatica_selecionada = st.sidebar.selectbox("Filtrar por Tática", taticas_disponiveis)

# 3. Carregar o Excel principal enviado pelo usuário
df = None
uploaded_file = st.file_uploader("📥 Envie o arquivo Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Exportar")
    df["Tipo Normalizado"] = df["Tipo"].str.lower()

    if tatica_selecionada != "Tática não definida":
        # Organizações que fazem parte da tática selecionada
        orgs_tatica = organizacoes_df.loc[organizacoes_df["Tática"] == tatica_selecionada, "Organização"].unique()

        # Filtra o DataFrame principal para manter só as organizações da tática
        df_filtrado = df[df["Organização"].isin(orgs_tatica)].copy()

        # Identifica organizações da tática sem dados no arquivo enviado
        orgs_com_dados = df_filtrado["Organização"].unique()
        orgs_sem_dados = set(orgs_tatica) - set(orgs_com_dados)

        if orgs_sem_dados:
            # Cria linhas vazias para as organizações sem dados
            df_vazios = pd.DataFrame({"Organização": list(orgs_sem_dados)})
            for col in df.columns:
                if col != "Organização":
                    df_vazios[col] = np.nan

            # Concatena os dados filtrados com os vazios para garantir todas as organizações da tática
            df_filtrado = pd.concat([df_filtrado, df_vazios], ignore_index=True)

        df = df_filtrado

    # Caso não haja dados após o filtro
    if df.empty:
        st.warning("Nenhuma organização com dados encontrada para essa tática.")
        st.stop()

    # 5. Determinar datas e organização (usado no PDF)
    data_inicio = pd.to_datetime(df["Data de Início"].iloc[0]).strftime("%d/%m/%Y")
    data_final = pd.to_datetime(df["Data Final"].iloc[0]).strftime("%d/%m/%Y")
    organizacao = str(df["Organização"].iloc[0])

    # Importar colunas_por_tipo
    colunas_por_tipo = {
        "Trator": [
            "AutoTrac™ Ativo (%)",
            "FieldCruise™ Ligado (%)",
            "Tempo Ligado do Efficiency Manager™ Automático (%)"
        ],
        "Pulverizador": [
            "Pulsante Ativo (%)",
            "Tempo de Controle de Seção Ativo (%)",
            "AutoTrac™ Ativo (%)"
        ],
        "Colheitadeira": [
            "AutoTrac™ Ativo (%)",
            "Active Terrain Adjustment™ Ligado (%)",
            "Harvest Smart Ligado (%)"
        ]
    }

    # 6. Escolher aba
    aba = st.sidebar.radio("Selecione a aba", options=["Consolidado", "Tipo Específico"])

    if aba == "Tipo Específico":
        tipo_selecionado = st.sidebar.selectbox("Selecione o Tipo de Equipamento", list(colunas_por_tipo.keys()))
        tipo_selecionado_normalizado = tipo_selecionado.lower()
        df_filtrado = df[df["Tipo Normalizado"].str.contains(tipo_selecionado_normalizado, na=False)]

        if df_filtrado.empty:
            st.warning("Nenhum dado encontrado para esse tipo.")
            st.stop()

        colunas = colunas_por_tipo[tipo_selecionado]

        for col in colunas:
            df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors="coerce")


        df_resultado = df_filtrado.groupby("Máquina")[colunas].mean().reset_index()
        df_resultado["Média (%)"] = df_resultado[colunas].mean(axis=1) * 100 # média correta, ignora NaN
        df_resultado["Média (%)"] = df_resultado["Média (%)"].apply(lambda x: round(x, 2))
        df_resultado = df_resultado.sort_values(by="Média (%)", ascending=False)

        media_geral_dict = {
            "Máquina": "Média Geral",
            **{col: f"{round(df_resultado[col].mean() *100, 2)}%" for col in colunas},
            "Média (%)": ""
        }

        media_geral = pd.DataFrame([media_geral_dict])
        df_numerico = df_resultado.copy()
        df_numerico[colunas] = df_numerico[colunas].fillna(0)  # apenas para exibir

        styled_df = pd.concat([df_numerico, media_geral], ignore_index=True)


        for col in colunas:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

        styled_df["Média (%)"] = styled_df["Média (%)"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (float, int)) else x)


        def color_scale(val):
            try:
                if val == "":
                    return "background-color: white"
                val = float(val.strip('%'))
                hue = (val / 100) * 120
                return f"background-color: hsl({hue}, 75%, 60%)"
            except:
                return "background-color: white"

        def highlight_media_geral(row):
            return ['background-color: #d3d3d3'] * len(row) if row["Máquina"] == "Média Geral" else [''] * len(row)

        styled = (
            styled_df.style
            .apply(highlight_media_geral, axis=1)
            .applymap(color_scale, subset=["Média (%)"])
            .set_properties(**{"text-align": "center"})
        )

        st.subheader(f"Média por Equipamento – {tipo_selecionado}")
        st.dataframe(styled, use_container_width=True)

        if not df_filtrado.empty:

            colunas_grafico = colunas  # tecnologias para o tipo selecionado

            # Média geral (em %) para cada tecnologia
            media_geral = df_resultado[colunas_grafico].mean() * 100

            # --- Removido o gráfico verde horizontal ---
            
            # 2) Gráficos separados por tecnologia para médias por modelo
            df_filtrado[colunas_grafico] = df_filtrado[colunas_grafico].apply(pd.to_numeric, errors='coerce')
            df_modelo = df_filtrado.groupby("Modelo")[colunas_grafico].mean() * 100

            for col in colunas_grafico:
                fig, ax = plt.subplots(figsize=(8, 5))
                df_modelo[col].plot(kind='bar', ax=ax, color='yellow', width=0.7)
                ax.set_ylim(0, 100)
                ax.set_ylabel("Percentual (%)")
                ax.set_title(f"Média por Modelo - {col}")
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Texto da média geral na parte superior do gráfico
                media = media_geral[col]
                ax.text(0.95, 0.95, f"Média Geral: {media:.1f}%", transform=ax.transAxes,
                        fontsize=12, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

                st.pyplot(fig)



        if st.button("📄 Gerar Relatório em PDF"):
           with tempfile.TemporaryDirectory() as tmpdir:
                imagem_path = os.path.join(tmpdir, "tabela.png")
                salvar_tabela_com_matplotlib(styled_df, imagem_path, organizacao, data_inicio, data_final)

                graficos_paths = []
                
                # Gráficos verticais de média por modelo (máximo 3)
                for i, col in enumerate(colunas[:3]):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df_modelo[col].plot(kind='bar', ax=ax, color='yellow', width=0.7)
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("Percentual (%)")
                    ax.set_title(f"Média por Modelo - {col}")
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    media = media_geral[col]
                    ax.text(0.95, 0.95, f"Média Geral: {media:.1f}%", transform=ax.transAxes,
                            fontsize=12, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
                    path = os.path.join(tmpdir, f"grafico_{i}.png")
                    salvar_grafico(fig, path)
                    graficos_paths.append(path)

                # Gráfico horizontal verde: Média geral por máquina
                df_media_maquina = df_resultado[["Máquina", "Média (%)"]].copy()
                df_media_maquina["Média (%)"] = df_media_maquina["Média (%)"].astype(float)
                df_media_maquina = df_media_maquina.sort_values(by="Média (%)", ascending=True).set_index("Máquina")

                fig, ax = plt.subplots(figsize=(10, 6))
                df_media_maquina["Média (%)"].plot(kind='barh', color='green', ax=ax)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Média (%)")
                ax.set_title(f"Média Geral das Tecnologias por Máquina - {tipo_selecionado}")
                ax.grid(axis='x', linestyle='--', alpha=0.7)

                for i, value in enumerate(df_media_maquina["Média (%)"]):
                    ax.text(value + 1, i, f'{value:.2f}%', va='center', fontsize=9)

                path_horizontal = os.path.join(tmpdir, "grafico_horizontal.png")
                salvar_grafico(fig, path_horizontal)
                graficos_paths.append(path_horizontal)

                # PDF
                fundo_capa = "capa_fundo.png"
                fundo_paginas = "fundo_conteudo.png"

                pdf = gerar_pdf_completo(
                    caminho_tabela=imagem_path,
                    graficos_paths=graficos_paths,
                    capa_fundo=fundo_capa,
                    fundo_paginas=fundo_paginas,
                    organizacao=organizacao,
                    periodo=f"{data_inicio} a {data_final}",
                    tipo_equipamento=tipo_selecionado
                )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    pdf.output(tmp_pdf.name)
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button(
                            label="⬇️ Baixar PDF Final",
                            data=f,
                            file_name=f"Relatorio_{tipo_selecionado}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
 

    if aba == "Consolidado":
        st.subheader("Relatório Consolidado")

        # Função robusta para mapear tipos, garantindo pegar todos os tratores
        def mapear_tipo(tipo_original):
            if pd.isna(tipo_original):
                return None
            if "Tratores Com Tração Em Duas Rodas" in tipo_original:
                return "Trator"
            elif "Colheitadeira" in tipo_original:
                return "Colheitadeira"
            elif "Pulverizador" in tipo_original:
                return "Pulverizador"
            else:
                return None

        colunas_por_tipo = {
            "Trator": [
                "AutoTrac™ Ativo (%)",
                "FieldCruise™ Ligado (%)",
                "Tempo Ligado do Efficiency Manager™ Automático (%)"
            ],
            "Pulverizador": [
                "Pulsante Ativo (%)",
                "Tempo de Controle de Seção Ativo (%)",
                "AutoTrac™ Ativo (%)"
            ],
            "Colheitadeira": [
                "AutoTrac™ Ativo (%)",
                "Active Terrain Adjustment™ Ligado (%)",
                "Harvest Smart Ligado (%)"
            ]
        }

        # Preparar dados
        df = df.copy()
        df["Tipo_Cat"] = df["Tipo"].apply(mapear_tipo)  # ajuste principal aqui

        todas_tecnologias = list(set(sum(colunas_por_tipo.values(), [])))
        for col in todas_tecnologias:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Quantidade de máquinas por Organização e Tipo_Cat
        df_qtd = (
            df.groupby(["Organização", "Tipo_Cat"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["Trator", "Pulverizador", "Colheitadeira"], fill_value=0)
        )

        # Calcular médias por organização para cada tecnologia, via média das médias por máquina
        medias_por_tipo = {}
        colunas_renomeadas = []

        for tipo in ["Trator", "Pulverizador", "Colheitadeira"]:
            df_tipo = df[df["Tipo_Cat"] == tipo].copy()
            df_tipo["Máquina"] = df_tipo["Máquina"].astype(str)

            # Filtra colunas que existem
            colunas_existentes = [col for col in colunas_por_tipo[tipo] if col in df_tipo.columns]
            df_tipo = df_tipo.dropna(how='all', subset=colunas_existentes)

            medias_por_maquina = df_tipo.groupby(["Organização", "Máquina"])[colunas_existentes].mean()
            medias_por_org = medias_por_maquina.groupby("Organização").mean()

            medias_por_tipo[tipo] = medias_por_org

            # Monta a lista de nomes renomeados para essas colunas existentes
            colunas_renomeadas += [f"{col}_{tipo}" for col in colunas_existentes]

        # Concatena as médias
        df_medias = pd.concat(medias_por_tipo.values(), axis=1)

        # Renomeia as colunas do dataframe concatenado
        df_medias.columns = colunas_renomeadas

        # Reconstrói lista colunas_renomeadas apenas com as existentes (precaução)
        colunas_renomeadas = [col for col in colunas_renomeadas if col in df_medias.columns]
        df_medias = df_medias[colunas_renomeadas]
        # Organizações da tática para garantir presença no df_fina

        # Combina quantidade de máquinas e médias
        df_final = df_qtd.join(df_medias, how="outer").fillna(0)

        if tatica_selecionada != "Tática não definida":
            orgs_tatica = organizacoes_df.loc[organizacoes_df["Tática"] == tatica_selecionada, "Organização"].unique()
            df_final = df_final.reindex(orgs_tatica).fillna(0)
        else:
            # Para "Tática não definida", manter todas as organizações do arquivo enviado pelo usuário
            orgs_upload = df["Organização"].unique()
            df_final = df_final.reindex(orgs_upload).fillna(0)


        # Agora calcula a média geral e ordena normalmente
        df_final["MÉDIA POR ORGANIZAÇÃO (%)"] = df_final[colunas_renomeadas].mean(axis=1)

        # Ordena top 70
        df_final = df_final.sort_values("MÉDIA POR ORGANIZAÇÃO (%)", ascending=False).head(70)

        # Linha TOTAL calculada igual ao específico
        total_maquinas = df.groupby("Tipo_Cat").size().reindex(["Trator", "Pulverizador", "Colheitadeira"]).fillna(0)

        medias_totais = {}
        for tipo, cols in colunas_por_tipo.items():
            df_tipo = df[df["Tipo_Cat"] == tipo].copy()
            df_tipo["Máquina"] = df_tipo["Máquina"].astype(str)
            colunas_existentes = [col for col in cols if col in df_tipo.columns]
            medias_maquina = df_tipo.groupby(["Organização", "Máquina"])[colunas_existentes].mean()
            for col in colunas_existentes:
                medias_totais[f"{col}_{tipo}"] = medias_maquina[col].mean()

        linha_total = pd.DataFrame([{
            "Trator": int(total_maquinas.get("Trator", 0)),
            "Pulverizador": int(total_maquinas.get("Pulverizador", 0)),
            "Colheitadeira": int(total_maquinas.get("Colheitadeira", 0)),
            **medias_totais,
            "MÉDIA POR ORGANIZAÇÃO (%)": np.nan
        }], index=["TOTAL"])

        df_final = pd.concat([df_final, linha_total])

        # Ajusta exibição: na linha TOTAL, substitui NaN por 0 só para exibir
        if "TOTAL" in df_final.index:
            df_final.loc["TOTAL", colunas_renomeadas] = df_final.loc["TOTAL", colunas_renomeadas].fillna(0)

        # Arredonda para evitar notação científica, ignorando NaNs
        for col in colunas_renomeadas:
            if col in df_final.columns:
                df_final[col] = df_final[col].apply(lambda x: round(x, 4) if pd.notna(x) else x)

        df_formatado = df_final.copy()

        def formatar_valor(x, coluna=None):
            if pd.isna(x):
                return ""
            try:
                # Se for coluna de máquinas, mostra como inteiro
                if coluna in ["Trator", "Pulverizador", "Colheitadeira"]:
                    return str(int(x))
                # Caso contrário, float com 4 casas decimais
                return f"{x:.4f}"
            except:
                return x

        for col in df_formatado.columns:
            df_formatado[col] = df_formatado[col].apply(lambda x: formatar_valor(x, coluna=col))

        def formatar_percentual(x):
            # Tenta converter para float se for string
            if isinstance(x, str):
                x = x.replace("%", "").strip()
                if x == "":
                    return ""
                try:
                    x = float(x)
                except:
                    return ""
            # Agora x é número ou NaN
            if pd.notna(x):
                return f"{x * 100:.2f}%"  # Multiplica por 100 antes de formatar
            return ""

        df_formatado["MÉDIA POR ORGANIZAÇÃO (%)"] = df_formatado["MÉDIA POR ORGANIZAÇÃO (%)"].apply(formatar_percentual)

        def cor_mapa(val):
            if isinstance(val, str) and val.endswith("%"):
                try:
                    val_num = float(val.replace("%", ""))
                    if val_num >= 50:
                        return "background-color: #63be7b"
                    elif val_num >= 25:
                        return "background-color: #ffeb84"
                    elif val_num > 0:
                        return "background-color: #f4b084"
                    else:
                        return "background-color: #e26b5d"
                except:
                    return ""
            return ""

        styled = (
            df_formatado.style
            .applymap(cor_mapa, subset=["MÉDIA POR ORGANIZAÇÃO (%)"])
            .set_properties(**{"text-align": "center"})
        )

        st.dataframe(styled, use_container_width=True)

        # --------- GRÁFICOS DE MÉDIAS POR TECNOLOGIA E TIPO --------------
        grafico_trator = "grafico_trator.png"
        grafico_pulv = "grafico_pulverizador.png"
        grafico_colh = "grafico_colheitadeira.png"

        fig_trator, fig_pulv, fig_colh = None, None, None  # Inicializa fora do loop

        for tipo in ["Trator", "Pulverizador", "Colheitadeira"]:
            df_tipo = df[df["Tipo_Cat"] == tipo]
            medias_tecnologias = df_tipo[colunas_por_tipo[tipo]].mean().fillna(0)
            total_maquinas_tipo = df_tipo.shape[0]

            labels_originais = medias_tecnologias.index.str.replace(f"_{tipo}", "")
            labels_quebrados = [label.replace(" ", "\n") for label in labels_originais]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(labels_quebrados, medias_tecnologias.values, color="yellow")

            for bar in bars:
                altura = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, altura + 0.01, f"{altura * 100:.0f}%", ha='center', fontsize=9)

            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Média (%)")
            ax.set_title(f"{tipo} - Médias das Tecnologias\nQuantidade de máquinas: {total_maquinas_tipo}")
            ax.tick_params(axis='x', labelsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            # Salva os gráficos para uso posterior no PDF
            if tipo == "Trator":
                fig_trator = fig
                fig_trator.savefig(grafico_trator, dpi=300, bbox_inches='tight', pad_inches=0.3)
            elif tipo == "Pulverizador":
                fig_pulv = fig
                fig_pulv.savefig(grafico_pulv, dpi=300, bbox_inches='tight', pad_inches=0.3)
            elif tipo == "Colheitadeira":
                fig_colh = fig
                fig_colh.savefig(grafico_colh, dpi=300, bbox_inches='tight', pad_inches=0.3)

        plt.close()


        # Caminhos e imagens base
        caminho_tabela = "tabela_consolidado.png"
        caminho_pdf = "relatorio_consolidado.pdf"
        fundo_capa = "capa_fundo.png"
        fundo_paginas = "fundo_conteudo.png"

        # Botão para gerar PDF
        if st.button("📄 Gerar PDF Consolidado", key="botao_pdf_consolidado"):

            with st.spinner("Gerando relatório..."):

                # Salvar imagem da tabela exibida
                salvar_tabela_com_matplotlib_cons(df_formatado, caminho_tabela)

                # Informações do cabeçalho
                total_orgs = df_final.shape[0] - 1 if "TOTAL" in df_final.index else df_final.shape[0]
                total_tratores = int(df_final.loc["TOTAL", "Trator"])
                total_pulvs = int(df_final.loc["TOTAL", "Pulverizador"])
                total_colhs = int(df_final.loc["TOTAL", "Colheitadeira"])
                data_inicio = pd.to_datetime(df["Data de Início"], dayfirst=True).min().strftime("%d/%m/%Y")
                data_fim = pd.to_datetime(df["Data Final"], dayfirst=True).max().strftime("%d/%m/%Y")

                # Função de geração do PDF
                def gerar_pdf_completo_cons(
                    caminho_tabela,
                    graficos_dict,
                    capa_fundo,
                    fundo_paginas,
                    total_orgs,
                    total_tratores,
                    total_pulvs,
                    total_colhs,
                    data_inicio,
                    data_fim,
                    tatica,
                    caminho_saida="relatorio_consolidado.pdf"
                ):
                    pdf = PDFConsolidado(capa_fundo=capa_fundo, fundo_paginas=fundo_paginas)
                    pdf.add_capa(
                        total_orgs=total_orgs,
                        total_tratores=total_tratores,
                        total_pulvs=total_pulvs,
                        total_colhs=total_colhs,
                        data_inicio=data_inicio,
                        data_fim=data_fim,
                        tatica=tatica_selecionada
                    )
                    pdf.add_tabela(caminho_tabela)
                    pdf.add_graficos_com_titulo(graficos_dict)
                    pdf.output(caminho_saida)

                # Gera o PDF consolidado com os gráficos corretos
                gerar_pdf_completo_cons(
                    caminho_tabela=caminho_tabela,
                    graficos_dict={
                        "Trator": grafico_trator,
                        "Pulverizador": grafico_pulv,
                        "Colheitadeira": grafico_colh,
                    },
                    capa_fundo=fundo_capa,
                    fundo_paginas=fundo_paginas,
                    total_orgs=total_orgs,
                    total_tratores=total_tratores,
                    total_pulvs=total_pulvs,
                    total_colhs=total_colhs,
                    data_inicio=data_inicio,
                    data_fim=data_fim,
                    tatica=tatica_selecionada,
                    caminho_saida=caminho_pdf
                )

            # Botão para baixar PDF
            with open(caminho_pdf, "rb") as f:
                st.download_button(
                    label="📥 Baixar PDF Consolidado",
                    data=f,
                    file_name="relatorio_UtilizacaoTecConsolidado.pdf",
                    mime="application/pdf"
                )

else:
    st.info("Aguardando upload do arquivo Excel.")

