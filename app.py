import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# Configuração da página para ocupar toda a largura e ter ícone
st.set_page_config(page_title="Asset Manager Pro", layout="wide", page_icon="📈")

# Estilo CSS customizado para dar uma "cara" de dashboard
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. FUNÇÕES DE VALUATION (AÇÕES)
# ==========================================
def buscar_dividendos_ultimos_5_anos(ticker):
    url = f"https://playinvest.com.br/dividendos/{ticker.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resposta = requests.get(url, headers=headers, timeout=5)
        if resposta.status_code != 200: return None
    except: return None

    soup = BeautifulSoup(resposta.text, 'html.parser')
    container = soup.find("div", class_="card featured-card per-year-chart")
    if not container: return None
    tabela = container.find("table")
    if not tabela: return None

    linhas = tabela.find("tbody").find_all("tr")
    dados = []
    for linha in linhas:
        colunas = linha.find_all("td")
        if len(colunas) >= 2:
            try:
                ano = int(colunas[0].text.strip())
                valor = float(colunas[1].text.strip().replace("R$", "").replace(",", "."))
                dados.append((ano, valor))
            except: continue

    if not dados: return None
    dados.sort(key=lambda x: x[0], reverse=True)
    ultimos_5 = dados[:5]
    if not ultimos_5: return None
    media = sum([v for _, v in ultimos_5]) / len(ultimos_5)
    return {"media": round(media, 4), "historico": ultimos_5}

def extrair_dados_valuation(ticker, taxa_bazin, taxa_gordon, taxa_crescimento):
    url = f"https://investidor10.com.br/acoes/{ticker.lower()}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resposta = requests.get(url, headers=headers, timeout=5)
        if resposta.status_code != 200: return None
    except: return None
    soup = BeautifulSoup(resposta.text, 'html.parser')

    def get_text(soup, title):
        el = soup.find("span", title=title)
        if el:
            body = el.find_parent("div").find_next("div", class_="_card-body")
            val = body.find("span") if body else None
            return val.text.strip().replace('%', '').replace(',', '.') if val else "0"
        return "0"
    
    def get_val_by_label(soup, label):
        el = soup.find(string=re.compile(fr"(?i){label}"))
        if el:
            val = el.find_parent().find_next("div", class_="value")
            return val.span.text.strip().replace('%', '').replace(',', '.') if val else "0"
        return "0"

    try:
        pl = float(get_text(soup, "P/L"))
        dy = float(get_text(soup, "DY"))
        vpa = float(get_val_by_label(soup, "VPA"))
        cotacao = soup.find("div", class_="_card cotacao")
        preco = float(cotacao.find("div", class_="_card-body").span.text.strip().replace("R$", "").replace(",", ".")) if cotacao else 0.0
        
        dados_divs = buscar_dividendos_ultimos_5_anos(ticker)
        historico_raw = []
        if dados_divs:
            dpa = dados_divs["media"]
            historico_raw = dados_divs["historico"]
        else:
            dpa = (dy / 100) * preco 
            historico_raw = []

        preco_bazin = round(dpa / taxa_bazin, 2) if dpa > 0 else 0
        lpa = round(preco / pl, 2) if pl > 0 else 0
        preco_graham = round(math.sqrt(22.5 * lpa * vpa), 2) if lpa > 0 and vpa > 0 else 0
        taxa_liq = taxa_gordon - taxa_crescimento
        preco_gordon = round(dpa / taxa_liq, 2) if dpa > 0 and taxa_liq > 0 else 0

        def calc_margem(teto): return round(((teto - preco) / preco) * 100, 2) if teto > 0 else 0

        return {
            "Ticker": ticker.upper(), "Preço Atual": preco, "DPA Est.": dpa,
            "Graham": preco_graham, "Margem Graham (%)": calc_margem(preco_graham),
            "Bazin": preco_bazin, "Margem Bazin (%)": calc_margem(preco_bazin),
            "Gordon": preco_gordon, "Margem Gordon (%)": calc_margem(preco_gordon),
            "Historico_Raw": historico_raw
        }
    except: return None

# ==========================================
# 2. FUNÇÕES DE MARKOWITZ
# ==========================================
def calcular_cagr(serie, fator_anual):
    if len(serie) < 1: return 0.0
    retorno_total = (1 + serie).prod()
    n = len(serie)
    if fator_anual == 1: return retorno_total - 1
    expoente = fator_anual / n
    try:
        return (retorno_total ** expoente) - 1
    except:
        return 0.0

def gerar_tabela_performance(df_retornos, fator_anual):
    stats = []
    for ativo in df_retornos.columns:
        serie = df_retornos[ativo]
        ret_total = calcular_cagr(serie, fator_anual)
        
        periodos_12m = 12 if fator_anual == 12 else 252
        if len(serie) >= periodos_12m:
            ret_12m = calcular_cagr(serie.tail(periodos_12m), fator_anual)
        else: ret_12m = np.nan
            
        periodos_24m = 24 if fator_anual == 12 else 504
        if len(serie) >= periodos_24m:
            ret_24m = calcular_cagr(serie.tail(periodos_24m), fator_anual)
        else: ret_24m = np.nan
            
        stats.append({
            "Ativo": ativo,
            "Média Histórica (Total)": ret_total * 100,
            "Últimos 12 Meses": ret_12m * 100 if not np.isnan(ret_12m) else None,
            "Últimos 24 Meses": ret_24m * 100 if not np.isnan(ret_24m) else None
        })
    return pd.DataFrame(stats)

def calc_portfolio(w, r, cov, rf):
    rp = np.sum(w * r)
    vp = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    sp = (rp - rf) / vp if vp > 0 else 0
    return rp, vp, sp

def min_sp(w, r, c, rf): return -calc_portfolio(w, r, c, rf)[2]
def min_vol(w, r, c, rf): return calc_portfolio(w, r, c, rf)[1]

def monte_carlo(mu_anual, vol_anual, valor_ini, aporte_mensal_ini, anos, inflacao_anual, n_sim=500):
    if np.isnan(mu_anual) or np.isnan(vol_anual) or vol_anual == 0:
        return np.zeros(12), np.zeros(12), np.zeros(12), 12

    dt = 1/12
    steps = int(anos * 12)
    caminhos = np.zeros((n_sim, steps + 1))
    caminhos[:, 0] = valor_ini
    aporte_atual = aporte_mensal_ini
    
    for t in range(1, steps + 1):
        if t > 1 and (t-1) % 12 == 0: aporte_atual *= (1 + inflacao_anual)
        z = np.random.normal(0, 1, n_sim)
        drift = (mu_anual - 0.5 * vol_anual**2) * dt
        diffusion = vol_anual * np.sqrt(dt) * z
        caminhos[:, t] = caminhos[:, t-1] * np.exp(drift + diffusion) + aporte_atual
        
    return np.percentile(caminhos, 95, axis=0), np.percentile(caminhos, 50, axis=0), np.percentile(caminhos, 5, axis=0), steps

def gerar_hover_text(nome, ret, vol, sharpe, pesos, ativos):
    texto = f"<b>{nome}</b><br>"
    texto += f"Retorno: {ret:.1%}<br>"
    texto += f"Risco (Vol): {vol:.1%}<br>"
    texto += f"Sharpe: {sharpe:.2f}<br><br>"
    texto += "<b>Composição:</b><br>"
    for i, ativo in enumerate(ativos):
        if pesos[i] > 0.01:
            texto += f"{ativo}: {pesos[i]:.1%}<br>"
    return texto

# ==========================================
# 3. INTERFACE PRINCIPAL
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4256/4256900.png", width=100) # Ícone decorativo
st.sidebar.title("Asset Manager Pro")
st.sidebar.markdown("---")
opcao = st.sidebar.radio("Ferramentas:", ["Valuation Fundamentalista", "Markowitz (Otimização)"])

if opcao == "Valuation Fundamentalista":
    st.title("📊 Valuation de Ações")
    st.markdown("Comparativo de preço justo via métodos consagrados.")
    
    with st.container(border=True):
        st.subheader("⚙️ Parâmetros")
        c1, c2, c3 = st.columns(3)
        tb = c1.number_input("Taxa Bazin (Dec)", 0.01, 0.50, 0.08, format="%.2f")
        tg = c2.number_input("Taxa Gordon (Dec)", 0.01, 0.50, 0.12, format="%.2f")
        tc = c3.number_input("Cresc. Gordon (Dec)", 0.00, 0.10, 0.02, format="%.2f")
        tickers = st.text_area("Tickers (separados por vírgula)", "BBAS3, ITSA4, WEG3, VALE3, TAEE11")
    
    if st.button("🚀 Calcular Valuation", type="primary"):
        lista = [t.strip() for t in tickers.split(',') if t.strip()]
        res_valuation = []
        res_dividendos = [] 
        bar = st.progress(0)
        
        for i, tick in enumerate(lista):
            dados = extrair_dados_valuation(tick, tb, tg, tc)
            if dados:
                hist = dados.pop("Historico_Raw") 
                res_valuation.append(dados)
                linha_div = {"Ticker": dados["Ticker"], "Média Usada": dados["DPA Est."]}
                for ano, valor in hist: linha_div[str(ano)] = valor
                res_dividendos.append(linha_div)
            bar.progress((i+1)/len(lista))
            
        if res_valuation:
            df = pd.DataFrame(res_valuation)
            
            # DASHBOARD VISUAL
            st.markdown("### 🎯 Resultados")
            
            # Gráfico de Barras Comparativo
            tickers_list = df['Ticker'].tolist()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tickers_list, y=df['Preço Atual'], name='Preço Atual', marker_color='gray'))
            fig.add_trace(go.Bar(x=tickers_list, y=df['Graham'], name='Graham', marker_color='#2ecc71'))
            fig.add_trace(go.Bar(x=tickers_list, y=df['Bazin'], name='Bazin', marker_color='#3498db'))
            fig.add_trace(go.Bar(x=tickers_list, y=df['Gordon'], name='Gordon', marker_color='#9b59b6'))
            
            fig.update_layout(barmode='group', title="Comparativo: Preço Atual vs Preço Justo",
                              yaxis_tickprefix="R$ ", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Tabela Formatada
            st.markdown("#### Detalhamento Numérico")
            format_dict = {
                "Preço Atual": "R$ {:.2f}", "DPA Est.": "R$ {:.4f}",
                "Graham": "R$ {:.2f}", "Bazin": "R$ {:.2f}", "Gordon": "R$ {:.2f}",
                "Margem Graham (%)": "{:.2f}%", "Margem Bazin (%)": "{:.2f}%", "Margem Gordon (%)": "{:.2f}%"
            }
            cols_existentes = {k: v for k, v in format_dict.items() if k in df.columns}
            st.dataframe(df.style.format(cols_existentes), use_container_width=True)
            
            with st.expander("🔍 Auditoria de Dividendos (Histórico usado)", expanded=False):
                if res_dividendos:
                    df_divs = pd.DataFrame(res_dividendos).set_index("Ticker")
                    st.dataframe(df_divs.style.format("R$ {:.4f}", na_rep="-"), use_container_width=True)
        else: st.warning("Nenhum dado encontrado.")

elif opcao == "Markowitz (Otimização)":
    st.title("📉 Otimizador de Portfólio")
    
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        arquivo = c1.file_uploader("📂 Upload do Excel", type=['xlsx'])
        with c2:
            st.markdown("**Calibragem**")
            tipo_dados = st.radio("Conteúdo:", ["Preços Históricos (R$)", "Retornos (%)"], horizontal=True)
            freq_option = st.selectbox("Periodicidade:", ["Diário (252)", "Mensal (12)", "Sem Anualização"])
            if freq_option.startswith("Diário"): fator_anual = 252
            elif freq_option.startswith("Mensal"): fator_anual = 12
            else: fator_anual = 1

    if 'otimizacao_feita' not in st.session_state: st.session_state.otimizacao_feita = False
    
    if arquivo:
        try:
            df_raw = pd.read_excel(arquivo)
            cols_numericas = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            cols_selecionadas = st.multiselect("Selecione os ATIVOS:", options=df_raw.columns, default=cols_numericas)
            
            if len(cols_selecionadas) < 2:
                st.warning("Selecione pelo menos 2 ativos.")
                st.stop()
                
            df_ativos = df_raw[cols_selecionadas].dropna()
            retornos = df_ativos.pct_change().dropna() if tipo_dados.startswith("Preços") else df_ativos
            
            # CARD DE HISTÓRICO
            st.info("💡 **Dica:** Use o histórico abaixo para ajustar sua 'Visão de Retorno' na próxima etapa.")
            df_perf = gerar_tabela_performance(retornos, fator_anual)
            st.dataframe(df_perf.set_index("Ativo").style.format("{:.2f}%", na_rep="-"), use_container_width=True)

            cov_matrix = retornos.cov() * fator_anual
            media_historica = df_perf["Média Histórica (Total)"].values
            
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()
        
        st.markdown("### 🎛️ Configuração do Otimizador")
        with st.container(border=True):
            df_config = pd.DataFrame({
                "Ativo": cols_selecionadas,
                "Peso Atual (%)": [round(100/len(cols_selecionadas), 2)] * len(cols_selecionadas),
                "Visão Retorno (%)": [round(m, 2) for m in media_historica], 
                "Min (%)": [0.0] * len(cols_selecionadas),
                "Max (%)": [100.0] * len(cols_selecionadas)
            })
            config_editada = st.data_editor(df_config, num_rows="fixed", hide_index=True, use_container_width=True)
            rf_input = st.number_input("Taxa Livre de Risco (%)", 0.0, 50.0, 10.0, format="%.2f") / 100
        
        if st.button("✨ Executar Otimização", type="primary"):
            visoes = config_editada["Visão Retorno (%)"].values / 100
            pesos_user = config_editada["Peso Atual (%)"].values / 100
            bounds = [(r["Min (%)"]/100, r["Max (%)"]/100) for _, r in config_editada.iterrows()]
            
            if sum(pesos_user) != 1: pesos_user = pesos_user / sum(pesos_user)
            n = len(cols_selecionadas)
            w0 = np.ones(n) / n
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            try:
                # Otimizações
                res_opt = minimize(min_sp, w0, args=(visoes, cov_matrix, rf_input), method='SLSQP', bounds=bounds, constraints=constraints)
                w_opt = res_opt.x
                r_opt, v_opt, s_opt = calc_portfolio(w_opt, visoes, cov_matrix, rf_input)
                
                r_user, v_user, s_user = calc_portfolio(pesos_user, visoes, cov_matrix, rf_input)
                res_min = minimize(min_vol, w0, args=(visoes, cov_matrix, rf_input), method='SLSQP', bounds=bounds, constraints=constraints)
                r_min, v_min, s_min = calc_portfolio(res_min.x, visoes, cov_matrix, rf_input)

                if np.isnan(r_opt): st.error("Erro matemático.")
                else:
                    st.session_state.otimizacao_feita = True
                    st.session_state.resultados = {
                        'ativos_lista': cols_selecionadas,
                        'r_opt': r_opt, 'v_opt': v_opt, 's_opt': s_opt, 'w_opt': w_opt,
                        'r_user': r_user, 'v_user': v_user, 's_user': s_user,
                        'r_min': r_min, 'v_min': v_min,
                        'visoes': visoes, 'bounds': bounds, 'cov': cov_matrix,
                        'pesos_user': pesos_user
                    }
            except Exception as e: st.error(f"Erro no Solver: {e}")

        if st.session_state.otimizacao_feita:
            res = st.session_state.resultados
            
            # --- DASHBOARD DE RESULTADOS ---
            st.markdown("---")
            st.markdown("### 🏆 Resultados da Otimização")
            
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Melhor Sharpe", f"{res['s_opt']:.2f}")
            col_kpi2.metric("Retorno Esperado", f"{res['r_opt']:.1%}")
            col_kpi3.metric("Volatilidade (Risco)", f"{res['v_opt']:.1%}")
            
            c_chart1, c_chart2 = st.columns([2, 1])
            
            with c_chart1:
                # Fronteira Eficiente
                max_ret = max(res['visoes']) 
                if max_ret > 2.0: max_ret = 2.0 
                if max_ret < res['r_opt']: max_ret = res['r_opt'] * 1.05
                
                rets_target = np.linspace(res['r_min'], max_ret, 40)
                vol_curve, ret_curve, hover_texts = [], [], []
                
                # ... (Lógica da fronteira mantida) ...
                n_salvo = len(res['ativos_lista'])
                w0_grafico = np.ones(n_salvo) / n_salvo
                for r_target in rets_target:
                    cons_curve = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                                  {'type': 'eq', 'fun': lambda x: calc_portfolio(x, res['visoes'], res['cov'], rf_input)[0] - r_target})
                    result = minimize(min_vol, w0_grafico, args=(res['visoes'], res['cov'], rf_input), method='SLSQP', bounds=res['bounds'], constraints=cons_curve)
                    if result.success:
                        r_c, v_c, s_c = calc_portfolio(result.x, res['visoes'], res['cov'], rf_input)
                        ret_curve.append(r_c); vol_curve.append(v_c)
                        hover_texts.append(gerar_hover_text("Ponto na Curva", r_c, v_c, s_c, result.x, res['ativos_lista']))

                fig = go.Figure()
                if len(vol_curve) > 0:
                    fig.add_trace(go.Scatter(x=vol_curve, y=ret_curve, mode='lines', name='Fronteira', line=dict(color='#2980b9'), hoverinfo='text', text=hover_texts))
                
                fig.add_trace(go.Scatter(x=[res['v_opt']], y=[res['r_opt']], mode='markers', marker=dict(size=14, color='#f1c40f', line=dict(width=1, color='black')), name='Melhor Sharpe', hoverinfo='text', text=gerar_hover_text("Ideal", res['r_opt'], res['v_opt'], res['s_opt'], res['w_opt'], res['ativos_lista'])))
                fig.add_trace(go.Scatter(x=[res['v_user']], y=[res['r_user']], mode='markers', marker=dict(size=12, color='black', symbol='x'), name='Sua Carteira', hoverinfo='text', text=gerar_hover_text("Atual", res['r_user'], res['v_user'], res['s_user'], res['pesos_user'], res['ativos_lista'])))
                
                fig.update_layout(title="Fronteira Eficiente", xaxis_title="Risco", yaxis_title="Retorno", template="plotly_white", xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"), height=400)
                st.plotly_chart(fig, use_container_width=True)

            with c_chart2:
                # Gráfico de Rosca (Donut) para Alocação
                fig_pie = go.Figure(data=[go.Pie(labels=res['ativos_lista'], values=res['w_opt'], hole=.4)])
                fig_pie.update_layout(title="Alocação Ideal Sugerida", height=400, showlegend=False)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.subheader("🔮 Simulação de Futuro (Monte Carlo)")
            
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                # INPUTS FORMATADOS NO DISPLAY (Streamlit não formata a digitação, mas o default fica bonito)
                inv_ini = c1.number_input("Investimento Inicial (R$)", value=10000.0, format="%.2f")
                aporte = c2.number_input("Aporte Mensal (R$)", value=1000.0, format="%.2f")
                anos = c3.number_input("Anos", value=10)
                inflacao = c4.number_input("Inflação Média (%)", value=5.0, format="%.2f") / 100
            
            if st.button("🎲 Rodar Simulação", type="primary"):
                if np.isnan(res['r_opt']): st.error("Erro nos dados.")
                else:
                    opt_top, opt_mid, opt_low, steps = monte_carlo(res['r_opt'], res['v_opt'], inv_ini, aporte, int(anos), inflacao)
                    usr_top, usr_mid, usr_low, _ = monte_carlo(res['r_user'], res['v_user'], inv_ini, aporte, int(anos), inflacao)
                    
                    x = np.linspace(0, int(anos), steps + 1)
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_mid, mode='lines', name='Ideal (Esperado)', line=dict(color='#27ae60', width=3)))
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_low, mode='lines', name='Ideal (Pessimista)', line=dict(color='#abebc6', width=0), fill='tonexty')) # Area chart trick? No, simple line
                    fig_sim.add_trace(go.Scatter(x=x, y=usr_mid, mode='lines', name='Atual (Esperado)', line=dict(color='black', dash='dash')))
                    
                    fig_sim.update_layout(title="Projeção Patrimonial", xaxis_title="Anos", yaxis_title="Patrimônio", template="plotly_white", 
                                          hovermode="x unified", separators=",.", yaxis=dict(tickprefix="R$ ", tickformat=",.0f"))
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    final_val = opt_mid[-1]
                    st.success(f"💰 **Patrimônio Final Estimado (Carteira Ideal):** R$ {final_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))