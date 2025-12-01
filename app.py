import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import hmac

# ==========================================
# 0. CONFIGURAÇÃO E SEGURANÇA
# ==========================================
st.set_page_config(page_title="Asset Manager Pro", layout="wide", page_icon="📈")

# CSS para o visual "Dashboard Premium"
st.markdown("""
<style>
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stMetric label {
        font-size: 16px !important;
        color: #666;
    }
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIN ---
def check_password():
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("🔒 Sistema Protegido")
        st.text_input("Digite a senha de acesso:", type="password", on_change=password_entered, key="password")
        if "password_correct" in st.session_state:
            st.error("Senha incorreta.")
    return False

if not check_password():
    st.stop()

# ==========================================
# 1. FUNÇÕES DE CÁLCULO
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

def calcular_cagr(serie, fator_anual):
    if len(serie) < 1: return 0.0
    retorno_total = (1 + serie).prod()
    n = len(serie)
    if fator_anual == 1: return retorno_total - 1
    expoente = fator_anual / n
    try: return (retorno_total ** expoente) - 1
    except: return 0.0

def gerar_tabela_performance(df_retornos, fator_anual):
    stats = []
    for ativo in df_retornos.columns:
        serie = df_retornos[ativo]
        ret_total = calcular_cagr(serie, fator_anual)
        periodos_12m = 12 if fator_anual == 12 else 252
        ret_12m = calcular_cagr(serie.tail(periodos_12m), fator_anual) if len(serie) >= periodos_12m else np.nan
        periodos_24m = 24 if fator_anual == 12 else 504
        ret_24m = calcular_cagr(serie.tail(periodos_24m), fator_anual) if len(serie) >= periodos_24m else np.nan
        stats.append({
            "Ativo": ativo, "Média Histórica (Total)": ret_total * 100,
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
    
    # Cálculo Determinístico (Tira-Teima: Juros Compostos Puros)
    linha_teorica = np.zeros(steps + 1)
    linha_teorica[0] = valor_ini
    taxa_mensal_equiv = (1 + mu_anual)**(1/12) - 1 # Converte taxa anual para mensal
    aporte_teorico = aporte_mensal_ini

    for t in range(1, steps + 1):
        if t > 1 and (t-1) % 12 == 0: 
            aporte_atual *= (1 + inflacao_anual)
            aporte_teorico *= (1 + inflacao_anual)
            
        # Monte Carlo (Com Volatilidade)
        z = np.random.normal(0, 1, n_sim)
        drift = (mu_anual - 0.5 * vol_anual**2) * dt
        diffusion = vol_anual * np.sqrt(dt) * z
        caminhos[:, t] = caminhos[:, t-1] * np.exp(drift + diffusion) + aporte_atual
        
        # Teórico (Sem Volatilidade)
        linha_teorica[t] = linha_teorica[t-1] * (1 + taxa_mensal_equiv) + aporte_teorico

    return np.percentile(caminhos, 95, axis=0), np.percentile(caminhos, 50, axis=0), np.percentile(caminhos, 5, axis=0), steps, linha_teorica

def gerar_hover_text(nome, ret, vol, sharpe, pesos, ativos):
    texto = f"<b>{nome}</b><br>Retorno: {ret:.1%}<br>Risco: {vol:.1%}<br>Sharpe: {sharpe:.2f}<br><br><b>Alocação:</b><br>"
    for i, ativo in enumerate(ativos):
        if pesos[i] > 0.01: texto += f"{ativo}: {pesos[i]:.1%}<br>"
    return texto

# ==========================================
# 3. INTERFACE E NAVEGAÇÃO
# ==========================================
st.sidebar.title("Asset Manager Pro")
opcao = st.sidebar.radio("Menu:", ["🏠 Início", "📊 Valuation (Ações)", "📉 Otimização (Markowitz)"])

if opcao == "🏠 Início":
    st.title("Asset Manager Pro")
    st.markdown("Bem-vindo ao seu sistema de gestão de ativos.")
    col1, col2 = st.columns(2)
    with col1:
        st.info("📊 **Valuation:** Calcule o preço justo de ações (Graham, Bazin, Gordon).")
    with col2:
        st.info("📉 **Otimização:** Encontre a carteira ideal e simule o futuro.")

elif opcao == "📊 Valuation (Ações)":
    st.title("📊 Valuation Fundamentalista")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        tb = c1.number_input("Taxa Bazin (Dec)", 0.01, 0.50, 0.08, step=0.01, format="%.2f")
        tg = c2.number_input("Taxa Gordon (Dec)", 0.01, 0.50, 0.12, step=0.01, format="%.2f")
        tc = c3.number_input("Cresc. Gordon (Dec)", 0.00, 0.10, 0.02, step=0.01, format="%.2f")
        tickers = st.text_area("Tickers", "BBAS3, ITSA4, WEG3")
    
    if st.button("Calcular", type="primary"):
        lista = [t.strip() for t in tickers.split(',') if t.strip()]
        res_valuation = []
        bar = st.progress(0)
        for i, tick in enumerate(lista):
            dados = extrair_dados_valuation(tick, tb, tg, tc)
            if dados: res_valuation.append(dados)
            bar.progress((i+1)/len(lista))
            
        if res_valuation:
            df = pd.DataFrame(res_valuation)
            st.markdown("### Resultados")
            
            fig = go.Figure()
            tickers_l = df['Ticker'].tolist()
            fig.add_trace(go.Bar(x=tickers_l, y=df['Preço Atual'], name='Preço Atual', marker_color='#7f8c8d', text=df['Preço Atual'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=tickers_l, y=df['Graham'], name='Graham', marker_color='#27ae60', text=df['Graham'], texttemplate='R$ %{y:.2f}'))
            fig.update_layout(title="Preço Atual vs Justo", barmode='group', template="plotly_white", yaxis_tickprefix="R$ ")
            st.plotly_chart(fig, use_container_width=True)

            format_dict = {"Preço Atual": "R$ {:.2f}", "Graham": "R$ {:.2f}", "Bazin": "R$ {:.2f}", "Gordon": "R$ {:.2f}", "Margem Graham (%)": "{:.2f}%"}
            cols = {k: v for k, v in format_dict.items() if k in df.columns}
            st.dataframe(df.style.format(cols), use_container_width=True)
        else: st.warning("Sem dados.")

elif opcao == "📉 Otimização (Markowitz)":
    st.title("📉 Otimizador de Portfólio")
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        arquivo = c1.file_uploader("Upload Excel", type=['xlsx'])
        with c2:
            tipo_dados = st.radio("Conteúdo:", ["Preços (R$)", "Retornos (%)"], horizontal=True)
            freq_option = st.selectbox("Periodicidade:", ["Diário (252)", "Mensal (12)", "Sem Anualização"])
            if freq_option.startswith("Diário"): fator_anual = 252
            elif freq_option.startswith("Mensal"): fator_anual = 12
            else: fator_anual = 1

    if 'otimizacao_feita' not in st.session_state: st.session_state.otimizacao_feita = False
    
    if arquivo:
        try:
            df_raw = pd.read_excel(arquivo)
            cols_numericas = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            cols_selecionadas = st.multiselect("Selecione ATIVOS:", options=df_raw.columns, default=cols_numericas)
            if len(cols_selecionadas) < 2: st.error("Mínimo 2 ativos."); st.stop()
            df_ativos = df_raw[cols_selecionadas].dropna()
            retornos = df_ativos.pct_change().dropna() if tipo_dados.startswith("Preços") else df_ativos
            
            df_perf = gerar_tabela_performance(retornos, fator_anual)
            with st.expander("Histórico de Rentabilidade", expanded=False):
                st.dataframe(df_perf.set_index("Ativo").style.format("{:.2f}%", na_rep="-"), use_container_width=True)
            cov_matrix = retornos.cov() * fator_anual
            media_historica = df_perf["Média Histórica (Total)"].values
        except Exception as e: st.error(f"Erro: {e}"); st.stop()
        
        with st.container(border=True):
            st.markdown("#### Premissas")
            df_config = pd.DataFrame({
                "Ativo": cols_selecionadas,
                "Peso Atual (%)": [round(100/len(cols_selecionadas), 2)] * len(cols_selecionadas),
                "Visão Retorno (%)": [round(m, 2) for m in media_historica], 
                "Min (%)": [0.0] * len(cols_selecionadas),
                "Max (%)": [100.0] * len(cols_selecionadas)
            })
            config_editada = st.data_editor(df_config, num_rows="fixed", hide_index=True, use_container_width=True)
            rf_input = st.number_input("Taxa Livre de Risco (%)", 0.0, 50.0, 10.0) / 100
        
        if st.button("🚀 Otimizar", type="primary"):
            visoes = config_editada["Visão Retorno (%)"].values / 100
            pesos_user = config_editada["Peso Atual (%)"].values / 100
            bounds = [(r["Min (%)"]/100, r["Max (%)"]/100) for _, r in config_editada.iterrows()]
            if abs(sum(pesos_user) - 100) > 1: pesos_user = pesos_user / sum(pesos_user)
            elif sum(pesos_user) > 10: pesos_user = pesos_user / 100
            n = len(cols_selecionadas); w0 = np.ones(n) / n
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            try:
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
                        'ativos_lista': cols_selecionadas, 'r_opt': r_opt, 'v_opt': v_opt, 's_opt': s_opt, 'w_opt': w_opt,
                        'r_user': r_user, 'v_user': v_user, 's_user': s_user, 'r_min': r_min, 'v_min': v_min,
                        'visoes': visoes, 'bounds': bounds, 'cov': cov_matrix, 'pesos_user': pesos_user
                    }
            except Exception as e: st.error(f"Solver: {e}")

        if st.session_state.otimizacao_feita:
            res = st.session_state.resultados
            st.markdown("### Resultados")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe", f"{res['s_opt']:.2f}")
            col2.metric("Retorno Esp.", f"{res['r_opt']:.1%}")
            col3.metric("Volatilidade", f"{res['v_opt']:.1%}")
            
            c_chart1, c_chart2 = st.columns([2, 1])
            with c_chart1:
                max_ret = max(res['visoes']); 
                if max_ret > 2.0: max_ret = 2.0
                if max_ret < res['r_opt']: max_ret = res['r_opt'] * 1.05
                rets_target = np.linspace(res['r_min'], max_ret, 40)
                vol_curve, ret_curve, hover_texts = [], [], []
                n_salvo = len(res['ativos_lista']); w0_grafico = np.ones(n_salvo) / n_salvo
                for r_target in rets_target:
                    cons_curve = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: calc_portfolio(x, res['visoes'], res['cov'], rf_input)[0] - r_target})
                    result = minimize(min_vol, w0_grafico, args=(res['visoes'], res['cov'], rf_input), method='SLSQP', bounds=res['bounds'], constraints=cons_curve)
                    if result.success:
                        r_c, v_c, s_c = calc_portfolio(result.x, res['visoes'], res['cov'], rf_input)
                        ret_curve.append(r_c); vol_curve.append(v_c)
                        hover_texts.append(gerar_hover_text("Curva", r_c, v_c, s_c, result.x, res['ativos_lista']))
                fig = go.Figure()
                if len(vol_curve) > 0: fig.add_trace(go.Scatter(x=vol_curve, y=ret_curve, mode='lines', name='Fronteira', line=dict(color='#3498db', width=3), hoverinfo='text', text=hover_texts))
                fig.add_trace(go.Scatter(x=[res['v_opt']], y=[res['r_opt']], mode='markers', marker=dict(size=15, color='#f1c40f', line=dict(width=2, color='black')), name='Ideal', hoverinfo='text', text=gerar_hover_text("Ideal", res['r_opt'], res['v_opt'], res['s_opt'], res['w_opt'], res['ativos_lista'])))
                fig.add_trace(go.Scatter(x=[res['v_user']], y=[res['r_user']], mode='markers', marker=dict(size=12, color='black', symbol='x'), name='Atual', hoverinfo='text', text=gerar_hover_text("Atual", res['r_user'], res['v_user'], res['s_user'], res['pesos_user'], res['ativos_lista'])))
                fig.update_layout(xaxis_title="Risco", yaxis_title="Retorno", template="plotly_white", xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"), height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c_chart2:
                fig_pie = go.Figure(data=[go.Pie(labels=res['ativos_lista'], values=res['w_opt'], hole=.4)])
                fig_pie.update_layout(title="Alocação Ideal", height=400, showlegend=False)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### 🔮 Projeção Monte Carlo")
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                inv_ini = c1.number_input("Inicial (R$)", 10000.0)
                aporte = c2.number_input("Mensal (R$)", 1000.0)
                anos = c3.number_input("Anos", 10)
                inflacao = c4.number_input("Inflação (%)", 5.0) / 100
            
            if st.button("Simular", type="primary"):
                if np.isnan(res['r_opt']): st.error("Erro.")
                else:
                    opt_top, opt_mid, opt_low, steps, linha_teorica = monte_carlo(res['r_opt'], res['v_opt'], inv_ini, aporte, int(anos), inflacao)
                    usr_top, usr_mid, usr_low, _, _ = monte_carlo(res['r_user'], res['v_user'], inv_ini, aporte, int(anos), inflacao)
                    x = np.linspace(0, int(anos), steps + 1)
                    
                    fig_sim = go.Figure()
                    # Linha Tira-Teima (Teórica)
                    fig_sim.add_trace(go.Scatter(x=x, y=linha_teorica, mode='lines', name='Teórico (Sem Volatilidade)', line=dict(color='#f1c40f', width=2, dash='dot')))
                    
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_mid, mode='lines', name='Ideal (Esperado)', line=dict(color='#27ae60', width=3)))
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_low, mode='lines', name='Ideal (Pessimista)', line=dict(color='#abebc6', width=0), fill='tonexty'))
                    fig_sim.add_trace(go.Scatter(x=x, y=usr_mid, mode='lines', name='Atual (Esperado)', line=dict(color='black', dash='dash')))
                    
                    fig_sim.update_layout(xaxis_title="Anos", yaxis_title="Patrimônio", template="plotly_white", hovermode="x unified", separators=",.", yaxis=dict(tickprefix="R$ ", tickformat=",.0f"))
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    st.success(f"💰 **Patrimônio Estimado (Cenário Ideal):** R$ {opt_mid[-1]:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
